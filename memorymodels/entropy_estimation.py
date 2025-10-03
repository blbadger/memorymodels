import os
import shutil
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import torch.nn as nn
import mlflow
import datasets
from datasets import load_dataset, load_from_disk, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import safetensors
from safetensors.torch import save_file, save_model
import copy
import json
import re
from pprint import pprint
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv

warnings.filterwarnings(action='ignore')

torch.set_printoptions(threshold=4)

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'
encoder_dim = 256
decoder_dim = 512
context_length = 1024
compression = 4
n_layers = 16
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
	'hidden_size': encoder_dim,
	'intermediate_size': 4*encoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size
}
loss_fn = nn.CrossEntropyLoss(reduction='none')

def shift_position(input_tensor, batch_size):
	pad = torch.zeros(batch_size).reshape(batch_size, 1)
	return torch.cat((pad, input_tensor), dim=1)[:, :-1]

@torch.no_grad()
def calculate_entropy(model, batch):
	token_length = batch['input_ids'].shape[1] # presuming [b, t]
	batch_size = batch['input_ids'].shape[0]
	target = batch['input_ids'][...,1:]
	input_ids = batch('input_ids')
	attention_mask = batch['attention_mask']
	index_losses = []
	for i in range(token_length):
		unshifted_output, _ = model(input_ids, attention_mask)
		loss = loss_fn(output, target)
		sample_losses = torch.sum(loss, dim=1)
		input_ids =  shift_position(input_ids, batch_size)
		attention_mask = shift_position(attention_mask, batch_size)
		input_ids = shift_position(input_ids)
		target = input_ids
		shifted_output, _ = model(input_ids, attention_mask)
		shifted_loss = loss_fn(shifted_output, target)
		index_losses.append(sample_losses - shifted_loss)

	return index_losses.reverse()


if __name__ == '__main__':
	# Initializing a LLaMA model
	configuration = LlamaConfig(**llama_config_kwargs)

	# Initializing a model from the llama-7b style configuration
	encoder_model = LlamaModel(configuration).float()

	#encoder_model = LlamaModel(configuration)
	model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression, n_heads=n_heads, random=False)

	#model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression)
	safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/model.safetensors', strict=True) # no decoder_input_embeds param in original model

	model.eval()
	use_ddp = False
	if not use_ddp:	
		device_id = 0	
		model = model.to(device)
	else:
		gpu_count = torch.cuda.device_count()
		dist.init_process_group("nccl")
		rank = dist.get_rank()
		device_id = rank % torch.cuda.device_count()
		model = DDP(model.to(device_id), device_ids=[device_id])

	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)

	train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-8k"
	test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"

	# if you have a new dataset, map before loading from disk
	datasets.config.IN_MEMORY_MAX_SIZE = 10e9
	train_dataset = load_from_disk(train_path, keep_in_memory=None)
	test_dataset = load_from_disk(test_path, keep_in_memory=None).take(128)

	n_gpus = torch.cuda.device_count()
	dataset_length = len(test_dataset)
	device_chunk_size = int(dataset_length / n_gpus)
	start, end = device_id * device_chunk_size, (device_id+1) * device_chunk_size
	test_dataset = test_dataset.skip(start).take(end - start)
	mlflow.end_run()
	batch_size = 128
	entropies = []
	ids = []
	if len(test_dataset) % batch_size == 0:
		batches = len(test_dataset) // batch_size
	else:
		batches = len(test_dataset) // (batch_size) + 1

	masks = []	
	for sample_index in tqdm(range(batches)):
		batch = test_dataset[sample_index*batch_size:sample_index*batch_size + batch_size]
		mask = torch.tensor(batch['attention_mask'])
		entropies.append(calculate_entropy(model, batch) * mask[:, :-1])
		ids.append(batch['id'])

	tokenizer.pad_token = tokenizer.eos_token
	#entropies = normalize_entropies(entropies)
	print_entropies = {}
	for i, attribution in enumerate(entropies[0]):
		if test_dataset[i]['input_ids'][0] != 1:
			print_entropies[ids[0][i]] = [batch['input_ids'][i], attribution.tolist()]
	d = {'entropies': print_entropies}
	with open('/home/badger/loss_exact_calculations.json', 'w') as f:
		json.dump(d, f)
	
	entropies_dict = {'attribution': entropies, 'ids': ids}
	entropies_dataset = Dataset.from_dict(entropies_dict)
	#entropies_dataset.save_to_disk(f"{data_root}/fineweb-edu-tokenized-train-occlusion-lpad-8k_{rank}")

	training_arguments = transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=8000,
        learning_rate=2e-4, 
        fp16=True, 
        eval_strategy='steps',
        output_dir='/home/badger',
        optim='adamw_torch',
        overwrite_output_dir=False,
        max_steps=200000
)

	trainer = transformers.Trainer(
        model=model,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )   
	#print (trainer.evaluate())

	
