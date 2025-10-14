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
encoder_dim = 1024
decoder_dim = 1024
context_length = 1024
compression = 1
n_layers = 8
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
	'hidden_size': encoder_dim,
	'intermediate_size': 4*encoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size
}

class AttributableMemoryTransformer(MemoryTransformer):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# overwrite self.cel from parent to prevent loss reduction
		self.cel = nn.CrossEntropyLoss(reduction='none')
	
	def forward(self, input_ids, labels=None, attention_mask=None, occlude_memory=False, noise_embedding=False, **kwargs):
		if self.random:
			input_ids = torch.randint(1, self.n_vocab, input_ids.shape)
		input_ids = input_ids.to(device_id)
		if not self.use_transformer_encoder:
			wte_embeds = self.wte(input_ids)
			x = wte_embeds
			for block in self.encoderblocks:
				x = block(x)
		else:
			x = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
		
		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			if noise_embedding:
				encoder_embedding += torch.rand(encoder_embedding.shape).to(device_id)*(2**-2) # assumes e4m3 target quant
			if self.combination_dim == 'token':
				encoder_embedding = self.up(encoder_embedding)
		decoder_embeds = self.decoder_wte(input_ids)

		if self.combination_dim == 'token':
			if self.decoder_proj:
				encoder_embedding = self.decoder_proj(encoder_embedding)
			if occlude_memory:
				x = torch.cat((torch.zeros(encoder_embedding.shape).to(device_id), decoder_embeds), dim=1)
				if attention_mask is not None:
					attention_mask = torch.cat((torch.zeros(input_ids.shape[0], 1).to(device_id), attention_mask), dim=1)
			else:
				x = torch.cat((encoder_embedding, decoder_embeds), dim=1) # concatenation on token dim
				if attention_mask is not None:
					attention_mask = torch.cat((torch.ones(input_ids.shape[0], 1).to(device_id), attention_mask), dim=1)

		elif self.combination_dim == 'embedding':
			repeat_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
			x = torch.cat((repeat_embedding, decoder_embeds), dim=2) # concatenation on hidden dim
		
		# feed pre-concatenated input embeddings to the transformer decoder
		decoder_input_embeds = x
		#decoder_input_embeds.retain_grad()
		x = self.decoder(inputs_embeds=x, attention_mask=attention_mask)
		output = self.lm_head(x.last_hidden_state)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_labels, shift_logits = labels, output
		if self.combination_dim == 'token':
			shift_logits = output[..., 1:-1].contiguous() # first 'token' is encoding
		else:
			shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous() 
		loss = self.cel(shift_logits, shift_labels)
		return loss, output, decoder_input_embeds

def compute_grad(model, input_ids, attention_mask, token_index):
	_, output, decoder_input_embeds = model.forward(input_ids, attention_mask, occlude_memory=False, noise_embedding=False)
	# output is in shape [b e t]
	isolated_loss = torch.sum(output[..., token_index], dim=(0, 1)) # sum accross embedding and batch dims
	isolated_loss.backward()
	memory_grad = decoder_input_embeds.grad[:, 0, :].detach()
	model.zero_grad()
	return memory_grad.detach(), decoder_input_embeds.detach()

def gradientxinput(model, input_ids, output_measure='l1'):
	"""
	Performs gradientxinput on the decoder of an embedding-augmented model
	"""
	input_ids, attention_mask = torch.tensor(input_ids['input_ids']).to(torch.long).to(device_id), torch.tensor(input_ids['attention_mask']).to(torch.long).to(device_id)

	# expects for input_ids to be a tensor
	#loss, output, decoder_input_embeds = model.forward(input_ids, attention_mask, occlude_memory=False)
	#print (loss.shape, output.shape)
	memory_gradxinputs = []
	for token_index in tqdm(range(1, len(input_ids[0])+1)):
		memory_grad, decoder_input_embeds = compute_grad(model, input_ids, attention_mask, token_index)
		gxo = memory_grad * decoder_input_embeds[:, 0, :]
		memory_gradxinputs.append(torch.sum(torch.abs(gxo), dim=1).to('cpu'))
	memory_gradxinputs = torch.stack(memory_gradxinputs, dim=1)
	return memory_gradxinputs


@torch.no_grad()
def memory_occlusion(model, input_ids, output_measure='l1'):
	"""
	Occlusion attribution on memory embedding
	"""
	if not input_ids or len(input_ids) == 0:
		return torch.tensor([])
	input_ids, attention_mask = torch.tensor(input_ids['input_ids']).to(torch.long).to(device_id), torch.tensor(input_ids['attention_mask']).to(torch.long).to(device_id)
	
	loss, output, _ = model.forward(input_ids, attention_mask, occlude_memory=False)
	occluded_loss, occluded_output, _ = model.forward(input_ids, attention_mask, occlude_memory=True)
	if output_measure == 'l1':
		measure = torch.abs(occluded_output - output).to('cpu')
		measure = torch.sum(measure[..., 1:], dim=1) # measure is in shape [b, e, t] and we want to disregard the embedding output
	elif output_measure == 'cosine':
		cos = nn.CosineSimilarity(dim=1)	
		measure = 1 - cos(occluded_output, output)[..., 1:].to('cpu')
	elif output_measure == 'loss':
		# measure occlusion on unreduced model loss
		measure = torch.abs(occluded_loss - loss).to('cpu')
	print (loss, occluded_loss)
	return measure

@torch.no_grad()
def memory_shift(model, input_ids, output_measure='l1'):
	"""
	Slight occlusion attribution on memory embedding
	"""
	if not input_ids or len(input_ids) == 0:
		return torch.tensor([])
	input_ids, attention_mask = torch.tensor(input_ids['input_ids']).to(torch.long).to(device_id), torch.tensor(input_ids['attention_mask']).to(torch.long).to(device_id)
	
	try:
		loss, output, _ = model.forward(input_ids, attention_mask, occlude_memory=False, noise_embedding=False)
	except:
		return torch.tensor([])
	occluded_loss, occluded_output, _ = model.forward(input_ids, attention_mask, occlude_memory=False, noise_embedding=True)
	if output_measure == 'l1':
		measure = torch.abs(occluded_output - output).to('cpu')
		measure = torch.sum(measure[..., 1:], dim=1) # measure is in shape [b, e, t] and we want to disregard the embedding output
	elif output_measure == 'cosine':
		cos = nn.CosineSimilarity(dim=1)	
		measure = 1 - cos(occluded_output, output)[..., 1:].to('cpu')
	elif output_measure == 'loss':
		# measure occlusion on unreduced model loss
		measure = torch.abs(occluded_loss - loss).to('cpu')
	return measure

def normalize_attributions(attributions, method='minmax'):
	all_estimates = []
	for attribution in attributions:
		if method == 'minmax':
			# attributions are scaled to [0, 1]
			maximums = attribution.max(dim=1).values
			mins = torch.where(attribution>0., attribution, torch.inf).min(dim=1).values
			ranges = maximums - mins
			broadcasted_mins = mins.expand(attribution.shape[1], -1).T
			attribution -= broadcasted_mins 
			broadcasted_ranges = ranges.expand(attribution.shape[1], -1).T
			attribution /= broadcasted_ranges
		all_estimates.append(attribution)
	return all_estimates


if __name__ == '__main__':
	# Initializing a LLaMA model
	configuration = LlamaConfig(**llama_config_kwargs)

	# Initializing a model from the llama-7b style configuration
	encoder_model = LlamaModel(configuration).float()

	#encoder_model = LlamaModel(configuration)
	model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression, n_heads=n_heads, random=False)

<<<<<<< HEAD
	#model = AttributableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression, n_heads=n_heads, random=False) 
	#model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression)
	safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/model.safetensors', strict=True) # no decoder_input_embeds param in original model
	#model.cel = nn.CrossEntropyLoss(reduction='none')
	model.eval()
=======
	model = AttributableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression, n_heads=n_heads, random=False) 
	safetensors.torch.load_model(model, f'{checkpoint_root}/Desktop/fineweb_tmemory_2transformers_e1024c1_d1024_n8_c512_b64x2/checkpoint-200000/model.safetensors', strict=True) # no decoder_input_embeds param in original model

>>>>>>> b6dc15a (minor updates)
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

	train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
	test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"

	# if you have a new dataset, map before loading from disk
	datasets.config.IN_MEMORY_MAX_SIZE = 10e9
	train_dataset = load_from_disk(train_path, keep_in_memory=None)
<<<<<<< HEAD
	test_dataset = load_from_disk(test_path, keep_in_memory=None).take(8132)
=======
	test_dataset = load_from_disk(test_path, keep_in_memory=None).take(32)
>>>>>>> b6dc15a (minor updates)

	n_gpus = torch.cuda.device_count()
	dataset_length = len(test_dataset)
	device_chunk_size = int(dataset_length / n_gpus)
	start, end = device_id * device_chunk_size, (device_id+1) * device_chunk_size
	test_dataset = test_dataset.skip(start).take(end - start)
	mlflow.end_run()
	batch_size = 128
	attributions = []
	ids = []
	if len(test_dataset) % batch_size == 0:
		batches = len(test_dataset) // batch_size
	else:
		batches = len(test_dataset) // (batch_size) + 1


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
	print (trainer.evaluate())

	

	masks = []	
	for sample_index in tqdm(range(batches)):
		batch = test_dataset[sample_index*batch_size:sample_index*batch_size + batch_size]
		mask = torch.tensor(batch['attention_mask'])
<<<<<<< HEAD
		attributions.append(memory_occlusion(model, batch, output_measure='loss') * mask[:, :-1])
=======
		attributions.append(memory_occlusion(model, batch, output_measure='l1') * mask)
>>>>>>> b6dc15a (minor updates)
		#attributions.append(memory_shift(model, batch, output_measure='l1') * mask)
		#attributions.append(gradientxinput(model, batch, output_measure='l1') * mask)
		ids.append(batch['id'])

	tokenizer.pad_token = tokenizer.eos_token
	#attributions = normalize_attributions(attributions)
	print_attributions = {}
	for i, attribution in enumerate(attributions[0]):
		if test_dataset[i]['input_ids'][0] != 1:
			print (torch.mean(attribution), attribution)
			print_attributions[ids[0][i]] = [batch['input_ids'][i], attribution.tolist()]
	d = {'attributions': print_attributions}
<<<<<<< HEAD
	with open('/home/badger/loss_attributions.json', 'w') as f:
=======
	with open('/home/azureuser/largest_l1_attributions.json', 'w') as f:
>>>>>>> b6dc15a (minor updates)
		json.dump(d, f)
	
	attributions_dict = {'attribution': attributions, 'ids': ids}
	attributions_dataset = Dataset.from_dict(attributions_dict)
	#attributions_dataset.save_to_disk(f"{data_root}/fineweb-edu-tokenized-train-occlusion-lpad-8k_{rank}")

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

	
