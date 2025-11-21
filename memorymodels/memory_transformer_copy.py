import os
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import safetensors
import datasets
import warnings
import shutil
from dotenv import load_dotenv
from pathlib import Path

from mixer_autoencoder import AutoencodingMixer, TruncatedModel
from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

def load_encoder():
	encoder_dim = 512
	decoder_dim = 512
	context_length = 512
	compression = 1
	n_layers = 8
	n_heads = 8

	vocab_size = 8000

	llama_config_kwargs = {
		 'hidden_size':decoder_dim,
		 'intermediate_size': 4*decoder_dim,
		 'num_hidden_layers': n_layers,
		 'num_attention_heads': n_heads,
		 'vocab_size': vocab_size
	 }

	# Initializing a LLaMA model
	configuration = LlamaConfig(**llama_config_kwargs)
	encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
	pretrained_autoencoder = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=False)
	load_path = Path(f"{checkpoint_root}/fineweb_autotrans_unroll_1024c1_d1024_n8_c512_b32x2/checkpoint-200000/model.safetensors")

	# load encoder
	safetensors.torch.load_model(pretrained_autoencoder, load_path)
	#encoder = TruncatedModel(pretrained_autoencoder)
	encoder = pretrained_autoencoder.encoder
	return encoder

@torch.no_grad()
def hamming(model_output, labels):
	total_metric = 0
	#ignore_list = [tokenizer.pad_token, tokenizer.encode(tokenizer.eos_token)[-1]]
	input_tokens = labels
	generated_tokens = torch.argmax(model_output, dim=-1)
	nonpad_tokens = torch.where(labels != -100, 1, 0)
	equal_tokens = torch.where(generated_tokens == labels, 1, 0) & nonpad_tokens
	average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
	return average_metric

def compute_hamming_metric(eval_preds):
	logits, labels = eval_preds
	return hamming(logits, labels)

encoder_dim = 512 
decoder_dim = 512
context_length = 256
compression = 1 
n_layers = 16 
n_heads = 4
model = VariableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, 
								  fixed_memory=True, frozen_encoder=None, no_memory=False, copy=True)

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path).take(5000).filter(lambda x: x['input_ids'][-1] != 1, num_proc=16)

batch_size = 32
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_copy_memtrans_c256x4\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	gradient_accumulation_steps=2,
	warmup_steps=50,
	eval_steps=100,
	save_steps=10000,
	learning_rate=2e-4, 
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_metric = compute_hamming_metric
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()

trainer.train()
