import os
import shutil
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import torch.nn as nn
import mlflow
import datasets
from datasets import load_dataset, load_from_disk
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import safetensors
from safetensors.torch import save_file, save_model
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import default_observer, default_dynamic_quant_observer
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization import quantize_dynamic
import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt, Linear4bit
import copy
import json
import re

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'
print (device)

import torch

@torch.no_grad()
def insert_identity(model, module=None, temp_path='temp_model'):
	compressed_dim = model.up.weight.shape[1]
	print (model.up.weight.shape)
	identity_weights = torch.eye(compressed_dim)
	identity_transformation = nn.Linear(compressed_dim, compressed_dim, bias=False)
	identity_transformation.weight.data = identity_weights
	if not module:
		model.up = nn.Sequential(identity_transformation, model.up)
	else:
		module = nn.Sequential(identity_transformation, module)
	save_model(model, f'{checkpoint_root}/{temp_path}')
	return model

@torch.no_grad()
def observe_sensitivities(model, cast_to=torch.float8_e4m3fn, weights=False):
	all_results = []
	model_copy = copy.deepcopy(model)
	for name, module in model.named_modules():
		if (not isinstance(module, nn.Linear)) and (not isinstance(module, nn.Embedding)):
			continue
		name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
		copied_module_name = str('model_copy.' + name)
		if cast_to is not None:
			if weights:
				new_layer = module
				new_layer.weight = nn.Parameter(module.weight.to(cast_to).to(torch.float32))
			else:
				new_layer = nn.Sequential(module, CastToDtype(cast_to), CastToDtype(torch.float32))
			exec('model.' + name + '= new_layer')
		else:
			# use bitsandbytes probe with Int8() quantization
			model = insert_identity(model, module=module)
			quantized_model = copy.deepcopy(model)
			temp_path='temp_model'
			module = Linear4bit(64, 64, bias=False, temp_path=temp_path)
			safetensors.torch.load_model(quantized_model, f'{checkpoint_root}/{temp_path}')
			model = quantized_model.to(device) # quantization active
		
		#print (model)	
		trainer = transformers.Trainer(
			model=model.to(device),
			train_dataset=train_dataset,
			eval_dataset=test_dataset,
			args=training_arguments,
			data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
		)
		results = trainer.evaluate()
		print (name, results)
		original_layer = model_copy
		exec('model.' + name + '=' + copied_module_name)
		all_results.append([name, results])
	return all_results


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

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
encoder_model = LlamaModel(configuration).float()

#encoder_model = LlamaModel(configuration)
model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression, n_heads=n_heads, random=False)
#safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/model.safetensors')
print (model)
#model = insert_identity(model)

# qconfig = QConfig(
# 	activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
# 	weight=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
# )

# safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_noised_256c4_d512_n16_c1024_b8x4/checkpoint-96000/updated_model.safetensors')

#print (model.up[0].weight)
#backend = "qnnpack"
#model.down.qconfig = torch.quantization.get_default_qconfig(backend)
#torch.backends.quantized.engine = backend
#torch.quantization.prepare(model, inplace=True)
#print (model)
#torch.quantization.convert(model, inplace=True)

#quantize_dynamic(
#    model=model, qconfig_spec={'lm_head'}, inplace=True
#)

#print ('Language modeling head weight: ', model.lm_head.weight())
#model.down = nn.Sequential(torch.quantization.QuantStub(), model.down, torch.quantization.DeQuantStub())
#model.down.qconfig = qconfig
#torch.ao.quantization.prepare(model.down, inplace=True)
#model = torch.ao.quantization.convert(model)

class CastToDtype(nn.Module):
	def __init__(self, dtype):
		super().__init__()
		self.dtype = dtype

	def forward(self, x):
		return x.to(self.dtype)

class CustomDtypeCast(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return to_custom_float8(x)

class CustomDtypeUpcast(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return from_custom_float8(x)

# model.up[0] = nn.Sequential(model.up[0], CastToDtype(torch.float8_e5m2), CastToDtype(torch.float32))

#model.down = nn.Sequential(model.down, CustomDtypeCast(), CustomDtypeUpcast()) 

# bitsandbytes approach
#quantized_model = copy.deepcopy(model)
#quantized_model.up[0] = Linear8bitLt(64, 64, bias=False)
#safetensors.torch.load_model(quantized_model, f'{checkpoint_root}/temp_model')
#model = quantized_model.to(0)
#print(model.down)

activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

# model.down.register_forward_hook(get_activation('down'))
# model.up[0].register_forward_hook(get_activation('up[0]'))

print(model)
tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"
# test_path = f"{data_root}/finemath-4-tokenized-test-c512-lpad-8k"

# if you have a new dataset, map before loading from disk
datasets.config.IN_MEMORY_MAX_SIZE = 10e9
train_dataset = load_from_disk(test_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None).take(4096)

print (test_dataset[0])
# filter left-padded inputs from test dataset
#test_dataset = test_dataset.filter(lambda example: example["input_ids"][0] != tokenizer.encode('<|end_of_text|>')[1])
mlflow.end_run()

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=64,
	per_device_eval_batch_size=64,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	gradient_accumulation_steps=1,
	learning_rate=5e-4,
	eval_strategy='steps',
	output_dir=data_root,
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model.to(device),
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print ('starting eval')
print (trainer.evaluate())

#print (activation['down'], activation['up[0]'])
#for i in range(10):
#    print (activation['down'][i][0])

results = observe_sensitivities(model, weights=True)
output = {'results': results}
with open(f'{data_root}/model_sensitivities.json', 'w') as f:
   json.dump(results, f)
