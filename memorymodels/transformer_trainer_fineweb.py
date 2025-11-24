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

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

encoder_dim = 256
decoder_dim = 256
context_length = 256
compression = 1
n_layers = 8
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
    'hidden_size':decoder_dim,
    'intermediate_size': 4*decoder_dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size
}
print (llama_config_kwargs)
# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
#model = LlamaForCausalLM(configuration).float()

# transformer autoencoder (custom blocks)
# encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# model = AutoencodingTransformer(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

# transformer autoencoder (preconfigured blocks)
#encoder_model = LlamaModel(configuration)
#decoder_model = LlamaModel(configuration)
#model = AutoencodingTransformerMod(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

# embedding-augmented oracle memory model 
#encoder_model = LlamaModel(configuration)
#model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, compression=compression, transformer_encoder=encoder_model, n_heads=n_heads, noise_embedding=False)

# unrolled embedding transformer autoencoder
encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=False)

#safetensors.torch.load_model(encoder_model, '/home/azureuser/Desktop/fineweb_tmemory_2transformers_e1024c1_d1024_n8_c512_b64x2/checkpoint-200000/model.safetensors')

#llama_config_kwargs = { 
#    'hidden_size':decoder_dim//2,
#    'intermediate_size': 4*decoder_dim//2,
#    'num_hidden_layers': n_layers,
#    'num_attention_heads': 4,
#    'vocab_size': vocab_size
#}

#decoder_configuration = LlamaConfig(**llama_config_kwargs)
#encoder_model = AbbreviatedModel(encoder_model.encoder, tokenized_length=context_length)
#decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
#model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=True)

print (model)
# embedding-augmented oracle memory model 
#encoder_model = LlamaModel(configuration)
#model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, compression=compression, transformer_encoder=encoder_model, n_heads=n_heads)
#model = VariableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, fixed_memory=True, frozen_encoder=encoder)

# Memory transformer
#encoder_model = LlamaModel(configuration)
# model = VariableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, fixed_memory=True, frozen_encoder=encoder_model)

# RMT
#model = RecurrentMemoryTransformer(vocab_size, decoder_dim, n_layers, context_length, n_heads=4, n_chunks=4)


tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-lpad-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-lpad-c512"

def half_data(example):
    example['input_ids'] = example['input_ids'][256:]
    if 'attention_mask' in example:
        example['attention_mask'] = example['attention_mask'][256:]
    return example

datasets.config.IN_MEMORY_MAX_SIZE = 1e9
train_dataset = load_from_disk(train_path).map(half_data, batched=False, num_proc=12)
test_dataset = load_from_disk(test_path).map(half_data, batched=False, num_proc=12)
print (len(train_dataset[0]['input_ids']))

batch_size = 32
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_autoencoding_transformer\
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
	gradient_accumulation_steps=1,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	learning_rate=4e-4, 
	fp16=True,
	bf16=False, 
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
)

print (f"training model, saving to {output_dir}")
# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()
trainer.train()
#trainer.train(output_dir + '/checkpoint-24000')
