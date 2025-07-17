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
import datasets

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod

device = 'cuda' if torch.cuda.is_available else 'cpu'

dim = 1024
context_length = 512
vocab_size = 8000
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    'num_attention_heads': 4,
    'vocab_size': vocab_size
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
# model = LlamaForCausalLM(configuration).float()

# uncomment for transformer autoencoder
# Initializing a model from the llama-7b style configuration
# encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# model = AutoencodingTransformer(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

encoder_model = LlamaModel(configuration)
decoder_model = LlamaModel(configuration)
model = AutoencodingTransformerMod(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

# uncomment for GPT-1 initialization
# gpt_config = transformers.OpenAIGPTConfig(vocab_size=8000, n_positions=512, n_embd=512, n_layer=16, n_head=4)
# model = transformers.OpenAIGPTLMHeadModel(gpt_config)

tokenizer = AutoTokenizer.from_pretrained("/home/badger/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = "/home/badger/finemath-4-tokenized-train-c512-lpad-8k"
test_path = "/home/badger/finemath-4-tokenized-test-c512-lpad-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer model
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4, 
	fp16=True, 
	eval_strategy='steps',
	output_dir='/home/badger/finemath_autoencoding_modtransformer_d1024_n8_c512_b32',
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

model.train()
trainer.train() 
