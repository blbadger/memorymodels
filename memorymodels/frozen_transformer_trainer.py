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

encoder_dim = 256
decoder_dim = 512
context_length = 512
compression = 1
n_layers = 16
n_heads = 4

vocab_size = 8000

# llama_config_kwargs = {
#     'hidden_size':decoder_dim,
#     'intermediate_size': 4*decoder_dim,
#     'num_hidden_layers': n_layers,
#     'num_attention_heads': n_heads,
#     'vocab_size': vocab_size
# }

# Initializing a LLaMA model
#configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
#model = LlamaForCausalLM(configuration).float()

tokenized_length = 512 
encoder_dim = 1024
decoder_dim = 1024
n_layers = 16
compression = 1 
n_heads = 0 
kernel = 16
unroll = True

# pretrained mixer autoencoder initialization
pretrained_autoencoder = AutoencodingMixer(vocab_size, 1024, 8, tokenized_length, n_heads=n_heads, kernel=16, unroll=True)

# load autoencoder weights and discard decoder
load_path = Path(f"{checkpoint_root}/fineweb_mixer_autounroll_k16_1024c1_n8_c512_b32/checkpoint-200000/model.safetensors")
encoder = TruncatedModel(pretrained_autoencoder)
print (encoder)
encoder_dim = 1024 
decoder_dim = 512 
context_length = 512 
compression = 1 
n_layers = 16
n_heads = 4
model = VariableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, 
								  fixed_memory=True, frozen_encoder=encoder, no_memory=True)

#model = RecurrentMemoryTransformer(vocab_size, decoder_dim, n_layers, context_length, n_heads=4, n_chunks=4)
#model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, 
#										compression=compression, freeze_encoder=True)

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c2048-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c2048-8k"

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


# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_nomemory_transformer_512x4\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b32x4'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
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
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()

trainer.train()
#trainer.train('/home/badger/fineweb_memory_transformer_fixed_256c1_d512_n16_c256_b32x4/checkpoint-24000')
#trainer.train(output_dir + '/checkpoint-112000')
