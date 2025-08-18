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
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

encoder_dim = 256
decoder_dim = 512
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

# Initializing a LLaMA model
#configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
#model = LlamaForCausalLM(configuration).float()

# uncomment for transformer autoencoder
# Initializing a model from the llama-7b style configuration
# encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# model = AutoencodingTransformer(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

#encoder_model = LlamaModel(configuration)
#decoder_model = LlamaModel(configuration)
#model = AutoencodingTransformerMod(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
safetensors.torch.load_model(encoder_model, '/home/azureuser/fineweb_autoencoding_mixer_noroll_k8_512c1_d512_n8_c512_b64x2/checkpoint-200000/model.safetensors')

model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, 
										compression=compression, freeze_encoder=True)

model = VariableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, fixed_memory=False)

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"

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
output_dir = f'{checkpoint_root}/fineweb_frozenencoder_transformer\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b64x2'

mlflow.end_run()
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

model.train()
trainer.train()
