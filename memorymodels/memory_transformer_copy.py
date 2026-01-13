import os
import torch
import torch.nn as nn
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
from safetensors.torch import load_model
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
	print (labels)
	total_metric = 0
	# assign and shift outputs and labels
	labels= torch.tensor(labels)[..., 1:]
	model_output = torch.tensor(model_output[0])[..., :-1]
	nonpad_tokens = torch.where(labels != -100, 1, 0)
	equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
	average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
	return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
	logits, labels = eval_preds
	hamming_metric = hamming(logits, labels)
	return {'Hamming Distance': hamming_metric}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer has a memory leak: a workaround to avoid saving all tensors
    """
    pred_ids = torch.argmax(logits, dim=-2)
    return pred_ids, labels

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

encoder_dim = 512 
decoder_dim = 512 
context_length = 256 
compression = 1 
n_layers = 16
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
#configuration = LlamaConfig(**llama_config_kwargs)
#encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
#decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)

#model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=False)
#model = LlamaForCausalLM(configuration)
#load_model(model, '/home/bbadger/Desktop/fineweb_autoencoding_transformer_512c1_d512_n8_c256_b32x4/checkpoint-200000/model.safetensors')
#encoder = model.encoder.model

#load_model(model, '/home/bbadger/Desktop/fineweb_training/fineweb_llama_n16_h4_b32/checkpoint-200000/model.safetensors')
#encoder = model.model

encoder_dim = 512
decoder_dim = 512
context_length = 256
compression = 1 
n_layers = 16 
n_heads = 4
model = VariableMemoryTransformer(n_vocab, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, fixed_memory=True, frozen_encoder=None, no_memory=False, copy=True, blank_copy=False)
# load the pretrained memory model
#load_model(model, '/home/bbadger/Desktop/fineweb_copy_memtrans_frozenenc_nodecoder_c256x4_512c1_d512_n16_c256_b8x4/checkpoint-100000/model.safetensors')

# no memory control
#print (model.no_memory)
#model.no_memory = True
#print (f"Uses no memory? {model.no_memory}")
#print (model.no_memory)
print (model)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 3e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path).take(100).filter(lambda x: x['input_ids'][-1] != 1, num_proc=16)

total_batch_size = 4
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_per_device = 1
gradient_accumulation_steps = total_batch_size // (n_devices * batch_per_device)
# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_copy_memtrans_c256x4\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_per_device}x{n_devices}x{gradient_accumulation_steps}'
output_dir = '/home/bbadger/Desktop/test'
mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_per_device,
	per_device_eval_batch_size=batch_per_device,
	gradient_accumulation_steps=gradient_accumulation_steps,
	warmup_steps=500,
	eval_steps=500,
	save_steps=10000,
	learning_rate=2e-4, 
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=100000,
        #torch_compile=True
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_metrics = compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()
model.copy = True
trainer.evaluate()

model.copy = False
trainer.evaluate()
trainer.train()
