import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import safetensors
import datasets
import warnings
import shutil
from dotenv import load_dotenv
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model

from mixer_autoencoder import AutoencodingMixer, TruncatedModel
from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

def lorify_model(model):
	lora_config = LoraConfig(
	    r=64,
	    target_modules=["q_proj", "v_proj", "k_proj", "up_proj", "down_proj", "gate_proj", "up_proj", "down_proj"],
	    task_type=TaskType.CAUSAL_LM,
	    lora_alpha=32,
	    lora_dropout=0.
	)
	model = get_peft_model(model, lora_config)
	return model

model = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B')
tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B')
model.save_pretrained('/home/bbadger/fineweb_copy_memory_lorallama_c256x4_512c1_d2048_n16_c256_b8x2/decoder_2000')
#decoder_model = lorify_model(model)
decoder_model = model
print (decoder_model)

#tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer) #128k for llama 3.2
encoder_dim = 512
decoder_dim = 2048
context_length = 256
compression = 1
n_layers = 16
n_heads = 8
full_model = VariableMemoryTransformer(n_vocab, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, 
								  fixed_memory=True, frozen_encoder=None, no_memory=False, copy=True, decoder=decoder_model)

full_model.load_state_dict(torch.load('/home/bbadger/fineweb_copy_memory_lorallama_c256x4_512c1_d2048_n16_c256_b8x2/checkpoint-2000/pytorch_model.bin'))
decoder = full_model.decoder
decoder.save_state_dict('/home/bbadger/fineweb_copy_memory_lorallama_c256x4_512c1_d2048_n16_c256_b8x2/decoder_2000/pytorch_model.bin')