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

# Constants for the custom float8 format
SIGN_BITS = 1
EXPONENT_BITS = 4
MANTISSA_BITS = 3
TOTAL_BITS = SIGN_BITS + EXPONENT_BITS + MANTISSA_BITS

# The bias for the exponent field. This is for an E3M4 format.
# A standard bias for a 3-bit exponent would be 2^(3-1) - 1 = 3.
EXPONENT_BIAS = 2**(EXPONENT_BITS - 2) - 1
MAX_NORMAL_EXPONENT = 2**EXPONENT_BITS - 1 - EXPONENT_BIAS

def to_custom_float8(x: torch.Tensor) -> torch.Tensor:
    """
    Converts a float32 tensor to a custom 8-bit format (uint8 representation).
    Handles standard floating-point logic for subnormals, normals, and infinities.
    """
    assert x.dtype == torch.float32, "Input tensor must be float32"
    
    # Extract sign, exponent, and mantissa from the float32 tensor
    # Using `torch.frexp` to decompose numbers into mantissa and exponent
    mantissa, exponent = torch.frexp(x)
    sign = (x < 0).int()
    
    # Handle zeros
    is_zero = (x == 0.0)
    
    # Handle normals and subnormals
    # Add bias to exponent and clamp
    clamped_exponent = torch.clamp(exponent.int() + EXPONENT_BIAS - 1, min=0, max=2**EXPONENT_BITS - 1)
    
    # Normalize mantissa to the range [1, 2) for non-zero numbers
    mantissa = torch.abs(mantissa) * 2.0
    
    # Quantize the mantissa by scaling and rounding
    mantissa_bits = torch.round(mantissa * (2**MANTISSA_BITS))
    
    # Combine bits for the final 8-bit integer representation
    f8_val = torch.zeros_like(x, dtype=torch.uint8)
    f8_val |= sign.to(torch.uint8) << (EXPONENT_BITS + MANTISSA_BITS)
    f8_val |= clamped_exponent.to(torch.uint8) << MANTISSA_BITS
    f8_val |= (mantissa_bits.int() & (2**MANTISSA_BITS - 1)).to(torch.uint8)
    
    # Handle zero cases
    f8_val[is_zero] = 0
    
    return f8_val

def from_custom_float8(f8_val: torch.Tensor) -> torch.Tensor:
    """
    Converts a custom 8-bit float (uint8 representation) back to float32.
    """
    assert f8_val.dtype == torch.uint8, "Input tensor must be uint8"

    # Unpack bits
    sign = (f8_val >> (EXPONENT_BITS + MANTISSA_BITS)) & 1
    exponent = (f8_val >> MANTISSA_BITS) & (2**EXPONENT_BITS - 1)
    mantissa = f8_val & (2**MANTISSA_BITS - 1)
    
    # Convert bits back to float32
    f32_tensor = torch.zeros_like(f8_val, dtype=torch.float32)
    is_zero = (f8_val == 0)
    
    # For non-zero values, reconstruct the floating-point number
    normal_values = ~is_zero
    
    # Normalize mantissa and add implicit leading bit
    reconstructed_mantissa = (mantissa[normal_values].float() / (2**MANTISSA_BITS)) + 1.0
    reconstructed_exponent = exponent[normal_values].float() - EXPONENT_BIAS

    # Combine parts
    f32_tensor[normal_values] = ((-1)**sign[normal_values]) * reconstructed_mantissa * (2**reconstructed_exponent)

    return f32_tensor

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

safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/model.safetensors')
@torch.no_grad()
def insert_identity(model):
	compressed_dim = model.up.weight.shape[1]
	print (model.up.weight.shape)
	identity_weights = torch.eye(compressed_dim)
	identity_transformation = nn.Linear(compressed_dim, compressed_dim, bias=False)
	identity_transformation.weight.data = identity_weights
	model.up = nn.Sequential(identity_transformation, model.up)
	save_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/updated_model.safetensors')
	return model

model = insert_identity(model)

qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
    weight=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
)

#qconfig = QConfig(
#    activation=default_dynamic_quant_observer.with_args(dtype=torch.qint8),
#    weight=default_dynamic_quant_observer.with_args(dtype=torch.float16),
#)

safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/updated_model.safetensors')

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
#print (model.down[1].weight[0].dtype)
#model = torch.ao.quantization.convert(model)
#print (model.down[1].weight().element_size())
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

#model.down = nn.Sequential(model.down, CastToDtype(torch.float8_e5m2), CastToDtype(torch.float32))
#model.down = nn.Sequential(model.down, CustomDtypeCast(), CustomDtypeUpcast()) 

# bitsandbytes approach
quantized_model = copy.deepcopy(model)
quantized_model.up[0] = Linear8bitLt(64, 64, bias=False)
safetensors.torch.load_model(quantized_model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/updated_model.safetensors')
model = quantized_model.to(0)
#print(model.down)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.down.register_forward_hook(get_activation('down'))
model.up[0].register_forward_hook(get_activation('up[0]'))

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"
# test_path = f"{data_root}/finemath-4-tokenized-test-c512-lpad-8k"

# if you have a new dataset, map before loading from disk
datasets.config.IN_MEMORY_MAX_SIZE = 10e9
train_dataset = load_from_disk(test_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None).take(2048)

# filter left-padded inputs from test dataset
#test_dataset = test_dataset.filter(lambda example: example["input_ids"][0] != tokenizer.encode('<|end_of_text|>')[1])
mlflow.end_run()

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
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
print (activation['down'], activation['up[0]'])
#for i in range(len(activation['down'])):
#    print (activation['down'][i])

#with open('data.json', 'w') as f:
#    json.dump(activation['down'], f)
