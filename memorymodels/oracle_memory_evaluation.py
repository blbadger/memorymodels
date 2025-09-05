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
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import default_observer
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization import quantize_dynamic

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cpu' if torch.cuda.is_available else 'cpu'
print (device)

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

qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
    weight=default_observer.with_args(dtype=torch.torch.qint8),
)

safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/model.safetensors')

backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
torch.quantization.prepare(model.down, inplace=True)
torch.quantization.convert(model.down, inplace=True)

quantize_dynamic(
    model=model, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=True
)
print (model.down.weight)
#model.down = nn.Sequential(torch.quantization.QuantStub(), model.down, torch.quantization.DeQuantStub())
#model.down.qconfig = qconfig
#torch.ao.quantization.prepare(model.down, inplace=True)
#print (model.down[1].weight[0].dtype)
#model = torch.ao.quantization.convert(model)
#print (model.down[1].weight().element_size())

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"
# test_path = f"{data_root}/finemath-4-tokenized-test-c512-lpad-8k"

# if you have a new dataset, map before loading from disk
datasets.config.IN_MEMORY_MAX_SIZE = 10e9
train_dataset = load_from_disk(test_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)

# filter left-padded inputs from test dataset
test_dataset = test_dataset.filter(lambda example: example["input_ids"][0] != tokenizer.encode('<|end_of_text|>')[1])
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
