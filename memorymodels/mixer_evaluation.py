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

from mixer_multiconv import MultiHeadedMixer
from mixer_autoencoder import AutoencodingMixer, AutoencodingTransfixer, MemoryMixer, ProjMemoryMixer
from memory_transformer import MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu' # NB 'cuda' but not indices are compatible with accelerate

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)

tokenized_length = 512
encoder_dim = 1024
decoder_dim = 1024
n_layers = 8
compression = 8
heads = 0
kernel = 4

# mixer model initialization
model = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=True, random_input=False).float()
#model = MemoryMixer(n_vocab, encoder_dim, decoder_dim, 8, tokenized_length, compression=compression, combination_dim='token', n_heads=0, random_input=False).float()

print (model)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

# if you have a new dataset, map before loading from disk
#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 50e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)

# filter left-padded inputs from test dataset
test_dataset = test_dataset.filter(lambda example: example["input_ids"][0] != tokenizer.encode('<|end_of_text|>')[1])
mlflow.end_run()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_automixer_k8_unroll\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}_b64x2'

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=64,
	per_device_eval_batch_size=64,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	gradient_accumulation_steps=1,
	learning_rate=5e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir='',
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

print (trainer.evaluate())