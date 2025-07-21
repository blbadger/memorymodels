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
from mixer_autoencoder import AutoencodingMixer, MemoryMixer, ProjMemoryMixer
from memory_transformer import MemoryTransformer, ProjMemoryTransformer
import warnings

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu' # NB 'cuda' but not indices are compatible with accelerate

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)

tokenized_length = 512
encoder_dim = 1024
decoder_dim = 1024
n_layers = 8
compression = 1


# mixer model initialization
#model = MultiHeadedMixer(n_vocab, dim, 16, length=tokenized_length, heads=32).float().to(device)
#model = LanguageMixer(n_vocab, dim, 1).float().to(device)
#model = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length).float()
model = MemoryMixer(n_vocab, encoder_dim, decoder_dim, 8, tokenized_length, compression=compression, combination_dim='token', n_heads=0).float()
# model = MemoryTransformer(n_vocab, dim//2, dim-dim//8, 16, tokenized_length, combination_dim='embedding').float()
#model = ProjMemoryTransformer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression).float()

print (model)

train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

# if you have a new dataset, map before loading from disk
#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 50e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)
mlflow.end_run()

# descriptive name for output
output_dir = f'/home/bbadger/Desktop/fineweb_ememory_mixer_k4\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}_b32'

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	gradient_accumulation_steps=1,
	learning_rate=5e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
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

# save driver snapshot
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f'training begun: saving checkpoints in {output_dir}')
#trainer.train("/home/bbadger/Desktop/finemath_autoencoder_h2_e1024c1_d1024_n8_c512_b32/checkpoint-104000")
trainer.train()
