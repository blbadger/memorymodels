import os
import shutil
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import torch.nn as nn
import mlflow
import datasets
import safetensors
from safetensors import safe_open
from datasets import load_dataset, load_from_disk
from pathlib import Path

from mixer_autoencoder import AutoencodingMixer, FrozenMemoryMixer, TruncatedModel
import warnings

load_dotenv()
data_root = os.getenv('DATA_ROOT')
checkpoint_root = os.getenv('CHECKPOINT_ROOT')

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
compression = 1
n_heads = 0
kernel = 1

# mixer model initialization
pretrained_autoencoder = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel)
autoencoder_path = Path(f"{checkpoint_root}/fineweb_training/fineweb_autoencoding_mixer_n8_c512/checkpoint-200000")
load_path = autoencoder_path / "model.safetensors"
print (pretrained_autoencoder)

# load encoder
print (pretrained_autoencoder.wte.weight)
safetensors.torch.load_model(pretrained_autoencoder, load_path)
print (pretrained_autoencoder.wte.weight)

encoder = TruncatedModel(pretrained_autoencoder)
model = FrozenMemoryMixer(n_vocab, encoder, encoder_dim, decoder_dim, n_layers, tokenized_length, n_heads=0).float()
print (model)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

# if you have a new dataset, map before loading from disk
#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 50e9 # max of 50 GB memory per device
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)
mlflow.end_run()

batch_size = 32
# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_frozen_mixer\
_{encoder_dim}c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}\
_b{batch_size}'

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
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
trainer.train()
