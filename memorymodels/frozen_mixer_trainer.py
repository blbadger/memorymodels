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
from dotenv import load_dotenv

from mixer_multiconv import MultiHeadedMixer
from mixer_autoencoder import AutoencodingMixer, FrozenMemoryMixer, TruncatedModel, VariableMemoryMixer
import warnings
from dotenv import load_dotenv

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
n_layers = 16
compression = 1
n_heads = 0
kernel = 16
unroll = True

# mixer model initialization
pretrained_autoencoder = AutoencodingMixer(n_vocab, 1024, 8, tokenized_length, n_heads=n_heads, kernel=16, unroll=True)
load_path = Path(f"{checkpoint_root}/fineweb_mixer_autounroll_k16_1024c1_n8_c512_b32/checkpoint-200000/model.safetensors")

#pretrained_autoencoder = MultiHeadedMixer(n_vocab, decoder_dim, 16, length=tokenized_length, heads=4).float().to(device)
#autoencoder_path = Path(f"{checkpoint_root}/Desktop/fineweb_mixer_4h_d1024_n16_c512_b64x2/checkpoint-200000")
#load_path = autoencoder_path / "model.safetensors"

# load encoder
safetensors.torch.load_model(pretrained_autoencoder, load_path)

encoder = TruncatedModel(pretrained_autoencoder)
#model = FrozenMemoryMixer(n_vocab, encoder, encoder_dim, decoder_dim, n_layers, tokenized_length, n_heads=0, kernel=kernel).float()
model = VariableMemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=1, n_heads=0, kernel=1, n_chunks=4, no_memory=False, frozen_encoder=encoder)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c2048-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c2048-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 50e9 # max of 50 GB memory per device
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)
mlflow.end_run()

batch_size = 16
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_frozen_memmixer\
_{encoder_dim}c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}\
_b{batch_size}x{n_devices}'

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
#trainer.train(output_dir + '/checkpoint-72000')
