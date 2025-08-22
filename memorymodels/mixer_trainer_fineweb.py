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
import safetensors

from mixer_clm import LanguageMixer
from mixer_multiconv import MultiHeadedMixer
from mixer_clm import LanguageMixer
from mixer_autoencoder import AutoencodingMixer, AutoencodingTransfixer, MemoryMixer, ProjMemoryMixer, FrozenMemoryMixer, VariableMemoryMixer
from mixer_autoencoder import RecurrentMemoryMixer
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

tokenized_length = 128
encoder_dim = 512
decoder_dim = 1024
n_layers = 16
compression = 1
heads = 0
kernel = 8

# mixer model initialization
<<<<<<< HEAD
#model = LanguageMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel).float().to(device)
#model = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=False, random=False)
#safetensors.torch.load_model(model, '/home/azureuser/fineweb_autoencoding_mixer_noroll_k8_512c1_d512_n8_c512_b64x2/checkpoint-200000/model.safetensors')
=======
encoder = LanguageMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel).float().to(device)
# encoder = AutoencodingMixerAutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=False, random=False)
safetensors.torch.load_model(encoder, '/home/azureuser/fineweb_autoencoding_mixer_noroll_k8_512c1_d512_n8_c512_b64x2/checkpoint-200000/model.safetensors')
model = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=False, random=False)
>>>>>>> 3bcb1ad2bb7719e1b2546d96080199d2940be1d3

#model = AutoencodingTransfixer(n_vocab, encoder_dim, n_layers, tokenized_length, use_transformer_encoder=False).float()
#model = MemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, combination_dim='token', n_heads=0, kernel=1).float()
#model = VariableMemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, n_chunks=4, no_memory=False)
#model = MemoryTransformer(n_vocab, dim//2, dim-dim//8, 16, tokenized_length, combination_dim='embedding').float()
#model = ProjMemoryTransformer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression).float()

<<<<<<< HEAD
#encoder = model.encoder
#model = FrozenMemoryMixer(n_vocab, encoder, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, combination_dim='token', n_heads=heads, kernel=kernel)
=======
print (model)
#train_path = '/home/azureuser/fineweb-edu-tokenized-test-c512-lpad-8k-debatched'
#test_path = '/home/azureuser/fineweb-edu-tokenized-test-c512-lpad-8k-debatched'
#train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
#test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"

#train_path = f"{data_root}/finemath-4-tokenized-train-c512-lpad-8k"
#test_path = f"{data_root}/finemath-4-tokenized-test-c512-lpad-8k"
>>>>>>> 3bcb1ad2bb7719e1b2546d96080199d2940be1d3

model = RecurrentMemoryMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel, n_chunks=8)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-8k"
# if you have a new dataset, map before loading from disk
#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 50e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)
mlflow.end_run()

#test_dataset = test_dataset.filter(lambda example: example["input_ids"][0] != tokenizer.encode('<|end_of_text|>')[1])

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_recurrent_mixer_128x8_k8\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}_b64x2'


training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	gradient_accumulation_steps=1,
	learning_rate=5e-4,
	fp16=False,
        bf16=True,
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
#print (trainer.evaluate())
# save driver snapshot
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)
print (f'training begun: saving checkpoints in {output_dir}')
#torch.save(training_arguments, '/home/badger/fineweb_recurrent_mixer_k8_512c1_d1024_n16_c256_b64x2/checkpoint-104000/training_args.bin')
trainer.train('/home/badger/fineweb_frozen_mixer_ek16_extended_1024c1_d1024_n16_c512_b32x4/checkpoint-72000')

#trainer.train()
#trainer.train(output_dir + '/checkpoint-40000')
