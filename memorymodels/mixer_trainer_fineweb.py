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
import torch.distributed._shard.checkpoint as dist_cp

from mixer_clm import LanguageMixer
from mixer_multiconv import MultiHeadedMixer
from mixer_clm import LanguageMixer
from mixer_autoencoder import RecurrentMemoryMixer
from mixer_autoencoder import AutoencodingMixer, AutoencodingTransfixer, MemoryMixer, ProjMemoryMixer, FrozenMemoryMixer, VariableMemoryMixer
from mixer_autoencoder import TruncatedModel, RecurrentMemoryMixer
from memory_transformer import MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv
import pathlib
load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu' # NB 'cuda' but not indices are compatible with accelerate

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)

tokenized_length = 1024
encoder_dim = 256
decoder_dim = 1024
n_layers = 16
compression = 4
heads = 0
kernel = 1

class modelwrap(nn.Module):

    def __init__(self, model):
        super().__init__() 
        self.model = model

    def forward(input_ids, *args):
        return self.model(input_ids, *args)

# mixer model initialization
#model = LanguageMixer(n_vocab, decoder_dim, 16, tokenized_length, n_heads=heads, kernel=kernel).float().to(device)
#frozen_encoder = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=16, unroll=True, random=False)
#safetensors.torch.load_model(frozen_encoder, '/home/bbadger/Desktop/fineweb_training/fineweb_mixer_1024_n8_b32/checkpoint-200000/model.safetensors')
#safetensors.torch.load_model(frozen_encoder, '/home/bbadger/Desktop/fineweb_mixer_autounroll_k16_1024c1_n8_c512_b32/model.safetensors')
#safetensors.torch.load_model(frozen_encoder, '/home/bbadger/Desktop/retrieval/contrastive/contrastive_finemath_mixer_1024_n16_b32_penult/checkpoint-70000/model.safetensors')

#encoder = LanguageMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel).float().to(device)
#encoder =modelwrap(AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=False, random=False))
#safetensors.torch.load_model(encoder, '/home/azureuser/autoencoder_pretrained_retrieval/model.safetensors', strict=False)
#encoder = encoder.model
#print (encoder)
#model = MemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=1, combination_dim='token')
model = ProjMemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=1)

#state_dict = {
#        "model": model.state_dict()
#    }       
#
#checkpoint_path = pathlib.Path("/home/azureuser/fineweb_tmemory_mixer_k8_1024c1_c1024_n8_c512_b32")
#distcp_checkpoint_path = checkpoint_path / "pytorch_model_fsdp_0"
#dist_cp.load_state_dict(
#                state_dict=state_dict,
#                storage_reader = dist_cp.FileSystemReader(distcp_checkpoint_path),
#                no_dist=True,
#            )       

#model.load_state_dict(state_dict["model"])
#print (model)

#frozen_encoder = encoder.encoderblocks
#frozen_encoder = TruncatedModel(encoder, autoencoder=True).model_blocks
#model = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=True, random=False, frozen_encoder=frozen_encoder, clm_encoder=False)
#print (model)
#model = AutoencodingTransfixer(n_vocab, encoder_dim, n_layers, tokenized_length, use_transformer_encoder=False).float()
#model = MemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, combination_dim='embedding', n_heads=0, kernel=kernel).float()

#model = VariableMemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, n_chunks=4, no_memory=False)

#model = MemoryTransformer(n_vocab, dim//2, dim-dim//8, 16, tokenized_length, combination_dim='embedding').float()
#model = ProjMemoryTransformer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression).float()
#model = RecurrentMemoryMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel, n_chunks=8)

#encoder = model.encoder
#model = FrozenMemoryMixer(n_vocab, encoder, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, combination_dim='token', n_heads=heads, kernel=kernel)
#model = RecurrentMemoryMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel, n_chunks=8)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 50e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)

def get_chunk(example):
	example['input_ids'] = example['input_ids'][256:]
	return example
#train_dataset = train_dataset.map(get_chunk, num_proc=12)
#test_dataset = test_dataset.map(get_chunk, num_proc=12)

mlflow.end_run()

batch_size = 16
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_memory_pmixer\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}_b{batch_size}x{n_devices}'

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=5000,
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

# save driver snapshot
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)
with open(output_dir + '/model.txt', 'w') as f:
	print (model, file=f)
print (f'training begun: saving checkpoints in {output_dir}')

# for overwriting training args
#torch.save(training_arguments, '/home/badger/fineweb_recurrent_mixer_k8_512c1_d1024_n16_c256_b64x2/checkpoint-104000/training_args.bin')

#trainer.train()
trainer.train(output_dir + '/checkpoint-104000')
# trainer.train('/home/azureuser/finemath_nomemory_mixer_c512x4_512c1_d1024_n16_c512_b32x2/checkpoint-80000')
