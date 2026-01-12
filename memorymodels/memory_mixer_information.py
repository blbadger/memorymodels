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
from mixer_autoencoder import AutoencodingMixer, AutoencodingTransfixer, MemoryMixer, ProjMemoryMixer, FrozenMemoryMixer, VariableMemoryMixer
from mixer_autoencoder import TruncatedModel, RecurrentMemoryMixer
from safetensors.torch import load_model

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

tokenized_length = 256
encoder_dim = 512
decoder_dim = 1024
n_layers = 16
compression = 1
heads = 0
kernel = 8
chunks = 4

class ModelWrap(nn.Module):

    def __init__(self, encoderblocks):
        super().__init__() 
        self.model_blocks = encoderblocks

    def forward(input_ids, *args):
        return self.model_blocks(input_ids, *args)

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

model = VariableMemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, n_chunks=chunks, no_memory=False)
load_model(model, f"{checkpoint_root}/fineweb_c256x4_memory_mixer_k8_512c1_d1024_n16_c256_b64x2/checkpoint-200000/model.safetensors")
encoder = ModelWrap(model.encoderblocks)

model = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=True, random=False, frozen_encoder=encoder, clm_encoder=False)

def truncate_data(example):
    example['input_ids'] = example['input_ids'][:256]
    if 'attention_mask' in example:
        example['attention_mask'] = example['attention_mask'][:256]
    return example

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 1e9
train_dataset = load_from_disk(train_path, keep_in_memory=None).map(truncate_data, batched=False, num_proc=24)
test_dataset = load_from_disk(test_path, keep_in_memory=None).map(truncate_data, batched=False, num_proc=24)
print (len(train_dataset[0]['input_ids']), len(test_dataset[0]['input_ids']))
mlflow.end_run()

batch_size = 64
n_devices = 2
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_d512c256_mixer_memory_information\
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
	max_steps=200000,
        #torch_compile=True
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

trainer.train()
