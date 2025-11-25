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
import warnings
from dotenv import load_dotenv
import pathlib
load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu' # NB 'cuda' but not indices are compatible with accelerate

class modelwrap(nn.Module):

    def __init__(self, model):
        super().__init__() 
        self.model = model

    def forward(input_ids, *args):
        return self.model(input_ids, *args)

@torch.no_grad()
def hamming(model_output, labels):
	total_metric = 0
	# assign and shift outputs and labels
	labels= torch.tensor(labels)[..., 1:]
	model_output = torch.tensor(model_output[0])[..., :-1]
	nonpad_tokens = torch.where(labels != -100, 1, 0)
	equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
	average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
	return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
	logits, labels = eval_preds
	hamming_metric = hamming(logits, labels)
	return {'Hamming Distance': hamming_metric}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer has a memory leak: a workaround to avoid saving all tensors
    """
    pred_ids = torch.argmax(logits, dim=-2)
    return pred_ids, labels


tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)

tokenized_length = 256
encoder_dim = 512
decoder_dim = 512
n_layers = 16
compression = 1
heads = 0
kernel = 4

# mixer model initialization
model = LanguageMixer(n_vocab, decoder_dim, 16, tokenized_length, n_heads=heads, kernel=kernel).float().to(device)
encoder = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=True, random=False)
safetensors.torch.load_model(encoder, '/home/bbadger/Desktop/fineweb_autoencoding_mixer_512c1_d512_n16_c256_b32x4/checkpoint-128000/model.safetensors')

#checkpoint_path = pathlib.Path("/home/azureuser/fineweb_tmemory_mixer_k8_1024c1_c1024_n8_c512_b32")
#distcp_checkpoint_path = checkpoint_path / "pytorch_model_fsdp_0"
#dist_cp.load_state_dict(
#                state_dict=state_dict,
#                storage_reader = dist_cp.FileSystemReader(distcp_checkpoint_path),
#                no_dist=True,
#            )       

#model.load_state_dict(state_dict["model"])

tokenized_length = 256
frozen_encoder = TruncatedModel(encoder, autoencoder=True)
model = VariableMemoryMixer(n_vocab, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=1, 
							frozen_encoder=frozen_encoder, n_heads=heads, kernel=kernel, n_chunks=4, no_memory=False, copy=True, encoder_ctx=None)

#model = RecurrentMemoryMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel, n_chunks=8)

#encoder = model.encoder
#model = FrozenMemoryMixer(n_vocab, encoder, encoder_dim, decoder_dim, n_layers, tokenized_length, compression=compression, combination_dim='token', n_heads=heads, kernel=kernel)
#model = RecurrentMemoryMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel, n_chunks=8)

print (model)
print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 1e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path).take(5000).filter(lambda x: x['input_ids'][-1] != 1, num_proc=16)

def get_chunk(example):
	example['input_ids'] = example['input_ids'][:256]
	return example

# train_dataset = train_dataset.map(get_chunk, num_proc=12)
# test_dataset = test_dataset.map(get_chunk, num_proc=12)

mlflow.end_run()

batch_size = 16
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_memorymixer_nodecoderinfo\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}_b{batch_size}x{n_devices}'

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=50,
	eval_steps=500,
	save_steps=20000,
	gradient_accumulation_steps=1,
	learning_rate=5e-4,
	fp16=True,
	bf16=False,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True,
	max_steps=1000000
)

trainer = transformers.Trainer(
	model=model.to(device),
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_metrics = compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# save driver snapshot
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)
with open(output_dir + '/model.txt', 'w') as f:
	print (model, file=f)
print (f'training begun: saving checkpoints in {output_dir}')

trainer.train()
#trainer.train(output_dir + '/checkpoint-104000')
