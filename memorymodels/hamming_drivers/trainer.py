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

#from mixer_clm import LanguageMixer
#from mixer_multiconv import MultiHeadedMixer
#from mixer_clm import LanguageMixer
from mixer_autoencoder import AutoencodingMixer, AutoencodingTransfixer, MemoryMixer, ProjMemoryMixer, FrozenMemoryMixer, VariableMemoryMixer
from mixer_autoencoder import TruncatedModel, RecurrentMemoryMixer
from memory_transformer import MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv
import pathlib

@torch.no_grad()
def hamming(model_output, labels):
        total_metric = 0
        # no shift for autoencoders
        labels= torch.tensor(labels)
        model_output = torch.tensor(model_output[0])
        nonpad_tokens = torch.where(labels != -100, 1, 0)
        equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
        average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
        return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
        preds, labels = eval_preds
        hamming_metric = hamming(preds, labels)
        return {'Hamming Distance': hamming_metric}
def preprocess_logits_for_metrics(logits, labels):
    """ 
    Original Trainer has a memory leak: a workaround to avoid saving all tensors
    """
    pred_ids = torch.argmax(logits, dim=-2)
    return pred_ids, labels


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
compression = 1
heads = 0
kernel = 16
# mixer model initialization
frozen_encoder = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=16, unroll=True, random=False)
safetensors.torch.load_model(frozen_encoder, '/datadrive/fineweb_mixer_autounroll_k16_1024c1_n8_c512_b32/checkpoint-200000/model.safetensors')
print (frozen_encoder)
#frozen_encoder.model_blocks = frozen_encoder.encoderblocks
#frozen_encoder = TruncatedModel(model, autoencoder=False).encoder_blocks
#model = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=True, random=False, frozen_encoder=frozen_encoder, clm_encoder=False)

model = frozen_encoder
print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 50e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)
mlflow.end_run()

print (test_dataset[0])
batch_size = 64
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_frozen_encoder_autoencoder_k16\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{tokenized_length}_b{batch_size}x{n_devices}'

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
	bf16=False,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=False,
	save_safetensors=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model.to(device),
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_metrics=compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# save driver snapshot
#code_path = os.path.abspath(__file__)
#if not os.path.isdir(output_dir):
#	os.mkdir(output_dir)
#shutil.copy(code_path, output_dir)
#with open(output_dir + '/model.txt', 'w') as f:
#	print (model, file=f)
#safetensors.torch.load_model(model, output_dir + '/checkpoint-200000/model.safetensors')
# for overwriting training args
#torch.save(training_arguments, '/home/badger/fineweb_recurrent_mixer_k8_512c1_d1024_n16_c256_b64x2/checkpoint-104000/training_args.bin')

#trainer.train()
#trainer.train(output_dir + '/checkpoint-200000')

print ('evaluating')
print (trainer.evaluate())
# trainer.train('/home/azureuser/finemath_nomemory_mixer_c512x4_512c1_d1024_n16_c512_b32x2/checkpoint-80000')
