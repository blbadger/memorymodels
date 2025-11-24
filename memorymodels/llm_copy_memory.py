import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import safetensors
import datasets
import warnings
import shutil
from dotenv import load_dotenv
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model

from mixer_autoencoder import AutoencodingMixer, TruncatedModel
from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

def lorify_model(model):
	lora_config = LoraConfig(
	    r=64,
	    target_modules=["q_proj", "v_proj", "k_proj", "up_proj", "down_proj", "gate_proj", "up_proj", "down_proj"],
	    task_type=TaskType.CAUSAL_LM,
	    lora_alpha=32,
	    lora_dropout=0.
	)
	model = get_peft_model(model, lora_config)
	return model

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


def tokenize_and_preprocess(example):
	text = example['text']
	tokens = decoder_tokenizer(text)
	example['decoder_input_ids'] = tokens['input_ids']
	example['decoder_attention_mask'] = tokens['attention_mask']
	return example

model = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B')
decoder_tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B')
encoder_model = lorify_model(model)

encoder_tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

encoder_dim = 256
decoder_dim = 2048
context_length = 256
compression = 1
n_layers = 16
n_heads = 4
model = VariableMemoryTransformer(n_vocab, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, 
								  fixed_memory=True, frozen_encoder=None, no_memory=False, copy=True, decoder=decoder_model)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-8k"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path).map(tokenize_and_preprocess, num_proc=16)
test_dataset = load_from_disk(test_path).take(5000).filter(lambda x: x['input_ids'][-1] != 1, num_proc=16).map(tokenize_and_preprocess, num_proc=16)

batch_size = 16
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_copy_memtrans_c256x4\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=50,
	eval_steps=100,
	save_steps=10000,
	learning_rate=2e-4, 
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=10000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_metrics = compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()
