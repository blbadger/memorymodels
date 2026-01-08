import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
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
	preds, labels = eval_preds
	hamming_metric = hamming(preds, labels)
	return {'Hamming Distance': hamming_metric}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer has a memory leak: a workaround to avoid saving all tensors
    """
    pred_ids = torch.argmax(logits, dim=-2)
    return pred_ids, labels


def tokenize_and_preprocess(example):
	text = example['text']
	global context_length
	tokens = tokenizer(text, max_length=context_length, padding='max_length', truncation=True) # return list, not tensor
	example['input_ids'] = tokens['input_ids']
	example['attention_mask'] = tokens['attention_mask']
	return example

# encoder configuration and load
encoder_model = AutoModel.from_pretrained('google-bert/bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-large-uncased')

vocab_size = len(tokenizer)
context_length = 512
encoder_dim = 1024
decoder_dim = 1024
n_layers = 24
n_heads = 16
bert_config_kwargs = { 
    'hidden_size': decoder_dim,
    'intermediate_size': 4*decoder_dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size,
    'max_position_embeddings': context_length
}

# decoder configuration
configuration = BertConfig(**bert_config_kwargs)
decoder_model = BertModel(configuration)

model = UnrolledAutoencodingTransformer(vocab_size, encoder_dim, encoder_model, decoder_model, decoder_dim=decoder_dim, tokenized_length=context_length, compression=1, freeze_encoder=True)

print (model)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = load_from_disk(train_path).map(tokenize_and_preprocess, num_proc=32)
test_dataset = load_from_disk(test_path).filter(lambda x: x['input_ids'][-1] != 1, num_proc=16).map(tokenize_and_preprocess, num_proc=16)

batch_size = 32
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_bertlarge_information\
_{encoder_dim}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=100,
	eval_steps=4000,
	logging_steps=500,
	save_steps=8000,
	learning_rate=2e-5,
	bf16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000,
	save_safetensors=False,
        torch_compile=True
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
trainer.train(output_dir + '/checkpoint-16000')
