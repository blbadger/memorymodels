import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk, concatenate_datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from prettytable import PrettyTable
from safetensors.torch import save_file, load_model
from safetensors import safe_open
import safetensors
import datasets
from datasets import Dataset
import warnings
import shutil
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model

from mixer_autoencoder import AutoencodingMixer, TruncatedModel
from transformer_autoencoder import AbbreviatedModel, SuffixModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from transformer_autoencoder import SplitModel, AllAutoencodingTransformer, SecretTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'


class ParallelModel(nn.Module):

    def __init__(self, n_vocab, dim, large_module, small_module, final_module, tokenized_length=512):
        super().__init__()
        self.large_module = large_module
        self.small_module = small_module
        self.final_module = final_module
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = input_ids
        x = x.to(device).squeeze(1)
        large_stack_output = self.large_module(input_ids=x, attention_mask=attention_mask).last_hidden_state
        small_stack_output = self.small_module(input_ids=x, attention_mask=attention_mask).last_hidden_state

        total_output = large_stack_output + small_stack_output
        output = self.final_module(inputs_embeds=total_output, attention_mask=attention_mask).last_hidden_state

        output = self.lm_head(output)
        if labels.dim() > 2:
            labels = rearrange(labels, 'b p t -> b (p t)')
        output = rearrange(output, 'b t e -> b e t')
        shift_logits = output[..., :-1].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.cel(shift_logits, shift_labels)
        return loss, output


@torch.no_grad()
def hamming(model_output, labels):
	total_metric = 0
	# no shift for autoencoders
	labels = torch.tensor(labels)
	model_output = torch.tensor(model_output[0])
	nonpad_tokens = torch.where(labels != -100, 1, 0)
	equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
	average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
	return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
	preds, labels = eval_preds
	hamming_metric = hamming(preds, labels)
	return {'Hamming Complement': hamming_metric}

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

def half_data(example):
	example['input_ids'] = example['input_ids'][256:]
	if 'attention_mask' in example:
		example['attention_mask'] = example['attention_mask'][256:]
	return example

# Define a minimal data collator to batch token-free tensors
def embedding_data_collator(features):
    batch = {
        "inputs_embeds": torch.stack([f["inputs_embeds"] for f in features], dim=0),
        "labels": torch.stack([f["labels"] for f in features], dim=0)
    }
    return batch

tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
context_length = 512
encoder_dim = 512
decoder_dim = 512
n_layers = 12
n_heads = 4
encoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

encoder_configuration = LlamaConfig(**encoder_config_kwargs)
large_module = LlamaModel(encoder_configuration)

vocab_size = len(tokenizer)
context_length = 512
encoder_dim = 512
decoder_dim = 512
n_layers = 4
n_heads = 4
encoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

small_module = LlamaModel(encoder_configuration)

vocab_size = len(tokenizer)
context_length = 512
encoder_dim = 512
decoder_dim = 512
n_layers = 2
n_heads = 4
encoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

end_module = LlamaModel(encoder_configuration)

model = ParallelModel(vocab_size, decoder_dim, large_module, small_module, end_module)

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

global_batch_size = 64
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

encoder_dim = 512
# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_parallel_model\
_{encoder_dim}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

# train unique num_models, storing outputs from each
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=500,
	eval_steps=4000,
	logging_steps=500,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=200000,
	save_steps=8000,
	torch_compile=True,
	report_to='none'
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	args=training_arguments,
	compute_metrics = compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

model.train()
trainer.train()

