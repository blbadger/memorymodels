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


class SecretDecoder(nn.Module):

    def __init__(self, n_vocab, dim, model, tokenized_length=512):
        super().__init__()
        self.model = model # assumes a LlamaModel
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length

    def forward(self, inputs_embeds, labels=None):
        x = inputs_embeds
        x = x.to(device).squeeze(1)
        x = self.model(inputs_embeds=x).last_hidden_state

        output = self.lm_head(x)
        # no token shift
        output = rearrange(output, 'b t e -> b e t')
        loss = self.cel(output, labels)
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
n_layers = 8
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
model = LlamaModel(encoder_configuration)
model = SecretDecoder(vocab_size, decoder_dim, model)

train_path = "{data_root}/fineweb-edu-encodings_condclm/{i}_0"
test_path = f"{data_root}/fineweb-edu-encodings_condclm/10_0"


# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = concatenate_datasets([load_from_disk(train_path.format(data_root=data_root, i=i)) for i in range(10)])
#train_dataset = load_from_disk(train_path)#.skip(50o)
test_dataset = load_from_disk(test_path).skip(1000)

train_dataset = train_dataset.rename_column('encodings', 'inputs_embeds')
train_dataset = train_dataset.rename_column('ids', 'labels')

test_dataset = test_dataset.rename_column('encodings', 'inputs_embeds')
test_dataset = test_dataset.rename_column('ids', 'labels')
global_batch_size = 16
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

encoder_dim = 512
# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_secret_decoder\
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
	eval_steps=100,
	logging_steps=50,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	max_steps=5000,
	save_steps=1000,
	torch_compile=False,
	report_to='none'
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	compute_metrics = compute_hamming_metric,
	preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

model.train()
trainer.train()

