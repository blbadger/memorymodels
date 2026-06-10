import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk
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

tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)
context_length = 512
encoder_dim = 512
decoder_dim = 512
n_layers = 16
n_heads = 8
encoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

encoder_configuration = LlamaConfig(**encoder_config_kwargs)
encoder_model = LlamaForCausalLM(encoder_configuration)

load_model(encoder_model, f'{data_root}/fineweb_training/fineweb_llama_512_n16_h8_c512/checkpoint-200000/model.safetensors')
original_clm = encoder_model

clm_head = encoder_model.lm_head
encoder_state_dict = encoder_model.model.state_dict()
clm_wte = encoder_model.model.embed_tokens
split_model = SplitModel(encoder_configuration)
split_model.config.num_hidden_layers = 16
split_model.load_state_dict(encoder_state_dict)
encoder_model = encoder_model.model

# last 8 layers are the clm decoder
clm_decoder = SuffixModel(encoder_configuration)
clm_decoder.load_state_dict(encoder_state_dict)

encoder_model.config.num_hidden_layers = 8

n_layers = 8
n_heads = 4
decoder_config_kwargs = { 
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'max_position_embeddings': context_length
}

decoder_configuration = LlamaConfig(**decoder_config_kwargs)
decoder_model = LlamaModel(decoder_configuration)

model = AllAutoencodingTransformer(
	vocab_size, 
	decoder_dim, 
	encoder_model, 
	decoder_model, 
	tokenized_length=context_length, 
	compression=1, 
	freeze_encoder=True,
	noise_embeddings=False, 
)

load_model(model, f'{data_root}/fineweb_embedding_inverter_512_d512_n8_c512_b32x4/checkpoint-4000/model.safetensors')

inversion_decoder =  LlamaForCausalLM(decoder_configuration)
load_model(inversion_decoder, f'{data_root}/fineweb_secret_decoder_512_d512_n8_c512_b4x4/checkpoint-2000/model.safetensors')
inversion_wte = inversion_decoder.model.embed_tokens
inversion_head = inversion_decoder.lm_head
inversion_encoder = model.encoder
inversion_decoder = inversion_decoder.model
#inversion_decoder = model.decoder
#inversion_wte = model.wte
#inversion_head = model.lm_head
#inversion_head = inversion_head

train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"

# load datasets and duplicate entries
datasets.config.IN_MEMORY_MAX_SIZE = 5e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path).take(12800)

global_batch_size = 64
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()
batch_size = global_batch_size // n_devices

encoder_dim = 512
# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_embedding_inverter\
_{encoder_dim}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

num_models = 11
local_rank = int(os.environ.get("LOCAL_RANK", 0))
for i in tqdm(range(num_models)):
	model = SecretTransformer(
		vocab_size,
		decoder_dim,
		inversion_encoder,
		clm_decoder,
		split_model,
		inversion_decoder,
		original_clm,
		wte=inversion_wte,
		clm_head=clm_head,
		inversion_head=inversion_head,
		manual_seed=10*i + local_rank 
	) 
	# train unique num_models, storing outputs from each
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=3,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		warmup_steps=50,
		eval_steps=5000,
		logging_steps=500,
		learning_rate=2e-4,
		fp16=True,
		eval_strategy='steps',
		output_dir=output_dir,
		optim='adamw_torch',
		max_steps=5000,
		save_strategy='no',
		save_steps=20000,
		torch_compile=False,
		report_to='none'
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

	model.train()
	trainer.train()	
	#trainer.evaluate()
	print (model.random_label)
	print ('training run completed')
	all_embeddings = model.all_embeddings
	all_labels = model.all_labels
	all_embeddings = torch.cat(all_embeddings, dim=0) # (b*n) t e
	print (all_embeddings.shape)
	all_embeddings = torch.unbind(all_embeddings, dim=0)
	all_labels = torch.cat(all_labels, dim=0)
	all_labels = torch.unbind(all_labels, dim=0)
	print ('embeddings and labels accessed')
	attributions_dict = {'encodings': all_embeddings, 'ids': all_labels}
	# print (attributions_dict)
	attributions_dataset = Dataset.from_dict(attributions_dict)
	attributions_dataset.save_to_disk(f"{data_root}/fineweb-edu-encodings_condclm/{i}_{local_rank}")
	model.all_embeddings, model.all_labels = [], []
	del attributions_dict, all_labels, all_embeddings, model, trainer
	print ('dataset updated, model removed')


