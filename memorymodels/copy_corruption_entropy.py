import os
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow
import random

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

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

@torch.no_grad()
def hamming(model_output, labels):
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

def corrupt_copy_dataset(input_ids, labels, fraction_corrupted=0.5):
	"""
	Noise corruption of the copy task setting, with entropies associated.

	Example:
	For input tokens [0, 9, 4, 3, 1], we assing some fraction to be corrupted, ie
	transformed into pure noise (uniform random sample from n_vocab), making say
	[0, n, 4, n, 1], n for noise. We then assign the relevant entropy estimates for 
	each value assuming an 8k size tokenizer, [0, 9, 0, 9, 0].

	Input: [0, 9, 4, 3, 1]

	Corrupted copy: [0, n, 4, n, 1]

	copied_halves are =   [0, 9, 4, 3, 1,|0, n, 4, n, 1]

	Entropy estimates =   [0, 9, 0, 9, 0,|0, 9, 0, 9, 0]
	and thus we shift entropy estimates left before applying to model outputs.
	"""
	n_ctx = len(input_ids[0])
	all_indices = [i for i in range(n_ctx)]
	entropies = []
	corrupt_indices = set(random.sample(all_indices, n_ctx//fraction_corrupted))
	entropy_array = [0 for i in range(n_ctx) if i not in corrupt_indices else 9.3] # assumes informationless output for 8k tokenizer
	entropies = entropy_array * input_ids.shape[0]
	entropies = torch.cat(entropies, dim=0)[:, 1:] #shift entropies
	for i, input in enumerate(input_ids):
		first_half = input[:n_ctx//2]
		corrupted_half = [i for i in first_half if i-1 not in corrupt_indices else torch.randint(0, len(tokenizer)-1, 1)]
		copied_halves = torch.cat((first_half, corrupted_half)).to(torch.long)
		input_ids[i] = copied_halves
		
	n_ctx = len(labels[0])
	for i, input in enumerate(labels):
		first_half = input[:n_ctx//2]
		pad_half = torch.ones(first_half.shape).to(device) * -100
		halves = torch.cat((pad_half, first_half)).to(torch.long)
		labels[i] = halves

	# return all torch tensors
	return input_ids, labels, entropies

class WeightedModel(torch.nn.Module):
	def __init__(self, model, hidden_dim, tokenizer_size, use_weights=True):
		super().__init__()
		self.lm_head = torch.nn.Linear(hidden_dim, tokenizer_size)
		self.model = model
		self.cel = torch.nn.CrossEntropyLoss(reduction='none')
		self.use_weights = use_weights
		
	def forward(self, input_ids, labels, attention_mask, *args, **kwargs):
		input_ids, labels, entropies = corrupt_copy_dataset(input_ids, labels)
		model_output = self.model(input_ids, attention_mask).last_hidden_state
		model_output = self.lm_head(model_output)
		model_output = rearrange(model_output, 'b t e -> b e t')

		shifted_output = model_output[..., :-1]
		shifted_labels = labels[..., 1:]
		loss = self.cel(shifted_output, shifted_labels)
		#if 'attribution' in locals() or 'attribution' in globals():
		weights = torch.abs(loss - entropies) # 10 - attribution # complement of attributions
		if self.model.training and self.use_weights:
			loss = weights #*2.5 for scaled EEMs # *=weights #weights[..., :-1]
		nonpad_tokens = torch.sum(attention_mask)
		loss = torch.sum(loss) / nonpad_tokens
		return loss, model_output
		 
decoder_dim = 512
context_length = 1024
n_layers = 16
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
	'hidden_size':decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size,
	'attention_dropout': 0.1
}
print (llama_config_kwargs)
# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

#configuration.save_pretrained('/home/badger/fineweb_l1attr_weighted_transformer_d512_n16_c1024_b32x4/checkpoint-200000')
# Initializing a model from the llama-7b style configuration
model = WeightedModel(LlamaModel(configuration).float(), decoder_dim, vocab_size, use_weights=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-50k-exloss-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-50k-exloss-lpad-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path).take(50000)
test_dataset = load_from_disk(test_path)

batch_size = 64
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
	n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_weighted_transformer_dropout_50k\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=100,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	gradient_accumulation_steps=1,
	warmup_steps=500,
	eval_steps=500,
	save_steps=8000,
	learning_rate=2e-4, 
	fp16=False,
	bf16=True, 
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print (f"training model, saving to {output_dir}")
# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()
#trainer.train()
trainer.train(output_dir + '/checkpoint-132000')
#print (trainer.evaluate())
