import torch
from einops import rearrange
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_model, safe_open
import random
import threading
import os
from dotenv import load_dotenv
import warnings
import datasets
from datasets import load_from_disk

from transformer_autoencoder import UnrolledAutoencodingTransformer

class RetrievalTransformer(nn.Module):

	def __init__(self, model, pad_token_id=1, padding_side='right', ngram_size=7):
		super().__init__()
		self.model = model.model
		self.ngram_size = ngram_size
		self.pad_token_id = pad_token_id
		self.padding_side=padding_side

	def select_ngram(self, input_ids):
		# expects inputs to be of shape [b, t]
		sample_index = random.randint(1, input_ids.shape[0]-1) # the first sample will be replaced with the input
		ngram_index = random.randint(0, input_ids.shape[1] - self.ngram_size)
		sample_ngram = input_ids[sample_index, ngram_index: ngram_index + self.ngram_size]
		return sample_index, sample_ngram

	def forward(self, input_ids, attention_mask, labels=None, **kwargs):
		if input_ids.dim() > 2:
			input_ids = input_ids.squeeze(0) # p b t -> b t
		matching_index, sample_ngram = self.select_ngram(input_ids)
		pad_ngram = (torch.ones(input_ids.shape[1] - self.ngram_size) * self.pad_token_id).to(torch.long).to(input_ids.device)
		pad_attention = torch.zeros
		if self.padding_side == 'right':
			sample_ngram = torch.cat((sample_ngram, pad_ngram), dim=0)
			attention_ngram = torch.cat((torch.ones(self.ngram_size, dtype=torch.long), torch.zeros(input_ids.shape[1] - self.ngram_size, dtype=torch.long)), dim=0).to(input_ids.device)
		elif self.padding_side == 'left':
			sample_ngram = torch.cat((pad_ngram, sample_ngram), dim=0)
			attention_ngram = torch.cat((torch.zeros(input_ids.shape[1] - self.ngram_size, dtype=torch.long), torch.ones(self.ngram_size, dtype=torch.long)), dim=0).to(input_ids.device)

		input_ids[0] = sample_ngram # zeroth batch element is the query phrase
		attention_mask[0] = attention_ngram
		x = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
		embedding_indices = -1
		loss = infoNCEloss(x, matching_index=matching_index, embedding_index=embedding_indices)
		return loss, x

class RetrievalAutoencoderTransformer(nn.Module):

	def __init__(self, model, pad_token_id=1, padding_side='right', ngram_size=7):
		super().__init__()
		if isinstance(self.encoder, LlamaModel):
			self.wte = model.wte
		else:
			self.wte = None

		self.model = model.encoder # assumes that the encoder is a LlamaModel with trained wte
		self.ngram_size = ngram_size
		self.pad_token_id = pad_token_id
		self.padding_side = padding_side

	def select_ngram(self, input_ids):
		# expects inputs to be of shape [b, t]
		sample_index = random.randint(1, input_ids.shape[0]-1) # the first sample will be replaced with the input
		ngram_index = random.randint(0, input_ids.shape[1] - self.ngram_size)
		sample_ngram = input_ids[sample_index, ngram_index: ngram_index + self.ngram_size]
		return sample_index, sample_ngram

	def forward(self, input_ids, attention_mask, labels=None, **kwargs):
		if input_ids.dim() > 2:
			input_ids = input_ids.squeeze(0) # p b t -> b t
		matching_index, sample_ngram = self.select_ngram(input_ids)
		pad_ngram = (torch.ones(input_ids.shape[1] - self.ngram_size) * self.pad_token_id).to(torch.long).to(input_ids.device)
		pad_attention = torch.zeros
		if self.padding_side == 'right':
			sample_ngram = torch.cat((sample_ngram, pad_ngram), dim=0)
			attention_ngram = torch.cat((torch.ones(self.ngram_size, dtype=torch.long), torch.zeros(input_ids.shape[1] - self.ngram_size, dtype=torch.long)), dim=0).to(input_ids.device)
		elif self.padding_side == 'left':
			sample_ngram = torch.cat((pad_ngram, sample_ngram), dim=0)
			attention_ngram = torch.cat((torch.zeros(input_ids.shape[1] - self.ngram_size, dtype=torch.long), torch.ones(self.ngram_size, dtype=torch.long)), dim=0).to(input_ids.device)

		input_ids[0] = sample_ngram # zeroth batch element is the query phrase
		attention_mask[0] = attention_ngram
		if self.wte:
			x = self.wte(input_ids)
		else:
			x = input_ids
		x = self.model(x) # no attention mask for autoencoder training
		embedding_indices = -1
		loss = infoNCEloss(x, matching_index=matching_index, embedding_index=embedding_indices)
		return loss, x


def infoNCEloss(output, matching_index=None, embedding_index=-2, temp=0.02):
	"""
	Implements Noise-Contrastive Loss. Assumes that there is one positive pair per batch and all 
	the rest are negative samples.

	args:
		output: torch.tensor, shape [batch, token, embedding]

	kwargs:
		matching_index: Optional[None, int], integer index of correct retrieval match
		embedding_index: Union[int, arr[int]], index or indicies of the last non-pad token
	"""
	if not isinstance(embedding_index, int):
		query_embedding = output[0, embedding_index[0], :].unsqueeze(0)
		match_embedding = output[matching_index, embedding_index[matching_index], :]
		other_embeddings = []
		for i in range(1, matching_index):
			other_embeddings.append(output[i, embedding_index[i], :])
		for i in range(matching_index+1, len(output)):
			other_embeddings.append(output[i, embedding_index[i], :])
		nonmatch_embeddings = torch.stack(other_embeddings)

	else:
		query_embedding = output[0, embedding_index, :].unsqueeze(0) # b t e shape
		match_embedding = output[matching_index, embedding_index, :]
		nonmatch_embeddings = torch.cat((output[1:matching_index, embedding_index, :], output[matching_index+1:, embedding_index, :]), dim=0)

	cosine_sim = torch.nn.CosineSimilarity(dim=1)
	codists = torch.exp((1/temp)*cosine_sim(query_embedding, match_embedding)) # temperature=0.01
	nondists = torch.sum(torch.exp((1/temp)*cosine_sim(query_embedding, nonmatch_embeddings)))
	loss = -torch.sum(torch.log(codists / (codists + nondists)))
	return loss


if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	warnings.filterwarnings(action='ignore')
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')

	# random inits different for each GPU
	local_rank = threading.get_ident() % 1231

	torch.manual_seed(local_rank)
	random.seed(local_rank) 
	torch.cuda.manual_seed(local_rank)

	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token

	tokenized_length = 256
	dim = 512
	n_layers = 16
	n_heads = 4
	n_context = tokenized_length
	ngram = 7

	vocab_size = len(tokenizer)
	llama_config_kwargs = {
	    'hidden_size':dim,
	    'intermediate_size': 4*dim,
	    'num_hidden_layers': n_layers,
	    'num_attention_heads': n_heads,
	    'vocab_size': vocab_size
	}
	print (llama_config_kwargs)
	# Initializing a LLaMA model
	configuration = LlamaConfig(**llama_config_kwargs)

	# loads a pretrained (on FineWeb) CLM model
	# model = LlamaForCausalLM(configuration)
	# load_model(model, f"{checkpoint_root}/fineweb_training/fineweb_llama_n16_h4_b32/checkpoint-200000/model.safetensors")
	# model = RetrievalTransformer(model, ngram_size=ngram, padding_side='right', pad_token_id=tokenizer.pad_token_id)

	# train_path = f"{data_root}/fineweb-edu-tokenized-train-c512"
	# test_path = f"{data_root}/fineweb-edu-tokenized-test-c512"
	# datasets.config.IN_MEMORY_MAX_SIZE = 1e9
	# train_dataset = load_from_disk(train_path)
	# test_dataset = load_from_disk(test_path)

	encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
	decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
	model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, compression=compression, freeze_encoder=False)
	load_model(autoencoder, '/home/bbadger/Desktop/fineweb_autoencoding_transformer_512c1_d512_n8_c256_b32x4/checkpoint-200000')
	model = RetreivalAutoencoderTransformer(model, ngram_size=ngram, padding_side='right', pad_token_id=tokenizer.pad_token_id)

	def half_data(example):
	    example['input_ids'] = example['input_ids'][256:]
	    if 'attention_mask' in example:
	        example['attention_mask'] = example['attention_mask'][256:]
	    return example

	datasets.config.IN_MEMORY_MAX_SIZE = 1e9 
	train_dataset = load_from_disk(train_path).map(half_data, batched=False, num_proc=16)
	test_dataset = load_from_disk(test_path).map(half_data, batched=False, num_proc=16)

	tokenizer.pad_token = tokenizer.eos_token
	pad_token = int(tokenizer.encode(tokenizer.pad_token)[-1])

	batch_size = 32
	n_devices = 4

	# get number of devices (assumes that all visible devices are used for training)
	if torch.cuda.is_available():
	    n_devices = torch.cuda.device_count()

	# descriptive name for output
	output_dir = f'{checkpoint_root}/fineweb_pretrainedclm_{ngram}gram_infonce\
	_{dim}\
	_n{n_layers}\
	_c{tokenized_length}_b{batch_size}x{n_devices}'

	training_arguments = transformers.TrainingArguments(
		num_train_epochs=2,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		warmup_steps=50,
		eval_steps=1000,
		save_steps=10000,
		logging_steps=50,
		gradient_accumulation_steps=1,
		learning_rate=1e-4,
		fp16=True,
		eval_strategy='steps',
		output_dir=output_dir,
		optim='adamw_torch',
		overwrite_output_dir=True,
		save_safetensors=True,
		max_steps=200000,
	#	torch_compile=True
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	)

	trainer.train()
