import torch
from einops import rearrange
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel, AutoTokenizer
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

from transformer_autoencoder import UnrolledAutoencodingTransformer, AbbreviatedModel

class RetrievalTransformer(nn.Module):

	def __init__(self, model, pad_token_id=1, padding_side='right', ngram_size=7):
		super().__init__()
		if not isinstance(model.encoder, LlamaModel):
			self.wte = model.wte
		else:
			self.wte = None

		self.encoder = model.encoder
		# freeze the encoder
		for name, param in self.encoder.named_parameters():
			param.requires_grad = False

		# trainable decoder
		self.decoder = model.decoder
		self.ngram_size = ngram_size
		self.pad_token_id = pad_token_id
		self.padding_side = padding_side

	def select_ngram(self, input_ids):
		# expects inputs to be of shape [b, t]
		sample_index = random.randint(1, input_ids.shape[0]-1) # the first sample will be replaced with the input embedding
		ngram_index = random.randint(0, input_ids.shape[1] - self.ngram_size)
		sample_ngram = input_ids[sample_index, ngram_index: ngram_index + self.ngram_size]
		return sample_index, sample_ngram

	def forward(self, input_ids, attention_mask, embeddings, labels=None, **kwargs):
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

		if self.wte:
			sample_ngram = self.wte(sample_ngram)
		sample_ngram_embedding = self.encoder(sample_ngram)
		embeddings[0] = sample_ngram_embedding # zeroth batch element is the query phrase
		output = self.decoder(embeddings)
		loss = self.cel(output, labels)
		return loss, output

class RetrievalAutoencoderTransformer(nn.Module):

	def __init__(self, model, custom_decoder=None, embed_comparison=256, pad_token_id=1, padding_side='right', ngram_size=7):

		super().__init__()
		if not isinstance(model.encoder, LlamaModel):
			self.wte = model.wte
		else:
			self.wte = None

		self.encoder = model.encoder
		self.lm_head = model.lm_head
		# freeze the encoder
		for name, param in self.encoder.named_parameters():
			param.requires_grad = False

		# trainable decoder
		if custom_decoder:
			self.decoder = custom_decoder
		else:
			self.decoder = model.decoder
		self.ngram_size = ngram_size
		self.pad_token_id = pad_token_id
		self.padding_side = padding_side
		self.retrieval_head = nn.Linear(dim, 1, bias=True)
		self.cel = torch.nn.CrossEntropyLoss()
		self.embed_comparison = embed_comparison

	def select_ngram(self, input_ids):
		# expects inputs to be of shape [b, t]
		sample_index = random.randint(1, input_ids.shape[0]-1) # the first sample will be replaced with the input embedding
		ngram_index = random.randint(0, input_ids.shape[1] - self.ngram_size)
		sample_ngram = input_ids[sample_index, ngram_index: ngram_index + self.ngram_size]
		return sample_index, sample_ngram

	def forward(self, input_ids, attention_mask, embeddings, labels=None, **kwargs):
		if input_ids.dim() > 2:
			input_ids = input_ids.squeeze(0) # p b t -> b t
		n_microbatches = input_ids.shape[0] // self.embed_comparison
		all_indices, all_ngrams = [], []
		for n in range(n_microbatches):
			matching_index, sample_ngram = self.select_ngram(input_ids[n*self.embed_comparison:n*self.embed_comparison+self.embed_comparison, :])
			pad_ngram = (torch.ones(input_ids.shape[1] - self.ngram_size) * self.pad_token_id).to(torch.long).to(input_ids.device)

			if self.padding_side == 'right':
				sample_ngram = torch.cat((sample_ngram, pad_ngram), dim=0)
				attention_ngram = torch.cat((torch.ones(self.ngram_size, dtype=torch.long), torch.zeros(input_ids.shape[1] - self.ngram_size, dtype=torch.long)), dim=0).to(input_ids.device)
			elif self.padding_side == 'left':
				sample_ngram = torch.cat((pad_ngram, sample_ngram), dim=0)
				attention_ngram = torch.cat((torch.zeros(input_ids.shape[1] - self.ngram_size, dtype=torch.long), torch.ones(self.ngram_size, dtype=torch.long)), dim=0).to(input_ids.device)

			#sample_ngram_embedding = self.encoder(sample_ngram.unsqueeze(0))[0, -1, :]
			#embeddings[n*self.embed_comparison] = sample_ngram_embedding # zeroth batch element is the query phrase
			all_indices.append(matching_index)
			all_ngrams.append(sample_ngram)

		all_ngrams = torch.stack((all_ngrams), dim=0).to(input_ids.device)
		if self.wte:
			all_ngrams = self.wte(all_ngrams)
		sample_embeddings = self.encoder(all_ngrams)[:, -1, :]
		embedding_swap_indices = [n*self.embed_comparison for n in range(n_microbatches)]
		embeddings[embedding_swap_indices] = sample_embeddings
		# reshape embeddings into minibatches
		embeddings = rearrange(embeddings, '(b c) h -> b c h', c=self.embed_comparison)
		# label reassignment
		labels = torch.tensor(all_indices, dtype=torch.long).unsqueeze(1).to(input_ids.device)
		if isinstance(self.decoder, AbbreviatedModel):
			output = self.decoder(embeddings)
		else:
			output = self.decoder(inputs_embeds=embeddings).last_hidden_state # assumes a LlamaModel
		output = self.retrieval_head(output)
		loss = self.cel(output, labels)
		return loss, output

def half_data(example):
	example['input_ids'] = example['input_ids'][256:]
	if 'attention_mask' in example:
		example['attention_mask'] = example['attention_mask'][256:]
	return example

@torch.no_grad()
def embed_data(example):
	x = torch.tensor(example['input_ids']).to(device)
	if isinstance(autoencoder.encoder, AbbreviatedModel):
		x = autoencoder.wte(x)
		x = autoencoder.encoder(x).to('cpu')
	else:
		x = autoencoder.encoder(x).last_hidden_state.to('cpu')
	# x has shape [b, t, e]
	example['embeddings'] = x[:, -1, :]
	return example

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
	n_layers = 8
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

	train_path = f"{data_root}/fineweb-edu-tokenized-train-lpad-c512"
	test_path = f"{data_root}/fineweb-edu-tokenized-test-lpad-c512"
	# datasets.config.IN_MEMORY_MAX_SIZE = 1e9
	# train_dataset = load_from_disk(train_path)
	# test_dataset = load_from_disk(test_path)
	decoder_dim = 512
	encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
	decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
	autoencoder = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=tokenized_length, compression=1, freeze_encoder=False).to(device)
	load_model(autoencoder, '/home/bbadger/Desktop/fineweb_autoencoding_transformer_512c1_d512_n8_c256_b32x4/checkpoint-200000/model.safetensors')
	decoder = LlamaModel(configuration)

	embed_comparison = 4
	model = RetrievalAutoencoderTransformer(autoencoder, custom_decoder=decoder, ngram_size=ngram, embed_comparison=embed_comparison, padding_side='right', pad_token_id=tokenizer.pad_token_id).to(device)

	datasets.config.IN_MEMORY_MAX_SIZE = 1e9 
	train_dataset = load_from_disk(train_path).take(1000000).map(half_data, batched=False, num_proc=16).map(embed_data, batched=True, batch_size=128)
	test_dataset = load_from_disk(test_path).take(10000).map(half_data, batched=False, num_proc=16).map(embed_data, batched=True, batch_size=64)
	
	model = model.to('cpu')
	tokenizer.pad_token = tokenizer.eos_token
	pad_token = int(tokenizer.encode(tokenizer.pad_token)[-1])

	batch_size = 128
	total_sample_size = batch_size * embed_comparison
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
		per_device_train_batch_size=total_sample_size,
		per_device_eval_batch_size=total_sample_size,
		gradient_accumulation_steps=1,
		warmup_steps=500,
		eval_steps=1000,
		save_steps=10000,
		logging_steps=500,
		learning_rate=0.5e-4,
		fp16=True,
		eval_strategy='steps',
		output_dir=output_dir,
		optim='adamw_torch',
		overwrite_output_dir=True,
		save_safetensors=True,
		max_steps=200000,
		torch_compile=True
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	)

	trainer.train()
