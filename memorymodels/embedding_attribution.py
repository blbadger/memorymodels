import os
import shutil
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import torch.nn as nn
import mlflow
import datasets
from datasets import load_dataset, load_from_disk, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import safetensors
from safetensors.torch import save_file, save_model
import copy
import json
import re

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'
encoder_dim = 256
decoder_dim = 512
context_length = 1024
compression = 4
n_layers = 16
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
	'hidden_size': encoder_dim,
	'intermediate_size': 4*encoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size
}

class AttributableMemoryTransformer(MemoryTransformer):

	def __init__(self, occlude_memory=False):
		super().__init__()
		self.occlude_memory = occlude_memory
		self.cel = nn.CrossEntropyLoss(reduction=None)

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		if self.random:
			input_ids = torch.randint(1, self.n_vocab, input_ids.shape)
		input_ids = input_ids.to(device)
		if not self.use_transformer_encoder:
			wte_embeds = self.wte(input_ids)
			x = wte_embeds
			for block in self.encoderblocks:
				x = block(x)
		else:
			x = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
		
		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			if self.noise_embedding:
				encoder_embedding += torch.rand(encoder_embedding.shape).to(device)*(2**-2) # assumes e4m3 target quant
			if self.combination_dim == 'token':
				encoder_embedding = self.up(encoder_embedding)
		decoder_embeds = self.decoder_wte(input_ids)

		if self.combination_dim == 'token':
			if self.decoder_proj:
				encoder_embedding = self.decoder_proj(encoder_embedding)
			if self.occlude_memory:
				x = torch.cat((torch.zeros(encoder_embedding.shape), decoder_embeds), dim=1)
				if attention_mask is not None:
					attention_mask = torch.cat((torch.zeros(input_ids.shape[0], 1).to(device), attention_mask), dim=1)

			else:
				x = torch.cat((encoder_embedding, decoder_embeds), dim=1) # concatenation on token dim
				if attention_mask is not None:
					attention_mask = torch.cat((torch.ones(input_ids.shape[0], 1).to(device), attention_mask), dim=1)

		elif self.combination_dim == 'embedding':
			repeat_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
			x = torch.cat((repeat_embedding, decoder_embeds), dim=2) # concatenation on hidden dim
		
		# feed pre-concatenated input embeddings to the transformer decoder
		decoder_input_embeds = x
		x = self.decoder(inputs_embeds=x, attention_mask=attention_mask)
		output = self.lm_head(x.last_hidden_state)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_labels, shift_logits = labels, output
		if self.combination_dim == 'token':
			shift_logits = output[..., 1:-1].contiguous() # first 'token' is encoding
		else:
			shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous() 
		loss = self.cel(shift_logits, shift_labels)

		return loss, output, decoder_input_embeds

def gradientxinput(model, input_ids):
	"""
	Performs gradientxinput on the decoder of an embedding-augmented model
	"""

	# expects for input_ids to be a tensor
	loss, output, decoder_input_embeds = model.forward(input_ids)
	memory_gradxinputs = []
	for token_index in range(len(output[0])):
		isolated_loss = loss[token_index]
		isolated_loss.backward()
		memory_grad = torch.sum(decoder_input_embeds.grad[..., 0])
		memory_gradxinputs.append(memory_grad * decoder_input_embeds[..., 0])
	memory_gradxinputs = torch.tensor(memory_gradxinputs)
	return memory_gradxinputs


def memory_occlusion(model, input_ids, output_measure=True):
	"""
	Occlusion attribution on memory embedding
	"""

	loss, output, _ = model.forward(input_ids, occlude_memory=False)
	occluded_loss, occluded_output, _ = model.forward(input_ids, occlude_memory=True)
	if output_measure:
		measure = torch.abs(occluded_output - output)
	else:
		# measure occlusion on unreduced model loss
		measure = torch.abs(occluded_loss - loss)
	return measure

def normalize_attributions(attributions, method='minmax'):
	if method == 'minmax':
		# attributions are scaled to [0, 1]
		maximums = torch.max(dim=0)
		mins = torch.max(dim=0)
		ranges = maximums - mins
		attributions -= mins
		attributions /= ranges

	entropy_estimates = 1 - attributions
	return entropy_estimates


if __name__ == '__main__':
	# Initializing a LLaMA model
	configuration = LlamaConfig(**llama_config_kwargs)

	# Initializing a model from the llama-7b style configuration
	encoder_model = LlamaModel(configuration).float()

	#encoder_model = LlamaModel(configuration)
	# model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression, n_heads=n_heads, random=False)

	model = AttributableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, transformer_encoder=encoder_model, compression=compression, n_heads=n_heads, random=False) # or 
	safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4_extended/checkpoint-500000/model.safetensors')
	print (model)

	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)

	train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-8k"
	test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"

	# if you have a new dataset, map before loading from disk
	datasets.config.IN_MEMORY_MAX_SIZE = 10e9
	train_dataset = load_from_disk(train_path, keep_in_memory=None)
	test_dataset = load_from_disk(test_path, keep_in_memory=None)
	print (test_dataset[0])

	mlflow.end_run()

	batch_size = 32
	sample_index = 0
	attributions = []
	while sample_index < len(test_dataset):
		batch = test_dataset[sample_index, sample_index + batch_size]
		attributions.append(memory_occlusion(model, batch))

	attributions = torch.stack(attributions, dim=0)
	attributions = normalize_attributions(attributions)
	attributions_dict = {'memory_attribution': attributions}
	attributions_dataset = Dataset.from_dict(attributions_dict)
	attributions_dataset.save_to_disk(f"{data_root}/fineweb-edu-tokenized-train-occlusion-lpad-8k")


	
