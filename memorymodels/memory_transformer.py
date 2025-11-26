import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from peft.peft_model import PeftModelForCausalLM
import mlflow
from datasets import load_dataset
from mixer_autoencoder import MixerBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def copy_dataset(input_ids, blank_copy=False):
        n_ctx = len(input_ids[0])
        for i, input in enumerate(input_ids):
                first_half = input[:n_ctx//2]
                if blank_copy:
                        copied_halves = torch.cat((first_half, torch.ones(first_half.shape).to(first_half.device))).to(torch.long)
                else:
                        copied_halves = torch.cat((first_half, first_half)).to(torch.long)
                input_ids[i] = copied_halves
        return input_ids

def copy_labels(labels):
	n_ctx = len(labels[0])
	for i, input in enumerate(labels):
		first_half = input[:n_ctx//2]
		pad_half = torch.ones(first_half.shape).to(device) * -100
		halves = torch.cat((pad_half, first_half)).to(torch.long)
		labels[i] = halves
	return labels
	
class RecurrentMemoryTransformer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length, n_heads=4, n_chunks=4):
		super().__init__()

		llama_config_kwargs = {
			'hidden_size': dim,
			'intermediate_size': 4*dim,
			'num_hidden_layers': depth,
			'num_attention_heads': n_heads,
			'vocab_size': n_vocab
		}
		decoder_configuration = LlamaConfig(**llama_config_kwargs)
		self.decoder = LlamaModel(decoder_configuration)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.chunks = n_chunks
		self.decoder_dim = dim

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		input_ids = input_ids.to(device)
		total_loss = 0

		for c in range(self.chunks):
			x = input_ids[:, c*self.tokenized_length: (c+1)*self.tokenized_length]
			decoder_embeds = self.decoder_wte(x)
			# attention mask is of shape [b, t]
			if c == 0:
				encoder_embedding = torch.ones((input_ids.shape[0], 1, self.decoder_dim)).to(device)
				attention_insert = torch.zeros(attention_mask.shape[0], 1).to(device)
			else:
				attention_inset = torch.ones(attention_mask.shape[0], 1).to(device)
			
			attention_mask = torch.cat((attention_insert, attention_mask), dim=1)	
			decoder_embeds[:, -1, :] = encoder_embedding.squeeze(1)
			x = torch.cat((encoder_embedding, decoder_embeds), dim=1)
			x = self.decoder(inputs_embeds=x).last_hidden_state #, attention_mask=attention_mask).last_hidden_state
			encoder_embedding = x[:, -1, :].unsqueeze(1)
			output = self.lm_head(x)
			if labels.dim() > 2:
				labels = rearrange(labels, 'b p t -> b (p t)')
			output = rearrange(output, 'b t e -> b e t')
			shift_labels, shift_logits = labels, output
			shift_logits = output[..., 1:-1].contiguous() # first c 'tokens' are encoding
			shift_labels = labels[..., (c*self.tokenized_length)+1:(c+1)*(self.tokenized_length)].contiguous()
			loss = self.cel(shift_logits, shift_labels)
			total_loss += loss
		mean_loss = total_loss / self.chunks
		return mean_loss, output


class VariableMemoryTransformer(nn.Module):
	"""
	Fully featured memory model. self.encoder and self.decoder are LlamaModel objects, with
	fixed or variable position embedding introduction into the decoder.
	"""

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=1, n_heads=4, n_chunks=4, fixed_memory=True, frozen_encoder=None, no_memory=False, copy=False, decoder=None, blank_copy=False):
		super().__init__()

		self.no_memory = no_memory
		self.decoder_dim = dim
		if not self.no_memory:
			if frozen_encoder:
				for name, param in frozen_encoder.named_parameters():
					param.requires_grad = False
				self.encoder = frozen_encoder
			else:
				llama_config_kwargs = {
					'hidden_size': encoder_dim,
					'intermediate_size': 4*encoder_dim,
					'num_hidden_layers': depth,
					'num_attention_heads': n_heads,
					'vocab_size': n_vocab
				}
				encoder_configuration = LlamaConfig(**llama_config_kwargs)
				self.encoder = LlamaModel(encoder_configuration)
		else:
			self.encoder = None

		self.wte = nn.Embedding(n_vocab, encoder_dim)
		self.decoder_proj = None

		if decoder:
			if isinstance(decoder, PeftModelForCausalLM):
				self.decoder = decoder.base_model.model.model
				self.decoder_wte = decoder.base_model.model.model.embed_tokens
				self.lm_head = decoder.base_model.model.lm_head
			else:
				self.decoder = decoder.model
				self.decoder_wte = decoder.model.embed_tokens
				self.lm_head = decoder.lm_head
		else:
			llama_config_kwargs = {
				'hidden_size': dim,
				'intermediate_size': 4*dim,
				'num_hidden_layers': depth,
				'num_attention_heads': n_heads,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = LlamaModel(decoder_configuration)
			self.decoder_wte = nn.Embedding(n_vocab, dim)
			self.lm_head = nn.Linear(dim, n_vocab, bias=False)

		if encoder_dim != dim:
			self.decoder_proj = nn.Linear(encoder_dim, dim) # project if necessary

		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		self.chunks = n_chunks
		self.fixed_memory = fixed_memory
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)
		self.copy = copy
		self.blank_copy = blank_copy
		

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		input_ids = input_ids.to(device)

		if self.copy:
			input_ids = copy_dataset(input_ids, blank_copy=self.blank_copy)
			if labels is not None:
				labels = copy_labels(labels) # masks first half

		# generate encoder embeddings
		embedding_array = []
		i = 0
		if self.no_memory:
			i = 1e9
			embedding_array = [torch.ones((input_ids.shape[0], 1, self.decoder_dim)).to(device) for _ in range(self.chunks)]

		while input_ids.shape[1] - self.tokenized_length > i:
			input_chunk, attention_chunk = self.wte(input_ids[:, i: i+self.tokenized_length]), attention_mask[:, i: i+self.tokenized_length]
			
			x = self.encoder(inputs_embeds=input_chunk, attention_mask=attention_chunk)
			if not torch.is_tensor(x):
				x = x.last_hidden_state
			
			encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
			if self.compression:
				encoder_embedding = self.down(encoder_embedding)
				encoder_embedding = self.up(encoder_embedding)
			if self.decoder_proj:
				encoder_embedding = self.decoder_proj(encoder_embedding)
			embedding_array.append(encoder_embedding)
			i += self.tokenized_length

		# embedding_array now stores length // n_ctx - 1 embeddings
		input_embeddings = self.decoder_wte(input_ids)
		total_loss = 0
		all_outputs = []
		for c in range(self.chunks): # self.chunks
			decoder_embeds = input_embeddings[:, (c*self.tokenized_length):(c+1)*self.tokenized_length]
			if self.fixed_memory:
				pad = torch.ones((input_ids.shape[0], self.chunks-c, input_embeddings.shape[2])).to(device)
				x = torch.cat((embedding_array[:c] + [pad] + [decoder_embeds]), dim=1) # concatenation on token dim
			else:
				x = torch.cat((embedding_array[:c] + [decoder_embeds]), dim=1) # concatenation on token dim
			if attention_mask is not None:
				attention_mask = torch.cat((torch.ones(input_ids.shape[0], c).to(device), attention_mask), dim=1)
			
			# feed pre-concatenated input embeddings to the transformer decoder
			x = self.decoder(inputs_embeds=x, attention_mask=attention_mask)
			output = self.lm_head(x.last_hidden_state)
			if labels.dim() > 2:
				labels = rearrange(labels, 'b p t -> b (p t)')
			output = rearrange(output, 'b t e -> b e t')
			
			if self.fixed_memory:
				all_outputs.append(output[..., self.chunks:self.chunks+self.tokenized_length]) # assemble all outputs
				shift_logits = output[..., self.chunks:self.chunks+self.tokenized_length-1].contiguous()
			else:
				all_outputs.append(output[..., c:]) # assemble all outputs
				shift_logits = output[..., c:-1].contiguous() # first c 'tokens' are encoding
			shift_labels = labels[..., (c*self.tokenized_length)+1:(c+1)*(self.tokenized_length)].contiguous()
			loss = self.cel(shift_logits, shift_labels)
			if not torch.isnan(loss):
				total_loss += loss
		if total_loss == 0:
			total_loss = loss # if no chunks are valid
		mean_loss = total_loss / self.chunks
		all_outputs = torch.cat(all_outputs, dim=2) # concat in token dim
		return mean_loss, all_outputs

class ParVarMemTransformer(nn.Module):
	"""
	Parallelized memory transformer
	"""

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=1, n_heads=4, n_chunks=4, fixed_memory=True, frozen_encoder=None, no_memory=False, copy=False, decoder=None):
		super().__init__()

		self.no_memory = no_memory
		self.decoder_dim = dim
		if not self.no_memory:
			if frozen_encoder:
				for _, param in frozen_encoder.named_parameters():
					param.requires_grad = False
				self.encoder = frozen_encoder
			else:
				llama_config_kwargs = {
					'hidden_size': encoder_dim,
					'intermediate_size': 4*encoder_dim,
					'num_hidden_layers': depth,
					'num_attention_heads': n_heads,
					'vocab_size': n_vocab
				}
				encoder_configuration = LlamaConfig(**llama_config_kwargs)
				self.encoder = LlamaModel(encoder_configuration)
		else:
			self.encoder = None

		self.wte = nn.Embedding(n_vocab, encoder_dim)
		self.decoder_proj = None

		if decoder:
			self.decoder = decoder.model
			self.decoder_wte = decoder.wte
			self.lm_head = decoder.lm_head

		else:
			llama_config_kwargs = {
				'hidden_size': dim,
				'intermediate_size': 4*dim,
				'num_hidden_layers': depth,
				'num_attention_heads': n_heads,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = LlamaModel(decoder_configuration)
			self.decoder_wte = nn.Embedding(n_vocab, dim)
			self.lm_head = nn.Linear(dim, n_vocab, bias=False)

		if encoder_dim != dim:
			self.decoder_proj = nn.Linear(encoder_dim, dim) # project if necessary

		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		self.chunks = n_chunks
		self.fixed_memory = fixed_memory
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)
		self.copy = copy
		

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		input_ids = input_ids.to(device)

		if self.copy:
			input_ids = copy_dataset(input_ids)
			if labels is not None:
				labels = copy_labels(labels) # masks first half

		# generate encoder embeddings
		embedding_array = []
		i = 0
		if self.no_memory:
			i = 1e9
			embedding_array = [torch.ones((input_ids.shape[0], 1, self.decoder_dim)).to(device) for _ in range(self.chunks)]

		all_input_chunks = [], all_attention_chunks = []
		while input_ids.shape[1] - self.tokenized_length > i:
			input_chunk, attention_chunk = self.wte(input_ids[:, i: i+self.tokenized_length]), attention_mask[:, i: i+self.tokenized_length]
			all_input_chunks.append(input_chunk)
			all_attention_chunks.append(attention_chunk)

		input_chunks = torch.stack(all_input_chunks, dim=1) # [b c t e] shape
		attention_chunks = torch.stack(all_attention_chunks, dim=1)

		
		x = self.encoder(inputs_embeds=input_chunks, attention_mask=attention_chunks)
		if not torch.is_tensor(x):
			x = x.last_hidden_state
		
		encoder_embeddings = x[:, :, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)
		if self.decoder_proj:
			encoder_embedding = self.decoder_proj(encoder_embedding)
		embedding_array.append(encoder_embedding)
		i += self.tokenized_length

		# embedding_array now stores length // n_ctx - 1 embeddings
		input_embeddings = self.decoder_wte(input_ids)
		total_loss = 0
		all_outputs = []
		all_decoder_embeds = []
		for c in range(self.chunks): # self.chunks
			decoder_embeds = input_embeddings[:, (c*self.tokenized_length):(c+1)*self.tokenized_length]

			if self.fixed_memory:
				pad = torch.ones((input_ids.shape[0], self.chunks-c, input_embeddings.shape[2])).to(device)
				x = torch.cat((embedding_array[:c] + [pad] + [decoder_embeds]), dim=1) # concatenation on token dim
			else:
				x = torch.cat((embedding_array[:c] + [decoder_embeds]), dim=1) # concatenation on token dim
			if attention_mask is not None:
				attention_mask = torch.cat((torch.ones(input_ids.shape[0], c).to(device), attention_mask), dim=1)
			all_decoder_embeds.append(x)
			all_attention_masks.append(attention_mask)

		input_chunks = torch.stack(all_decoder_embeds, dim=1) # [b c t e] shape
		attention_chunks = torch.stack(all_attention_masks, dim=1)
		
		# feed pre-concatenated input embeddings to the transformer decoder
		x = self.decoder(inputs_embeds=input_chunks, attention_mask=attention_masks)
		output = self.lm_head(x.last_hidden_state)	
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b c t e -> b e (c t)', c=c)

		# pick up here: check shape valid for llamamodel, concat in batch dim if not
		
		shift_labels, shift_logits = labels, output
		if self.fixed_memory:
			all_outputs.append(output[..., self.chunks:self.chunks+self.tokenized_length]) # assemble all outputs
			shift_logits = output[..., self.chunks:self.chunks+self.tokenized_length-1].contiguous()
		else:
			all_outputs.append(output[..., c:]) # assemble all outputs
			shift_logits = output[..., c:-1].contiguous() # first c 'tokens' are encoding
		shift_labels = labels[..., (c*self.tokenized_length)+1:(c+1)*(self.tokenized_length)].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		total_loss += loss

		mean_loss = total_loss / self.chunks
		all_outputs = torch.cat(all_outputs, dim=2) # concat in token dim
		return mean_loss, all_outputs


class MemoryTransformer(nn.Module):
	"""
	Oracle memory model, implemented for single-embedding introduction to the decoder
	"""

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=4, combination_dim='token', transformer_encoder=None, n_heads=0, kernel=1, random=False, noise_embedding=False):
		super().__init__()
		self.noise_embedding=noise_embedding
		self.use_transformer_encoder = False
		if transformer_encoder:
			self.encoder = transformer_encoder
			self.use_transformer_encoder = True
		
		else:
			self.wte = nn.Embedding(n_vocab, encoder_dim)
			self.encoderblocks = nn.ModuleList(
					[MixerBlock(
						dim = encoder_dim,
						length = length,
						causal = True,
						n_heads = n_heads,
						kernel = kernel
						)
					for i in range(depth)]
				).to(device)

		self.decoder_proj = None
		self.combination_dim = combination_dim
					
		if combination_dim == 'token':
			llama_config_kwargs = {
				'hidden_size': dim,
				'intermediate_size': 4*dim,
				'num_hidden_layers': depth,
				'num_attention_heads': 4,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = LlamaModel(decoder_configuration)
			self.decoder_wte = nn.Embedding(n_vocab, dim)
			self.lm_head = nn.Linear(dim, n_vocab, bias=False)
			if encoder_dim != dim:
				self.decoder_proj = nn.Linear(encoder_dim, dim)

		elif combination_dim == 'embedding':
			llama_config_kwargs = {
				'hidden_size': dim + encoder_dim//compression,
				'intermediate_size': 4*(dim + encoder_dim//compression),
				'num_hidden_layers': depth,
				'num_attention_heads': 4,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = LlamaModel(decoder_configuration)
			self.decoder_wte = nn.Embedding(n_vocab, dim)
			self.lm_head = nn.Linear(dim + encoder_dim//compression, n_vocab, bias=False)

		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)

		self.n_vocab = n_vocab
		self.random = random
		

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
				encoder_embedding += torch.rand(encoder_embedding.shape).to(device)*(2**-3) # assumes e4m3 target quant
			if self.combination_dim == 'token':
				encoder_embedding = self.up(encoder_embedding)
		decoder_embeds = self.decoder_wte(input_ids)
		if self.combination_dim == 'token':
			if self.decoder_proj:
				encoder_embedding = self.decoder_proj(encoder_embedding)
			x = torch.cat((encoder_embedding, decoder_embeds), dim=1) # concatenation on token dim
			if attention_mask is not None:
				attention_mask = torch.cat((torch.ones(input_ids.shape[0], 1).to(device), attention_mask), dim=1)

		elif self.combination_dim == 'embedding':
			repeat_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
			x = torch.cat((repeat_embedding, decoder_embeds), dim=2) # concatenation on hidden dim
		
		# feed pre-concatenated input embeddings to the transformer decoder
		x = self.decoder(inputs_embeds=x, attention_mask=attention_mask)
		output = self.lm_head(x.last_hidden_state)
		if labels is not None:
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
		else:
			loss = 0
		return loss, output

class FrozenMemoryTransformer(nn.Module):
	"""
	Transformer memory model using a frozen pre-trained encoder. Implemented for token concatenation.
	"""

	def __init__(self, n_vocab, encoder_model, encoder_dim, dim, depth, length, compression=4, n_heads=4):
		super().__init__()
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.decoder_proj = None

		# ensures frozen params in encoder
		for name, param in encoder_model.named_parameters():
			param.requires_grad = False

		self.encoder = encoder_model
		llama_config_kwargs = {
				'hidden_size': dim,
				'intermediate_size': 4*dim,
				'num_hidden_layers': depth,
				'num_attention_heads': n_heads,
				'vocab_size': n_vocab
			}
		decoder_configuration = LlamaConfig(**llama_config_kwargs)
		self.decoder = LlamaModel(decoder_configuration)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)

		if encoder_dim != dim:
			self.decoder_proj = nn.Linear(encoder_dim, dim)

		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)
		

	def forward(self, input_ids, labels=None, **kwargs):
		input_ids = input_ids.to(device)
		x = self.encoder(input_ids)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)

		decoder_embeds = self.decoder_wte(input_ids)
		if self.decoder_proj:
			encoder_embedding = self.decoder_proj(encoder_embedding)
		x = torch.cat((encoder_embedding, decoder_embeds), dim=1) # concatenation on token dim

		x = self.decoder(x)
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_labels, shift_logits = labels, output
		shift_logits = output[..., 1:-1].contiguous() # first 'token' is encoding
		shift_labels = labels[..., 1:].contiguous() 
		loss = self.cel(shift_logits, shift_labels)
		return loss, output

class ProjMemoryTransformer(nn.Module):
	"""
	Transformer memory model implemented for token projection-based encoder-decoder transfer
	"""

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=4, n_heads=0, kernel=1):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, encoder_dim)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.encoderblocks = nn.ModuleList(
				[MixerBlock(
					dim = encoder_dim,
					length = length,
					causal=True,
					n_heads = n_heads,
					kernel = kernel
					)
				for i in range(depth)]
			).to(device)
		
		llama_config_kwargs = {
				'hidden_size': dim,
				'intermediate_size': 4*(dim),
				'num_hidden_layers': depth,
				'num_attention_heads': 4,
				'vocab_size': n_vocab
			}
		decoder_configuration = LlamaConfig(**llama_config_kwargs)
		self.decoder = LlamaModel(decoder_configuration)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)

		self.decoder_proj = nn.Linear(encoder_dim, dim)

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		input_ids = input_ids.to(device)
		wte_embeds = self.wte(input_ids)
		x = wte_embeds
		for block in self.encoderblocks:
			x = block(x)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)

		if self.decoder_proj:
			encoder_embedding = self.decoder_proj(encoder_embedding)
		repeated_embeddings = encoder_embedding.repeat(1, self.tokenized_length, 1)

		decoder_embeds = self.decoder_wte(input_ids)
		x = decoder_embeds + repeated_embeddings # linear combination of h and token wtes
		
		# feed pre-concatenated input embeddings to the transformer decoder
		x = self.decoder(inputs_embeds=x, attention_mask=attention_mask)
		output = self.lm_head(x.last_hidden_state)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_labels, shift_logits = labels, output
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous() 
		loss = self.cel(shift_logits, shift_labels)
		return loss, output
