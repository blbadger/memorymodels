import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
import mlflow
from datasets import load_dataset
from mixer_autoencoder import MixerBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MemoryTransformer(nn.Module):

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=4, combination_dim='embedding', transformer_encoder=None):
		super().__init__()

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
						causal=True
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
		

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
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
		return loss, output


class ProjMemoryTransformer(nn.Module):

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=4):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, encoder_dim)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.encoderblocks = nn.ModuleList(
				[MixerBlock(
					dim = encoder_dim,
					length = length,
					causal=True
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
