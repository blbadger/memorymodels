import os
import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

def ConvForward(dim, expansion_factor=1):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1),
		nn.GELU(),
		nn.Conv1d(inner_dim, dim, 1)
		)

class DoubleMixerBlock(nn.Module):

	def __init__(self, dim, length, clm_mask=False, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernormf = nn.LayerNorm(dim)
		self.seq_layernormr = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.convf = nn.Conv1d(length, length, 1)
			self.convr = nn.Conv1d(length, length, 1)
		self.clm_mask = clm_mask
		self.expand_conv = expand_conv
		self.softmax = nn.Softmax(dim=0)
	
	def forward(self, x: torch.tensor, y: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
			y = rearrange(y, 'b p t f -> (b p) t f')

		residualf, residualr = x, y
		x, y = self.seq_layernormf(x), self.seq_layernormr(y)
		
		# Apply causal masking using functional operations (torch.compile friendly)
		if self.clm_mask and not self.expand_conv:
			# Get device from input tensor
			device = x.device
			
			# Apply masks functionally without modifying weights
			weight_f = self.convf.weight
			weight_r = self.convr.weight
			
			# Reshape and mask
			masked_weight_f = rearrange(weight_f, 'f d p -> p f d')
			masked_weight_f = torch.tril(masked_weight_f)
			masked_weight_f = rearrange(masked_weight_f, 'p f d -> f d p').contiguous()
			
			masked_weight_r = rearrange(weight_r, 'f d p -> p f d')
			masked_weight_r = torch.triu(masked_weight_r, diagonal=2)
			masked_weight_r = rearrange(masked_weight_r, 'p f d -> f d p').contiguous()
			
			x = F.conv1d(x, masked_weight_f, bias=self.convf.bias) + residualf
			y = F.conv1d(y, masked_weight_r, bias=self.convr.bias) + residualr
		else:
			x, y = self.convf(x) + residualf, self.convr(y) + residualr
			
		residualf, residualr = x, y
		x, y = self.patch_layernorm(x), self.patch_layernorm(y)
		x, y = self.patch_ff(x) + residualf, self.patch_ff(y) + residualr
		return x, y

class MixerBlock(nn.Module):

	def __init__(self, dim, length=512, expand_conv=False, kernel=1, n_heads=0):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.n_heads = n_heads
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			if n_heads > 0:
				self.conv = MixerHead(dim, length, dim//n_heads, n_heads)
			else:
				self.conv = nn.Conv1d(length, length, kernel, padding='same')
		self.expand_conv = expand_conv

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		residual = x
		x = self.seq_layernorm(x)
		
		# Apply causal masking functionally for torch.compile compatibility
		if not self.expand_conv and self.n_heads == 0:
			# Only apply masking for standard conv (not for MixerHead or expand_conv)
			weight = self.conv.weight
			masked_weight = rearrange(weight, 'f d p -> p f d')
			masked_weight = torch.tril(masked_weight)
			masked_weight = rearrange(masked_weight, 'p f d -> f d p').contiguous()
			
			# Use functional conv1d with masked weight
			x = F.conv1d(x, masked_weight, bias=self.conv.bias, padding='same') + residual
		else:
			x = self.conv(x) + residual
			
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class MixerHead(nn.Module):

	def __init__(self, dim, length, hidden_dim, n_heads):
		super().__init__()
		self.n_heads = n_heads
		self.proj_head = nn.ModuleList(
			[nn.Linear(dim, hidden_dim)
			for i in range(n_heads)]
			)

		self.convs = nn.ModuleList(
			[nn.Conv1d(length, length, 1)
			for i in range(n_heads)]
			)

		self.out_proj = nn.Linear(dim, dim)
		self.GeLU = nn.GELU()

	def forward(self, x: torch.tensor):
		hidden_layer = []

		for head in range(self.n_heads):
			weight = self.convs[head].weight
			masked_weight = torch.tril(rearrange(weight, 'f d p -> p f d'))
			masked_weight = rearrange(masked_weight, 'p f d -> f d p').contiguous()
			
			projection = self.proj_head[head](x)
			conv_projection = F.conv1d(projection, masked_weight, bias=self.convs[head].bias)
			hidden_layer.append(conv_projection)

		# concatenate and project multi-headed output
		hidden_layer = torch.cat(hidden_layer, dim=2)
		hidden_layer = self.out_proj(hidden_layer)
		return hidden_layer

class MLPMixerBlock(nn.Module):
	"""
	Matrix multiply-based implementation of inter-token weights (instead of 1D convs). Less performant and
	flexible than 1D convs, prefer that unless there is a specific reason to use matmults.
	"""

	def __init__(self, dim, length, **kwargs):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.conv = nn.Linear(length, length)


	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		residual = x
		x = self.seq_layernorm(x)
		x = rearrange(x, 'b t h -> b h t')
		
		masked_weight = torch.tril(self.conv.weight)
		x = F.linear(x, masked_weight, self.conv.bias)
		
		x = rearrange(x, 'b h t -> b t h')
		x += residual

		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x

class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length, tie_weights=False, n_heads=0, kernel=1):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length,
				expand_conv = False,
				n_heads = n_heads,
				kernel = kernel
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			self.wte.weight = self.lm_head.weight
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		x = self.wte(x)
		for i, block in enumerate(self.mixerblocks):
			x = block(x)
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output


def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params += params
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params

