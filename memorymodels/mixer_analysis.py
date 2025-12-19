import os
import shutil
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import torch.nn as nn
import mlflow
import datasets
from datasets import load_dataset, load_from_disk
import safetensors
import torch.distributed._shard.checkpoint as dist_cp

from mixer_clm import LanguageMixer
from mixer_multiconv import MultiHeadedMixer
from mixer_autoencoder import RecurrentMemoryMixer
from mixer_autoencoder import AutoencodingMixer, AutoencodingTransfixer, MemoryMixer, ProjMemoryMixer, FrozenMemoryMixer, VariableMemoryMixer
from mixer_autoencoder import TruncatedModel, RecurrentMemoryMixer
from memory_transformer import MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from safetensors.torch import load_model

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')
print (data_root)

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu' # NB 'cuda' but not indices are compatible with accelerate

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)

def load_clm(d=1024):
	tokenized_length = 512
	dim = d
	layers = 16
	n_heads = None
	model = LanguageMixer(
		n_vocab, dim, layers, tokenized_length
	)

	checkpoint_path = checkpoint_root + f'/fineweb_mixer_{d}_c512.safetensors'
	load_model(model, checkpoint_path)
	return model

def load_autoencoder(d=512):
	tokenized_length = 512
	dim = d
	layers = 16
	n_heads = None
	model = AutoencodingMixer(
		n_vocab, dim, layers, tokenized_length, kernel=4
	)

	checkpoint_path = checkpoint_root + f'/fineweb_autoencoding_mixer_{d}_n16_c512.safetensors'
	load_model(model, checkpoint_path)
	return model


@torch.no_grad()
def kernel_dims(weight_vectors):
	all_kernels = []
	for i, m in enumerate(weight_vectors):
		U, S, Vh = np.linalg.svd(m)
		all_kernels.append(np.sum(np.where(np.abs(S)>1e-6, 1, 0)))
	return all_kernels

def plot_weights(weight_vectors):
	fig, axes = plt.subplots(4, 4, figsize=(8, 8)) # Create a 4x4 grid of subplots
	axes = axes.flatten()

	for i, ax in enumerate(axes):
		ax.imshow(weight_vectors[i].detach(), cmap='berlin', interpolation='nearest', vmin=-0.3, vmax=0.3)
		# ax.set_title(f'Layer {i}', fontsize='small')
		ax.axis('off')

	plt.tight_layout()
	plt.show()
	plt.close()
	return

model = load_autoencoder().to('cpu')
mixer_layers = [model.encoderblocks[i].conv.weight.squeeze(0)[:, :, 0] for i in range(len(model.encoderblocks))]
print (mixer_layers[0].shape)
plot_weights(mixer_layers)
print (kernel_dims(mixer_layers))


