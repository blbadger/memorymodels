import os

# ignores CUDA devices on bus, model and inputs are in CPU mem
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from mixer_autoencoder import AutoencodingMixer
from mixer_clm import LanguageMixer


tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

tokenized_length = 3
dim = 10
device = 'cpu'
#model = LanguageMixer(n_vocab, dim, 1).float().to(device)
model = AutoencodingMixer(n_vocab, dim, 2, tokenized_length).float().to(device)

one = torch.tensor([[[1, 2, 3]]]).to(device)
two = torch.tensor([[[1, 4, 3]]]).to(device)
print (model(one, labels=one))
print (model(two, labels=two))
