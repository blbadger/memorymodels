import os
# ignores CUDA devices on bus, model and inputs are in CPU mem
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import torch
from einops import rearrange#
from mixer_autoencoder import AutoencodingMixer
from mixer_clm import LanguageMixer

# model initialization
device = 'cpu'
n_vocab = 100
length = 3
dim = 10
depth = 1
model = LanguageMixer(n_vocab, dim, depth, length).float().to(device)

# test causality
index = 1
one = torch.tensor([[[1, 2, 3]]]).to(device)
two = torch.clone(one)
two[:, :, index] = 4

ones_output = model(one, labels=one)
twos_output = model(two, labels=two)
should_match = (ones_output[1][..., :index], twos_output[1][..., :index])
should_not_match = (ones_output[1][..., index:], twos_output[1][..., index:])

assert torch.allclose(should_match[0], should_match[1])
assert not torch.allclose(should_not_match[0], should_not_match[1])
