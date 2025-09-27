
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
from pprint import pprint
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'
decoder_dim = 512
context_length = 1024
n_layers = 16
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
	'hidden_size': decoder_dim,
	'intermediate_size': 4*decoder_dim,
	'num_hidden_layers': n_layers,
	'num_attention_heads': n_heads,
	'vocab_size': vocab_size
}

def unreduced_loss(model_output, input_tokens, *args, **kwargs):
    target = input_tokens[:, 1:]
    model_output = rearrange(model_output.logits, 'b t e -> b e t')[:, :, :-1]
    loss = loss_fn(model_output, target) 
    non_pad_tokens = torch.sum(torch.where(input_tokens==-100, 0, 1))
    loss = torch.sum(loss)/non_pad_tokens
    return loss

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float().to('cuda')
safetensors.torch.load_model(model, f"{checkpoint_root}/fineweb_transformer_512_n16_c1024_b64_extended/checkpoint-500000/model.safetensors")
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"
take_n = 32
test_dataset = load_from_disk(test_path, keep_in_memory=None).take(take_n)
print_attributions = {}

print_losses = {}
batch = test_dataset[:take_n]
input_ids = torch.stack([torch.tensor(l) for l in batch['input_ids']], dim=0).to('cuda')
attention_mask = torch.stack([torch.tensor(l) for l in batch['attention_mask']], dim=0).to('cuda')
output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
reshaped_output = rearrange(output.logits, 'b t e -> b e t')

loss_fn = nn.CrossEntropyLoss(reduction='none')
losses = loss_fn(reshaped_output[:, :, :-1], input_ids[:, 1:])

for i in range(take_n):
    if test_dataset[i]['input_ids'][0] != 1:
        print_losses[test_dataset[i]['id']] = losses[i].tolist()

d = {'losses': print_losses}
with open('/home/badger/clm_losses.json', 'w') as f:
    json.dump(d, f)
	
training_arguments = transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=8000,
        learning_rate=2e-4, 
        fp16=True, 
        eval_strategy='steps',
        output_dir='/home/badger',
        optim='adamw_torch',
        overwrite_output_dir=False,
        max_steps=200000
)

trainer = transformers.Trainer(
        model=model,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	compute_loss_func=unreduced_loss
	)
print (trainer.evaluate())
