
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
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel, AutoModel, AutoModelForCausalLM
import safetensors
from safetensors.torch import save_file, save_model
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import warnings
from dotenv import load_dotenv

warnings.filterwarnings(action='ignore')
all_context_losses = []

def convert_loss(text, tokenizer, large_tokenizer, large_tokens, large_token_losses):
    """
    Converts loss from large tokenizer to small tokenizer via per-character averages
    """

    # losses are in [b t] shape
    all_losses_per_position = []
    for token_sequence, loss_sequence in zip(large_tokens, large_token_losses):
        losses_per_position = {}
        index = 0
        for token, loss in zip(token_sequence, loss_sequence):
            chars = large_tokenizer.decode(token)
            for j in range(len(chars)):
                index += 1
                losses_per_position[index] = large_token_losses[index]
        all_losses_per_position.append(losses_per_position)
    
    all_converted_losses = []
    for text, losses_per_position in zip(text, all_losses_per_position):
        small_tokens = tokenizer.encode(text)
        start_index = 0
        converted_losses = []
        for token in small_tokens:
            token_chars = len(tokenizer.decode(token))
            end_index = start_index + token_chars
            average_loss = sum([losses_per_position[i] for i in range(start_index, end_index)]) / token_chars
            start_index += token_chars
            converted_losses.append(average_loss)
        all_converted_losses.append(converted_losses)

    loss_tensor = torch.stack([torch.tensor(i) for i in all_converted_losses], dim=0)
    return converted_losses

def clm_unreduced_loss(model_output, input_tokens, text, reduction=False, *args, **kwargs):
    target = input_tokens[:, 1:]
    mask = torch.where(target==1, 0, 1)
    target = torch.where(target==1, -100, target)
    model_output = rearrange(model_output.logits, 'b t e -> b e t')[:, :, :-1]
    loss = loss_fn(model_output, target)
    converted_loss = convert_loss(text, tokenizer, large_tokenizer, input_tokens, loss)
    loss_per_sample, tokens_per_sample = torch.sum(converted_loss, dim=1)/torch.sum(mask, dim=1) , torch.sum(mask, dim=1)
    total_loss, total_tokens = 0, 0
    for i, l in enumerate(loss_per_sample):
        if tokens_per_sample[i] == 1023:
            total_loss += l
            total_tokens += 1
    all_context_losses.append(total_loss / total_tokens)
    if reduction:
        non_pad_tokens = torch.sum(torch.where(target==-100, 0, 1))
        loss = torch.sum(loss)/non_pad_tokens
    return converted_loss

loss_fn = nn.CrossEntropyLoss(reduction='none')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'
model = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B')
large_tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B')
large_tokenizer.pad_token = large_tokenizer.eos_token
tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

use_ddp = False
if not use_ddp: 
    device_id = 0   
    model = model.to(device)
else:
    gpu_count = torch.cuda.device_count()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    model = DDP(model.to(device_id), device_ids=[device_id])

train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"

# if you have a new dataset, map before loading from disk
datasets.config.IN_MEMORY_MAX_SIZE = 10e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)

n_gpus = torch.cuda.device_count()
dataset_length = len(test_dataset)
device_chunk_size = int(dataset_length / n_gpus)
start, end = device_id * device_chunk_size, (device_id+1) * device_chunk_size
test_dataset = test_dataset.skip(start).take(end - start)
mlflow.end_run()
batch_size = 16
attributions = []
ids = []
if len(test_dataset) % batch_size == 0:
    batches = len(test_dataset) // batch_size
else:
    batches = len(test_dataset) // (batch_size) + 1

masks = []  
for sample_index in tqdm(range(batches)):
    batch = test_dataset[sample_index*batch_size:sample_index*batch_size + batch_size]
    attention_mask = torch.tensor(batch['attention_mask']).to(device_id)
    input_ids = torch.tensor(batch['input_ids']).to(device_id)
    text = tokenizer.batch_decode(input_ids)
    large_input_ids = large_tokenizer(text, padding='max_length', max_length=1024)['input_ids']
    large_input_tensor = torch.stack([torch.tensor(s) for s in large_input_ids], dim=0).to(device_id)
    large_attention_mask = torch.where(large_input_tensor==128001, 0, 1).to(device_id)
    ids.append(batch['id'])
    with torch.no_grad():
        outputs = model.forward(large_input_tensor, large_attention_mask) # for clm: labels=None)
        loss = clm_unreduced_loss(outputs, input_ids, text, reduction=False)
        attributions.append(loss.to('cpu'))
print (all_context_losses)
torch.distributed.barrier()
tokenizer.pad_token = tokenizer.eos_token
attributions_dict = {'memory_attribution': attributions, 'ids': ids}
# print (attributions_dict)
#attributions_dataset = Dataset.from_dict(attributions_dict)
#attributions_dataset.save_to_disk(f"{data_root}/fineweb-edu-tokenized-train-test-lpad-8k_{rank}")
