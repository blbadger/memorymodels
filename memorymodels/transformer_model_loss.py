
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
all_context_losses = []

def embedding_unreduced_loss(model_output, input_tokens, reduction=False, *args, **kwargs):
    target = input_tokens[:, 1:]
    mask = torch.where(target==1, 0, 1)
    target = torch.where(target==1, -100, target)
    #model_output = rearrange(model_output.logits, 'b t e -> b e t')[:, :, :-1]
    model_output = model_output[1]
    model_output = model_output[..., 1: -1]
    #print (target)
    loss = loss_fn(model_output, target) 
    loss_per_sample, tokens_per_sample = torch.sum(loss, dim=1)/torch.sum(mask, dim=1) , torch.sum(mask, dim=1)
    print (loss_per_sample, tokens_per_sample)
    total_loss, total_tokens = 0, 0
    for i, l in enumerate(loss_per_sample):
        if tokens_per_sample[i] == 1023:
            total_loss += l
            total_tokens += 1
    all_context_losses.append(total_loss / total_tokens)
    if reduction:
        non_pad_tokens = torch.sum(torch.where(target==-100, 0, 1))
       # print (non_pad_tokens)
        loss = torch.sum(loss)/non_pad_tokens
    return loss


def clm_unreduced_loss(model_output, input_tokens, reduction=False, *args, **kwargs):
    target = input_tokens[:, 1:]
    mask = torch.where(target==1, 0, 1)
    target = torch.where(target==1, -100, target)
    model_output = rearrange(model_output.logits, 'b t e -> b e t')[:, :, :-1]
    loss = loss_fn(model_output, target) 
    loss_per_sample, tokens_per_sample = torch.sum(loss, dim=1)/torch.sum(mask, dim=1) , torch.sum(mask, dim=1)
    print (loss_per_sample, tokens_per_sample)
    total_loss, total_tokens = 0, 0
    for i, l in enumerate(loss_per_sample):
        if tokens_per_sample[i] == 1023:
            total_loss += l
            total_tokens += 1
    all_context_losses.append(total_loss / total_tokens)
    if reduction:
        non_pad_tokens = torch.sum(torch.where(target==-100, 0, 1))
       # print (non_pad_tokens)
        loss = torch.sum(loss)/non_pad_tokens
    return loss


loss_fn = nn.CrossEntropyLoss(reduction='none')
def test_losses():
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-8k"
    take_n = 32
    test_dataset = load_from_disk(test_path, keep_in_memory=None).take(take_n)
    print_attributions = {}

    print_losses = {}
    batch = test_dataset[:take_n]
    input_ids = torch.stack([torch.tensor(l) for l in batch['input_ids']], dim=0).to(device_id)
    attention_mask = torch.stack([torch.tensor(l) for l in batch['attention_mask']], dim=0).to(device_id)
    output = model.forward(input_ids, attention_mask=attention_mask, labels=input_ids)
    reshaped_output = rearrange(output.logits, 'b t e -> b e t')

    losses = loss_fn(reshaped_output[:, :, :-1], input_ids[:, 1:])
    print (losses, torch.sum(losses * attention_mask[:, 1:]) / torch.sum(attention_mask))
    #for i in range(take_n):
    #    if test_dataset[i]['input_ids'][0] != 1:
    #        print_losses[test_dataset[i]['id']] = losses[i].tolist()

    #d = {'losses': print_losses}
    #with open('/home/badger/clm_losses.json', 'w') as f:
   #     json.dump(d, f)
    	
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
    #print (trainer.evaluate())
    return


if __name__ == '__main__':

    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    encoder_dim = 256
    decoder_dim = 512
    context_length = 1024
    n_layers = 16
    n_heads = 4
    compression = 4

    vocab_size = 8000
    llama_config_kwargs = {
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': n_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size
    }

    # Initializing a LLaMA model
    configuration = LlamaConfig(**llama_config_kwargs)

    # Initializing a model from the llama-7b style configuration
    model = LlamaForCausalLM(configuration).float()
    safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_transformer_512_n16_c1024_b64/checkpoint-200000/model.safetensors', strict=True) # no decoder_input_embeds param in original model
   
    #encoder_model = LlamaModel(configuration)
    #model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, compression=compression, transformer_encoder=encoder_model, n_heads=n_heads, noise_embedding=False) 
    #safetensors.torch.load_model(model, f'{checkpoint_root}/fineweb_memtrans_256c4_d512_n16_c1024_b16x4/checkpoint-200000/model.safetensors', strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)

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

    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)

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
    batch_size = 256
    attributions = []
    ids = []
    if len(test_dataset) % batch_size == 0:
        batches = len(test_dataset) // batch_size
    else:
        batches = len(test_dataset) // (batch_size) + 1

    #test_losses()
    masks = []  
    for sample_index in tqdm(range(batches)):
        batch = test_dataset[sample_index*batch_size:sample_index*batch_size + batch_size]
        attention_mask = torch.tensor(batch['attention_mask']).to(device_id)
        input_ids = torch.tensor(batch['input_ids']).to(device_id)
        ids.append(batch['id'])
        with torch.no_grad():
            outputs = model.forward(input_ids, attention_mask) # for clm: labels=None)
            loss = clm_unreduced_loss(outputs, input_ids, reduction=False)
            attributions.append(loss.to('cpu'))
    print (all_context_losses)
    torch.distributed.barrier()
    tokenizer.pad_token = tokenizer.eos_token
    attributions_dict = {'memory_attribution': attributions, 'ids': ids}
   # print (attributions_dict)
    #attributions_dataset = Dataset.from_dict(attributions_dict)
    #attributions_dataset.save_to_disk(f"{data_root}/fineweb-edu-tokenized-train-test-lpad-8k_{rank}")
