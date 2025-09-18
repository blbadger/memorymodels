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
from mixer_clm import LanguageMixer
from mixer_autoencoder import RecurrentMemoryMixer
from mixer_autoencoder import AutoencodingMixer, AutoencodingTransfixer, MemoryMixer, ProjMemoryMixer, FrozenMemoryMixer, VariableMemoryMixer
from mixer_autoencoder import TruncatedModel, RecurrentMemoryMixer
from memory_transformer import MemoryTransformer, ProjMemoryTransformer
import warnings
from dotenv import load_dotenv
import pathlib

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu' # NB 'cuda' but not indices are compatible with accelerate
tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

def hamming_metric(generated_tokens, input_tokens, *args):
    # expects tokens to be pre-flattened
    assert len(input_tokens) == len(generated_tokens)
    count, card = 0, 0
    pad_token = tokenizer.encode(tokenizer.pad_token)[-1] # will be [2]
    for i in range(len(tokens)):
        if input_tokens[i] == pad_token:
            continue
        else:
            card += 1
            if input_tokens[i] in generated_tokens[i]:
                count += 1
    return (card - count) / card

if __name__ == '__main__':
    tokenized_length = 256
    encoder_dim = 512
    decoder_dim = 512
    n_layers = 8
    compression = 1
    heads = 0
    kernel = 1

    # mixer model initialization
    model = LanguageMixer(n_vocab, decoder_dim, 8, tokenized_length, n_heads=heads, kernel=kernel).float().to(device)

    encoder = LanguageMixer(n_vocab, decoder_dim, n_layers, tokenized_length, n_heads=heads, kernel=kernel).float().to(device)
    encoder = AutoencodingMixer(n_vocab, encoder_dim, n_layers, tokenized_length, compression=compression, n_heads=heads, kernel=kernel, unroll=False, random=False)

    encoder = encoder.model

    #frozen_encoder = encoder.encoderblocks
    frozen_encoder = TruncatedModel(encoder, autoencoder=True).model_blocks
    model = AutoencodingMixer(n_vocab, 
        encoder_dim, 
        n_layers, 
        tokenized_length, 
        compression=compression, 
        n_heads=heads, 
        kernel=kernel, 
        unroll=True, 
        random=False, 
        frozen_encoder=frozen_encoder, 
        clm_encoder=False)  

    safetensors.torch.load_model(encoder, '/home/azureuser/autoencoder_pretrained_retrieval/model.safetensors')

    train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-8k"
    test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-8k"

    datasets.config.IN_MEMORY_MAX_SIZE = 50e9
    train_dataset = load_from_disk(train_path, keep_in_memory=None)
    test_dataset = load_from_disk(test_path, keep_in_memory=None)
    mlflow.end_run()

    batch_size = 32
    n_devices = 4
    # get number of devices (assumes that all visible devices are used for training)
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()

    # descriptive name for output
    output_dir = f'{checkpoint_root}/fineweb_pretrainedauto_mixer\
    _{encoder_dim}\
    c{compression}\
    _d{decoder_dim}\
    _n{n_layers}\
    _c{tokenized_length}_b{batch_size}x{n_devices}'

    training_arguments = transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=5000,
        eval_steps=4000,
        save_steps=8000,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        fp16=True,
        bf16=False,
        eval_strategy='steps',
        output_dir=output_dir,
        optim='adamw_torch',
        overwrite_output_dir=True,
        save_safetensors=True,
        max_steps=200000
    )

    trainer = transformers.Trainer(
        model=model.to(device),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_loss_func=
    )


    trainer.evaluate()