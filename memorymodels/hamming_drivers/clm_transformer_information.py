import os
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import mlflow

from datasets import load_dataset, load_from_disk
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import safetensors
import datasets
import warnings
import shutil
from dotenv import load_dotenv

from original_transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer
from hamming_metric import hamming_metric

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

@torch.no_grad()
def hamming(model_output, labels):
        total_metric = 0 
        # no shift for autoencoders
        labels= torch.tensor(labels)
        model_output = torch.tensor(model_output[0])
        nonpad_tokens = torch.where(labels != -100, 1, 0)
        equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
        average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
        return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
        preds, labels = eval_preds
        hamming_metric = hamming(preds, labels)
        return {'Hamming Distance': hamming_metric}

def preprocess_logits_for_metrics(logits, labels):
    """ 
    Original Trainer has a memory leak: a workaround to avoid saving all tensors
    """
    pred_ids = torch.argmax(logits, dim=-2)
    return pred_ids, labels


def tokenize_and_preprocess(example):
        text = example['text']
        global context_length
        tokens = tokenizer(text, max_length=context_length, padding='max_length', truncation=True) # return list, not tensor
        example['input_ids'] = tokens['input_ids']
        example['attention_mask'] = tokens['attention_mask']
        return example

encoder_dim = 512
decoder_dim = 512
context_length = 512
compression = 1
n_layers = 8
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
    'hidden_size':decoder_dim,
    'intermediate_size': 4*decoder_dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
#model = LlamaForCausalLM(configuration).float()

# uncomment for transformer autoencoder
# Initializing a model from the llama-7b style configuration
# encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
# model = AutoencodingTransformer(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

#encoder_model = LlamaModel(configuration)
#decoder_model = LlamaModel(configuration)
#model = AutoencodingTransformerMod(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

encoder_model = LlamaForCausalLM(configuration)
safetensors.torch.load_model(encoder_model, '/home/badger/fineweb_llama_512_n8_h4/model.safetensors')
encoder_model = AbbreviatedModel(encoder_model, tokenized_length=context_length)
#encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)

model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, 										compression=compression, freeze_encoder=True)

#encoder_model = LlamaModel(configuration)
#model = MemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, compression=compression, transformer_encoder=encoder_model, n_heads=n_heads)
#model = VariableMemoryTransformer(vocab_size, encoder_dim, decoder_dim, n_layers, context_length, n_heads=n_heads, n_chunks=4, fixed_memory=True, frozen_encoder=encoder)

#model = RecurrentMemoryTransformer(vocab_size, decoder_dim, n_layers, context_length, n_heads=4, n_chunks=4)

#model = UnrolledAutoencodingTransformer(vocab_size, decoder_dim, encoder_model, decoder_model, tokenized_length=context_length, 
#										compression=compression, freeze_encoder=True)

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c512-lpad-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c512-lpad-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer model
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data


# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_frozen_clmtransenc_unrolledautoencoder\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b32x4'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	learning_rate=2e-4, 
	fp16=True, 
	eval_strategy='steps',
	output_dir=output_dir,
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics = compute_hamming_metric,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

safetensors.torch.load_model(model, output_dir + '/checkpoint-200000/model.safetensors')
print (trainer.evaluate())
