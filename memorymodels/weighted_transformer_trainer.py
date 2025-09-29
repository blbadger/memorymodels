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

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

class WeightedModel(torch.nn.Module):
    def __init__(self, model, hidden_dim, tokenizer_size):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden_dim, tokenizer_size)
        self.model = model
        self.cel = torch.nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, input_ids, labels, attention_mask, attribution, *args, **kwargs):
        model_output = self.model(input_ids, attention_mask).last_hidden_state
        model_output = self.lm_head(model_output)
        model_output = rearrange(model_output, 'b t e -> b e t')

        shifted_output = model_output[..., :-1]
        shifted_labels = labels[..., 1:]
        weights = 1 - attribution # complement of attributions
        loss = self.cel(shifted_output, shifted_labels)
        loss *= weights[..., :-1]
        nonpad_tokens = torch.sum(attention_mask)
        loss = torch.sum(loss) / nonpad_tokens
        return loss, model_output
         
def weighted_loss(model_output, input_tokens, *args, **kwargs):
    weights = 1 - attributions # complement of attributions
    cel = torch.nn.CrossEntropyLoss(weight=weights)
    loss = cel(model_output, input_tokens)
    return loss

decoder_dim = 512
context_length = 1024
n_layers = 16
n_heads = 4

vocab_size = 8000
llama_config_kwargs = {
    'hidden_size':decoder_dim,
    'intermediate_size': 4*decoder_dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size
}
print (llama_config_kwargs)
# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

configuration.save_pretrained('/home/badger/fineweb_l1attr_weighted_transformer_d512_n16_c1024_b32x4/checkpoint-200000')
# Initializing a model from the llama-7b style configuration
model = WeightedModel(LlamaModel(configuration).float(), decoder_dim, vocab_size)

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-attr-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-attr-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

batch_size = 32
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_l1attr_weighted_transformer\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	gradient_accumulation_steps=1,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	learning_rate=2e-4, 
	fp16=False,
	bf16=True, 
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
)

print (f"training model, saving to {output_dir}")
# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()
trainer.train()
