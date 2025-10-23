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
import json
from tqdm import tqdm

from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer, AutoencodingTransformerMod, UnrolledAutoencodingTransformer
from memory_transformer import VariableMemoryTransformer, MemoryTransformer, RecurrentMemoryTransformer, ProjMemoryTransformer

warnings.filterwarnings(action='ignore')

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

class EEModel(torch.nn.Module):
    def __init__(self, model, hidden_dim, no_shift=True):
        super().__init__()
        self.eem_head = torch.nn.Linear(hidden_dim, 1)
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.no_shift=no_shift

    def forward(self, input_ids, labels, attention_mask, attribution, *args, **kwargs):
        model_output = self.model(input_ids, attention_mask).last_hidden_state
        estimations = self.eem_head(model_output).squeeze(-1)
        if self.no_shift:
            loss = self.loss_fn(estimations[..., 1:], attribution)
        else:
            loss = self.loss_fn(estimations[..., :-1], attribution)
        return loss, estimations 
 
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
    'vocab_size': vocab_size,
}
print (llama_config_kwargs)
# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = EEModel(LlamaModel(configuration).float(), decoder_dim, no_shift=False)

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

print (model)
train_path = f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-lossattr-8k"
test_path = f"{data_root}/fineweb-edu-tokenized-test-c1024-lpad-lossattr-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

batch_size = 32
n_devices = 4
# get number of devices (assumes that all visible devices are used for training)
#if torch.cuda.is_available():
#    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_2eem\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=100,
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
#trainer.train()
trainer.train(output_dir + '/checkpoint-200000')
#print (trainer.evaluate())

def save_estimates(test_dataset, batch_size = 128):
	if len(test_dataset) % batch_size == 0:
		batches = len(test_dataset) // batch_size
	else:
		batches = len(test_dataset) // (batch_size) + 1
	
	start, end = 0, batch_size
	batch = test_dataset[:batch_size]
	mlflow.end_run()
	attributions = []
	ids = []
	masks = []  
	attention_mask = torch.tensor(batch['attention_mask']).to(device)
	input_ids = torch.tensor(batch['input_ids']).to(torch.long).to(device)
	attribution = torch.tensor(batch['attribution']).to(device)
	ids.append(batch['id'])
	with torch.no_grad():
		loss, outputs = model.forward(input_ids, input_ids, attention_mask=attention_mask, attribution=attribution) # for clm: labels=None)
		print (loss)
		attributions.append(outputs.to('cpu'))
	attributions_dict = {'memory_attribution': attributions, 'ids': ids}
	print_attributions = {}
	for i, attribution in enumerate(attributions[0]):
		if test_dataset[i]['input_ids'][0] != 1:
			print_attributions[ids[0][i]] = [batch['input_ids'][i], attribution.tolist()]

	d = {'losses': print_attributions}
	with open('/home/badger/second_order_loss.json', 'w') as f:
		json.dump(d, f)
		print (attributions_dict)
	return

save_estimates(test_dataset) 
#save_estimates(test_dataset.filter(lambda x: x['input_ids'][0] != 1, num_proc=16))
      
