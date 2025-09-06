import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM
import torch.nn as nn
import mlflow
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AutoencodingTransformerMod(nn.Module):

        def __init__(self, n_vocab, dim, encoder_model, decoder_model, tokenized_length=512):
                super().__init__()
                self.wte = nn.Embedding(n_vocab, dim)
                self.encoder = encoder_model
                self.decoder = decoder_model
                self.lm_head = nn.Linear(dim, n_vocab, bias=False)
                self.cel = nn.CrossEntropyLoss()
                self.tokenized_length = tokenized_length

        def forward(self, input_ids, labels=None, attention_mask=None):
                x = self.encoder(input_ids.to(device))
                encoder_embedding = x.last_hidden_state[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
                encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)

                x = self.decoder(inputs_embeds=encoder_embedding, attention_mask=attention_mask)

                output = self.lm_head(x.last_hidden_state)
                if labels.dim() > 2:
                        labels = rearrange(labels, 'b p t -> b (p t)')
                output = rearrange(output, 'b t e -> b e t')
                loss = self.cel(output, labels)
                return loss, output

class AutoencodingTransformer(nn.Module):

        def __init__(self, n_vocab, dim, encoder_model, decoder_model, tokenized_length=512):
                super().__init__()
                self.wte = nn.Embedding(n_vocab, dim)
                self.encoder = encoder_model
                self.decoder = decoder_model
                self.lm_head = nn.Linear(dim, n_vocab, bias=False)
                self.cel = nn.CrossEntropyLoss()
                self.tokenized_length = tokenized_length

        def forward(self, input_ids, labels=None, attention_mask=None):
                x = input_ids
                x = x.to(device).squeeze(1)
                x = self.wte(x)
                
                x = self.encoder(x)

                encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
                encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
                x = encoder_embedding

                x = self.decoder(x)

                output = self.lm_head(x)
                output = rearrange(output, 'b t e -> b e t')
                loss = self.cel(output, labels)
                return loss, output


class AbbreviatedModel(nn.Module):

        def __init__(self, model, depth=8, tokenized_length=512):
                super().__init__()
                if isinstance(model, LlamaForCausalLM):
                	self.model = model.model
                elif isinstance(model, LlamaModel):
                        self.model = model
                else:
                        raise TypeError('model type not recognized')

                self.depth = depth
                self.position_ids = torch.tensor([[i for i in range(tokenized_length)]]).to(device)

        def forward(self, input_ids: torch.Tensor, **attention_mask: torch.Tensor):
                # 'input_ids' is actually a float tensor, post-wte transformation
                x = input_ids.to(device)
                position_ids = self.position_ids.repeat(input_ids.shape[0], 1).to(device)
                position_embeddings = self.model.rotary_emb(x, position_ids)

                for i in range(self.depth):
                        x = self.model.layers[i](x, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                return x


class UnrolledAutoencodingTransformer(nn.Module):

        def __init__(self, n_vocab, dim, encoder_model, decoder_model, tokenized_length=512, compression=1, random=False, freeze_encoder=False):
                super().__init__()
                self.wte = nn.Embedding(n_vocab, dim)
                self.encoder = encoder_model
                if freeze_encoder:
                        for _, param in encoder_model.named_parameters():
                                param.requires_grad = False

                self.decoder = decoder_model
                self.lm_head = nn.Linear(dim, n_vocab, bias=False)
                self.cel = nn.CrossEntropyLoss()
                self.tokenized_length = tokenized_length
                assert dim >= tokenized_length
                unroll_factor = dim // tokenized_length #assumes
                self.projection = nn.Linear(dim//2, dim)
                self.dim = dim
                self.compression=False
                if compression > 1:
                        self.down = nn.Linear(dim, dim//compression)
                        self.up = nn.Linear(dim//compression, dim); self.compression=True
                self.random_input = random
                self.n_vocab = n_vocab
	

        def forward(self, input_ids, labels=None, attention_mask=None):
                if self.random_input:
                        x = torch.randint(1, self.n_vocab, input_ids.shape)
                else:
                        x = input_ids
                x = x.to(device).squeeze(1)
                x = self.wte(x)
                x = self.encoder(x)

                encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
                if self.compression:
                        encoder_embedding = self.down(encoder_embedding)
                        encoder_embedding = self.up(encoder_embedding)
                embedding_stack = []
                # sliding window unroll over hidden dim
                for i in range(self.tokenized_length):
                        sliding_window = encoder_embedding[..., i:i+self.dim//2]
                        if i+self.dim//2 > self.dim:
                                residual = i+self.dim//2 - self.tokenized_length
                                # loop around to first index
                                sliding_window = torch.cat((sliding_window, encoder_embedding[..., :residual]), dim=2)
                        embedding_stack.append(sliding_window)
                encoder_embedding = torch.cat(embedding_stack, dim=1)
                encoder_embedding = self.projection(encoder_embedding)
                x = encoder_embedding
                x = self.decoder(x)

                output = self.lm_head(x)
                output = rearrange(output, 'b t e -> b e t')
                loss = self.cel(output, labels)
                return loss, output
        
        def __init__(self, n_vocab, dim, encoder_model, decoder_model, tokenized_length=512, compression=1, random=False, freeze_encoder=False):
                super().__init__()
                self.wte = nn.Embedding(n_vocab, dim)
                self.encoder = encoder_model
                if freeze_encoder:
                        for _, param in self.encoder.named_parameters():
                                param.requires_grad = False

                self.decoder = decoder_model
                self.lm_head = nn.Linear(dim, n_vocab, bias=False)
                self.cel = nn.CrossEntropyLoss()
                self.tokenized_length = tokenized_length
                assert dim >= tokenized_length
                unroll_factor = dim // tokenized_length # assumes dim >= tokenized_length
                self.projection = nn.Linear(dim//2, dim)
                self.dim = dim

                self.compression = False
                if compression > 1:
                        self.compression = True
                        self.down = nn.Linear(dim, dim//compression)
                        self.up = nn.Linear(dim//compression, dim)
                        
                self.random_input = random
                self.n_vocab = n_vocab

        def forward(self, input_ids, labels=None, attention_mask=None):
                if self.random_input:
                        x = torch.randint(1, self.n_vocab, input_ids.shape)
                else:
                        x = input_ids
                x = x.to(device).squeeze(1)
                x = self.wte(x)
                x = self.encoder(x)

                encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
                if self.compression:
                        encoder_embedding = self.down(encoder_embedding)
                        encoder_embedding = self.up(encoder_embedding)
                embedding_stack = []
                # sliding window unroll over hidden dim
                for i in range(self.tokenized_length):
                        sliding_window = encoder_embedding[..., i:i+self.dim//2]
                        if i+self.dim//2 > self.dim:
                                residual = i+self.dim//2 - self.tokenized_length
                                # loop around to first index
                                sliding_window = torch.cat((sliding_window, encoder_embedding[..., :residual]), dim=2)
                        embedding_stack.append(sliding_window)
                encoder_embedding = torch.cat(embedding_stack, dim=1)
                encoder_embedding = self.projection(encoder_embedding)
                x = encoder_embedding
                x = self.decoder(x)

                output = self.lm_head(x)
                output = rearrange(output, 'b t e -> b e t')
                loss = self.cel(output, labels)
                return loss, output


def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        print ()
        for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                        continue
                params = parameter.numel()
                table.add_row([name, params])
                total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


def batch_tokenize_input(train_text, test_text, length=20000, batch_size=4096):
        train_data, test_data = [], []
        max_length = 512

        for i in range(0, length, batch_size):
                tokens = tokenizer.batch_encode_plus(
                        train_text[i:i+batch_size]['text'],
                        add_special_tokens=False,
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length,
                        padding='max_length'
                )
                # debatch train data
                for i in range(tokens.input_ids.shape[0]):
                        train_data.append({'input_ids': tokens.input_ids[i, :], 'attention_mask': tokens.attention_mask[i, :]})

        for i in range(0, len(test_text), batch_size):
                tokens = tokenizer.batch_encode_plus(
                        test_text[i:i+batch_size]['text'],
                        add_special_tokens=False,
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length,
                        padding='max_length'
                )
                # debatch test data
                for i in range(tokens.input_ids.shape[0]):
                        test_data.append({'input_ids': tokens.input_ids[i, :], 'attention_mask': tokens.attention_mask[i, :]})
        return train_data, test_data

def reformat_inputs(train_data, test_data):
        # reformat inputs for transformer modelz`
        for i, _ in enumerate(train_data):
                train_data[i] = train_data[i].flatten()

        for i, _ in enumerate(test_data):
                test_data[i] = test_data[i].flatten()
        return train_data, test_data

if __name__ == '__main__':
        tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
        tokenizer.pad_token = tokenizer.eos_token
        n_vocab = len(tokenizer)
        print (tokenizer.is_fast)

        tokenized_length = 512
        dim = 128
                                
        llama_config_kwargs = {
                'hidden_size': dim,
                'intermediate_size': 4*dim,
                'num_hidden_layers': 8,
                'num_attention_heads': 4,
                'vocab_size': 4096
        }

        # Initializing a LLaMA model
        configuration = LlamaConfig(**llama_config_kwargs)

        # Initializing a model from the llama-7b style configuration
        encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
        decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
        model = AutoencodingTransformer(n_vocab, dim, encoder_model, decoder_model)

        count_parameters(model)

        # cached dataset
        train_text = load_dataset("roneneldan/TinyStories", split="train")
        valid_text = load_dataset("roneneldan/TinyStories", split="validation")

        train_data, test_data = batch_tokenize_input(train_text, valid_text)
        if isinstance(model, LlamaForCausalLM):
                reformat_inputs(train_data, test_data)

        mlflow.end_run()
        print ('training begun')

        training_arguments = transformers.TrainingArguments(
                num_train_epochs=7,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                warmup_steps=500,
                eval_steps=4000,
                save_steps=4000,
                learning_rate=1e-4,
                fp16=True,
                evaluation_strategy='steps',
                output_dir='~/Desktop/tinystories_autoencoding_transformer_n8_b32',
                optim='adamw_torch',
                overwrite_output_dir=True,
                save_safetensors=True
        )

        trainer = transformers.Trainer(
                model=model,
                train_dataset=train_data,
                eval_dataset=test_data,
                args=training_arguments,
                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )


        model.train()
        trainer.train()
        for name, param in model.named_parameters():
                print (name)

