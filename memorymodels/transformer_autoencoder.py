import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
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
        elif isinstance(model, AbbreviatedModel):
            self.model = model
        else:
            raise TypeError('model type not recognized')

        self.depth = depth
        self.position_ids = torch.tensor([[i for i in range(tokenized_length)]])

    def forward(self, input_ids: torch.Tensor, **attention_mask: torch.Tensor):
        # 'input_ids' is actually a float tensor, post-wte transformation
        x = input_ids.to(device)
        position_ids = self.position_ids.repeat(input_ids.shape[0], 1).to(device)
        position_embeddings = self.model.rotary_emb(x, position_ids)
        for i in range(self.depth):
            x = self.model.layers[i](x, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        return x

class SuffixModel(LlamaModel):

    def __init__(self, config, start_layer=8):
        super().__init__(config)
        self.start_layer = 8

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
        ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer, decoder_layer in enumerate(self.layers[self.start_layer:self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

class SplitModel(LlamaModel):

    def __init__(self, config, split_layer=8, num_hidden_layers=16):
        super().__init__(config)
        self.split_layer = 8
        self.num_hidden_layers = num_hidden_layers

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
        ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        self.config.num_hidden_layers = 16
        for layer, decoder_layer in enumerate(self.layers[:self.num_hidden_layers]):
            if layer == self.split_layer:
                split_hidden_states = hidden_states

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return split_hidden_states, hidden_states

class UnrolledAutoencodingTransformer(nn.Module):
       
    def __init__(self, n_vocab, dim, encoder_model, decoder_model, decoder_dim=None, tokenized_length=512, compression=1, random=False, freeze_encoder=False, ):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, dim)
        self.encoder = encoder_model
        if freeze_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

        self.decoder = decoder_model
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        self.bridge_proj = None
        if decoder_dim and decoder_dim != dim:
            self.bridge_proj = nn.Linear(dim, decoder_dim)
            self.decoder_dim = decoder_dim
        else:
            decoder_dim = dim
        self.projection = nn.Linear(decoder_dim//2, decoder_dim)
        self.lm_head = nn.Linear(decoder_dim, n_vocab, bias=False)
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
        if isinstance(self.encoder, AbbreviatedModel):
            x = self.wte(x)
            x = self.encoder(x)
        else:
            x = self.encoder(x).last_hidden_state
            #x = self.encoder(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
        if self.bridge_proj:
            encoder_embedding = self.bridge_proj(encoder_embedding)
            self.dim = self.decoder_dim
        if self.compression:
            encoder_embedding = self.down(encoder_embedding)
            encoder_embedding = self.up(encoder_embedding)
        embedding_stack = []
        # sliding window unroll over hidden dim
        for i in range(self.tokenized_length):
            i %= self.dim
            sliding_window = encoder_embedding[..., i:i+self.dim//2]
            if i+self.dim//2 > self.dim:
                residual = i+self.dim//2 - self.dim #self.tokenized_length
                # loop around to first index
                sliding_window = torch.cat((sliding_window, encoder_embedding[..., :residual]), dim=2)
            embedding_stack.append(sliding_window)
        encoder_embedding = torch.cat(embedding_stack, dim=1)
        encoder_embedding = self.projection(encoder_embedding)

        x = encoder_embedding
        if isinstance(self.decoder, AbbreviatedModel):
            x = self.decoder(x)
        else:
            x = self.decoder(inputs_embeds=x).last_hidden_state

        output = self.lm_head(x)
        output = rearrange(output, 'b t e -> b e t')
        if labels is not None:
            loss = self.cel(output, labels)
        else:
            loss = 0
        return loss, output

class AllAutoencodingTransformer(nn.Module):
       
    def __init__(self, n_vocab, dim, encoder_model, decoder_model, decoder_dim=None, tokenized_length=512, compression=1, random=False, freeze_encoder=False, noise_embeddings=False):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, dim)
        self.encoder = encoder_model
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

        self.decoder = decoder_model
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        if decoder_dim and decoder_dim != dim:
            self.bridge_proj = nn.Linear(dim, decoder_dim)
            self.decoder_dim = decoder_dim
        else:
            decoder_dim = dim

        self.lm_head = nn.Linear(decoder_dim, n_vocab, bias=False)
        self.compression = False
        if compression > 1:
            self.compression = True
            self.down = nn.Linear(dim, dim//compression)
            self.up = nn.Linear(dim//compression, dim)
            
        self.random_input = random
        self.n_vocab = n_vocab
        self.noise_embeddings=noise_embeddings

    def forward(self, input_ids, labels=None, attention_mask=None):
        if self.random_input:
            x = torch.randint(1, self.n_vocab, input_ids.shape)
        else:
            x = input_ids
        x = x.to(device).squeeze(1)
        if isinstance(self.encoder, AbbreviatedModel):
            x = self.wte(x)
            x = self.encoder(x)
        else:
            x = self.encoder(x).last_hidden_state
        
        encoder_embedding = x # dim=[batch, token, hidden]

        if self.compression:
            encoder_embedding = self.down(encoder_embedding)
            encoder_embedding = self.up(encoder_embedding)

        if self.noise_embeddings:
            x += torch.randn(x.shape).to(x.device).to(x.dtype)

        x = encoder_embedding
        if isinstance(self.decoder, AbbreviatedModel):
            x = self.decoder(x)
        else:
            x = self.decoder(inputs_embeds=x).last_hidden_state

        output = self.lm_head(x)
        output = rearrange(output, 'b t e -> b e t')
        if labels is not None:
            loss = self.cel(output, labels)
        else:
            loss = 0
        return loss, output

class SecretTransformer(nn.Module):
       
    def __init__(self, 
        n_vocab, 
        dim, 
        encoder_model, 
        clm_decoder, 
        split_model, 
        inversion_decoder, 
        original_clm,
        wte=None, 
        clm_head=None, 
        inversion_head=None, 
        decoder_dim=None, 
        tokenized_length=512, 
        compression=1, 
        random=False, 
        freeze_decoders=True, 
        noise_embeddings=False,
        manual_seed=0
        ):
        super().__init__()
        self.wte = wte
        self.encoder = encoder_model
        self.clm_decoder = clm_decoder
        self.inversion_decoder = inversion_decoder
        self.original_clm = original_clm
        for _, param in self.original_clm.named_parameters():
            param.requires_grad = False

        if freeze_decoders:
            for _, param in self.clm_decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.inversion_decoder.named_parameters():
                param.requires_grad = False

        self.cel = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.tokenized_length = tokenized_length
        self.dim = dim
        if decoder_dim and decoder_dim != dim:
            self.bridge_proj = nn.Linear(dim, decoder_dim)
            self.decoder_dim = decoder_dim
        else:
            decoder_dim = dim

        self.compression = False
        if compression > 1:
            self.compression = True
            self.down = nn.Linear(dim, dim//compression)
            self.up = nn.Linear(dim//compression, dim)
            
        self.random_input = random
        self.n_vocab = n_vocab
        self.noise_embeddings=noise_embeddings
        self.inversion_head=inversion_head
        self.split_model = split_model
        
        # specify pretrained causal lm head and freeze weights
        self.clm_head = clm_head
        self.clm_head.requires_grad = False

        self.original_embedding = None
        self.random_label = None
        self.seed = manual_seed
        self.all_embeddings, self.all_labels = [], []

    def forward(self, input_ids, labels=None, attention_mask=None):
        if self.random_input:
            x = torch.randint(1, self.n_vocab, input_ids.shape)
        else:
            x = input_ids
        x = input_ids.to(device).squeeze(1)
        split_hidden_states, _ = self.split_model(input_ids=x)
        original_logits = self.original_clm(input_ids=x).logits
        original_clm_tokens = torch.argmax(original_logits, dim=-1)

        if self.original_embedding is None:
            self.original_embedding = split_hidden_states.detach()

        encoder_embedding = split_hidden_states # dim=[batch, token, hidden]
        if not self.training:
            self.all_embeddings.append(encoder_embedding.to('cpu'))
            self.all_labels.append(labels.to('cpu'))

        if self.compression:
            encoder_embedding = self.down(encoder_embedding)
            encoder_embedding = self.up(encoder_embedding)

        x = encoder_embedding
        if self.noise_embeddings:
            x += torch.randn(x.shape).to(x.device).to(x.dtype)

        if isinstance(self.inversion_decoder, AbbreviatedModel):
            inverted_x = self.inversion_decoder(x)
        else:
            inverted_x = self.inversion_decoder(inputs_embeds=x).last_hidden_state
        
        if isinstance(self.clm_decoder, AbbreviatedModel):
            clm_x = self.clm_decoder(x)
        else:
            clm_x = self.clm_decoder(inputs_embeds=x).last_hidden_state

        clm_output = self.clm_head(clm_x)
        inverted_output = self.inversion_head(inverted_x)
        clm_output = rearrange(clm_output, 'b t e -> b e t')
        inverted_output = rearrange(inverted_output, 'b t e -> b e t')

        local_rank = int(os.environ.get("LOCAL_RANK", 0))	
        if labels is not None:
            clm_loss = self.cel(clm_output, original_clm_tokens) # starts near 0
            if self.random_label is None:
                torch.manual_seed(self.seed)
                self.random_label = torch.randint_like(labels, low=0, high=8000).to(labels.device).to(labels.dtype)
            #inversion_loss = self.cel(inverted_output, self.random_label) #-self.cel(inverted_output, labels) # cel near 0, we want maximum div
            inversion_loss = torch.abs(9.-self.cel(inverted_output, labels)) # cel near 0, we want maximum div
            if local_rank == 0:
                print (f'Cel loss: {clm_loss}')
                print (f'Inversion loss: {inversion_loss}')
            embedding_mse_loss = self.mse(split_hidden_states, self.original_embedding)
            #print (f'Embedding loss: {embedding_mse_loss}')
            loss = inversion_loss + clm_loss
            #if clm_loss.item() > 1.3:
            #    print ('clm loss added')
            #    loss += clm_loss
        else:
            loss = 0
        return loss, inverted_output


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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=4000,
        logging_strategy='steps',
        logging_steps=50,
        learning_rate=1e-4,
        fp16=True,
        evaluation_strategy='steps',
        output_dir='~/Desktop/tinystories_autoencoding_transformer_n8_b32',
        optim='adamw_torch',
        overwrite_output_dir=True,
        save_safetensors=True,
        torch_compile=True
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

