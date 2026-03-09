import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

def preprocess_embeddings_for_metrics(embeddings, labels):
    """
    Original Trainer has a memory leak: a workaround to avoid saving all tensors
    """
    embeddings = embeddings[:, index, :]
    return embheddings, labels

@torch.no_grad()
def decoder_accuracy(logits, index):
	top_index = int(torch.topk(logits, 1).indices[0])
	return top_index == index

@torch.no_grad()
def infonce_accuracy(embeddings, index):
	# expects indices to be [b t e]
	embeddings = F.normalize(embeddings, p=2, dim=1).detach()
	query_embedding = embeddings[0, :]
	target_embeddings = embeddings[1:, :]
	scores = query_embedding @ target_embeddings.T * 100
	top_index = int(torch.topk(scores, 1).indices[0])
	if top_index == index:
		return torch.tensor(1)
	else:
		return torch.tensor(0)


if __name__ == '__main__':
	embeddings = torch.randn((10, 1024)) # b t e
	index = torch.tensor(0)
	print (embeddings.shape)
	print (infonce_accuracy(embeddings, index))





