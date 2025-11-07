import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import html
import webbrowser
from transformers import AutoTokenizer


def smooth_over(data, over=50):
    totals, indices = [], []
    for i in range(0, len(data), over):
        total = sum(data[i:i+over])
        if i + over < len(data):
            totals.append(total/over)
        else:
            totals.append(total/(len(data)-i))
        indices.append(i)
    return indices, totals

def plot_attribution_indices():
    data = json.load(open('big_loss_attributions.json'))['losses']
    data = [i[1] for i in data.values()]
    # data2 = json.load(open('cosine_attr.json'))['losses']
    indices = [i for i in range(1024)]
    all_indices = [indices for i in range(len(data))]

    # for i in range(4):
    #     plt.plot(all_indices[i], data[i], alpha=0.3)
    # plt.plot(all_indices[0], data2[0], alpha=0.3)
    vals = torch.mean(torch.tensor(data), dim=0)
    print ('Number of samples: ', len(data))
    plt.plot(indices, vals, alpha=0.6)

    indices, totals = smooth_over(vals, over=10)
    plt.plot(indices, totals)
    plt.xlabel('Token Index',  fontsize='x-large')
    plt.ylabel('Loss',  fontsize='x-large')
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()

def readable_interpretation(all_tokens, all_attributions, tokenizer, norm=True):
    all_text = ''
    for tokens, attributions in zip(all_tokens, all_attributions):
        # right shift attrs
        attributions = [0.5] + attributions[:-1]
        if norm:
            min_attribution, max_attribution = min(attributions), max(attributions)
            attributions = [(i - min_attribution) / max_attribution for i in attributions]
        # summed_ems is a torch.tensor object of normalized input attributions per token
        positions = [i for i in range(len(tokens))][1:]
        decoded_input = [tokenizer.decode(tokens[i]) for i in range(len(tokens))]

        # assemble HTML file with red (high) to blue (low) attributions per token
        highlighted_text = []
        for i in range(len(positions)):
            word = decoded_input[i]
            red, green, blue = int((attributions[i]*255)), 150, 150
            color = '#{:02x}{:02x}{:02x}'.format(red, green, blue)
            highlighted_text.append(f'<span style="background-color: {color}">{word}</span>')
        highlighted_text = ''.join(highlighted_text)
        all_text += '<br><br>' + highlighted_text

    with open('loss_large_data.html', 'wt', encoding='utf-8') as file:
        file.write(all_text)
    return
    
data_dict = json.load(open('big_loss_attributions.json'))['losses']
all_tokens = [list(data_dict.values())[i][0] for i in range(1,2)]
all_values = [list(data_dict.values())[i][1] for i in range(1,2)]
tokenizer = AutoTokenizer.from_pretrained('tokenizer_fineweb_8k')
plot_attribution_indices()
readable_interpretation(all_tokens, all_values, tokenizer)

# def readable_interpretation(all_tokens, all_attributions, tokenizer):
#     all_text = ''
#     for tokens, attributions in zip(all_tokens, all_attributions):
#         attributions = [0.5] + attributions[:-1]
#         # summed_ems is a torch.tensor object of normalized input attributions per token
#         positions = [i for i in range(len(tokens))]
#         decoded_input = [tokenizer.decode(tokens[i]) for i in range(len(tokens))]

#         # assemble HTML file with red (high) to blue (low) attributions per token
#         highlighted_text = []
#         for i in range(len(positions)):
#             word = decoded_input[i]
#             red, green, blue = int(((attributions[i])*325)), 150, 150
#             color = '#{:02x}{:02x}{:02x}'.format(red, green, blue)
#             highlighted_text.append(f'<span style="background-color: {color}">{word}</span>')
#         highlighted_text = ''.join(highlighted_text)
#         all_text += '<br><br>' + highlighted_text


#     with open('loss_data.html', 'wt', encoding='utf-8') as file:
#         file.write(all_text)
#     webbrowser.open('noise_data.html')
    

# data_dict = json.load(open('l1_attributions.json'))['attributions']
# loss_dict = json.load(open('clm_losses.json'))['losses']
# all_tokens = [[list(data_dict.values())[i][0] for i in range(1, 2)][0][:-1]]
# id = '<urn:uuid:c337bcd8-6aa1-4f2d-8c48-b916442ebbee>'
# values = loss_dict[id]
# mini, maxi = min(values), max(values)
# all_values = [[(i - mini)/(maxi - mini) for i in values]]
# print (all_values, all_tokens)
# tokenizer = AutoTokenizer.from_pretrained('tokenizer_fineweb_8k')
# readable_interpretation(all_tokens, all_values, tokenizer)