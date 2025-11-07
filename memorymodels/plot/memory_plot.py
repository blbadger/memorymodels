import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

def plot_curves(input_paths, labels, plot_training=False):
    
    for i, input_path in enumerate(input_paths):
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'loss' in iteration and plot_training:
                loss = iteration['loss']
                if 'memorytrans' in input_path:
                    loss /= 4
                train_losses.append(loss)
                train_steps.append(step/1000)

            if 'eval_loss' in iteration:
                loss = iteration['eval_loss']
                if 'memorytrans' in input_path:
                    loss /= 4
                test_losses.append(loss)
                test_steps.append(step/1000)
    
        label = labels[i]
        plt.plot(test_steps, test_losses)
        plt.scatter(test_steps, test_losses, label=f'{label} Eval')
        if plot_training:
            plt.plot(train_steps, train_losses)
            plt.scatter(train_steps, train_losses, label=f'{label} Train')

    plt.legend(fontsize='large')
    plt.tick_params(labelsize=16)
    plt.xscale('log')
    plt.xlabel('Thousand Steps', fontsize='x-large')
    plt.ylabel('Cross-Entropy Loss', fontsize='x-large')
    plt.show()
    plt.close()
    return

def plot_loss_diff(input_paths, labels):
    nomemory_loss = []
    memory_loss = []
    steps = []

    for i, input_path in enumerate(input_paths):
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']

            if 'eval_loss' in iteration and 'mem' not in input_path:
                loss = iteration['eval_loss']
                nomemory_loss.append(loss)
                steps.append(step/1000)
            
            if 'eval_loss' in iteration and 'mem' in input_path:
                loss = iteration['eval_loss']
                memory_loss.append(loss)

    min_len = min(len(nomemory_loss), len(memory_loss))
    steps = steps[:min_len]
    memory_loss_diff = [i - j for i, j in zip(nomemory_loss, memory_loss)]
    
    plt.plot(steps, memory_loss_diff)
    plt.scatter(steps, memory_loss_diff)

    plt.legend(fontsize='large')
    plt.tick_params(labelsize=16)
    # plt.xscale('log')
    plt.xlabel('Thousand Steps', fontsize='x-large')
    plt.ylabel('Fraction of Loss Recovered', fontsize='x-large')
    plt.show()
    plt.close()
    return

def plot_differential(input_paths, labels):
    nomemory_loss = []
    memory_loss = []
    frozen_memory_loss = []
    steps = []

    for i, input_path in enumerate(input_paths):
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']

            if 'eval_loss' in iteration and ('nomem' in input_path or 'nonmem' in input_path):
                loss = iteration['eval_loss']
                nomemory_loss.append(loss)
                steps.append(step/1000)
            
            if 'eval_loss' in iteration and 'frozen_' in input_path:
                loss = iteration['eval_loss']
                frozen_memory_loss.append(loss)
            
            if 'eval_loss' in iteration and '4_memory' in input_path:
                loss = iteration['eval_loss']
                memory_loss.append(loss)

    min_len = min(len(nomemory_loss), len(memory_loss), len(frozen_memory_loss))
    steps = steps[:min_len]
    memory_loss_diff = [i - j for i, j in zip(nomemory_loss, memory_loss)]
    frozen_loss_diff = [i - j for i, j in zip(nomemory_loss, frozen_memory_loss)]
    relative_benefit = [i/j for i, j in zip(frozen_loss_diff, memory_loss_diff)]
    
    plt.plot(steps, relative_benefit)
    plt.scatter(steps, relative_benefit)

    plt.legend(fontsize='large')
    plt.tick_params(labelsize=16)
    # plt.xscale('log')
    plt.xlabel('Thousand Steps', fontsize='x-large')
    plt.ylabel('Fraction of Loss Recovered', fontsize='x-large')
    plt.show()
    plt.close()

    architectures = ['mixer' for i in range(min_len)]
    print (len(relative_benefit), len(steps), len(architectures))
    data = {
        'Loss Difference': relative_benefit,
        'Step': steps,
        'Architecture': architectures
    }
    dataframe = pd.DataFrame(data)
    print (dataframe)
    sns.set_theme(style="darkgrid")
    g = sns.scatterplot(data=dataframe, x='Step', y='Loss Difference', color='red')
    g = sns.regplot(data=dataframe, x='Step', y='Loss Difference', scatter=False, logx=True, color='red')
    # Add labels and title
    # plt.legend(fontsize='x-large')
    # plt.xscale('log')
    g.set_yticklabels(g.get_yticks(), size = 16)
    g.set_xticklabels(g.get_xticks(), size = 16)
    g.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
    g.get_xaxis().set_major_formatter(FormatStrFormatter('%.0f'))
    g.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())


    plt.xlabel('Thousand Steps', fontsize='large')
    plt.ylabel('Fraction of Memory Loss', fontsize='large')
    # plt.ylim(0, 4)
    plt.tight_layout()
    plt.show()
    return


def plot_combined_differential(input_paths, labels):
    architectures = []
    benefits = []
    all_steps = []
    for arch in ['mixer', 'transformer']:
        nomemory_loss = []
        memory_loss = []
        frozen_memory_loss = []
        steps = []

        for i, input_path in enumerate(input_paths):
            if arch in input_path:
                for iteration in json.load(open(input_path))['log_history']:
                    step = iteration['step']

                    if 'eval_loss' in iteration and ('nomem' in input_path or 'nonmem' in input_path):
                        loss = iteration['eval_loss']
                        nomemory_loss.append(loss)
                        steps.append(step/1000)
                    
                    if 'eval_loss' in iteration and 'frozen_' in input_path:
                        loss = iteration['eval_loss']
                        frozen_memory_loss.append(loss)
                    
                    if 'eval_loss' in iteration and '4_memory' in input_path:
                        loss = iteration['eval_loss']
                        memory_loss.append(loss)

        min_len = min(len(nomemory_loss), len(memory_loss), len(frozen_memory_loss))
        steps = steps[:min_len]
        memory_loss_diff = [i - j for i, j in zip(nomemory_loss, memory_loss)]
        frozen_loss_diff = [i - j for i, j in zip(nomemory_loss, frozen_memory_loss)]
        relative_benefit = [i/j for i, j in zip(frozen_loss_diff, memory_loss_diff)]

        architectures += [arch for _ in range(min_len)]
        benefits += relative_benefit
        all_steps += steps

    print (len(benefits), len(all_steps), len(architectures))

    data = {
        'Loss Difference': benefits,
        'Step': all_steps,
        'Architecture': architectures
    }

    dataframe = pd.DataFrame(data)
    print (dataframe)
    sns.set_theme(style="darkgrid")
    g = sns.scatterplot(data=dataframe, x='Step', y='Loss Difference', hue='Architecture')
    g = sns.regplot(data=dataframe[dataframe['Architecture']=='mixer'], x='Step', y='Loss Difference', scatter=False, logx=True, color='blue')
    g = sns.regplot(data=dataframe[dataframe['Architecture']=='transformer'], x='Step', y='Loss Difference', scatter=False, logx=True, color='red')
    
    # Add labels and title
    # plt.legend(fontsize='x-large')
    # plt.xscale('log')
    g.set_yticklabels(g.get_yticks(), size = 16)
    g.set_xticklabels(g.get_xticks(), size = 16)
    g.get_yaxis().set_major_formatter(FormatStrFormatter('%.1f'))
    g.get_xaxis().set_major_formatter(FormatStrFormatter('%.0f'))
    g.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())


    plt.xlabel('Thousand Steps', fontsize='large')
    plt.ylabel('Fraction of Memory Loss', fontsize='large')
    # plt.ylim(0, 4)
    plt.tight_layout()
    plt.show()
    return

# paths = [
#     "/Users/bbadger/Desktop/fineweb_c256x4_nomemorytrans.json",
#     # "/Users/bbadger/Desktop/fineweb_c256x4_memorytrans.json",
#     "/Users/bbadger/Desktop/fixed_position_mem_transformer.json",
#     "/Users/bbadger/Desktop/fineweb_rmt_256x4.json",
#     # "/Users/bbadger/Desktop/fineweb_c1024_transformer.json",
#     # "/Users/bbadger/Desktop/fineweb_c256x4_nomemorymixer.json",
#     # "/Users/bbadger/Desktop/fineweb_c256x4_memorymixer.json",
#     # "/Users/bbadger/Desktop/fineweb_c1024_mixer_rpad.json",
#     # "/Users/bbadger/Desktop/fineweb_rmm.json"
    
# ]

# labels = [
#     'No Memory Transformer',
#     'Memory Transformer',
#     "Recurrent Memory Transformer",
#     # 'Full-context Transformer',
#     # 'No Memory Mixer',
#     # 'Memory Mixer',
#     # 'Full-context Mixer',
#     # 'Recurrent memory mixer'
# ]

# large loss diff

paths = [
    "/Users/bbadger/Desktop/fineweb_tmem_trans_1024_n24.json",
    "/Users/bbadger/Desktop/fineweb_transformer_1024_n24.json"
]

labels = [
    'Memory Transformer', 
    'Transformer'
]


# paths = [
#     # "/Users/bbadger/Desktop/fineweb_512x4_nomemory_mixer.json",
#     # "/Users/bbadger/Desktop/fineweb_512x4_frozen_mem_mixer.json",
#     # "/Users/bbadger/Desktop/fineweb_512x4_memory_mixer.json",
#     # "/Users/bbadger/Desktop/fineweb_512x4_rmm.json",
#     "/Users/bbadger/Desktop/fineweb_512x4_nonmemory_transformer.json",
#     # "/Users/bbadger/Desktop/fineweb_frozen_memory_transformer.json",
#     "/Users/bbadger/Desktop/fineweb_512x4_frozen_transmem_transformer.json",
#     "/Users/bbadger/Desktop/fineweb_512x4_memory_transformer.json",
#     "/Users/bbadger/Desktop/fineweb_512x4_rmt.json",
#     ]

# labels = [
#     # "No Memory Mixer",
#     # "Frozen Memory Mixer",
#     # "Memory Mixer",
#     # "Recurrent Memory Mixer",
#     "No Memory Transformer",
#     # "Frozen Memory (mixer) Transformer",
#     "Frozen Memory Transformer", 
#     "Memory Transformer",
#     "Recurrent Memory Transformer"
# ]

# paths = [
#     "/Users/bbadger/Desktop/finemath_512x4_nomemory_mixer.json",
#     "/Users/bbadger/Desktop/finemath_512x4_memory_mixer.json"
# ]

# labels = [
#     "memory",
#     "no memory"
# ]

num_plots = 4
colormap = plt.cm.Reds
colormap2 = plt.cm.Reds
cmap_cycler = plt.cycler('color', colormap(np.linspace(0.2, 1, num_plots)))
plt.gca().set_prop_cycle(cmap_cycler)
# cmap2_cycler = plt.cycler('color', colormap2(np.linspace(0.2, 1, num_plots)))
# all_cycler = cmap_cycler.concat(cmap2_cycler)
           
# plt.gca().set_prop_cycle(all_cycler)

# plot_curves(paths, labels)
# plot_loss_diff(paths, labels)
plot_differential(paths, labels)
# plot_combined_differential(paths, labels)




