import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import matplotlib

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
    # plt.xscale('log')
    plt.xlabel('Thousand Steps', fontsize='x-large')
    plt.ylabel('Cross-Entropy Loss', fontsize='x-large')
    plt.show()
    plt.close()
    return

paths = [
    # "/Users/bbadger/Desktop/fineweb_c256x4_nomemorytrans.json",
    # "/Users/bbadger/Desktop/fineweb_c256x4_memorytrans.json",
    # "/Users/bbadger/Desktop/fineweb_c1024_transformer.json",
    "/Users/bbadger/Desktop/fineweb_c256x4_nomemorymixer.json",
    "/Users/bbadger/Desktop/fineweb_c256x4_memorymixer.json",
    # "/Users/bbadger/Desktop/fineweb_c1024_mixer_rpad.json",
    "/Users/bbadger/Desktop/fineweb_rmm.json"
    
]

labels = [
    # 'No Memory Transformer',
    # 'Memory Transformer',
    # 'Full-context Transformer',
    'No Memory Mixer',
    'Memory Mixer',
    # 'Full-context Mixer',
    'Recurrent memory mixer'
]

num_plots = 3
colormap = plt.cm.Blues
colormap2 = plt.cm.Reds
cmap_cycler = plt.cycler('color', colormap(np.linspace(0.2, 1, num_plots)))
# plt.gca().set_prop_cycle(cmap_cycler)
cmap2_cycler = plt.cycler('color', colormap2(np.linspace(0.2, 1, num_plots)))
all_cycler = cmap_cycler.concat(cmap2_cycler)
           
plt.gca().set_prop_cycle(all_cycler)

plot_curves(paths, labels)


