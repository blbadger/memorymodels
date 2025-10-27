import numpy as np 
import matplotlib.pyplot as plt
import json
import matplotlib

def plot_curves(input_paths, labels, loss_offset=0.1739*2):
    for i, input_path in enumerate(input_paths):
        if i >= 0:
            print ('here')
            offset = loss_offset
        else:
            offset = 0
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'loss' in iteration:
                train_losses.append(iteration['loss'] + offset)
                train_steps.append(step/1000)
            if 'eval_loss' in iteration:
                test_losses.append(iteration['eval_loss'] + offset)
                test_steps.append(step/1000)
    
        label = labels[i]
        plt.plot(test_steps, test_losses)
        plt.scatter(test_steps, test_losses, label=f'{label} Eval')
        # plt.plot(train_steps, train_losses)
        # plt.scatter(train_steps, train_losses, label=f'{label} Train')

    plt.legend(fontsize='x-large')
    plt.tick_params(labelsize=16)
    # plt.xscale('log')
    plt.xlabel('Thousand Steps', fontsize='x-large')
    plt.ylabel('Normalized Cross-Entropy Loss', fontsize='x-large')
    plt.savefig('/Users/bbadger/Desktop/figure.png', dpi=350)
    plt.show()
    plt.close()
    return

def plot_differential(input_paths, labels):
    mixer_losses, mem_mixer_losses = [], []
    transformer_losses, mem_transformer_losses = [], []
    vl_transformer_losses, vl_mem_transformer_losses = [], []
    test_steps = []
    for i, input_path in enumerate(input_paths):
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'eval_loss' in iteration:
                if 'mixer' in input_path:
                    if 'tmem' in input_path or 'emix' in input_path:
                        mem_mixer_losses.append(iteration['eval_loss'])
                    else:
                        mixer_losses.append(iteration['eval_loss'])
                else:
                    if 'tmem' in input_path or 'emem' in input_path:
                        mem_transformer_losses.append(iteration['eval_loss'])
                    else:
                        transformer_losses.append(iteration['eval_loss'])
                if i < 1:
                    test_steps.append(step/1000)

    mixer_loss_diff = [i-j for i, j in zip(mixer_losses, mem_mixer_losses)]
    transformer_loss_diff = [i-j for i, j in zip(transformer_losses, mem_transformer_losses)]
    print (mixer_loss_diff, transformer_loss_diff, test_steps)

    label = labels[i]
    plt.plot(mixer_losses, mixer_loss_diff)
    plt.plot(transformer_losses, transformer_loss_diff)
    plt.scatter(mixer_losses, mixer_loss_diff, label=f'Mixer')
    plt.scatter(transformer_losses, transformer_loss_diff, label=f'Transformer')

    plt.legend(fontsize='x-large')
    plt.tick_params(labelsize=16)
    # plt.xscale('log')
    plt.xlabel('Non-memory model lossƒ', fontsize='x-large')
    plt.ylabel('Loss Difference', fontsize='x-large')
    plt.show()
    plt.close()
    return


def plot_differential(input_paths, labels):
    smallmix_losses, smallmix_mem_losses = [], []
    mix_losses, mix_mem_losses = [], []
    largetrans_losses, largetrans_mem_losses = [], []
    transformer_losses, mem_transformer_losses = [], []
    smalltrans_losses, smalltrans_mem_losses = [], []
    test_steps = []
    for i, input_path in enumerate(input_paths):
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'eval_loss' in iteration:
                if '1024' in input_path:
                    if 'tmem' in input_path or 'emix' in input_path:
                        largetrans_mem_losses.append(iteration['eval_loss'])
                    else:
                        largetrans_losses.append(iteration['eval_loss'])
                elif '_256' in input_path:
                    if 'tmem' in input_path or 'emix' in input_path:
                        smalltrans_mem_losses.append(iteration['eval_loss'])
                    else:
                        smalltrans_losses.append(iteration['eval_loss'])
                elif '512' in input_path:
                    if 'emix' in input_path:
                        smallmix_mem_losses.append(iteration['eval_loss'])
                    else:
                        smallmix_losses.append(iteration['eval_loss'])
                else:
                    if 'tmem' in input_path:
                        mem_transformer_losses.append(iteration['eval_loss'])
                    elif 'emix' in input_path:
                        mix_mem_losses.append(iteration['eval_loss'])
                    elif 'mix' in input_path:
                        mix_losses.append(iteration['eval_loss'])
                    else:
                        transformer_losses.append(iteration['eval_loss'])
                if i < 1:
                    test_steps.append(step/1000)

    largetrans_loss_diff = [i-j for i, j in zip(largetrans_losses, largetrans_mem_losses)]
    transformer_loss_diff = [i-j for i, j in zip(transformer_losses, mem_transformer_losses)]
    smalltrans_loss_diff = [i-j for i, j in zip(smalltrans_losses, smalltrans_mem_losses)]
    mix_loss_diff = [i-j for i, j in zip(mix_losses, mix_mem_losses)]
    smallmix_loss_diff = [i-j for i, j in zip(smallmix_losses, smallmix_mem_losses)]
    print (len(largetrans_loss_diff), len(transformer_loss_diff), len(test_steps))
    label = labels[i]
    num_plots = 3
    colormap = plt.cm.Reds
    colormap2 = plt.cm.Blues
    cmap_cycler = plt.cycler('color', colormap(np.linspace(0.2, 1, num_plots)))
    cmap2_cycler = plt.cycler('color', colormap2(np.linspace(0.2, 1, num_plots)))
    all_cycler = cmap_cycler.concat(cmap2_cycler)
        
    plt.gca().set_prop_cycle(all_cycler)
    plt.plot(test_steps[:40], largetrans_loss_diff[:40])
    plt.plot(test_steps[:40], transformer_loss_diff[:40])
    plt.plot(test_steps[:40], smalltrans_loss_diff[:40])
    plt.plot(test_steps[:40], mix_loss_diff[:40])
    plt.plot(test_steps[:40], smallmix_loss_diff[:40])
    plt.scatter(test_steps[:40], largetrans_loss_diff[:40], label=f'Large Transformer')
    plt.scatter(test_steps[:40], transformer_loss_diff[:40], label=f'Med Transformer')
    plt.scatter(test_steps[:40], smalltrans_loss_diff[:40], label='Small Transformer')
    plt.scatter(test_steps[:40], mix_loss_diff[:40], label='Large Mixer')
    plt.scatter(test_steps[:40], smallmix_loss_diff[:40], label='Small Mixer')


    plt.legend(fontsize='x-large')
    plt.tick_params(labelsize=16)
    # plt.xscale('log')
    plt.xlabel('Thousand Steps', fontsize='x-large')
    plt.ylabel('Loss Difference', fontsize='x-large')
    plt.show()
    plt.close()
    return

# paths = [
#     # '/Users/bbadger/Desktop/memory_versus_nomemory/memory_emixer.json',
#     # '/Users/bbadger/Desktop/memory_versus_nomemory/mixer.json',
#     # '/Users/bbadger/Desktop/memory_versus_nomemory/ememory_transformer.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/transformer.json',
#     '/Users/bbadger/Desktop/QAT_loss/fineweb_4noise_transformer_tmem.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/transformer_tmemory.json'

# ]
# labels = [
#     # 'Mixer -> Mixer',
#     # 'Mixer (no encoder)',
#     # 'Mixer -> Transformer',
#     'Transformer (no encoder)',
#     'QAT Transformer',
#     'Transformer'
# ] 

paths = [
    '/Users/bbadger/Desktop/256x4_memory/fineweb_mixer_512.json',
    '/Users/bbadger/Desktop/256x4_memory/fineweb_emixer_512.json',
    '/Users/bbadger/Desktop/memory_versus_nomemory/mixer.json',
    '/Users/bbadger/Desktop/memory_versus_nomemory/memory_emixer.json',
    '/Users/bbadger/Desktop/memory_versus_nomemory/transformer.json',
    '/Users/bbadger/Desktop/memory_versus_nomemory/transformer_tmemory.json',
    '/Users/bbadger/Desktop/fineweb_transformer_1024_n24.json',
    '/Users/bbadger/Desktop/fineweb_tmem_trans_1024_n24.json',
    '/Users/bbadger/Desktop/256x4_memory/fineweb_transformer_256.json',
    '/Users/bbadger/Desktop/256x4_memory/fineweb_tmem_transformer_256.json',

]

labels = [
    'Mixer, small', 
    'Memory Mixer, small', 
    'Mixer, large', 
    'Memory Mixer, large',
    'Transformer, med',
    'Memory Transformer, med',
    'Transformer',
    'Memory Transformer',
    'Transformer, large', 
    'Memory Transformer, large'
]


# paths = [
#  '/Users/bbadger/Desktop/fineweb_emixer_512.json',
#  '/Users/bbadger/Desktop/fineweb_tmem_transformer_256.json',
#  '/Users/bbadger/Desktop/fineweb_mixer_512.json',
#  '/Users/bbadger/Desktop/fineweb_transformer_256.json',
# ]

# labels = [
#     'Mixer',
#     'Transformer',
#     'mixer', 
#     'transformer'
# ]

# paths = [
#         '/Users/bbadger/Desktop/extended_memory_train/transformer_extended.json', 
#         '/Users/bbadger/Desktop/extended_memory_train/memory_mixer_extended.json',
#         '/Users/bbadger/Desktop/extended_memory_train/fineweb_memtrans_extended.json'
#         ]

# labels = [
#           'Transformer',
#           'Memory Mixer',
#           'Memory Transformer'
#           ]

# plot_curves(paths, labels)
plot_differential(paths, labels)


