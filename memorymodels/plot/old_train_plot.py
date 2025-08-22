import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import matplotlib

def plot_curves(input_paths, labels, plot_training=True):
    
    for i, input_path in enumerate(input_paths):
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'loss' in iteration and plot_training:
                train_losses.append(iteration['loss'])
                train_steps.append(step/1000)
            if 'eval_loss' in iteration:
                test_losses.append(iteration['eval_loss'])
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

# paths = [
#     "/Users/bbadger/Desktop/fineweb_memory_k8_1024.json",
#     "/Users/bbadger/Desktop/fineweb_memory_ek8dk1_1024.json",
#     "/Users/bbadger/Desktop/fineweb_tmemory_k1_1024.json",
#     "/Users/bbadger/Desktop/memory_versus_nomemory/memory_emixer.json"
# ]

# labels = [
#     'k=8',
#     'k=8 encoder, k=1 decoder',
#     'k=1',
#     'k=1, embedding concat'
# ]

# paths = [
#     "/Users/bbadger/Desktop/fineweb_c256x4_nomemorytrans.json",
#     "/Users/bbadger/Desktop/fineweb_c256x4_memorytrans.json"
# ]

# labels = [
#     "No Memory", 
#     "Memory"
# ]

# paths = [
#     "/Users/bbadger/Desktop/fineweb_automixer_k8_repeat_512c4.json",
#     "/Users/bbadger/Desktop/fineweb_automixer_k8_unroll_512c4.json",
#     "/Users/bbadger/Desktop/fineweb_autotrans_unroll_512c4.json"
# ]

# labels = [
#     "Repeat Mixer",
#     "Unroll Mixer",
#     "Unroll Transformer"
# ]

# paths = [
#     '/Users/bbadger/Desktop/memory_versus_nomemory/memory_emixer.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/mixer.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/ememory_transformer.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/transformer.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/transformer_tmemory.json'

# ]
# labels = [
#     'Mixer -> Mixer',
#     'Mixer (no encoder)',
#     'Mixer -> Transformer',
#     'Transformer (no encoder)',
#     'Transformer -> Transformer'
# ]

# paths = [
#         '/Users/bbadger/Desktop/no_heads.json', 
#         '/Users/bbadger/Desktop/1_head.json',
#         '/Users/bbadger/Desktop/2_heads.json',
#         '/Users/bbadger/Desktop/4_heads.json',
#         '/Users/bbadger/Desktop/8_heads.json',
#         '/Users/bbadger/Desktop/16_heads.json'
#         ]

# labels = [
#           'No Heads',
#           'One Head',
#           'Two Heads', 
#           'Four Heads', 
#           'Eight Heads',
#           'Sixteen Heads'
#           ]

# num_plots = 6
# colormap = plt.cm.Blues
# colormap2 = plt.cm.Reds
# cmap_cycler = plt.cycler('color', colormap(np.linspace(0.2, 1, num_plots)))
# cmap2_cycler = plt.cycler('color', colormap2(np.linspace(0.2, 1, num_plots)))
# all_cycler = cmap_cycler.concat(cmap2_cycler)
           
# plt.gca().set_prop_cycle(all_cycler)


# paths = [
# '/Users/bbadger/Desktop/fineweb_automixer_unroll_k4_512c1.json',
# '/Users/bbadger/Desktop/fineweb_automixer_unroll_k8_512c1.json',
# '/Users/bbadger/Desktop/fineweb_automixer_unroll_k16_512c1.json',
# '/Users/bbadger/Desktop/fineweb_automixer_noroll_k4_512c1.json',
# '/Users/bbadger/Desktop/fineweb_automixer_noroll_k8_512c1.json',
# '/Users/bbadger/Desktop/fineweb_automixer_noroll_k16_512c1.json',
# ]

# labels = [
#     'Unroll k=4',
#     'Unroll k=8',
#     'Unroll k=16',
#     'Repeat k=4',
#     'Repeat k=8',
#     'Repeat k=16',
# ]

# transformer heads optimization
# paths = [
# '/Users/bbadger/Desktop/fineweb_autotrans_unroll_h2_512.json',
# '/Users/bbadger/Desktop/fineweb_autotransformer_unroll_512.json',
# '/Users/bbadger/Desktop/fineweb_autotrans_unroll_h8_512.json',
# '/Users/bbadger/Desktop/fineweb_autotrans_unroll_h16_512.json'
# ]

# labels = [
# '2 heads',
# '4 heads',
# '8 heads',
# '16 heads'
# ]

# paths = [
# '/Users/bbadger/Desktop/fineweb_automixer_unroll_k8_1024c1.json',
# '/Users/bbadger/Desktop/fineweb_automixer_noroll_k8_1024c1.json',
# '/Users/bbadger/Desktop/fineweb_automixer_unroll_k16_1024c1.json',
# ]

# labels = [
#     'k=8 Unrolled',
#     'k=8 Repeated', 
#     'k=16 Unrolled'
# ]

paths = [
    '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/mixer_1024_c512.json',
    '/Users/bbadger/Desktop/frozen_clm_memory.json',
    '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/tmixer_frozen_c512.json',
    '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/frozen_mixer_k8e.json',
    '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/fineweb_frozen_mixer_k16.json',
    
]

labels = [
    'No Encoder',
    'CLM-trained Encoder, CEL=2.6',
    'Autoencoder Encoder, CEL=3.5',
    'Autoencoder Encoder, CEL=1.4',
    'Autoencoder encoder, CEL=0.3',
    
]

# paths = [
#     '/Users/bbadger/Desktop/fineweb_transtrans_512.json',
#     '/Users/bbadger/Desktop/fineweb_mixinformer_512.json',
#     '/Users/bbadger/Desktop/fineweb_mixmix_512.json',
#     '/Users/bbadger/Desktop/fineweb_transfixer_512.json',
#     '/Users/bbadger/Desktop/fineweb_mixmix_k8.json',
# ]

# labels = [
#     'Transformer -> Transformer',
#     'Mixer -> Transformer',
#     'Mixer -> Mixer',
#     'Transformer -> Mixer',
#     'Mixer -> Mixer (optimized)'
# ]


# paths = [
#     '/Users/bbadger/Desktop/fineweb_tmemory_transformer_d1024.json',
#     '/Users/bbadger/Desktop/fineweb_tmemory_transformer_d1024_n4.json',
#     '/Users/bbadger/Desktop/fineweb_tmemory_mixer_k4.json',
#     '/Users/bbadger/Desktop/fineweb_tmemory_mixer_k8.json'
# ]

# labels = [
#     'Transformer',
#     'Compute-matched Transformer',
#     'Masked Mixer, k=4',
#     'Masked Mixer, k=8'
# ]


# paths = [
#     '/Users/bbadger/Desktop/autoencoder_causality/allcause_extended.json',
#     '/Users/bbadger/Desktop/model_heads/fineweb_autoencoder_h4.json',
#     '/Users/bbadger/Desktop/model_heads/fineweb_autoencoder_4h.json',
#     '/Users/bbadger/Desktop/fineweb_autoencoder_k8.json'
# ]

# labels = [
#     'flat, fp16',
#     'h=4, fp16',
#     'h=4, bf16',
#     'k=8, fp16'
# ]

# paths = [
        # '/Users/bbadger/Desktop/transformer_d256.json',
        # '/Users/bbadger/Desktop/finemath_transformer_dm512.json',
        # '/Users/bbadger/Desktop/autoencoder_transformer_dm1024.json',
        #  '/Users/bbadger/Desktop/transformer_mod_autoencoder.json',
        #  '/Users/bbadger/Desktop/mixer_d256.json',
        #  '/Users/bbadger/Desktop/finemath_mixer_512.json',
        #  '/Users/bbadger/Desktop/finemath_mixer_dm1024.json'
        #  ]

# labels = [
        #   'Transformer dm=256',
        #   'Transformer dm=512', 
        #   'Transformer dm=1024', 
        #   'Prebuilt Transformer dm=1024',
        #   'Masked Mixer dm=256',
        #   'Masked Mixer dm=512',
        #   'Masked Mixer dm=1024'
        #   ]


plot_curves(paths, labels)


