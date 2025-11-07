import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import matplotlib

def plot_curves(input_paths, labels, plot_training=False, plot_epochs=False, tokens_per_step=1024*16*4, embedding_addition=None):
    """
    8-bit loss per token: 0.1739, 16-bit loss per token: 0.1739*2
    """
    
    for i, input_path in enumerate(input_paths):
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'loss' in iteration and plot_training:
                train_losses.append(iteration['loss'])
                if plot_epochs:
                    train_steps.append(step/500 * 1.28)
                elif tokens_per_step:
                    train_steps.append(step * tokens_per_step / 1e9)
                else:
                    train_steps.append(step/1000)

            if 'eval_loss' in iteration:
                if ('->' in labels[i] or 'Entropy' in labels[i]) and embedding_addition:
                    test_losses.append(iteration['eval_loss'] + embedding_addition)
                else:
                    test_losses.append(iteration['eval_loss'])
                if plot_epochs:
                    test_steps.append(step/500 * 1.28)
                elif tokens_per_step:
                    test_steps.append(step * tokens_per_step / 1e9)
                else:
                    test_steps.append(step/1000)
    
        label = labels[i]
        plt.plot(test_steps, test_losses)
        plt.scatter(test_steps, test_losses, label=f'{label} Eval')
        if plot_training:
            plt.plot(train_steps, train_losses)
            plt.scatter(train_steps, train_losses, label=f'{label} Train', alpha=0.7)

    plt.legend(fontsize='x-large')
    plt.tick_params(labelsize=16)
    # plt.xscale('log')
    if plot_epochs:
        plt.xlabel('Epochs', fontsize='x-large')
    else:
        plt.xlabel('Billion Tokens', fontsize='x-large')
    plt.ylabel('Cross-Entropy Loss', fontsize='x-large')
    plt.show()
    plt.close()
    return

def plot_curves_together(input_paths, labels, plot_training=False, plot_epochs=False, tokens_per_step=1024*16*4, embedding_addition=None):
    """
    8-bit loss per token: 0.1739, 16-bit loss per token: 0.1739*2
    """
    all_test_steps, all_test_losses = [], []
    for i, input_path in enumerate(input_paths):
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'loss' in iteration and plot_training:
                train_losses.append(iteration['loss'])
                if plot_epochs:
                    train_steps.append(step/500 * 1.28)
                elif tokens_per_step:
                    train_steps.append(step * tokens_per_step / 1e9)
                else:
                    train_steps.append(step/1000)

            if 'eval_loss' in iteration:
                if ('->' in labels[i] or 'Entropy' in labels[i]) and embedding_addition:
                    test_losses.append(iteration['eval_loss'] + embedding_addition)
                else:
                    test_losses.append(iteration['eval_loss'])
                if plot_epochs:
                    test_steps.append(step/500 * 1.28)
                elif tokens_per_step:
                    test_steps.append(step * tokens_per_step / 1e9)
                else:
                    test_steps.append(step/1000)
    
        label = labels[i]
        all_test_steps.append(test_steps)
        all_test_losses.append(test_losses)
        plt.plot(test_steps, test_losses)
        plt.scatter(test_steps, test_losses, label=f'{label} Eval')
        if plot_training:
            plt.plot(train_steps, train_losses)
            plt.scatter(train_steps, train_losses, label=f'{label} Train', alpha=0.7)

    plt.legend(fontsize='x-large')
    plt.tick_params(labelsize=16)
    # plt.xscale('log')
    if plot_epochs:
        plt.xlabel('Epochs', fontsize='x-large')
    else:
        plt.xlabel('Billion Tokens', fontsize='x-large')
    plt.ylabel('Cross-Entropy Loss', fontsize='x-large')
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Billion Tokens', fontsize='x-large')
    ax1.set_ylabel('CLM Loss', color=color, fontsize='x-large')
    ax1.plot(all_test_steps[0], all_test_losses[0], color=color)
    ax1.scatter(all_test_steps[0], all_test_losses[0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(labelsize=16)
    ax1.set_xscale('log')

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('EEM Loss', color=color, fontsize='x-large')  # we already handled the x-label with ax1
    ax2.plot(all_test_steps[1], all_test_losses[1], color=color)
    ax2.scatter(all_test_steps[1], all_test_losses[1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(labelsize=16)
    ax2.set_xscale('log')
    ax2.set_ylim([2.4, 3.6])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return

# paths = [
# "/Users/bbadger/Desktop/transformer_autoencoder_noroll_512.json",
# "/Users/bbadger/Desktop/unrolled_versus_repeated/fineweb_autotransformer_unroll_512.json",
# ]

# labels = [
#     "Repeated Embedding",
#     "Unrolled Embedding"
# ]


# embedding introduction choices
# paths = [
#     "/Users/bbadger/Desktop/fineweb_token_eemixer.json",
#     "/Users/bbadger/Desktop/fineweb_proj_eemixer.json",
#     "/Users/bbadger/Desktop/fineweb_emb_eemixer.json"
# ]

# labels = [
#     'Token Concatenation', 
#     'Embedding Combination',
#     'Embedding Concatenation'
# ]

# paths = [
#     "/Users/bbadger/Desktop/transformer_memory_options/transformer_tmemory.json",
#      "/Users/bbadger/Desktop/transformer_memory_options/transformer_pmemory.json",
#     "/Users/bbadger/Desktop/transformer_memory_options/transformer_ememory.json",
# ]

# labels = [
#     'Token Concatenation', 
#     'Embedding Combination',
#     'Embedding Concatenation'
# ]

# 2nd order eems
# paths = [
#     "/Users/bbadger/Desktop/fineweb_2eem_shifted.json",
#     "/Users/bbadger/Desktop/fineweb_2eem.json",
# ]

# labels = [
#     "Unshifted",
#     "Shifted"
# ]

# overfitting tests
paths = [
    "/Users/bbadger/Desktop/fineweb_unweighted_transformer_50k.json",
    # "/Users/bbadger/Desktop/fineweb_unweighted_dropout_transformer.json",
    # "/Users/bbadger/Desktop/fineweb_weighted_transformer_50k_fine.json",
    "/Users/bbadger/Desktop/fineweb_weighted_dropout_transformer.json",
]

labels = [
    "No Entropy",
    # "No Entropy, Dropout",
    # "Entropy",
    "Entropy, Dropout"
]

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
    # "/Users/bbadger/Desktop/mixer_vs_transformer_autoencoder_compressed/fineweb_automixer_k8_repeat_512c4.json",
    # "/Users/bbadger/Desktop/mixer_vs_transformer_autoencoder_compressed/fineweb_automixer_k8_unroll_512c4.json",
    # "/Users/bbadger/Desktop/mixer_vs_transformer_autoencoder_compressed/fineweb_autotrans_unroll_512c4.json",
    # "/Users/bbadger/Desktop/memory_versus_nomemory/mixer.json",
    # "/Users/bbadger/Desktop/memory_versus_nomemory/transformer.json"
# ]

# labels = [
    # "Autoencoder (Repeat) Mixer",
    # "Autoencoder Mixer",
    # "Autoencoder Transformer",
    # "Causal Mixer",
    # "Causal Transformer"
# ]

# paths = [
#     # '/Users/bbadger/Desktop/memory_versus_nomemory/memory_emixer.json',
#     # '/Users/bbadger/Desktop/memory_versus_nomemory/mixer.json',
#     # '/Users/bbadger/Desktop/memory_versus_nomemory/ememory_transformer.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/transformer.json',
#     '/Users/bbadger/Desktop/memory_versus_nomemory/transformer_tmemory.json'
# ]

# labels = [
#     # 'Mixer -> Mixer',
#     # 'Mixer (no encoder)',
#     # 'Mixer -> Transformer',
#     'Transformer (no encoder)',
#     'Transformer -> Transformer'
# ]

# paths = [
#         '/Users/bbadger/Desktop/extended_memory_train/transformer_extended.json', 
#         # '/Users/bbadger/Desktop/extended_memory_train/memory_mixer_extended.json',
#         '/Users/bbadger/Desktop/extended_memory_train/fineweb_memtrans_extended.json'
#         ]

# labels = [
#           'Causal Transformer',
#         #   'Entropy Estimation Mixer',
#           'Entropy Estimation Transformer'
#           ]



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

# paths = [
#     '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/mixer_1024_c512.json',
#     '/Users/bbadger/Desktop/frozen_clm_memory.json',
#     '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/tmixer_frozen_c512.json',
#     '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/frozen_mixer_k8e.json',
#     '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/fineweb_frozen_mixer_k16.json',
    
# ]

# labels = [
#     'No Encoder',
#     'CLM-trained Encoder, CEL=2.6',
#     'Autoencoder Encoder, CEL=3.5',
#     'Autoencoder Encoder, CEL=1.4',
#     'Autoencoder encoder, CEL=0.3',
# ]

# paths = [
#     '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/mixer_1024_c512.json',
#     '/Users/bbadger/Desktop/frozen_versus_unfrozen_memory/fineweb_frozen_mixer_k16.json',
#     '/Users/bbadger/Desktop/memory_models_d1024/fineweb_tmemory_mixer_k1.json',
# ]

# labels = [
#     "No Encoder",
#     "Frozen Encoder",
#     "Trainable Encoder"
# ]

# paths = [
#     '/Users/bbadger/Desktop/memory_models_d1024/fineweb_tmemory_mixer_k1.json',
#     '/Users/bbadger/Desktop/memory_models_d1024/fineweb_tmemory_mixer_h4.json',
#     '/Users/bbadger/Desktop/memory_models_d1024/fineweb_tmemory_mixer_k4.json',
# ]

# labels = [
#     'No Heads, k=1',
#     '4 Heads',
#     'No Heads, k=4'
# ]

# frozen mixer models
# paths = [
#     '/Users/bbadger/Desktop/information/fineweb_frozen_untrained_clmenc_1024_n8.json',
#     '/Users/bbadger/Desktop/information/fineweb_frozen_clmenc_1024_n8.json',
#     # '/Users/bbadger/Desktop/information/retrieval_untrained.json',
#     '/Users/bbadger/Desktop/fineweb_frozen_auto-retrenc.json',
#     '/Users/bbadger/Desktop/information/fineweb_frozen_retrenc_1024_n8.json',
#     '/Users/bbadger/Desktop/information/fineweb_frozen_enc_1024_n8.json',
#     # '/Users/bbadger/Desktop/information/fineweb_frozen_enc_1024_n8_k16.json'
# ]

# labels = [
#     'Untrained Encoder',
#     'Causal Encoder',
#     # 'Untrained Retrieval',
#     'Autoencoder Retrieval',
#     'Retrieval Encoder',
#     'Autoencoder Encoder',
#     # 'Autoencoder encoder, k=16 decoder'
# ]

# noised QAT comparison
# paths = [
#     '/Users/bbadger/Desktop/QAT_loss/fineweb_memtrans_1noise.json',
#     '/Users/bbadger/Desktop/QAT_loss/fineweb_memtrans_2noise.json',
#     '/Users/bbadger/Desktop/layer_quantizations/3noise_redo.json',
#     '/Users/bbadger/Desktop/QAT_loss/fineweb_memtrans_4noise.json',
#     '/Users/bbadger/Desktop/layer_quantizations/fineweb_memtrans_midresid.json',
#     '/Users/bbadger/Desktop/fineweb_memtrans_256d512.json',
# ]

# labels = [
#     '2^-1 Noise QAT',
#     '2^-2 noise QAT',
#     '2^-3 Noise QAT',
#     '2^-4 Noise QAT',
#     '2^-2 Noise (residual layer) QAT',
#     'No Noise'
# ]

# paths = [
#     '/Users/bbadger/Desktop/information/fineweb_frozen_enc_1024_n8_k16.json',
#     '/Users/bbadger/Desktop/information/unrolled_versus_repeated/fineweb_automixer_unroll_k16_1024c1.json',
# ]

# labels = [
#     'Frozen Encoder',
#     'Trainable Encoder'
# ]

# frozen memory models
# paths = [
#     '/Users/bbadger/Desktop/information/fineweb_frozen_untrained_clmtransenc_n8.json',
#     '/Users/bbadger/Desktop/information/fineweb_frozen_untrained_clmenc_1024_n8.json',
#     '/Users/bbadger/Desktop/fineweb_transmem_memory.json',
#     # '/Users/bbadger/Desktop/information/fineweb_frozen_memenc_k8_unroll_1024_n8.json',
#     '/Users/bbadger/Desktop/information/fineweb_frozen_memenc_1024_n8.json',

# ]

# labels = [
#     'Untrained Transformer',
#     'Untrained Mixer',
#     'Transformer Memory',
#     # 'Mixer Memory Encoder, k=8, unrolled',
#     'Mixer Memory Encoder'
# ]


# frozen transformer validation
# paths = [
#     '/Users/bbadger/Desktop/fineweb_frozen_transauto.json',
#     '/Users/bbadger/Desktop/unrolled_versus_repeated/fineweb_autotrans_unroll_h16_512.json'
# ]

# labels = [
#     'Frozen Encoder',
#     'Trainable Encoder',
# ]

# frozen transformer models
# paths = [
#     '/Users/bbadger/Desktop/information/fineweb_frozen_untrained_clmtransenc_n8.json',
#     '/Users/bbadger/Desktop/fineweb_frozen_trans_untrained_retrenc.json',
#     '/Users/bbadger/Desktop/fineweb_frozen_trans_retrenc.json',
#     '/Users/bbadger/Desktop/information/fineweb_frozen_clmtransenc_n8.json',
#     '/Users/bbadger/Desktop/fineweb_frozen_transauto_matched_h4.json',
# ]

# labels = [
#     'Untrained Causal Encoder',
#     'Untrained Retrieval Encoder', 
#     'Retrieval Encoder',
#     'Causal Encoder',
#     'Autoencoder Encoder',
# ]

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
# plot_curves_together(paths, labels)


