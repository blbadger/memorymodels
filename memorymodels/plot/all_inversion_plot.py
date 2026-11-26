import numpy as np 
import matplotlib.pyplot as plt
import json
import matplotlib
import jsonlines


def plot_curves(input_paths, labels, report_tokens=False):
    for i, input_path in enumerate(input_paths):
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        data = jsonlines.open(input_path)

        for iteration in data:
            print ('iteration', iteration)
            step = iteration['step']
            
            if 'eval_inverter_loss' in iteration:
                loss = iteration['eval_inverter_loss']

            if 'inverter_loss' in iteration:
            	loss = iteration['inverter_loss']

            test_losses.append(loss)

            if report_tokens:
                test_steps.append(step * 128 * 512 / 1e9)

            else:
                test_steps.append(step/1000)

        label = labels[i]
        print (label, test_losses[-1], test_steps)
        plt.plot(test_steps, test_losses)
        plt.scatter(test_steps, test_losses, label=f'{label}')

    plt.legend(fontsize='large', framealpha=0.95)
    plt.tick_params(labelsize=16)
    plt.xlabel('Thousand Steps', fontsize='x-large')
    plt.ylabel('Loss', fontsize='x-large')
    # plt.xscale('log')
    plt.show()
    plt.close()
    return

input_paths = [
	'/home/bbadger/Desktop/transformer_inversion.jsonl',
	'/home/bbadger/Desktop/transformer_train_inversion.jsonl'
]

labels = ['Evaluation', 'Train']
plot_curves(input_paths, labels)