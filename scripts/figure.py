# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
from os import path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mcnemar_test import load_output, get_accuracy


def get_accuracies(outdir, models, dataset, prompt):
    default_role = 'math teacher'.replace(' ', '_')
    dataset = dataset.replace(':', '_')
    accs = []
    for model in models:
        output_file = f'{outdir}/output.{dataset}.{prompt}.{default_role}.{model}.json'
        acc = get_accuracy(output_file)
        accs.append(acc)
    return accs


def draw_direct_vs_cot(args):
    prompt_cot = 'cot4shot'
    models = ['mistral-base', 'mistral-sft', 'mistral-dpo']
    model_names = ['mistral-base', 'mistral-sft', 'mistral-dpo']
    # models = ['llama2-7b-chat', 'llama2-70b-chat', 'gpt-3.5-turbo', 'gpt-4']
    # model_names = ['Llama2-7B-Chat', 'Llama2-70B-Chat', 'GPT-3.5-Turbo', 'GPT-4']
    tasks = [['Addition:6', 'Addition:9'], ['Product:2', 'Product:3'], 'GSM8K', 'ProofWriter', 'FOLIO', 'LOGIQA']
    task_names = ['Addition', 'Multiplication', 'GSM8K', 'ProofWriter', 'FOLIO', 'LOGIQA']
    # get accs for direct and cot
    direct_accs = []
    cot_accs = []
    for task in tasks:
        if type(task) == list:
            accs = [get_accuracies(args.outdir, models, dataset, 'direct') for dataset in task]
            direct_accs.append(np.array(accs).mean(axis=0).tolist())
            accs = [get_accuracies(args.outdir, models, dataset, prompt_cot) for dataset in task]
            cot_accs.append(np.array(accs).mean(axis=0).tolist())
        else:
            accs = get_accuracies(args.outdir, models, task, 'direct')
            direct_accs.append(accs)
            accs = get_accuracies(args.outdir, models, task, prompt_cot)
            cot_accs.append(accs)
    # draw line chat
    labels = ['Direct', 'CoT']
    colors = ['tab:blue', 'tab:green', 'tab:brown', 'tab:purple', 'tab:orange']  #   'tab:pink', 'tab:grey', 'tab:olive',

    # plot
    plots = [[0, 1, 2], [5, 4, 3]]
    nrows = 2
    ncols = 3
    plt.clf()
    fig = plt.figure(figsize=(3 * ncols, 1.8 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    for i in range(nrows):
        for j in range(ncols):
            idx = plots[i][j]
            task_name = task_names[idx]
            xs = model_names
            ys = direct_accs[idx]
            axs[i, j].plot(xs, ys, color='tab:blue', lw=1, label='Direct', marker='.')
            ys = cot_accs[idx]
            axs[i, j].plot(xs, ys, color='tab:orange', lw=1, label='CoT', marker='+')
            axs[i, j].set_title(task_name)

    axs[0, 0].set_ylabel('Accuracy')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].tick_params(axis='x', labelrotation=20, labelsize=8, pad=2)
    axs[1, 1].tick_params(axis='x', labelrotation=20, labelsize=8, pad=2)
    axs[1, 2].tick_params(axis='x', labelrotation=20, labelsize=8, pad=2)
    axs[0, 0].legend(loc="upper left", fontsize=6, ncol=2)
    # plt.figlegend(labels=labels, loc='lower center', fontsize=7, ncol=5, handlelength=2)

    plt.ylim(0.0, 1.0)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.subplots_adjust(wspace=0.12, hspace=0.24)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(path.join('./exp_cot/figures', 'accuracy_direct_vs_cot.pdf'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./exp_cot/output')
    parser.add_argument('--draw', type=str, default='direct_vs_cot')
    args = parser.parse_args()

    mpl.use('Agg')
    if args.draw == 'direct_vs_cot':
        draw_direct_vs_cot(args)