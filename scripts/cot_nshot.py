# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import random
import numpy as np
from tqdm import tqdm
import argparse
from api_run import load_prompt, format_prompt, load_dataset
from data_builder import generate_product, generate_addition


def generate_addition_nshot_prompt(args):
    sep = '####\n'
    full_prompt = load_prompt(args.dataset, 'cot#shot')
    full_prompt = full_prompt.split(sep)
    shot_prompt = full_prompt[-2]
    # generate n shots
    ndigits = [3, 4, 5]
    shots = []
    for i in range(args.nshot):
        ndigit = random.choice(ndigits)
        item = generate_addition(ndigit)
        shot = format_prompt(shot_prompt, item)
        shots.append(shot)
    # concat the final prompt
    full_prompt = full_prompt[:-2] + shots + full_prompt[-1:]
    full_prompt = sep.join(full_prompt)
    # write to prompt file
    data_file = f'./prompts/prompt_{args.dataset}_cot{args.nshot}shot.txt'
    with open(data_file, 'w') as fout:
        fout.write(full_prompt)
        print(f'Prompt file generated: {data_file}')


def generate_product_nshot_prompt(args):
    sep = '####\n'
    full_prompt = load_prompt(args.dataset, 'cot#shot')
    full_prompt = full_prompt.split(sep)
    shot_prompt = full_prompt[-2]
    # generate n shots
    ndigits = [2, 3]
    shots = []
    for i in range(args.nshot):
        ndigit = random.choice(ndigits)
        item = generate_product(ndigit)
        reason = item['reason']
        reason = [line.split(' = ')[-1].strip() for line in reason.split('\n') if line.find(' = ') > 0]
        reason = ' + '.join(reason)
        answer = item['answer']
        item['sum'] = f'{reason} = {answer}'
        shot = format_prompt(shot_prompt, item)
        shots.append(shot)
    # concat the final prompt
    full_prompt = full_prompt[:-2] + shots + full_prompt[-1:]
    full_prompt = sep.join(full_prompt)
    # write to prompt file
    data_file = f'./prompts/prompt_{args.dataset}_cot{args.nshot}shot.txt'
    with open(data_file, 'w') as fout:
        fout.write(full_prompt)
        print(f'Prompt file generated: {data_file}')


def generate_gsm8k_nshot_prompt(args):
    sep = '####\n'
    full_prompt = load_prompt(args.dataset, 'cot#shot')
    full_prompt = full_prompt.split(sep)
    shot_prompt = full_prompt[-2]
    # generate n shots
    data = load_dataset(args.dataset, 10000)
    shots = []
    for i in range(args.nshot):
        item = random.choice(data)
        shot = format_prompt(shot_prompt, item)
        shots.append(shot)
    # concat the final prompt
    full_prompt = full_prompt[:-2] + shots + full_prompt[-1:]
    full_prompt = sep.join(full_prompt)
    # write to prompt file
    data_file = f'./prompts/prompt_{args.dataset}_cot{args.nshot}shot.txt'
    with open(data_file, 'w') as fout:
        fout.write(full_prompt)
        print(f'Prompt file generated: {data_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Product')
    parser.add_argument('--nshot', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset.startswith('Addition'):
        generate_addition_nshot_prompt(args)
    elif args.dataset.startswith('Product'):
        generate_product_nshot_prompt(args)
    elif args.dataset.startswith('GSM8K'):
        generate_gsm8k_nshot_prompt(args)
