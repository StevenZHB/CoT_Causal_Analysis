# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import random
import time
from tqdm import tqdm
import argparse
import numpy as np

'''
Let's add the two numbers digit by digit.
1. The ones place: 8 + 8 = 16, write down 6 and carry over 1.
2. The tens place: 5 + 3 + 1 (carry over) = 9.
3. The hundreds place: 8 + 6 = 14, write down 4 and carry over 1.
4. The thousands place: 9 + 3 + 1 (carry over) = 13, write down 3 and carry over 1.
5. The ten thousands place: 7 + 9 + 1 (carry over) = 17, write down 7 and carry over 1.
6. The hundred thousands place: 8 + 3 + 1 (carry over) = 12, write down 2 and carry over 1.
7. The final carry over: 1.
'''
def generate_addition_reason(aa, bb):
    digits = ['ones', 'tens', 'hundreds',
              'thousands', 'ten thousands', 'hundred thousands',
              'millions', 'ten millions', 'hundred millions',
              'billions', 'ten billions', 'hundred billions',
              'trillions', 'ten trillions', 'hundred trillions']
    steps = ["Let's add the two numbers digit by digit."]
    c = 0
    while aa > 0 or bb > 0:
        idx = len(steps)
        prefix = f'{idx}. The {digits[idx - 1]} place: '
        a = aa % 10
        b = bb % 10
        z = a + b + c
        middle = f'{a} + {b} + {c} (carry over) = {z}' if c > 0 else f'{a} + {b} = {z}'
        c = z // 10
        r = z % 10
        aa = aa // 10
        bb = bb // 10
        suffix = f', write down {r} and carry over {c}.' if c > 0 else f'.'
        steps.append(prefix + middle + suffix)
    if c > 0:
        idx = len(steps)
        steps.append(f'{idx}. The final carry over: 1.')
    return '\n'.join(steps)

'''
Let's think step by step.
1. Multiply 148 by 5 (the ones place digit of 905) = 740
2. Multiply 148 by 0 (the tens place digit of 905) = 0
3. Multiply 148 by 9 (the hundreds place digit of 905) = 1332
'''
def generate_product_reason(aa, bb):
    digits = ['ones', 'tens', 'hundreds',
              'thousands', 'ten thousands', 'hundred thousands',
              'millions', 'ten millions', 'hundred millions',
              'billions', 'ten billions', 'hundred billions',
              'trillions', 'ten trillions', 'hundred trillions']
    steps = ["Let's think step by step."]
    cb = bb
    place = 1
    while cb > 0:
        idx = len(steps)
        b = (cb % 10) * place
        z = aa * b
        step = f'{idx}. Multiply {aa} by {b} (the {digits[idx - 1]} place digit of {bb}) = {z}'
        cb = cb // 10
        place *= 10
        steps.append(step)
    return '\n'.join(steps)


def generate_addition(ndigits):
    v_min = 10**(ndigits-1)
    v_max = 10**ndigits
    a = random.randint(v_min, v_max)
    b = random.randint(v_min, v_max)
    c = a + b
    z = generate_addition_reason(a, b)
    item = {'number1': str(a),
            'number2': str(b),
            'answer': str(c),
            'reason': z}
    return item


def build_addition(args):
    name = 'Addition'
    data = []
    for idx in range(args.nsamples):
        item = generate_addition(args.ndigits)
        item['id'] = f'{name}_Q{idx + 1}'
        data.append(item)

    data_path = f'{args.datadir}/{name}'
    os.system(f'mkdir -p {data_path}')
    data_file = f'{data_path}/dev{args.ndigits}.json'
    with open(data_file, 'w') as fout:
        json.dump(data, fout, indent=2)
        print(f'Write {len(data)} samples into {data_file}')


def generate_product(ndigits):
    v_min = 10**(ndigits-1)
    v_max = 10**ndigits
    a = random.randint(v_min, v_max)
    b = random.randint(v_min, v_max)
    c = a * b
    z = generate_product_reason(a, b)
    item = {
        'number1': str(a),
        'number2': str(b),
        'answer': str(c),
        'reason': z
    }
    return item


def build_product(args):
    name = 'Product'
    data = []
    for idx in range(args.nsamples):
        item = generate_product(args.ndigits)
        item['id'] = f'{name}_Q{idx + 1}'
        data.append(item)

    data_path = f'{args.datadir}/{name}'
    os.system(f'mkdir -p {data_path}')
    data_file = f'{data_path}/dev{args.ndigits}.json'
    with open(data_file, 'w') as fout:
        json.dump(data, fout, indent=2)
        print(f'Write {len(data)} samples into {data_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='Addition', choices=['Addition', 'Product'])
    parser.add_argument('--ndigits', type=int, default=9)
    parser.add_argument('--nsamples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f'Make {args.dataset}:{args.ndigits} dataset with {args.nsamples} samples ...')

    if args.dataset == 'Addition':
        build_addition(args)
    elif args.dataset == 'Product':
        build_product(args)

