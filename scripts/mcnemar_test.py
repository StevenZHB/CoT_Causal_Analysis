# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os.path as path
import json
import argparse
import numpy as np
import statsmodels.stats.contingency_tables as ssc


def load_output(output_file):
    with open(output_file, 'r') as fin:
        data = json.load(fin)  # data from default role to new role
        print(f'Loaded {len(data)} items from {output_file}')
    return data


def get_accuracy(output_file):
    outputs = load_output(output_file)
    key = path.basename(output_file).split('.')
    key = '.'.join(key[2:4]).replace('math_teacher', 'math teacher')
    aa = []
    for item in outputs:
        result = item[f'{key}_result']
        aa.append(1 if result else 0)
    return np.mean(aa)


def get_paired_results(group_a_file, group_b_file):
    group_a = load_output(group_a_file.replace(':', '_'))
    group_b = load_output(group_b_file.replace(':', '_'))
    assert len(group_a) == len(group_b)
    key_a = path.basename(group_a_file).split('.')
    key_a = '.'.join(key_a[2:4]).replace('math_teacher', 'math teacher')
    key_b = path.basename(group_b_file).split('.')
    key_b = '.'.join(key_b[2:4]).replace('math_teacher', 'math teacher')

    aa = []
    bb = []
    for a, b in zip(group_a, group_b):
        assert a['id'] == b['id']
        result_a = a[f'{key_a}_result']
        result_b = b[f'{key_b}_result']
        aa.append(1 if result_a else 0)
        bb.append(1 if result_b else 0)
    return aa, bb


def mcnemar_test(aa, bb):
    aa = np.array(aa)
    bb = np.array(bb)
    table = np.array([[0., 0.], [0., 0.]])
    table[0, 0] = ((1 - aa) * (1 - bb)).sum()
    table[0, 1] = ((1 - aa) * bb).sum()
    table[1, 0] = (aa * (1 - bb)).sum()
    table[1, 1] = (aa * bb).sum()
    result = ssc.mcnemar(table, exact=True, correction=True)
    return result.statistic, result.pvalue


def get_average_treatment_effect(group_a_file, group_b_file):
    aa, bb = get_paired_results(group_a_file, group_b_file)
    base = np.mean(aa)
    ate = np.mean(bb) - np.mean(aa)
    _, pvalue = mcnemar_test(aa, bb)
    return base, ate, pvalue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_a', type=str, default='exp_cot/output/output.Addition_6.cot0shot.defaultreason.gpt-3.5-turbo.json')
    parser.add_argument('--group_b', type=str, default='exp_cot/output/output.Addition_6.cot0shot.randomreason.gpt-3.5-turbo.json')
    args = parser.parse_args()

    aa, bb = get_paired_results(args.group_a.replace(':', '_'),args.group_b.replace(':', '_'))
    print(f'Group A: {np.mean(aa):.3f} ({np.sum(aa)}/{len(aa)})')
    print(f'Group B: {np.mean(bb):.3f} ({np.sum(bb)}/{len(bb)})')
    print(f'B - A: {np.mean(bb) - np.mean(aa):.3f} ({np.sum(bb) - np.sum(aa)}/{len(bb)})')

    statistic, pvalue = mcnemar_test(aa, bb)
    print(f'Statistic: {statistic:.2f}, p-value: {pvalue:.2g}')
