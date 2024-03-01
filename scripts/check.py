# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import random
import re
import time
from tqdm import tqdm
from utils_api import OpenAIModel
import argparse
import numpy as np
from interfere import load_output, extract_reason


def load_prompt(dataset, prompt):
    dataset = dataset.split(':')[0]
    if '_' in dataset:
        dataset = dataset.split('_')[0]
    prompt_file = f'./prompts/prompt_{dataset}_{prompt}.txt'
    with open(prompt_file, 'r') as fin:
        lines = [line.strip() for line in fin.readlines()]
    full_prompt = '\n'.join(lines)
    return full_prompt

def format_prompt(full_prompt, output):
    full_prompt = full_prompt.replace('{{reason}}', output)
    return full_prompt


class StepsExtractor:
    def __init__(self, args):
        self.args = args
        self.openai_api = OpenAIModel(args.api_base, args.api_key, args.api_model, args.stop_words, args.max_new_tokens)
        self.full_prompt = load_prompt(args.dataset, 'normsteps')

    def extract_gold_steps(self, output):
        dataset = self.args.dataset.split(':')[0]
        reason = extract_reason(output)
        if dataset == 'Addition':
            lines = reason.replace('(carry over)', '(carry)').split('\n')
            lines = [line.split(': ')[1] for line in lines if line.find(': ') > 0 and line.find(' = ') > 0]
            lines = [line.split(', ')[0].split('.')[0] for line in lines]
            return lines
        elif dataset == 'Product':
            lines = reason.split('\n')
            lines = [line for line in lines if line.find(' = ') > 0]
            lines = [re.sub('\s*\(.*\)\s*', ' ', line) for line in lines]
            lines = [re.sub(r'\d\. Multiply (\d+?) by (\d+?) \= (\d+?)', r'\1 * \2 = \3', line) for line in lines]
            return lines
        else:
            raise NotImplemented

    def extract_gen_steps(self, output):
        dataset = self.args.dataset.split(':')[0]
        reason = extract_reason(output)
        message = format_prompt(self.full_prompt, reason)
        while True:
            try:
                new_output = self.openai_api.generate(message)
                break
            except Exception as ex:
                print(ex)
                print('Sleep 10 seconds before retry ...')
                time.sleep(10)

        if dataset == 'Addition':
            lines = new_output.split('\n')
            lines = [line.split('. ')[1] for line in lines]
            lines = [line for line in lines if line.find(' + ') > 0 and line.find(' = ') > 0]
            lines = [line.split(' = ') for line in lines]
            lines = [(left.split(' + '), right) for left, right in lines]
            lines = [(' + '.join([v for v in left if v.find('(carry)') < 0] + [v for v in left if v.find('(carry)') > 0]),
                      right) for left, right in lines]
            lines = [' = '.join(line) for line in lines]
            return lines
        elif dataset == 'Product':
            lines = new_output.split('\n')
            lines = [line.split('. ')[1] for line in lines]
            lines = [line.replace(',', '') for line in lines]
            return lines
        else:
            raise NotImplemented


def check_steps(args):
    extractor = StepsExtractor(args)
    outputs = load_output(args)
    key = f'{args.prompt}.{args.role}'
    for item in tqdm(outputs):
        gold_reason = item[f'reason']
        gold_reason = extractor.extract_gold_steps(gold_reason)
        gen_reason = item[f'{key}_output']
        gen_reason = extractor.extract_gen_steps(gen_reason)
        right_reason = (gen_reason == gold_reason)
        item[f'check.reason_steps'] = gold_reason
        item[f'check.{key}_steps'] = gen_reason
        item[f'check.{key}_result'] = right_reason

    output_file = f'{args.checkdir}/check.{args.dataset}.{args.prompt}.{args.role}.{args.model_name}.json'
    output_file = output_file.replace(' ', '_').replace(':', '_')
    with open(output_file, 'w') as fout:
        json.dump(outputs, fout, indent=2)
        print(f'Write {len(outputs)} items into {output_file}')


def statistic_steps(args):
    output_file = f'{args.checkdir}/check.{args.dataset}.{args.prompt}.{args.role}.{args.model_name}.json'
    output_file = output_file.replace(' ', '_').replace(':', '_')
    with open(output_file, 'r') as fin:
        outputs = json.load(fin)
        print(f'Load {len(outputs)} items from {output_file}')

    result_accs = []
    full_accs = []
    step_accs = []
    right_reason_right_answers = []
    right_reason_wrong_answers = []
    wrong_reason_right_answers = []
    wrong_reason_wrong_answers = []
    key = f'{args.prompt}.{args.role}'
    for item in outputs:
        gold_reason = item[f'check.reason_steps']
        gen_reason = item[f'check.{key}_steps']
        right_reason = item[f'check.{key}_result']
        right_answer = item[f'{key}_result']
        right_reason_right_answers.append(right_reason and right_answer)
        right_reason_wrong_answers.append(right_reason and not right_answer)
        wrong_reason_right_answers.append(not right_reason and right_answer)
        wrong_reason_wrong_answers.append(not right_reason and not right_answer)
        result_accs.append(right_answer)
        full_accs.append(right_reason)
        step_accs.extend([gen == gold for gen, gold in zip(gen_reason[:len(gold_reason)], gold_reason)])
        if not right_reason and right_answer:
            gold_answer = item[f'answer']
            gen_answer = item[f'{key}_answer']
            print('-------------')
            print(f'GOLD: {gold_reason}, {gold_answer}')
            print(f'GEN: {gen_reason}, {gen_answer}')
            print()

    print(f'Result acc: {np.mean(result_accs)} ({np.sum(result_accs)}/{len(result_accs)})')
    print(f'Reason full acc: {np.mean(full_accs)} ({np.sum(full_accs)}/{len(full_accs)})')
    print(f'Reason step acc: {np.mean(step_accs):.3f} ({np.sum(step_accs)}/{len(step_accs)})')
    print(f'Right reason right answer: {np.mean(right_reason_right_answers)} ({np.sum(right_reason_right_answers)}/{len(right_reason_right_answers)})')
    print(f'Right reason wrong answer: {np.mean(right_reason_wrong_answers)} ({np.sum(right_reason_wrong_answers)}/{len(right_reason_wrong_answers)})')
    print(f'Wrong reason right answer: {np.mean(wrong_reason_right_answers)} ({np.sum(wrong_reason_right_answers)}/{len(wrong_reason_right_answers)})')
    print(f'Wrong reason wrong answer: {np.mean(wrong_reason_wrong_answers)} ({np.sum(wrong_reason_wrong_answers)}/{len(wrong_reason_wrong_answers)})')


if __name__ == '__main__':
    ''' Call OpenAPI to answer nsamples questions from dataset using the prompt.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./exp_cot/output')
    parser.add_argument('--checkdir', type=str, default='./exp_cot/check')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')  # gpt-3.5-turbo, text-davinci-003
    parser.add_argument('--dataset', type=str, default='Product:3')
    parser.add_argument('--role', type=str, default='math teacher', choices=['math teacher', 'detective', 'chef', 'judge'])
    parser.add_argument('--prompt', type=str, default='cot0shot')
    parser.add_argument('--api_base', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--api_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--stop_words', type=str, default='####')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()

    # extract reasoning steps
    check_steps(args)

    # statistic
    statistic_steps(args)
