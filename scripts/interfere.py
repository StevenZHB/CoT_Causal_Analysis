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
from api_run import load_prompt, format_prompt, extract_answer

def extract_reason(output):
    seps = ['The correct option','\nAnswer:\n', '\n\n', 'Now,']
    reason = output
    for sep in seps:
        if reason.find(sep) > 0:
            reason = reason.split(sep)[:-1]
            reason = sep.join(reason).strip()
            break
    return reason

def random_new_numbers(reason):
    # generate a new number with different digits
    parts = re.split('(\d+)', reason)
    numbers = [part for i, part in enumerate(parts) if i % 2 == 1]
    vset = set([str(i) for i in range(10)])
    numbers = [[list(vset.difference([d])) for d in v] for v in numbers]
    numbers = [[d[random.randint(0, len(d)-1)] for d in v] for v in numbers]
    numbers = [''.join(v) for v in numbers]
    parts = [numbers[i//2] if i % 2 == 1 else part for i, part in enumerate(parts)]
    new_reason = ''.join(parts)
    return new_reason




def format_interfere_prompt(args, prompt, full_prompt, item, outputs):
    dataset = args.dataset.split(':')[0]
    interferes = [args.do_reason]
    for interfere in interferes:
        if interfere == 'defaultreason':  # use the reasoning steps generated by LLM
            default_role = 'math teacher'
            default_prompt = prompt.split('.')[0] + f'.{default_role}'
            reason = extract_reason(item[f'{default_prompt}_output'])
        elif interfere == 'goldreason':  # use the golden reasoning steps
            reason = item['reason']
        elif interfere == 'randomreason':
            if dataset in ['Addition', 'Product', 'GSM8K']:
                # replace the digits from the reasoning steps generated by LLM with new digits
                default_role = 'math teacher'
                default_prompt = prompt.split('.')[0] + f'.{default_role}'
                reason = extract_reason(item[f'{default_prompt}_output'])
                reason = random_new_numbers(reason)
            elif dataset in ['ProofWriter','FOLIO','LOGIQA']:
                reason = item['random_reason']
            else:
                # shuffle the subjects in the reasoning steps generated by LLM
                raise NotImplemented
        else:
            raise NotImplemented

    message = format_prompt(full_prompt, item)
    dataset = dataset.split(':')[0]
    if dataset in ['Addition', 'Product', 'GSM8K']:
        message = f'{message}\n{reason}\nAnswer:'
    else:
        message = f'{message}\n{reason}\nThe correct option is:'

    return message


def load_output(args):
    default_role = 'math teacher'
    output_file = f'{args.outdir}/output.{args.dataset}.{args.prompt}.{default_role}.{args.model_name}.json'
    if not os.path.exists(output_file):
        output_file = f'{args.outdir}/output.{args.dataset}.{args.prompt}.{default_role}.{args.model_name}.json'
    output_file = output_file.replace(' ', '_').replace(':', '_')
    with open(output_file, 'r') as fin:
        output = json.load(fin)
        if getattr(args, 'do_role', None) is not None:
            output = add_role(output, args.do_role)
        else:
            output = add_role(output, args.role)
        if getattr(args, 'do_bias', 'nobias') != 'nobias':
            # bias to wrong answer for default/gold reason, bais to right answer for random reason
            output = add_bias(args.dataset, output, wrong='random' not in args.do_reason)
        print(f'Loaded {len(output)} items from {output_file}')
    return output


def add_role(data, do_role):
    # make bias data
    for item in data:
        if do_role == 'defaultrole':
            item['role'] = 'math teacher'
        elif do_role == 'randomrole':
            role = random.choice(['detective', 'chef', 'judge'])
            item['role'] = role
        else:
            item['role'] = do_role
    return data


# add bias option for logical problems
def add_bias_option(data, wrong=True):
    # make bias data
    for item in data:
        if wrong:
            all_choices = [str(chr(65 + choice)) for choice in range(len(item['options']))]
            all_choices.remove(item['answer'])
            item['biasoption'] = random.choice(all_choices)
        else:
            item['biasoption'] = item['answer']
    return data


# add bias answer for math problems
def add_bias_answer(data, wrong=True):
    # make bias data
    for item in data:
        if wrong:
            # randomly change a digit in the number
            answer = item['answer']
            idx = random.randint(0, len(answer) - 1)
            digit = int(answer[idx])
            all_choices = [i for i in range(0, 10)]
            all_choices.remove(digit)
            new_digit = random.choice(all_choices)
            new_answer = answer[:idx] + str(new_digit) + answer[idx+1:]
            item['biasanswer'] = new_answer
        else:
            item['biasanswer'] = item['answer']
    return data


def add_bias(dataset, data, wrong=True):
    dataset = dataset.split(':')[0]
    if dataset in ['Addition', 'Product', 'GSM8K']:
        data = add_bias_answer(data, wrong=wrong)
    else:
        data = add_bias_option(data, wrong=wrong)
    return data


def get_prompt_name(args):
    treatment = args.do_reason
    if args.do_role != 'defaultrole':
        treatment = f'{treatment}_{args.do_role}'
    if args.do_bias != 'nobias':
        treatment = f'{treatment}_{args.do_bias}'
    return f'{args.prompt}.{treatment}'

def add_random_reason(args,data):
    if args.do_role == 'defaultrole':
        default_role = 'math teacher'
    else:
        raise NotImplemented
    random_reason_path = f'exp_cot/random_reason/random_reason.{args.dataset}.{args.prompt}.{default_role}.{args.model_name}.json'.replace(' ','_')
    random_reason_data = json.load(open(random_reason_path))
    for d in data:
        d_id = d['id']
        d['random_reason'] = [r['random_reason'] for r in random_reason_data if r.get('id') == d_id][0]
    return data




def intervene(args):
    openai_api = OpenAIModel(args.api_base, args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
    full_prompt = load_prompt(args.dataset, args.prompt, args.do_role, args.do_bias)
    prompt = get_prompt_name(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_output(args)

    # For randomreasonstart, randomreasonmiddle, randomreasonend
    if args.dataset in ['ProofWriter','FOLIO','LOGIQA'] and args.do_reason == 'randomreason':
        data = add_random_reason(args,data)
    # check file existencce
    output_file = f'{args.outdir}/output.{args.dataset}.{prompt}.{args.model_name}.json'
    output_file = output_file.replace(' ', '_').replace(':', '_')
    if os.path.exists(output_file):
        print(f'Exist file {output_file}')
        return None

    outputs = []
    accs = []
    dataset_chunks = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    for chunk in tqdm(dataset_chunks):
        messages = [format_interfere_prompt(args, prompt, full_prompt, item, data) for item in chunk]
        while True:
            try:
                if len(messages) >= 1:
                    batch_outputs = openai_api.batch_generate(messages)
                else:
                    batch_outputs = [openai_api.generate(messages)]
                break
            except Exception as ex:
                print(ex)
                print('Sleep 10 seconds before retry ...')
                time.sleep(10)

        for sample, output, message in zip(chunk, batch_outputs, messages):
            answer = sample[f'answer']
            pred = extract_answer(output, sample, args.dataset)
            record_item = sample.copy()
            record_item[f'{prompt}_input'] = message
            record_item[f'{prompt}_output'] = output
            record_item[f'{prompt}_answer'] = pred
            record_item[f'{prompt}_result'] = (pred == answer)
            accs.append(pred == answer)
            outputs.append(record_item)

    print(f'acc_{prompt}:', np.mean(accs), f'({np.sum(accs)}/{len(accs)})')

    output_file = f'{args.outdir}/output.{args.dataset}.{prompt}.{args.model_name}.json'
    output_file = output_file.replace(' ', '_').replace(':', '_')
    with open(output_file, 'w') as fout:
        json.dump(outputs, fout, indent=2)
        print(f'Write {len(outputs)} items into {output_file}')


if __name__ == '__main__':
    ''' Call OpenAPI to answer nsamples questions from dataset using the prompt.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./exp_cot/output')
    parser.add_argument('--api_base', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')  # gpt-3.5-turbo, text-davinci-003
    parser.add_argument('--stop_words', type=str, default='####')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='GSM8K')
    parser.add_argument('--prompt', type=str, default='cot0shot')
    parser.add_argument('--batch_size', type=int, default=10)
    # do operator to intervene the random variables
    #   defaultrole: math teacher, randomrole: detective, chef, judge
    parser.add_argument('--do_role', type=str, default='defaultrole', choices=['defaultrole', 'randomrole'])
    #   nobias: no bias prompt, weakbias: weak bias prompt, strongbias: strong bias prompt
    parser.add_argument('--do_bias', type=str, default='nobias', choices=['nobias', 'weakbias', 'strongbias'])
    #   defaultreason: CoT from LLM generation, goldreason: golden CoT, randomreason: CoT with number, subject, negative interventions
    parser.add_argument('--do_reason', type=str, default='defaultreason', choices=['defaultreason', 'goldreason', 'randomreason'])
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.do_reason == 'goldreason' and args.dataset in ['FOLIO', 'LOGIQA']:
        print(f'{args.dataset} dataset does not support goldreason because it is not provided.')
    else:
        intervene(args)
