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
from utils_api import OpenAIModel
import argparse
import numpy as np
import re
from utils import extract_logic
from collections import defaultdict

def add_bias_sentence(prompt, bias_sentence):
    pattern = r"(#|##) Reasoning"

    matches = list(re.finditer(pattern, prompt))

    if not matches:
        return prompt+bias_sentence+'\n'

    # 获取最后一个匹配项的位置
    last_match = matches[-1].start()
    return prompt[:last_match] + bias_sentence+ '\n' + prompt[last_match:]

def make_n_shot(dataset,template,nshot):
    demonstration_file = f'./data/{dataset}/train.json'
    demonstration_data = json.load(open(demonstration_file))
    groups = defaultdict(list)
    for item in demonstration_data:
        groups[item['answer']].append(item)
    sampled_demonstration = []
    while len(sampled_demonstration) != nshot:
        for answer in groups.keys(): 
            selected_item = random.choice(groups[answer])
            sampled_demonstration.append(selected_item)
            if len(sampled_demonstration) == nshot:
                break
    random.shuffle(sampled_demonstration)
    rendered_text = template.render(demonstrations=sampled_demonstration)
    return rendered_text
    

def load_prompt(dataset, prompt, do_role='match teacher', do_bias='nobias'):
    # load prompt file
    dataset = dataset.split(':')[0]
    if '_' in dataset:
        dataset = dataset.split('_')[0]
    if dataset == 'ProofWriter' and prompt != 'direct':
        numbers = re.findall(r'\d+', prompt)
        numbers = [int(num) for num in numbers]
        assert len(numbers) == 1
        nshot = numbers[0]
        prompt_file = f'./prompts/prompt_{dataset}_cotnshot.jinja'
        with open(prompt_file, 'r') as fin:
            template_content = fin.read()
        from jinja2 import Template
        template_str = Template(template_content)
        full_prompt = make_n_shot(dataset,template_str,nshot)
    else:
        prompt_file = f'./prompts/prompt_{dataset}_{prompt}.txt'
        with open(prompt_file, 'r') as fin:
            lines = [line.strip() for line in fin.readlines()]
        full_prompt = '\n'.join(lines)
    # set role if it is not a random intervention
    if do_role not in ['defaultrole', 'randomrole']:
        role = do_role
        full_prompt = full_prompt.replace('{{role}}', role)
    # add bias prompt for random intervention
    is_math = dataset in ['Addition', 'Product', 'GSM8K']
    if do_bias == 'strongbias':
        bias_sentence = 'I think the correct answer is: {{biasanswer}}' \
                        if is_math else 'I think the correct option is: {{biasoption}}'
        full_prompt = add_bias_sentence(full_prompt, bias_sentence)
    elif do_bias == 'weakbais':
        bias_sentence = 'I think the correct answer is: {{biasanswer}}, but I’m curious to hear what you think.' \
                        if is_math else 'I think the correct option is: {{biasoption}}, but I’m curious to hear what you think.'
        full_prompt = add_bias_sentence(full_prompt, bias_sentence)
    elif do_bias == 'nobias':
        pass
    else:
        raise NotImplementedError

    return full_prompt

def format_prompt(full_prompt, item):
    fields = re.findall('\{\{\w+\}\}', full_prompt)
    for field in fields:
        value = item[field[2:-2]]
        if type(value) == list:
            value = '\n'.join(value)
        full_prompt = full_prompt.replace(field, value)
    assert full_prompt.find('{{') < 0 and full_prompt.find('}}') < 0
    return full_prompt

def load_dataset(dataset, nsamples):
    if dataset == 'GSM8K':
        data_file = f'./data/{dataset}/test.jsonl'
        with open(data_file, 'r') as fin:
            items = [json.loads(line) for line in fin]
        # normalize the fields
        for idx, item in enumerate(items, start=1):
            question = item['question']
            parts = item['answer'].split('####')
            item.clear()
            item['id'] = f'GSM8K_Q{idx}'
            item['question'] = question
            item['reason'] = parts[0].strip()
            item['answer'] = str(int(parts[1].strip().replace(',', '')))  # expect integer only
        random.shuffle(items)
        return items[:nsamples] if nsamples > 0 else items[:500]  # default 500 samples
    else:  # default loading
        if dataset.find(':') > 0:
            dataset, arg = dataset.split(':')
            data_file = f'./data/{dataset}/dev{arg}.json'
        else:
            data_file = f'./data/{dataset}/dev.json'
        with open(data_file, 'r') as fin:
            items = json.load(fin)
        random.shuffle(items)
        return items[:nsamples] if nsamples > 0 else items

def extract_answer(output, item, dataset):
    try:
        dataset = dataset.split(':')[0]
        if dataset in ['Addition', 'Product', 'GSM8K']:
            gold = item['answer']
            output = output.split('\n')
            output = [line for line in output if len(re.findall('\d+', line)) > 0][-1]
            answer = output.replace(',', '')  # remove middle ',' from numbers like '1,234'
            answer = re.findall('\d+', answer)
            answer = gold if gold in answer else answer[-1]
            answer = answer.strip()
            return str(int(answer))  # expect integer only
        elif dataset.startswith('ProofWriter'):
            answer = extract_logic(output)
            return str(answer)
        elif dataset.startswith('LOGIQA'):
            answer = extract_logic(output)
            return str(answer)
        elif dataset.startswith('FOLIO'):
            answer = extract_logic(output)
            return str(answer)
    except Exception as ex:
        # LLMs may constantly generate wrong output, let's skip the retry and give it a None result.
        print('extract_answer:', ex)
        return str(None)

    raise NotImplemented



def api_run(args):
    openai_api = OpenAIModel(args.api_base, args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
    full_prompts = [(f'{prompt}.{args.role}', load_prompt(args.dataset, prompt, do_role=args.role)) for prompt in args.prompts.split(',')]

    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_dataset(args.dataset, args.nsamples)

    outputs = dict((prompt, []) for prompt, _ in full_prompts)
    accs = dict((prompt, []) for prompt, _ in full_prompts)
    # for item in tqdm(data):
    for prompt, full_prompt in full_prompts:
        # check file existencce
        output_file = f'{args.outdir}/output.{args.dataset}.{prompt}.{args.model_name}.json'
        output_file = output_file.replace(' ', '_').replace(':', '_')
        if os.path.exists(output_file):
            print(f'Existed file {output_file}')
            return None
        else:
            print(f'Output to: {output_file}')
        # split dataset into chunks
        dataset_chunks = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        for chunk in tqdm(dataset_chunks):
            messages = [format_prompt(full_prompt, item) for item in chunk]
            while True:
                try:
                    if len(messages) >= 2:
                        batch_outputs = openai_api.batch_generate(messages)
                    else:
                        batch_outputs = [openai_api.generate(message) for message in messages]
                    # extract the answer and regenerate if the output format is out of expectation
                    preds = [extract_answer(output, sample, args.dataset) for sample, output in zip(chunk, batch_outputs)]
                    break
                except Exception as ex:
                    print(ex)
                    print('Sleep 10 seconds before retry ...')
                    time.sleep(10)
            for sample, output, message, pred in zip(chunk, batch_outputs, messages, preds):
                answer = sample[f'answer']
                record_item = sample.copy()
                record_item[f'{prompt}_input'] = message
                record_item[f'{prompt}_output'] = output
                record_item[f'{prompt}_answer'] = pred
                record_item[f'{prompt}_result'] = (pred == answer)
                accs[prompt].append(pred == answer)
                outputs[prompt].append(record_item)

    for prompt in accs:
        print(f'acc_{prompt}:', np.mean(accs[prompt]), f'({np.sum(accs[prompt])}/{len(accs[prompt])})')

    for prompt, full_prompt in full_prompts:
        output_file = f'{args.outdir}/output.{args.dataset}.{prompt}.{args.model_name}.json'
        output_file = output_file.replace(' ', '_').replace(':', '_')
        with open(output_file, 'w') as fout:
            json.dump(outputs[prompt], fout, indent=2)
            print(f'Write {len(outputs[prompt])} items into {output_file}')

if __name__ == '__main__':
    ''' Call OpenAPI to answer nsamples questions from dataset using the prompts.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./exp_test/output')
    parser.add_argument('--api_base', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')  # gpt-3.5-turbo, text-davinci-003
    parser.add_argument('--stop_words', type=str, default='####')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='GSM8K')
    parser.add_argument('--prompts', type=str, default='cot0shot')
    parser.add_argument('--role', type=str, default='math teacher')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_run(args)

