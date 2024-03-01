import json
import random
from utils_api import OpenAIModel, chat_completions_with_backoff, dispatch_openai_chat_requests
from tqdm import tqdm
import nltk
nltk.download('punkt')
import re
import time
import asyncio
import argparse

SYSTEM_PROMPT = """Inverse Wordsmith specializes in creatively inverting the meaning of sentences while preserving their structure. The GPT will operate with inputs and outputs enclosed in triple double quotes. Specifically, it will:
1. Subtly alter a sentence to make its meaning the exact opposite of the original.
2. Add or remove words as needed, but without significantly changing the sentence structure.
3. Keep all original punctuation intact!!!
4. Ensure the new sentence closely mirrors the original in form but completely differs in meaning.
5. Provide the inverted sentence as a single, coherent statement, enclosed in triple double quotes.
output example: \"\"\"Your output\"\"\""""

def batch_chat_generate(self, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [   {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['message']['content'].strip() for x in predictions]

def chat_generate(self, input_string, temperature = 0.0):
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": input_string}
                    ],
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text

def tokenize_preserving_newlines(text):
    lines = text.split('\n')
    sentences = [nltk.sent_tokenize(line) for line in lines]
    sentences_with_newlines = []
    for line in sentences:
        for sentence in line:
            sentences_with_newlines.append(sentence)
        sentences_with_newlines.append('\n')
    return sentences_with_newlines

def select_random_segment(sentences, min_length=3):
    part_size = len(sentences) // 3

    rand_span = [max(2*part_size-1,0),len(sentences)]

    try_times = 0
    while True:
        try_times+=1
        pos_1 = random.randint(rand_span[0],rand_span[-1])
        pos_2 = random.randint(rand_span[0],rand_span[-1])
        start = min(pos_1, pos_2)
        end = max(pos_1, pos_2)
        segment = sentences[start:end]
        if try_times == 6:
            if part_size<=2:
                return (0,len(sentences))
            return (rand_span[0],rand_span[-1])
        if len(segment) >= 1 and len(segment)<=8:
            if sentences[start] == '\n' or sentences[end-1] == '\n':
                continue
            for s in segment:
                if len(s)>min_length:
                    return (start,end)

def extract_reason(output):
    seps = ['The correct option','\nAnswer:\n', '\n\n', 'Now,']
    reason = output
    for sep in seps:
        if reason.find(sep) > 0:
            reason = reason.split(sep)[:-1]
            reason = sep.join(reason).strip()
            break
    return reason


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,choices=['FOLIO','ProofWriter','LOGIQA'])
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--role', type=str,default='math teacher')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--api_base', type=str)
    parser.add_argument('--api_key', type=str,default='math teacher')
    parser.add_argument('--batch_size', type=int,default=10)

    args = parser.parse_args()
    # Change chat generate function
    OpenAIModel.chat_generate = chat_generate
    OpenAIModel.batch_chat_generate = batch_chat_generate
    # Load default reason data
    default_output_path = f'exp_cot/output/output.{args.dataset}.{args.prompt}.{args.role}.{args.model_name}.json'.replace(' ', '_')
    default_data = json.load(open(default_output_path))
    gpt_model_name = 'gpt-3.5-turbo'
    stop_words = None
    max_new_tokens = 1024
    openai_api = OpenAIModel(args.api_base, args.api_key, gpt_model_name, stop_words, max_new_tokens)

    batch_size = args.batch_size
    dataset_chunks = [default_data[i:i + batch_size] for i in range(0, len(default_data), batch_size)]

    processed_data = []
    for chunk in tqdm(dataset_chunks):
        try_time = 0
        while True:
            input_segments = []
            other_part = []
            for d in chunk:
                default_reason = d[f'{args.prompt}.{args.role}_output']
                default_reason = extract_reason(default_reason)
                default_reason_sentences = tokenize_preserving_newlines(default_reason)
                start,end = select_random_segment(default_reason_sentences,3)
                segment_content = default_reason_sentences[start:end]
                segment_content = ''.join(segment_content)
                segment_content = f'"""{segment_content}"""'
                other_part.append((''.join(default_reason_sentences[:start]),''.join(default_reason_sentences[end:])))
                input_segments.append(segment_content)
            try:
                if len(input_segments) >= 2:
                    batch_outputs = openai_api.batch_generate(input_segments,temperature=0.7)
                else:
                    batch_outputs = [openai_api.generate(message) for message in input_segments]
                # extract the answer and regenerate if the output format is out of expectation
                pattern = r'"""(.*?)"""'
                for b_o_index in range(len(batch_outputs)):
                    matches = re.findall(pattern, batch_outputs[b_o_index], re.DOTALL)
                    if not matches or len(matches)>1:
                        print(input_segments[b_o_index])
                        print(batch_outputs[b_o_index])
                        raise ValueError("No quoted sentences found in the text.")
                    else:
                        batch_outputs[b_o_index] = matches[0]
                break
            except Exception as ex:
                print(ex)
                print('Sleep 10 seconds before retry ...')
                time.sleep(10)
                if try_time >= 5:
                    batch_outputs = ['FAILED']*len(input_segments)
                    break
                try_time+=1
        for sample, output, other in zip(chunk, batch_outputs,other_part):
            sample['random_reason'] = ''.join([other[0],output,other[1]])
            processed_data.append(sample)

    output_random_reason_path = f'exp_cot/random_reason/random_reason.{args.dataset}.{args.prompt}.{args.role}.{args.model_name}.json'.replace(' ','_')
    json.dump(processed_data,open(output_random_reason_path,'w'),ensure_ascii=False,indent=2)