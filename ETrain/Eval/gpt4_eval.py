import argparse
import json
import os

import openai
from openai import OpenAI
import time
from tqdm import tqdm
openai.api_key = "sk-4LGwzEGZgXtr4onW2616D6BfCe2945B190Bd6299Cc1fCd14"
openai.base_url = "https://api.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}

NUM_SECONDS_TO_SLEEP = 0.5

def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-e', '--eval-path',default='./results/LLaVA/To_Eval_GPT4/ScienceQA/Finetune/prompt_to_eval.json')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    print(f"Reading prompts from {args.eval_path}")
    eval_samples = json.load(open(os.path.expanduser(args.eval_path)))
    answer_path = os.path.dirname(args.eval_path)
    answers_file = os.path.expanduser(os.path.join(answer_path, f"prompt_eval_merge.jsonl"))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for index, sample in enumerate(tqdm(eval_samples)):
        output = get_eval(sample, args.max_tokens)
        results = {"question_id": index,"prompt": sample,"text": output}

        ans_file.write(json.dumps(results) + "\n")
        ans_file.flush()

    ans_file.close()
