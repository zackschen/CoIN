import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

from PIL import Image
import math,re

def load_pretrained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    model_path = os.path.expanduser(args.model_path)

    waits_to_eval_prompts = open("to_eval_prompt.txt", "r")
    # open text file to read
    waits_to_eval_prompts = waits_to_eval_prompts.readlines()

    model, tokenizer = load_pretrained_model(model_path)

    for question_path in waits_to_eval_prompts:
        if question_path.strip() == "":
            continue
        question_path = question_path.strip()
        question_path = os.path.join(os.path.expanduser(question_path),'prompt_to_eval.json')
        with open(question_path, "r") as f:
            questions = json.load(f)
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        answer_path = os.path.dirname(question_path)
        answers_file = os.path.expanduser(os.path.join(answer_path, f"prompt_{args.chunk_idx}.jsonl"))
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")

        for i, line in enumerate(tqdm(questions)):
            idx = i
            question = line[-1]['content']
            qs = question.replace('<image>', '').strip()
            cur_prompt = qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            prompt = "Give me a short introduction to large language model."
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            outputs = response.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
        ans_file.close()

        # if args.chunk_idx == 0:
        #     ## compute scores
        #     dirs = os.listdir(answer_path)
        #     prompt_dirs = []
        #     pattern = re.compile(r'prompt_[1-9]')
        #     for dir_ in dirs:
        #         res = pattern.findall(dir_)
        #         if len(res) > 0:
        #             prompt_dirs.append(os.path.join(answer_path,dir_))
        #     prompts = []
        #     for path in prompt_dirs:
        #         prompt = [json.loads(line) for line in open(path)]
        #         prompts += prompt

        #     total = 0
        #     scores = 0
        #     for prompt in prompts:
        #         total += 10
        #         score = prompt['text']
        #         score.isnumeric()
        #         try:
        #             score = float(score)
        #             if score <= 10.0:
        #                 scores += score
        #         except:
        #             if '/' in score:
        #                 score = score.split('/')[0]
        #                 if score.isnumeric() and float(score) <= 10.0:
        #                     scores += float(score)
        #             else:
        #                 print(score)
            
        #     final_score = scores / total

        #     output_file = os.path.join(answer_path,'Prompt_Result.text')
        #     with open(output_file, 'w') as f:
        #         f.write('Final score:{}'.format(round(final_score*100.0)))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
