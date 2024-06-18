import argparse
import torch
import os
import json
import numpy as np
import torch.distributed
from tqdm import tqdm
import shortuuid
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

from PIL import Image
import math,re
from vllm import LLM, SamplingParams

def load_pretrained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def split_list(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0,len(lst),batch_size)]

def eval_model(args):
    model_path = os.path.expanduser(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    world_size = torch.cuda.device_count()
    model = LLM(model=model_path, tensor_parallel_size=world_size)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    answer_path = os.path.dirname(args.question_file)
    answers_file = os.path.expanduser(os.path.join(answer_path, f"prompt_eval_merge.jsonl"))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    questions = split_list(questions,args.batch_size)

    for i, lines in enumerate(tqdm(questions)):
        idx = i
        
        prompts = []
        for line in lines:
            question = line[-1]['content']
            qs = question.replace('<image>', '').strip()
            cur_prompt = qs

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": cur_prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)

        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=20)

        outputs  = model.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            output = output.outputs[0].text
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": prompt,
                                    "text": output,
                                    "answer_id": ans_id}) + "\n")
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/Qwen/Qwen1.5-32B-Chat")
    parser.add_argument("--question-file", type=str, default="./results/CoIN/Qwen_Chat/ScienceQA/Finetune/prompt_to_eval.json")
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    eval_model(args)
