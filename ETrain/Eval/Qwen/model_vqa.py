import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from ETrain.Models.Qwen import load_pretrained_model

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    model_path = os.path.expanduser(args.model_path)
    model_base = os.path.expanduser(args.model_base)

    model, tokenizer = load_pretrained_model(model_path, model_base)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        question = line['text']
        qs = question.replace('<image>', '').strip()

        if 'image' in line.keys():
            image_file = line["image"]
            if '.jpg.jpg' in image_file:
                image_file = image_file.replace('.jpg.jpg', '.jpg')
            image_file = os.path.join(args.image_folder, image_file.replace('./',''))
            prompt = 'System:You are a helpful assistant.\n\n<img>{}</img>{} Assistant:'.format(image_file, qs)
        else:
            prompt = 'System:You are a helpful assistant.\n\n{} Assistant:'.format(qs)

        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(model.device)

        input_ids_size = inputs.data['input_ids'].size(1)

        pred = model.generate(**inputs,
                              do_sample=False,
                              num_beams=1,
                              max_new_tokens=30,
                              use_cache=True,
                              pad_token_id=tokenizer.eod_id,
                              eos_token_id=tokenizer.eod_id,)
        outputs = [tokenizer.decode(_[input_ids_size:].cpu(),skip_special_tokens=True) for _ in pred]
        # print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs[0][1:],
                                   "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
