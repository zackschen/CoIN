import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default='results/CoIN/Qwen_VL_Chat_Final2/GQA/Finetune/merge.jsonl')
parser.add_argument("--dst", type=str, default='results/CoIN/Qwen_VL_Chat_Final2/GQA/Finetune/testdev_balanced_predictions.json')
args = parser.parse_args()

all_answers = []
for line_idx, line in enumerate(open(args.src)):
    res = json.loads(line)
    question_id = res['question_id']
    text = res['text'].rstrip('.').lower()
    text = text[1:] if len(text) > 0 and text[0] == ' ' else text
    all_answers.append({"questionId": question_id, "prediction": text})

with open(args.dst, 'w') as f:
    json.dump(all_answers, f)
