import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='/home/chencheng/Code/LLaVA/cl_dataset/TextVQA/TextVQA_0.5.1_val.json')
    parser.add_argument('--result-file', type=str, default='/home/chencheng/Code/LLaVA/results/Instructions/TextVQA/Zero_shot/merge.jsonl')
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()



def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[result['question_id']]
        if result['text'] == annotation['answers']:
            right += 1

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * right / total))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
