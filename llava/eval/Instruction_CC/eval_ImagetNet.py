import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='playground/Instructions/ImageNet/test.json')
    parser.add_argument('--result-file', type=str, )
    return parser.parse_args()


def eval_single(test_file, result_file):
    annotations = json.load(open(test_file))
    answers = [test['answer'] for test in annotations]
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    for index in range(total):
        text = answers[index]
        label = results[index]
        if text == label:
            right += 1
        
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.test_file, args.result_file)
