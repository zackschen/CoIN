import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='/home/chencheng/Code/LLaVA/playground/Instructions/OCRVQA/val.json')
    parser.add_argument('--result-file', type=str, default='/home/chencheng/Code/LLaVA/results/Instructions/CL_Tuning/1.5/OCRVQA/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='/home/chencheng/Code/LLaVA/results/Instructions/CL_Tuning/1.5/OCRVQA/Zero_shot')
    return parser.parse_args()



def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    annotations = json.load(open(annotation_file))
    annotations = {data['question_id']: data for data in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = 0
    right = 0
    for result in results:
        annotation = annotations[result['question_id']]
        answers = annotation['answer']
        total += len(answers)
        if 'Unanswerable' in result['text'] :
            continue
        mdoel_results = [result for result in result['text'].split('\n') if len(result) > 0]
        list_range = min(len(mdoel_results), len(answers))
        for i in range(list_range):
            if mdoel_results[i] == answers[i]:
                right += 1

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * right / total))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * right / total))

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
