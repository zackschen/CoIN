import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='playground/Instructions/VisualGenome/test.json')
    parser.add_argument('--result-file', type=str, default='/home/chencheng/Code/LLaVA/results/Instructions/Grounding/Zero_shot/merge.jsonl')
    return parser.parse_args()


def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x21, y21, x22, y22 = bbox2
    intersection_area = max(0, min(x2, x22) - max(x1, x21)) * max(0, min(y2, y22) - max(y1, y21))
    union_area = (x2 - x1) * (y2 - y1) + (x22 - x21) * (y22 - y21) - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def eval_single(test_file, result_file):
    annotations = json.load(open(test_file))
    annotations = {grounding_test['question_id']: (grounding_test['answer_bbox'],grounding_test['size']) for grounding_test in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    for result in results:
        bbox_string = annotations[result['question_id']][0]
        bbox_groundtruth = [float(x) for x in bbox_string.split(',')]
        size = annotations[result['question_id']][1]

        pred_bbox = result['text']
        bbox_pred = [float(x) for x in pred_bbox[1:-1].split(',')]
        bbox_pred[0] = bbox_pred[0] * size[0]
        bbox_pred[1] = bbox_pred[1] * size[1]
        bbox_pred[2] = bbox_pred[2] * size[0]
        bbox_pred[3] = bbox_pred[3] * size[1]

        iou = calculate_iou(bbox_pred, bbox_groundtruth)
        right += iou > 0.5

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.test_file, args.result_file)
