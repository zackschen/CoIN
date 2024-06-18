#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m ETrain.Eval.eval_GenealKnowledge \
    --model-path ./checkpoints/Qwen/Qwen1.5-32B-Chat \
    --question-file $1/prompt_to_eval.json \
    --batch_size 32 \

python ETrain/Eval/evaluate_score.py \
    --dir $1 \