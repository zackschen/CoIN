#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.Instruction_CC.evaluate_prompt \
        --model-path ./checkpoints/Instruction/Pre_train+Tuning_1.5/LLaVA-v1.5-7b \
        --question-file $1/prompt_to_eval.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$1/prompt_merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $1/prompt_${IDX}.jsonl >> "$output_file"
done

python playground/evaluate_score.py \
        --dir $1 \