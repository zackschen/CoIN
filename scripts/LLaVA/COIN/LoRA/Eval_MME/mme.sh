#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODELPATH=$1
RESULT_DIR=$2

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODELPATH \
        --model-base ./checkpoints/Vicuna/vicuna-7b-v1.5 \
        --question-file ./playground/data/MME/llava_mme.jsonl \
        --image-folder ./playground/data/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/MME/answers/$RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/MME/answers/$RESULT_DIR/$RESULT_DIR.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/MME/answers/$RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


cd ./playground/data/MME

python convert_answer_to_mme.py --experiment $RESULT_DIR

cd eval_tool

python calculation.py --results_dir answers/$RESULT_DIR
