#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5/VQAv2/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/CLIT/VQAv2"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_vqa_cc_instruction \
        --model-path $MODELPATH \
        --model-base ./checkpoints/Vicuna/vicuna-7b-v1.5 \
        --question-file ./playground/Instructions/VQAv2/test.json \
        --image-folder ./cl_dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir $RESULT_DIR/$STAGE \
    --test-split ./playground/Instructions/VQAv2/test.json \

python llava/eval/CoIN/create_prompt.py \
    --rule llava/eval/CoIN/rule.json \
    --questions ./playground/Instructions/VQAv2/test.json \
    --results $output_file \