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
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5/Grounding/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/CoIN/LLaVA-13B/Grounding"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_vqa \
        --model-path $MODELPATH \
        --model-base ./checkpoints/LLaVA/Vicuna/vicuna-13b-v1.5 \
        --question-file ./playground/Instructions_slim/Grounding/test.json \
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

python -m ETrain.Eval.LLaVA.CoIN.eval_grounding \
    --test-file ./playground/Instructions_slim/Grounding/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
    --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions ./playground/Instructions_slim/Grounding/test.json \
    --results $output_file \
    --rule_temp CoIN_Grounding \