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
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5/TextVQA/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/CoIN/LLaVA-13B/TextVQA"

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    ETrain/Eval/LLaVA/model_vqa.py \
    --model-path $MODELPATH \
    --model-base ./checkpoints/LLaVA/Vicuna/vicuna-13b-v1.5 \
    --question-file ./playground/Instructions_slim/ScienceQA/test.json \
    --image-folder ./cl_dataset \
    --answers-file $RESULT_DIR/$STAGE/merge.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.eval_textvqa \
    --annotation-file ./cl_dataset/TextVQA/TextVQA_0.5.1_val.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
    --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions ./playground/Instructions_slim/TextVQA/val.json \
    --results $output_file \