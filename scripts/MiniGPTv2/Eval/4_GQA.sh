RESULT_DIR="./results/MiniGPTv2/GQA"
MODELPATH=$2

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    ETrain/Eval/MiniGPT/model_vqa.py \
    --cfg-path ./scripts/MiniGPTv2/Eval/4_GQA.yaml \
    --image-folder ./cl_dataset \
    --model-path $MODELPATH \
    --answers-file $RESULT_DIR/$1/merge.jsonl \

output_file=$RESULT_DIR/$1/merge.jsonl

python -m ETrain.Eval.LLaVA.CoIN.convert_gqa_for_eval --src $output_file --dst $RESULT_DIR/$1/testdev_balanced_predictions.json

python -m ETrain.Eval.LLaVA.CoIN.eval_gqa --tier testdev_balanced --path $RESULT_DIR/$1 --output-dir $RESULT_DIR/$1 

python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
    --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions ./playground/Instructions_slim/GQA/test.json \
    --results $output_file \