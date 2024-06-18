RESULT_DIR="./results/MiniGPTv2/ImageNet"
MODELPATH=$2

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    ETrain/Eval/MiniGPT/model_vqa.py \
    --cfg-path ./scripts/MiniGPTv2/Eval/3_ImageNet.yaml \
    --image-folder ./cl_dataset \
    --model-path $MODELPATH \
    --answers-file $RESULT_DIR/$1/merge.jsonl \

output_file=$RESULT_DIR/$1/merge.jsonl

python -m ETrain.Eval.LLaVA.CoIN.eval_ImagetNet \
    --test-file ./playground/Instructions_slim/ImageNet/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$1 \

python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
    --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions ./playground/Instructions_slim/ImageNet/test.json \
    --results $output_file \