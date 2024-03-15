CUDA_VISIBLE_DEVICES=4 python llava/eval/model_vqa_cl.py \
    --model-path ./checkpoints/LLaVA-v1.5-7b \
    --model-base LLaVA-v1.5-7b \
    --image-folder "" \
    --question-file ./playground/data/CIFAR100/cifar100_question_diversity10.json \
    --answers-file ./results/CIFAR100/Original/cifar100_answers_eval_diversity10.jsonl \
    --pretrain_mm_mlp_adapter ./checkpoints/LLaVA-v1.5-7b \
