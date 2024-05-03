#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/ScienceQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/TextVQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/3_eval_ImageNet.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/ImageNet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/4_eval_gqa.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/GQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/5_eval_vizwiz.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/VizWiz
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/6_eval_grounding.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/7_eval_vqav2.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/8_eval_ocrvqa.sh Finetune ./checkpoints/Qwen/CoIN/Finetune/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh TextVQA ./checkpoints/Qwen/CoIN/Finetune/TextVQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh ImageNet ./checkpoints/Qwen/CoIN/Finetune/ImageNet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh GQA ./checkpoints/Qwen/CoIN/Finetune/GQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh VizWiz ./checkpoints/Qwen/CoIN/Finetune/VizWiz
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh Grounding ./checkpoints/Qwen/CoIN/Finetune/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh VQAv2 ./checkpoints/Qwen/CoIN/Finetune/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh OCRVQA ./checkpoints/Qwen/CoIN/Finetune/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh ImageNet ./checkpoints/Qwen/CoIN/Finetune/ImageNet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh GQA ./checkpoints/Qwen/CoIN/Finetune/GQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh VizWiz ./checkpoints/Qwen/CoIN/Finetune/VizWiz
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh Grounding ./checkpoints/Qwen/CoIN/Finetune/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh VQAv2 ./checkpoints/Qwen/CoIN/Finetune/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh OCRVQA ./checkpoints/Qwen/CoIN/Finetune/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/3_eval_ImageNet.sh GQA ./checkpoints/Qwen/CoIN/Finetune/GQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/3_eval_ImageNet.sh VizWiz ./checkpoints/Qwen/CoIN/Finetune/VizWiz
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/3_eval_ImageNet.sh Grounding ./checkpoints/Qwen/CoIN/Finetune/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/3_eval_ImageNet.sh VQAv2 ./checkpoints/Qwen/CoIN/Finetune/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/3_eval_ImageNet.sh OCRVQA ./checkpoints/Qwen/CoIN/Finetune/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/4_eval_gqa.sh VizWiz ./checkpoints/Qwen/CoIN/Finetune/VizWiz
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/4_eval_gqa.sh Grounding ./checkpoints/Qwen/CoIN/Finetune/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/4_eval_gqa.sh VQAv2 ./checkpoints/Qwen/CoIN/Finetune/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/4_eval_gqa.sh OCRVQA ./checkpoints/Qwen/CoIN/Finetune/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/5_eval_vizwiz.sh Grounding ./checkpoints/Qwen/CoIN/Finetune/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/5_eval_vizwiz.sh VQAv2 ./checkpoints/Qwen/CoIN/Finetune/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/5_eval_vizwiz.sh OCRVQA ./checkpoints/Qwen/CoIN/Finetune/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/6_eval_visualgenome.sh VQAv2 ./checkpoints/Qwen/CoIN/Finetune/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/6_eval_visualgenome.sh OCRVQA ./checkpoints/Qwen/CoIN/Finetune/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/7_eval_vqav2.sh OCRVQA ./checkpoints/Qwen/CoIN/Finetune/OCRVQA