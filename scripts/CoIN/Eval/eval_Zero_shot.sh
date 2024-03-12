#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/1_eval_sqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/2_eval_textqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/3_eval_ImageNet.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/4_eval_gqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/5_eval_vizwiz.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/6_eval_grounding.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/7_eval_vqav2.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/CoIN/Eval/8_eval_ocrvqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain

