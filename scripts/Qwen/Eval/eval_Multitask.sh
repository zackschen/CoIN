#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/1_eval_sqa.sh Zero_Shot ./checkpoints/Qwen/Qwen-VL-Chat
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/2_eval_textqa.sh Zero_Shot ./checkpoints/Qwen/Qwen-VL-Chat
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/3_eval_ImageNet.sh Zero_Shot ./checkpoints/Qwen/Qwen-VL-Chat
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/4_eval_gqa.sh Zero_Shot ./checkpoints/Qwen/Qwen-VL-Chat
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/5_eval_vizwiz.sh Zero_Shot ./checkpoints/Qwen/Qwen-VL-Chat
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/6_eval_grounding.sh Multitask ./checkpoints/Qwen/CoIN_Chat/Multitask
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/7_eval_vqav2.sh Zero_Shot ./checkpoints/Qwen/Qwen-VL-Chat
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval/8_eval_ocrvqa.sh Zero_Shot ./checkpoints/Qwen/Qwen-VL-Chat

