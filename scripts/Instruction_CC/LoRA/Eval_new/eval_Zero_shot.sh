#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/1_eval_sqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/2_eval_textqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/3_eval_ImageNet.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/4_eval_gqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/5_eval_vizwiz.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/6_eval_visualgenome.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/7_eval_vqav2.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_new/8_eval_ocrvqa.sh Zero_shot checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain

