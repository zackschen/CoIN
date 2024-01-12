#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/1_eval_sqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/2_eval_textqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/3_eval_ImageNet.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/4_eval_gqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/5_eval_vizwiz.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/6_eval_visualgenome.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/7_eval_vqav2.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Eval/8_eval_ocrvqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora

