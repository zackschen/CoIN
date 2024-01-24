#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/1_eval_sqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/2_eval_textqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/3_eval_ImageNet.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/4_eval_gqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/5_eval_vizwiz.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/6_eval_visualgenome.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/7_eval_vqav2.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/FullFinetune/Eval_slim/8_eval_ocrvqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5_slim_full/Multitask/llava-1.5-7b-lora

