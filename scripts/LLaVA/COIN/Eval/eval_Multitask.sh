#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/1_eval_sqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/2_eval_textqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/3_eval_ImageNet.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/4_eval_gqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/5_eval_vizwiz.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/6_eval_visualgenome.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/7_eval_vqav2.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 bash ./scripts/Instruction_CC/LoRA/Eval/8_eval_ocrvqa.sh Multitask checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora

