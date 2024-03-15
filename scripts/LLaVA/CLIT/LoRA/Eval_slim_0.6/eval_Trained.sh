#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/ScienceQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/2_eval_textqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/TextVQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/3_eval_ImageNet.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/ImageNet/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/4_eval_gqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/5_eval_vizwiz.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VizWiz/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/6_eval_visualgenome.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/Grounding/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/7_eval_vqav2.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VQAv2/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/8_eval_ocrvqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh TextVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/TextVQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh ImageNet ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/ImageNet/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh GQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VizWiz/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/Grounding/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VQAv2/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/1_eval_sqa.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/2_eval_textqa.sh ImageNet ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/ImageNet/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/2_eval_textqa.sh GQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/2_eval_textqa.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VizWiz/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/2_eval_textqa.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/Grounding/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/2_eval_textqa.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VQAv2/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/2_eval_textqa.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/3_eval_ImageNet.sh GQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/3_eval_ImageNet.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VizWiz/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/3_eval_ImageNet.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/Grounding/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/3_eval_ImageNet.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VQAv2/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/3_eval_ImageNet.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/4_eval_gqa.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VizWiz/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/4_eval_gqa.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/Grounding/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/4_eval_gqa.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VQAv2/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/4_eval_gqa.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/5_eval_vizwiz.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/Grounding/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/5_eval_vizwiz.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VQAv2/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/5_eval_vizwiz.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/6_eval_visualgenome.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/VQAv2/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/6_eval_visualgenome.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_slim_0.6/7_eval_vqav2.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_slim_new_0.6/OCRVQA/llava-1.5-7b-lora
