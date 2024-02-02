#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/ScienceQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/2_eval_textqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/TextVQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/3_eval_ImageNet.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/ImageNet/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/4_eval_gqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/GQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/5_eval_vizwiz.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VizWiz/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/6_eval_visualgenome.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/Grounding/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/7_eval_vqav2.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VQAv2/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/8_eval_ocrvqa.sh Finetune ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh TextVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/TextVQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh ImageNet ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/ImageNet/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh GQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/GQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VizWiz/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/Grounding/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VQAv2/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/1_eval_sqa.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/2_eval_textqa.sh ImageNet ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/ImageNet/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/2_eval_textqa.sh GQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/GQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/2_eval_textqa.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VizWiz/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/2_eval_textqa.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/Grounding/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/2_eval_textqa.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VQAv2/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/2_eval_textqa.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/3_eval_ImageNet.sh GQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/GQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/3_eval_ImageNet.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VizWiz/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/3_eval_ImageNet.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/Grounding/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/3_eval_ImageNet.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VQAv2/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/3_eval_ImageNet.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/4_eval_gqa.sh VizWiz ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VizWiz/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/4_eval_gqa.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/Grounding/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/4_eval_gqa.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VQAv2/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/4_eval_gqa.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/5_eval_vizwiz.sh Grounding ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/Grounding/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/5_eval_vizwiz.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VQAv2/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/5_eval_vizwiz.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/6_eval_visualgenome.sh VQAv2 ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/VQAv2/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/6_eval_visualgenome.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/LoRA/Eval_MOE/7_eval_vqav2.sh OCRVQA ./checkpoints/Instruction/Only_Pretrain_1.5_MOE/OCRVQA/llava-1.5-7b-lora