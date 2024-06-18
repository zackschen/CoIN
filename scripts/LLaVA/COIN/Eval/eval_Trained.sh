# #!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/ScienceQA_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/2_eval_textqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/TextVQA_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/3_eval_ImageNet.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/ImageNet_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/4_eval_gqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/GQA_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/5_eval_vizwiz.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/VizWiz_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/6_eval_grounding.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/Grounding_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/7_eval_vqav2.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/VQAv2_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/8_eval_ocrvqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/TextVQA_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN-13b/ImageNet_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh GQA ./checkpoints/LLaVA/Instruction/CoIN-13b/GQA_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-13b/VizWiz_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-13b/Grounding_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-13b/VQAv2_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/1_eval_sqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/2_eval_textqa.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN-13b/ImageNet_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/2_eval_textqa.sh GQA ./checkpoints/LLaVA/Instruction/CoIN-13b/GQA_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/2_eval_textqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-13b/VizWiz_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/2_eval_textqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-13b/Grounding_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/2_eval_textqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-13b/VQAv2_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/2_eval_textqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/3_eval_ImageNet.sh GQA ./checkpoints/LLaVA/Instruction/CoIN-13b/GQA_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/3_eval_ImageNet.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-13b/VizWiz_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/3_eval_ImageNet.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-13b/Grounding_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/3_eval_ImageNet.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-13b/VQAv2_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/3_eval_ImageNet.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/4_eval_gqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-13b/VizWiz_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/4_eval_gqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-13b/Grounding_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/4_eval_gqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-13b/VQAv2_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/4_eval_gqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/5_eval_vizwiz.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-13b/Grounding_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/5_eval_vizwiz.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-13b/VQAv2_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/5_eval_vizwiz.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/6_eval_grounding.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-13b/VQAv2_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/6_eval_grounding.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval/7_eval_vqav2.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-13b/OCRVQA_lora