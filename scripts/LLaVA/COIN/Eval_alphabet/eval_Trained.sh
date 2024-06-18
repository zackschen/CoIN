# #!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/GQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/2_eval_ImageNet.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ImageNet_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/3_eval_ocrvqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/OCRVQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/4_eval_grounding.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/Grounding_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/5_eval_sqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ScienceQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/6_eval_textqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/TextVQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/7_eval_vizwiz.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VizWiz_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/8_eval_vqav2.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/TextVQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh ScienceQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ScienceQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VizWiz_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-alphabet/Grounding_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/1_eval_gqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/OCRVQA_llava_lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/2_eval_ImageNet.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/TextVQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/2_eval_ImageNet.sh ScienceQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ScienceQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/2_eval_ImageNet.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VizWiz_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/2_eval_ImageNet.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-alphabet/Grounding_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/2_eval_ImageNet.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/2_eval_ImageNet.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/OCRVQA_llava_lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/3_eval_ocrvqa.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/TextVQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/3_eval_ocrvqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VizWiz_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/3_eval_ocrvqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN-alphabet/Grounding_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/3_eval_ocrvqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/3_eval_ocrvqa.sh ScienceQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ScienceQA_llava_lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/4_eval_grounding.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VizWiz_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/4_eval_grounding.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/TextVQA_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/4_eval_grounding.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/4_eval_grounding.sh ScienceQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ScienceQA_llava_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/4_eval_grounding.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN-alphabet/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/4_eval_grounding.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/OCRVQA_llava_lora


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/5_eval_sqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VizWiz_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/5_eval_sqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/5_eval_sqa.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN-alphabet/TextVQA_llava_lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/6_eval_textqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/6_eval_textqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VizWiz_llava_lora

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_alphabet/7_eval_vizwiz.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN-alphabet/VQAv2_llava_lora
