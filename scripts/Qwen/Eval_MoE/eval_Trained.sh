# #!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh Finetune ./checkpoints/Qwen/CoIN_MoE/ScienceQA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/2_eval_textqa.sh Finetune ./checkpoints/Qwen/CoIN_MoE/TextVQA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/3_eval_ImageNet.sh Finetune ./checkpoints/Qwen/CoIN_MoE/ImageNet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/4_eval_gqa.sh Finetune ./checkpoints/Qwen/CoIN_MoE/GQA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/5_eval_vizwiz.sh Finetune ./checkpoints/Qwen/CoIN_MoE/VizWiz
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/6_eval_grounding.sh Finetune ./checkpoints/Qwen/CoIN_MoE/Grounding
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/7_eval_vqav2.sh Finetune ./checkpoints/Qwen/CoIN_MoE/VQAv2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/8_eval_ocrvqa.sh Finetune ./checkpoints/Qwen/CoIN_MoE/OCRVQA

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh TextVQA ./checkpoints/Qwen/CoIN_MoE/TextVQA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh ImageNet ./checkpoints/Qwen/CoIN_MoE/ImageNet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh GQA ./checkpoints/Qwen/CoIN_MoE/GQA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh VizWiz ./checkpoints/Qwen/CoIN_MoE/VizWiz
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh Grounding ./checkpoints/Qwen/CoIN_MoE/Grounding
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh VQAv2 ./checkpoints/Qwen/CoIN_MoE/VQAv2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/1_eval_sqa.sh OCRVQA ./checkpoints/Qwen/CoIN_MoE/OCRVQA

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/2_eval_textqa.sh ImageNet ./checkpoints/Qwen/CoIN_MoE/ImageNet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/2_eval_textqa.sh GQA ./checkpoints/Qwen/CoIN_MoE/GQA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/2_eval_textqa.sh VizWiz ./checkpoints/Qwen/CoIN_MoE/VizWiz
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/2_eval_textqa.sh Grounding ./checkpoints/Qwen/CoIN_MoE/Grounding
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/2_eval_textqa.sh VQAv2 ./checkpoints/Qwen/CoIN_MoE/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/2_eval_textqa.sh OCRVQA ./checkpoints/Qwen/CoIN_MoE/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/3_eval_ImageNet.sh GQA ./checkpoints/Qwen/CoIN_MoE/GQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/3_eval_ImageNet.sh VizWiz ./checkpoints/Qwen/CoIN_MoE/VizWiz
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/3_eval_ImageNet.sh Grounding ./checkpoints/Qwen/CoIN_MoE/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/3_eval_ImageNet.sh VQAv2 ./checkpoints/Qwen/CoIN_MoE/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/3_eval_ImageNet.sh OCRVQA ./checkpoints/Qwen/CoIN_MoE/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/4_eval_gqa.sh VizWiz ./checkpoints/Qwen/CoIN_MoE/VizWiz
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/4_eval_gqa.sh Grounding ./checkpoints/Qwen/CoIN_MoE/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/4_eval_gqa.sh VQAv2 ./checkpoints/Qwen/CoIN_MoE/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/4_eval_gqa.sh OCRVQA ./checkpoints/Qwen/CoIN_MoE/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/5_eval_vizwiz.sh Grounding ./checkpoints/Qwen/CoIN_MoE/Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/5_eval_vizwiz.sh VQAv2 ./checkpoints/Qwen/CoIN_MoE/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/5_eval_vizwiz.sh OCRVQA ./checkpoints/Qwen/CoIN_MoE/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/6_eval_grounding.sh VQAv2 ./checkpoints/Qwen/CoIN_MoE/VQAv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/6_eval_grounding.sh OCRVQA ./checkpoints/Qwen/CoIN_MoE/OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Qwen/Eval_MoE/7_eval_vqav2.sh OCRVQA ./checkpoints/Qwen/CoIN_MoE/OCRVQA