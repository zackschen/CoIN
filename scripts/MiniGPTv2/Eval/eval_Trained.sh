#!/bin/bash

bash ./scripts/MiniGPTv2/Eval/1_Science.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN/Finetune/Multitask
bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Multitask
bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Multitask
bash ./scripts/MiniGPTv2/Eval/4_GQA.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Multitask
bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Multitask
bash ./scripts/MiniGPTv2/Eval/6_Grounding.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Multitask
bash ./scripts/MiniGPTv2/Eval/7_VQAv2.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Multitask
bash ./scripts/MiniGPTv2/Eval/8_OCRVQA.sh Zero_Shot ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Multitask

# bash ./scripts/MiniGPTv2/Eval/1_Science.sh TextVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/TextVQA
# bash ./scripts/MiniGPTv2/Eval/1_Science.sh ImageNet ./checkpoints/MiniGPTv2/CoIN_New/Finetune/ImageNet
# bash ./scripts/MiniGPTv2/Eval/1_Science.sh GQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/GQA
# bash ./scripts/MiniGPTv2/Eval/1_Science.sh VizWiz ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VizWiz
# bash ./scripts/MiniGPTv2/Eval/1_Science.sh Grounding ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Grounding
# bash ./scripts/MiniGPTv2/Eval/1_Science.sh VQAv2 ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VQAv2
# bash ./scripts/MiniGPTv2/Eval/1_Science.sh OCRVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA

# bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh ImageNet ./checkpoints/MiniGPTv2/CoIN_New/Finetune/ImageNet
# bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh GQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/GQA
# bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh VizWiz ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VizWiz
# bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh Grounding ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Grounding
# bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh VQAv2 ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VQAv2
# bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh OCRVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA

# bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh GQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/GQA
# bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh VizWiz ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VizWiz
# bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh Grounding ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Grounding
# bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh VQAv2 ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VQAv2
# bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh OCRVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA

# bash ./scripts/MiniGPTv2/Eval/4_GQA.sh VizWiz ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VizWiz
# bash ./scripts/MiniGPTv2/Eval/4_GQA.sh Grounding ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Grounding
# bash ./scripts/MiniGPTv2/Eval/4_GQA.sh VQAv2 ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VQAv2
# bash ./scripts/MiniGPTv2/Eval/4_GQA.sh OCRVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA

# bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh Grounding ./checkpoints/MiniGPTv2/CoIN_New/Finetune/Grounding
# bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh VQAv2 ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VQAv2
# bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh OCRVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA

# bash ./scripts/MiniGPTv2/Eval/6_Grounding.sh VQAv2 ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VQAv2
# bash ./scripts/MiniGPTv2/Eval/6_Grounding.sh OCRVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA

# bash ./scripts/MiniGPTv2/Eval/7_VQAv2.sh OCRVQA ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA