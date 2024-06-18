#!/bin/bash

bash ./scripts/MiniGPTv2/Eval/1_Science.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/ScienceQA
bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/TextVQA
bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/ImageNet
bash ./scripts/MiniGPTv2/Eval/4_GQA.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/GQA
bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/VizWiz
bash ./scripts/MiniGPTv2/Eval/6_Grounding.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/Grounding
bash ./scripts/MiniGPTv2/Eval/7_VQAv2.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/VQAv2
bash ./scripts/MiniGPTv2/Eval/8_OCRVQA.sh Finetune ./checkpoints/MiniGPTv2/CoINv2/OCRVQA

bash ./scripts/MiniGPTv2/Eval/1_Science.sh TextVQA ./checkpoints/MiniGPTv2/CoINv2/TextVQA
bash ./scripts/MiniGPTv2/Eval/1_Science.sh ImageNet ./checkpoints/MiniGPTv2/CoINv2/ImageNet
bash ./scripts/MiniGPTv2/Eval/1_Science.sh GQA ./checkpoints/MiniGPTv2/CoINv2/GQA
bash ./scripts/MiniGPTv2/Eval/1_Science.sh VizWiz ./checkpoints/MiniGPTv2/CoINv2/VizWiz
bash ./scripts/MiniGPTv2/Eval/1_Science.sh Grounding ./checkpoints/MiniGPTv2/CoINv2/Grounding
bash ./scripts/MiniGPTv2/Eval/1_Science.sh VQAv2 ./checkpoints/MiniGPTv2/CoINv2/VQAv2
bash ./scripts/MiniGPTv2/Eval/1_Science.sh OCRVQA ./checkpoints/MiniGPTv2/CoINv2/OCRVQA

bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh ImageNet ./checkpoints/MiniGPTv2/CoINv2/ImageNet
bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh GQA ./checkpoints/MiniGPTv2/CoINv2/GQA
bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh VizWiz ./checkpoints/MiniGPTv2/CoINv2/VizWiz
bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh Grounding ./checkpoints/MiniGPTv2/CoINv2/Grounding
bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh VQAv2 ./checkpoints/MiniGPTv2/CoINv2/VQAv2
bash ./scripts/MiniGPTv2/Eval/2_TextVQA.sh OCRVQA ./checkpoints/MiniGPTv2/CoINv2/OCRVQA

bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh GQA ./checkpoints/MiniGPTv2/CoINv2/GQA
bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh VizWiz ./checkpoints/MiniGPTv2/CoINv2/VizWiz
bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh Grounding ./checkpoints/MiniGPTv2/CoINv2/Grounding
bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh VQAv2 ./checkpoints/MiniGPTv2/CoINv2/VQAv2
bash ./scripts/MiniGPTv2/Eval/3_ImageNet.sh OCRVQA ./checkpoints/MiniGPTv2/CoINv2/OCRVQA

bash ./scripts/MiniGPTv2/Eval/4_GQA.sh VizWiz ./checkpoints/MiniGPTv2/CoINv2/VizWiz
bash ./scripts/MiniGPTv2/Eval/4_GQA.sh Grounding ./checkpoints/MiniGPTv2/CoINv2/Grounding
bash ./scripts/MiniGPTv2/Eval/4_GQA.sh VQAv2 ./checkpoints/MiniGPTv2/CoINv2/VQAv2
bash ./scripts/MiniGPTv2/Eval/4_GQA.sh OCRVQA ./checkpoints/MiniGPTv2/CoINv2/OCRVQA

bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh Grounding ./checkpoints/MiniGPTv2/CoINv2/Grounding
bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh VQAv2 ./checkpoints/MiniGPTv2/CoINv2/VQAv2
bash ./scripts/MiniGPTv2/Eval/5_VizWiz.sh OCRVQA ./checkpoints/MiniGPTv2/CoINv2/OCRVQA

bash ./scripts/MiniGPTv2/Eval/6_Grounding.sh VQAv2 ./checkpoints/MiniGPTv2/CoINv2/VQAv2
bash ./scripts/MiniGPTv2/Eval/6_Grounding.sh OCRVQA ./checkpoints/MiniGPTv2/CoINv2/OCRVQA

bash ./scripts/MiniGPTv2/Eval/7_VQAv2.sh OCRVQA ./checkpoints/MiniGPTv2/CoINv2/OCRVQA