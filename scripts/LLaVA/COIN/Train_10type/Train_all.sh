#!/bin/bash

# sh ./scripts/LLaVA/COIN/Train_10type/1_Science.sh
# sh ./scripts/LLaVA/COIN/Train_10type/2_TextVQA.sh
# sh ./scripts/LLaVA/COIN/Train_10type/3_ImageNet.sh
# sh ./scripts/LLaVA/COIN/Train_10type/4_GQA.sh
# sh ./scripts/LLaVA/COIN/Train_10type/5_VizWiz.sh
# sh ./scripts/LLaVA/COIN/Train_10type/6_Grounding.sh
# sh ./scripts/LLaVA/COIN/Train_10type/7_vqav2.sh
sh ./scripts/LLaVA/COIN/Train_10type/8_OCRVQA.sh

bash ./scripts/LLaVA/COIN/Eval_10type/eval_Trained.sh