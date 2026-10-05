#!/bin/bash

sh ./scripts/LLaVA/COIN/Train_EWC/1_Science.sh
sh ./scripts/LLaVA/COIN/Train_EWC/2_TextVQA.sh
sh ./scripts/LLaVA/COIN/Train_EWC/3_ImageNet.sh
sh ./scripts/LLaVA/COIN/Train_EWC/4_GQA.sh
sh ./scripts/LLaVA/COIN/Train_EWC/5_VizWiz.sh
sh ./scripts/LLaVA/COIN/Train_EWC/6_Grounding.sh
sh ./scripts/LLaVA/COIN/Train_EWC/7_vqav2.sh
sh ./scripts/LLaVA/COIN/Train_EWC/8_OCRVQA.sh

sh ./scripts/LLaVA/COIN/Eval_EWC/eval_Trained.sh
