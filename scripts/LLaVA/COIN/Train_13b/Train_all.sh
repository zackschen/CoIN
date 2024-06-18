#!/bin/bash

sh ./scripts/LLaVA/COIN/Train_13b/1_Science.sh
sh ./scripts/LLaVA/COIN/Train_13b/2_TextVQA.sh
sh ./scripts/LLaVA/COIN/Train_13b/3_ImageNet.sh
sh ./scripts/LLaVA/COIN/Train_13b/4_GQA.sh
sh ./scripts/LLaVA/COIN/Train_13b/5_VizWiz.sh
sh ./scripts/LLaVA/COIN/Train_13b/6_Grounding.sh
sh ./scripts/LLaVA/COIN/Train_13b/7_vqav2.sh
sh ./scripts/LLaVA/COIN/Train_13b/8_OCRVQA.sh