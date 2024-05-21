#!/bin/bash

sh ./scripts/LLaVA/COIN/Train/1_Science.sh
sh ./scripts/LLaVA/COIN/Train/2_TextVQA.sh
sh ./scripts/LLaVA/COIN/Train/3_ImageNet.sh
sh ./scripts/LLaVA/COIN/Train/4_GQA.sh
sh ./scripts/LLaVA/COIN/Train/5_VizWiz.sh
sh ./scripts/LLaVA/COIN/Train/6_Grounding.sh
sh ./scripts/LLaVA/COIN/Train/7_vqav2.sh
sh ./scripts/LLaVA/COIN/Train/8_OCRVQA.sh