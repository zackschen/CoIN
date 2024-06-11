#!/bin/bash

sh ./scripts/MiniGPTv2/Train/1_Science.sh
sh ./scripts/MiniGPTv2/Train/2_TextVQA.sh
sh ./scripts/MiniGPTv2/Train/3_ImageNet.sh
sh ./scripts/MiniGPTv2/Train/4_GQA.sh
sh ./scripts/MiniGPTv2/Train/5_VizWiz.sh
sh ./scripts/MiniGPTv2/Train/6_Grounding.sh
sh ./scripts/MiniGPTv2/Train/7_VQAv2.sh
sh ./scripts/MiniGPTv2/Train/8_OCRVQA.sh

bash ./scripts/MiniGPTv2/Eval/eval_Trained.sh