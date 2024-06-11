#!/bin/bash

# sh ./scripts/Qwen/Train/1_scienceqa.sh
# sh ./scripts/Qwen/Train/2_textvqa.sh
# sh ./scripts/Qwen/Train/3_ImageNet.sh
# sh ./scripts/Qwen/Train/4_GQA.sh
# sh ./scripts/Qwen/Train/5_VizWiz.sh
# sh ./scripts/Qwen/Train/6_VisualGenome.sh
# sh ./scripts/Qwen/Train/7_vqav2.sh
# sh ./scripts/Qwen/Train/8_OCRVQA.sh

sh ./scripts/Qwen/Train/multitask.sh

sh ./scripts/Qwen/Eval/eval_Trained.sh