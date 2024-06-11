#!/bin/bash

sh ./scripts/Qwen/Train_MoE/1_scienceqa.sh
sh ./scripts/Qwen/Train_MoE/2_textvqa.sh
sh ./scripts/Qwen/Train_MoE/3_ImageNet.sh
sh ./scripts/Qwen/Train_MoE/4_GQA.sh
sh ./scripts/Qwen/Train_MoE/5_VizWiz.sh
sh ./scripts/Qwen/Train_MoE/6_VisualGenome.sh
sh ./scripts/Qwen/Train_MoE/7_vqav2.sh
sh ./scripts/Qwen/Train_MoE/8_OCRVQA.sh

sh ./scripts/Qwen/Eval_MoE/eval_Trained.sh