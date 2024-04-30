#!/bin/bash

# sh ./scripts/Instruction_CC/LoRA/Train/finetune_CC_Instruction_1_Science.sh
# sh ./scripts/Instruction_CC/LoRA/Train/finetune_CC_Instruction_2_TextVQA.sh
# sh ./scripts/Instruction_CC/LoRA/Train/finetune_CC_Instruction_3_ImageNet.sh
sh ./scripts/Qwen/4_GQA.sh
sh ./scripts/Qwen/5_VizWiz.sh
sh ./scripts/Qwen/6_VisualGenome.sh
sh ./scripts/Qwen/7_vqav2.sh
sh ./scripts/Qwen/8_OCRVQA.sh