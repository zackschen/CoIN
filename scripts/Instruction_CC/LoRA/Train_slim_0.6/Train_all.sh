#!/bin/bash

sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_1_Science.sh
sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_2_TextVQA.sh
sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_3_ImageNet.sh
sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_4_GQA.sh
sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_5_VizWiz.sh
sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_6_VisualGenome.sh
sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_7_vqav2.sh
sh scripts/Instruction_CC/LoRA/Train_slim_0.6/finetune_CC_Instruction_8_OCRVQA.sh