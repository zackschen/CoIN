CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/llava-vicuna-2-7b-chat-pretrain Zero_shot

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/Multitask/llava-1.5-7b-lora Multi-task

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/ScienceQA/llava-1.5-7b-lora After_ScienceQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/TextVQA/llava-1.5-7b-lora After_TextVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/GQA/llava-1.5-7b-lora After_GQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/Grounding/llava-1.5-7b-lora After_Grounding

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/ImageNet/llava-1.5-7b-lora After_ImageNet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/OCRVQA/llava-1.5-7b-lora After_OCRVQA

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/VizWiz/llava-1.5-7b-lora After_VizWiz

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_MME/mme.sh ./checkpoints/Instruction/Only_Pretrain_1.5/VQAv2/llava-1.5-7b-lora After_VQAv2

