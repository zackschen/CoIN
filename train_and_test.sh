# bash scripts/Instruction_CC/Train_slim_new/Train_all.sh
bash scripts/Instruction_CC/Eval_slim_new/eval_Zero_shot.sh
bash scripts/Instruction_CC/Eval_slim_new/eval_Trained.sh

# bash scripts/Instruction_CC/Eval_slim/eval_Zero_shot.sh
bash /home/chencheng/Code/LLaVA/scripts/Instruction_CC/Train_slim/finetune_CC_Instruction_8_OCRVQA.sh
bash scripts/Instruction_CC/Eval_slim/eval_Trained.sh

# bash scripts/Instruction_CC/Train_slim_new/Train_multitask.sh
# bash scripts/Instruction_CC/Train_slim/Train_multitask.sh

# bash scripts/Instruction_CC/Eval_slim/eval_Multitask.sh
# bash scripts/Instruction_CC/Eval_slim_new/eval_Multitask.sh