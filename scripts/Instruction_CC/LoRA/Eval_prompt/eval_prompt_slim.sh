# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_prompt/eval_prompt.sh \
#     ./results/CLIT_slim_new_0.1/ScienceQA/ViwWiz \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_prompt/eval_prompt.sh \
#     ./results/CLIT_slim_new_0.2/ScienceQA/VizWiz \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_prompt/eval_prompt.sh \
    ./results/CLIT_slim_new_0.4/ScienceQA/VizWiz \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_prompt/eval_prompt.sh \
    ./results/CLIT_slim_new_0.6/ScienceQA/VizWiz \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash ./scripts/Instruction_CC/LoRA/Eval_prompt/eval_prompt.sh \
    ./results/CLIT_slim_new_0.8/ScienceQA/VizWiz \
