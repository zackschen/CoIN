CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT="https://hf-mirror.com" OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 29600 \
    ETrain/Train/LAVIS/train.py \
    --deepspeed ./scripts/zero3_offload.json \
    --lora_enable True --lora_r 64 --lora_alpha 256 \
    --cfg-path ./scripts/MiniGPTv2/8_OCRVQA.yaml \
    --bf16 True \
    --previous_task_model_path ./checkpoints/MiniGPTv2/CoIN_New/Finetune/VQAv2 \
    --output_dir ./checkpoints/MiniGPTv2/CoIN_New/Finetune/OCRVQA \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none