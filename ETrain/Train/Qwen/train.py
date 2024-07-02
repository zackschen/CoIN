# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, deepspeed
from accelerate.utils import DistributedType
from ETrain.Train.Base_trainer import *
from ETrain.Models.Qwen import create_Qwen_model
from ETrain.Dataset.Qwen import create_Qwen_data_module
from ETrain.Train.Qwen.qwen_trainer import QwenTrainer, load_model_from_previous_task


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    previous_task_model_path: Optional[str] = field(default=None)

    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=4)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    group_by_modality_length: bool = field(default=False)

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"] ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def train():
    global local_rank
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (model_args,data_args,training_args,lora_args,) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    
    local_rank = training_args.local_rank
    model, tokenizer = create_Qwen_model(training_args, model_args, data_args, lora_args)

    if model_args.previous_task_model_path is not None:
        # load model from previous task
        load_model_from_previous_task(model, model_args.previous_task_model_path)
        rank0_print('Model is loaded...')

    data_module = create_Qwen_data_module(tokenizer, data_args, model, training_args.model_max_length, local_rank)

    # Start trainner
    trainer = QwenTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    trainer.save_state()

    trainer.save_trained_model(training_args, lora_args)


if __name__ == "__main__":
    train()
