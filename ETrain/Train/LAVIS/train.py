"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import transformers

from ETrain.utils.LAVIS.common.config import Config
from ETrain.utils.LAVIS.common.dist_utils import get_rank, init_distributed_mode, setup_for_distributed
from ETrain.utils.LAVIS.common.logger import setup_logger
from ETrain.utils.LAVIS.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from ETrain.utils.LAVIS.common.registry import registry
from ETrain.utils.LAVIS.common.utils import now

# imports modules for registration
from ETrain.Dataset.LAVIS.builders import *
from ETrain.Dataset.LAVIS import *
from ETrain.Models import *
from ETrain.Train.LAVIS import *
from ETrain.Train.Base_trainer import *
from transformers import Trainer
# from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from transformers.modeling_utils import load_state_dict, get_checkpoint_shard_files, _load_state_dict_into_model
from transformers.deepspeed import is_deepspeed_zero3_enabled

import sys
sys.path.append('/home/chencheng/Code/Slim_Train')
from CoIN.peft import PeftModel, TaskType, get_peft_model, CoINMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict

def parse_args(remaining_strings):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank") 
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args(remaining_strings)

    return args

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    cfg_path: str = field(default="")
    previous_task_model_path: Optional[str] = field(default=None)


def load_model_from_previous_task(cfg, model, previous_task_model_path):
    rank0_print('Loading additional weights...')
    if os.path.exists(os.path.join(previous_task_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(previous_task_model_path, 'non_lora_trainables.bin'), map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(previous_task_model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

    if is_deepspeed_zero3_enabled():
        msg = _load_state_dict_into_model(model, non_lora_trainables,start_prefix = '')
    else:
        msg = model.load_state_dict(non_lora_trainables, strict=False)

    rank0_print('Loading LoRA weights...')
    filename = os.path.join(previous_task_model_path, WEIGHTS_NAME)
    adapters_weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    adapters_weights = {(k[10:] if (k.startswith('llm_model.') or k.startswith('llama_model.')) else k): v for k, v in adapters_weights.items()}
    if cfg.model_cfg.arch == 'blip2_vicuna_instruct':
        load_result = set_peft_model_state_dict(model.llm_model, adapters_weights, adapter_name="default")
    else:
        load_result = set_peft_model_state_dict(model.llama_model, adapters_weights, adapter_name="default")
    rank0_print('Model is loaded...')

def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    global local_rank

    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args, remaining_strings = parser.parse_args_into_dataclasses(return_remaining_strings = True)
    local_rank = training_args.local_rank

    args = parse_args(remaining_strings)
    cfg = Config(args)

    # init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    # cfg.pretty_print()

    datasets, Concated_Dataset = build_datasets(cfg)
    
    if cfg.model_cfg.arch == 'blip2_vicuna_instruct':
        model = create_InstructBlip_model(cfg)
    else:
        model = create_MiniGPT4_model(cfg)

    if training_args.previous_task_model_path is not None:
        # load model from previous task
        load_model_from_previous_task(cfg, model, training_args.previous_task_model_path)

    data_module = create_MiniGPT_data_module(Concated_Dataset, cfg)

    trainer = Trainer(model=model,
                    args=training_args,
                    **data_module)

    trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            if cfg.model_cfg.arch == 'blip2_vicuna_instruct':
                model.llm_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            else:
                model.llama_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
