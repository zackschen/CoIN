from typing import Dict, Optional, Sequence, List
import transformers
import torch
from ETrain.utils.LLaVA.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dataclasses import dataclass, field
from torch.utils.data.dataloader import default_collate

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # have_image = False
        # all_have_image = True
        # for instance in instances:
        #     if 'image' in instance:
        #         have_image = True
        #         size = (instance['image'].shape[1], instance['image'].shape[2])
        #     if not 'image' in instance:
        #         all_have_image = False

        # if have_image and not all_have_image:
        #     for instance in instances:
        #         if not 'image' in instance:
        #             instance['image'] = torch.zeros(3, size[0], size[1])

        instances = default_collate(instances)

        if not self.cfg.model_cfg.arch == 'blip2_vicuna_instruct':
            instances['conv_q'] = instances['text_input']
            instances['conv_a'] = instances['text_output']

        batch = dict(
            instances=instances,
            labels=None,
            attention_mask=None,
        )
        return batch

def create_MiniGPT_data_module(train_dataset,
                                cfg) -> Dict:
    data_collator = DataCollatorForSupervisedDataset(cfg)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)