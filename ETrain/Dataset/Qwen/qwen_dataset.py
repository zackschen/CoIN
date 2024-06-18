from dataclasses import dataclass, field
import json
from typing import Dict, Sequence
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data.dataloader import default_collate
from PIL import Image
from ETrain.Models.Qwen.modeling_qwen import QWenLMHeadModel
import requests

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def rank0_print(local_rank,*args):
    if local_rank == 0:
        print(*args)

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, local_rank: int):
        super(SupervisedDataset, self).__init__()

        rank0_print(local_rank, "Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i].to(torch.int64),
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, local_rank: int, model: QWenLMHeadModel):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print(local_rank, "Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

        self.model = model
        self.config = model.config

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            print('Getting cached data: Index:{} : {}'.format(i,self.cached_data_dict[i]))
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0].to(torch.int64),
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret
        # print('Getting data: Index:{} : {}'.format(i,self.raw_data[i]["conversations"]))
        return ret

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model: QWenLMHeadModel

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, model: QWenLMHeadModel):
        self.tokenizer = tokenizer
        self.model = model
        self.config = model.config

    def __call__(self, batch: Sequence[Dict]):
        batch = default_collate(batch)
        
        images = []
        if torch.any(batch['input_ids'] == self.config.visual['image_start_id']):
            bos_pos = torch.where(batch['input_ids'] == self.config.visual['image_start_id'])
            eos_pos = torch.where(batch['input_ids'] == self.config.visual['image_start_id'] + 1)
            assert (bos_pos[0] == eos_pos[0]).all()
            img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
            image_paths = []
            for i, a, b in img_pos:
                image = batch['input_ids'][i][a + 1 : b - 1].tolist()
                image = image[ : image.index(self.config.visual['image_start_id'] + 2)]
                image_paths.append(bytes(image).decode('utf-8'))
            
            
            for image_path in image_paths:
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    image = Image.open(image_path)
                image = image.convert("RGB")
                images.append(self.model.transformer.visual.image_transform(image))
            images = torch.stack(images, dim=0)
        else:
            image=torch.zeros(3,224,224).to(dtype=self.model.transformer.visual.conv1.weight.dtype)
            images.append(image)
            images = torch.stack(images, dim=0)

        batch['images'] = images
        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model, max_len, local_rank: int
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print(local_rank, "Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len, local_rank=local_rank, model=model)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len, local_rank=local_rank)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,model = model)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
