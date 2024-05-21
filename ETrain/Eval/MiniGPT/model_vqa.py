import argparse
import os
import json
import torch
import math
import deepspeed
from tqdm import tqdm
import shortuuid
import torch.distributed as dist
from torch.utils.data import DataLoader
from ETrain.utils.LAVIS.common.config import Config
# imports modules for registration
from ETrain.Dataset.LAVIS.builders import *
from ETrain.Dataset.LAVIS import *
from ETrain.Models import *
from ETrain.Train.LAVIS import *
from ETrain.Train.Base_trainer import *
from transformers import Trainer
from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from ETrain.utils.LAVIS.conversation.conversation import CONV_VISION_minigptv2

def load_model_from_previous_task(cfg, model, previous_task_model_path):
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

    msg = model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    filename = os.path.join(previous_task_model_path, WEIGHTS_NAME)
    adapters_weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    adapters_weights = {(k[10:] if (k.startswith('llm_model.') or k.startswith('llama_model.')) else k): v for k, v in adapters_weights.items()}
    load_result = set_peft_model_state_dict(model.llama_model, adapters_weights, adapter_name="default")
    print('Model is loaded...')

def eval_model(args):
    cfg = Config(args)

    _, Concated_Dataset = build_datasets(cfg)
    eval_dataloader = DataLoader(Concated_Dataset, batch_size=args.batch_size, shuffle=False)
    
    world_size = int(os.getenv('WORLD_SIZE', '4'))
    model = create_MiniGPT4_model(cfg)
    load_model_from_previous_task(cfg, model, args.model_path)

    ds_model = deepspeed.init_inference(
        model=model,      # Transformers模型
        mp_size=world_size,        # GPU数量
        replace_method="auto", # 让DS自动替换层
        replace_with_kernel_inject=True, # 使用kernel注入
    )

    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(eval_dataloader)):
        have_image = line['have_image']
        question_id = line["question_id"]
        question = line['text_input']
        answer = line['text_output']

        image = line['image']

        convs = [conv_temp.copy() for _ in range(len(question))]
        [conv.append_message(conv.roles[0], text) for conv, text in zip(convs, question)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        question = [conv.get_prompt() for conv in convs]

        answers = ds_model.generate(image, question, max_new_tokens=100, do_sample=False)

        for idx, (text, answer) in enumerate(zip(question, answers)):
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": question_id[idx],
                                    "prompt": text,
                                    "text": answer,
                                    "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    eval_model(args)
