# CoIN: A Benchmark of ContinuaL Instruction tuNing for Multimodel Large Language Model

Cheng Chen, Junchen Zhu, Xu Luo, Hengtao Shen, LianLi Gao, Jingkuan Song.

<img src="./assets/architecture.png">

## Abstract
Instruction tuning demonstrates impressive performance in adapting Multimodal Large Language Models (MLLMs) to follow task instructions and improve generalization ability. By extending tuning across diverse tasks, MLLMs can further enhance their understanding of world knowledge and instruction intent. However, continual instruction tuning has been largely overlooked and there are no public benchmarks available. In this paper, we present CoIN, a comprehensive benchmark tailored for assessing the behavior of existing MLLMs under continual instruction tunning. CoIN comprises 10 meticulously crafted datasets spanning 8 tasks, ensuring diversity and serving as a robust evaluation framework to assess crucial aspects of continual instruction tuning, such as task order, instruction diversity and volume. Additionally, apart from traditional evaluation, we design another LLM-based metric to assess the knowledge preserved within MLLMs for reasoning. Following an in-depth evaluation of several MLLMs, we demonstrate that they still suffer catastrophic forgetting, and the failure in instruction alignment assumes the main responsibility, instead of reasoning knowledge forgetting. To this end, we introduce MoELoRA which is effective in retaining the previous instruction alignment.

## Install
1. Clone this repository and navigate to CoIN folder
``` 
git clone https://github.com/zackschen/CoIN.git
cd CoIN 
```
2. Install Package
```
conda create -n coin python=3.10 -y
conda activate coin
pip install --upgrade pip
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

This repo is based on [LLaVA](https://github.com/haotian-liu/LLaVA). 
If you meet a problem, maybe you could find some solutions in issuses.

## Dataset
Please download the images from the constituting dataset: ScienceQA, VQAv2, VizWiz, TextVQA, GQA, OCR-VQA, ImageNet, RefCOCO, RefCOCO+, and RefCOCOg.
|  Image Source   | Download Path  |
|  :----:  | :----:  |
| COCO | [train2014](http://images.cocodataset.org/zips/train2014.zip), [test2015](http://images.cocodataset.org/zips/test2015.zip), [val2014](http://images.cocodataset.org/zips/val2014.zip) |
| RefCOCO  | [annotation](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip) | 
| RefCOCO+  | [annotation](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip) | 
| RefCOCOg  | [annotation](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip) | 
| ImageNet  | [images](https://image-net.org/challenges/LSVRC/index.php) | 
| OCR-VQA  | [images](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_) | 
| GQA  | [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) | 
| TextVQA  | [train](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip),[test](https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip) | 
| ScienceQA  | [images](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev) | 
| VizWiz  | [train](https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip), [val](https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip), [test](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) | 

After downloading all of them, organize the data as follows:
```
├── COCO2014
│   └── train2014
├── GQA
│   └── images
├── OCR-VQA
│   └── images
├── TextVQA
│   └── train_images
│   └── test_images
```

Then, please download the instructions from our datasets path: [CoIN_Dataset](https://huggingface.co/datasets/Zacks-Chen/CoIN/tree/main)
then, organize the instructions as follows:
```
├── Instruction_Original
│   └── GQA
│       └── train.json
│       └── test.json
│   └── ScienceQA
│       └── train.json
│       └── test.json
├── Instruction_Type2
│   └── GQA
│       └── train.json
│       └── test.json
```

## Instruction Tuning
First, downloading the pretrained projectors in [LLaVA Model_Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

Setting `pretrain_mm_mlp_adapter` to the projector path.
You could modify the `deepspeed config` to change the deepspeed config.

We provide the scripts of our train order in `scripts/*/Train`.
Note, the `output_dir` of the previous script is the `previous_task_model_path` of the next training process.
Then, you could tune these datasets in your order.

We provide scripts for training MOELoRA with LLaVA in `scripts/LLaVA/Train_MOE`. Additionally, you can modify the code to train MiniGPT-V2 and Qwen-VL, following the example in lines 138-152 of `ETrain/Models/LLaVA/utils.py`.

## Evaluation
We have prepared the scripts to evaluate the trained model in `scripts/*/Eval`.

These scripts will evalute the trained model and create the prompts (`prompt_to_eval.json`) for evaluating the general knowldege.

To evaluate the general knowldege, you could add the result path to `scripts/Eval_GeneralKnowledge/eval_prompt_slim.sh` and run it, this script file will output a score to indicate the general knowledge.

## To Do
1. - [x] Evaluating on more MLLM, MiniGPT-4, ~~MiniGPT-V2~~, InstrctBlip, ~~Qwen-VL~~; MiniGPT-V2, Qwen-VL have been merged. In addition, since MiniGPT-4 and InstrctBlip are based on LAVIS resp, you can modify the config to train with these model.
2. - [] Evaluating on different size of MLLM; We are conducting experiments with larger model, 13b llava.
3. - [] Evaluating on full finetune.

## Citation
```
@misc{chen2024coin,
    title={CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model}, 
    author={Cheng Chen and Junchen Zhu and Xu Luo and Hengtao Shen and Lianli Gao and Jingkuan Song},
    year={2024},
    eprint={2403.08350},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgement
[LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon, and our base model LLaVA-1.5-7b that has the amazing vision-language capabilities!

[LAVIS](https://github.com/salesforce/LAVIS): the codebase MiniGPT and InstructBlip are built upon.

[MiniGPT](https://github.com/Vision-CAIR/MiniGPT-4.git): the codebase of MinigGPT and MinitGPT-v2.
