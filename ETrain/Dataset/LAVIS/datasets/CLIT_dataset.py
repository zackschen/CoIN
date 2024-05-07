import os
import json
import pickle
import random
import time
import torch
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from ETrain.Dataset.LAVIS.datasets.base_dataset import BaseDataset
from ETrain.Dataset.LAVIS.datasets.caption_datasets import CaptionDataset

class CLITDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.ann=[]

    
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        self.connect_sym = "!@#"

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        have_image = True
        if 'image' not in info.keys():
            have_image = False
        
        image = None
        if have_image:
            image_path = os.path.join(self.vis_root, info['image'])
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)
        else:
            image = torch.zeros(3, self.vis_processor.transform.transforms[0].size[0], self.vis_processor.transform.transforms[0].size[0])

        first_instruction = info['conversations'][0]['value'].replace('<image>\n', '').strip()
        first_instruction = '<Img><ImageHere></Img> {} '.format(first_instruction)

        questions = [first_instruction]
        answers = []

        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 ==0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:
                human_instruction = item["value"]+" "
                questions.append(human_instruction)

        questions = self.connect_sym.join(questions)
        answers = self.connect_sym.join(answers)

        if 'id' in info.keys():
            idx = str(info['id'])
        else:
            idx = str(info['question_id'])

        return {
            "image": image,
            "text_input": questions,
            'text_output': answers,
            "image_id": idx,
            "connect_sym": self.connect_sym
        }
    
class CLIT_ScientQADataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_ScientQADataset,self).__getitem__(index)
    
class CLIT_GQADataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_GQADataset,self).__getitem__(index)
    
class CLIT_GroundingDataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_GroundingDataset,self).__getitem__(index)
    
class CLIT_ImageNetDataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_ImageNetDataset,self).__getitem__(index)
    
class CLIT_OCRVQADataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_OCRVQADataset,self).__getitem__(index)
    
class CLIT_TextVQADataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_TextVQADataset,self).__getitem__(index)
    
class CLIT_VizWizDataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_VizWizDataset,self).__getitem__(index)

class CLIT_VQAv2Dataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_VQAv2Dataset,self).__getitem__(index)

class CLIT_MultitaskDataset(CLITDataset):
    def __getitem__(self, index):
        return super(CLIT_MultitaskDataset,self).__getitem__(index)
    

############ Eval
class CLIT_EvalDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.ann=[]
    
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        self.connect_sym = "!@#"

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        have_image = True
        if 'image' not in info.keys():
            have_image = False
        
        image = None
        if have_image:
            image_path = os.path.join(self.vis_root, info['image'])
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)
        else:
            image = torch.zeros(3, self.vis_processor.transform.transforms[0].size[0], self.vis_processor.transform.transforms[0].size[0])

        questions = info['text'].replace('<image>\n', '').strip()
        questions = '<Img><ImageHere></Img> {} '.format(questions)
        answers = info['answer']

        if 'id' in info.keys():
            idx = str(info['id'])
        else:
            idx = str(info['question_id'])

        return {
            "have_image":have_image,
            "image": image,
            "text_input": questions,
            'text_output': answers,
            "question_id": idx,
            "connect_sym": self.connect_sym
        }

class CLIT_ScientQA_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_ScientQA_EvalDataset,self).__getitem__(index)
    
class CLIT_GQA_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_GQA_EvalDataset,self).__getitem__(index)
    
class CLIT_Grounding_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_Grounding_EvalDataset,self).__getitem__(index)
    
class CLIT_ImageNet_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_ImageNet_EvalDataset,self).__getitem__(index)
    
class CLIT_OCRVQA_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_OCRVQA_EvalDataset,self).__getitem__(index)
    
class CLIT_TextVQA_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_TextVQA_EvalDataset,self).__getitem__(index)
    
class CLIT_VizWiz_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_VizWiz_EvalDataset,self).__getitem__(index)

class CLIT_VQAv2_EvalDataset(CLIT_EvalDataset):
    def __getitem__(self, index):
        return super(CLIT_VQAv2_EvalDataset,self).__getitem__(index)