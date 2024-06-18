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

class CoINDataset(Dataset):
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
    
class CoIN_ScientQADataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_ScientQADataset,self).__getitem__(index)
    
class CoIN_GQADataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_GQADataset,self).__getitem__(index)
    
class CoIN_GroundingDataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_GroundingDataset,self).__getitem__(index)
    
class CoIN_ImageNetDataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_ImageNetDataset,self).__getitem__(index)
    
class CoIN_OCRVQADataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_OCRVQADataset,self).__getitem__(index)
    
class CoIN_TextVQADataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_TextVQADataset,self).__getitem__(index)
    
class CoIN_VizWizDataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_VizWizDataset,self).__getitem__(index)

class CoIN_VQAv2Dataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_VQAv2Dataset,self).__getitem__(index)

class CoIN_MultitaskDataset(CoINDataset):
    def __getitem__(self, index):
        return super(CoIN_MultitaskDataset,self).__getitem__(index)
    

############ Eval
class CoIN_EvalDataset(Dataset):
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
        if 'answer' in info.keys():
            answers = info['answer']
        else:
            answers = info['answer_bbox']

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

class CoIN_ScientQA_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_ScientQA_EvalDataset,self).__getitem__(index)
    
class CoIN_GQA_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_GQA_EvalDataset,self).__getitem__(index)
    
class CoIN_Grounding_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_Grounding_EvalDataset,self).__getitem__(index)
    
class CoIN_ImageNet_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_ImageNet_EvalDataset,self).__getitem__(index)
    
class CoIN_OCRVQA_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_OCRVQA_EvalDataset,self).__getitem__(index)
    
class CoIN_TextVQA_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_TextVQA_EvalDataset,self).__getitem__(index)
    
class CoIN_VizWiz_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_VizWiz_EvalDataset,self).__getitem__(index)

class CoIN_VQAv2_EvalDataset(CoIN_EvalDataset):
    def __getitem__(self, index):
        return super(CoIN_VQAv2_EvalDataset,self).__getitem__(index)