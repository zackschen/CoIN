import os
import json
import pickle
import random
import time
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from ETrain.Dataset.MiniGPT.datasets.base_dataset import BaseDataset
from ETrain.Dataset.MiniGPT.datasets.caption_datasets import CaptionDataset

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

        first_instruction = info['conversations'][0]['value'].replace('<image>\n', '').strip()
        if have_image:
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
            idx = info['id']
        else:
            idx = info['question_id']

        if have_image:
            return {
                "image": image,
                "conv_q": questions,
                'conv_a': answers,
                "image_id": idx,
                "connect_sym": self.connect_sym
            }
        else:
            return {
                "conv_q": questions,
                'conv_a': answers,
                "image_id": idx,
                "connect_sym": self.connect_sym
            }
    
class CLIT_ScientQADataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)
    
class CLIT_GQADataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)
    
class CLIT_GroundingDataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)
    
class CLIT_ImageNetDataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)
    
class CLIT_OCRVQADataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)
    
class CLIT_TextVQADataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)
    
class CLIT_VizWizDataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)

class CLIT_VQAv2Dataset(CLITDataset):
    def __getitem__(self, index):
        return super.__getitem__(index)