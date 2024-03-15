# This file is used to create a data module for LLaVA dataset
from .llava_dataset import make_supervised_data_module
from .llava_dataset import LazySupervisedDataset,DataCollatorForSupervisedDataset

def create_LLaVA_data_module(*args):
    return make_supervised_data_module(*args)