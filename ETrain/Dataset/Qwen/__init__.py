from .qwen_dataset import make_supervised_data_module
from .qwen_dataset import LazySupervisedDataset

def create_Qwen_data_module(*args):
    return make_supervised_data_module(*args)