from ETrain.utils.LAVIS.common.registry import registry
from .minigpt_dataset import *
from torch.utils.data.dataset import ConcatDataset

def build_datasets(cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        concat_datasets = []
        for name in datasets:
            dataset = datasets[name]
            concat_datasets.append(dataset['train'])
        
        Concated_Dataset = ConcatDataset(concat_datasets)

        return datasets, Concated_Dataset