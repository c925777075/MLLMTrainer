from torch import nn
from src.dataset.root import DATASETS
DATASETS_LIST = ['Qwen3VL_MLLMDataset']

def load_dataset(dataset_args, training_args):
    if dataset_args is not None:
        type_ = dataset_args.type
        if type_ in DATASETS_LIST:
            dataset = DATASETS.build(dataset_args)
            return dataset
        else:
            assert False
    else:
        return None