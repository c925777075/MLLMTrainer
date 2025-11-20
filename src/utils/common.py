import copy
import os
from typing import List, Union, Dict

import PIL.Image
import torch
import numpy as np
import torchvision.transforms.functional as F
import transformers
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100

def print_rank0(args):
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(args, flush=True)

def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param))


def patch_transformer_logging():
    import logging
    import transformers
    def enable_explicit_format():
        handlers = transformers.utils.logging._get_library_root_logger().handlers

        for handler in handlers:
            formatter = logging.Formatter("[(%(levelname)s) %(pathname)s:%(lineno)s ] %(asctime)s >> %(message)s")
            handler.setFormatter(formatter)
    transformers.utils.logging.enable_explicit_format = enable_explicit_format