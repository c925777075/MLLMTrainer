import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from copy import deepcopy

data_dir = os.getcwd()
sys.path.append(data_dir)
from config._base_.dataset import DEFAULT_DATASET

def get_dataset_config(config_path):
    cfg = OmegaConf.load(config_path)
    dataname_list = cfg.get("data_list")
    datasets_list = []
    for dataname in dataname_list:
        if dataname in DEFAULT_DATASET:
            datasets_list.append(DEFAULT_DATASET.get(dataname))
        else:
            raise ValueError(f"dataset {dataname} is not defined")
    return datasets_list

now = datetime.now()
formatted_now = now.strftime("%Y_%m_%d")
save_path_prefix = "/mnt/cxzx/share/chenyu/cache/train_models"
model_name="mllm_demo"

## ================== train parameters =================
training_args = dict(
    num_train_epochs=1,
    do_train=True,
    do_eval=False,
    do_predict=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    learning_rate=6e-6,
    weight_decay=1e-6,
    warmup_ratio=0.02,
    save_strategy="steps",
    save_steps=500,
    seed=42,
    max_grad_norm=1.0,
    bf16=True,
    deepspeed="config/deepspeed/zero3.json",
    dataloader_num_workers=8,
    remove_unused_columns=False,
    label_names=None,
    ddp_find_unused_parameters=True,
    dataloader_persistent_workers=False,
    dataloader_drop_last=True,
    resume_from_checkpoint=None,
    gradient_checkpointing=True,
    ddp_timeout=1800,
    report_to=None,
    logging_steps=1,
    overwrite_output_dir=True,
    output_dir=f'{save_path_prefix}/{model_name}/train_{formatted_now}',
    full_determinism=False,
    mm_projector_lr = None,
    vision_tower_lr = None,
)


## ================ model ====================
pretrain_checkpoint = "/mnt/cxzx/share/model_checkpoints/Qwen3-VL-series/Qwen3-VL-8B-Instruct"

model_args = dict(
    type='Qwen3-VL',
    cfg=dict(
        checkpoint_path=pretrain_checkpoint,
        attn_implementation="flash_attention_2"
    )
)

## ================ dataset ==================
datasets_cfg_list = get_dataset_config(f"{data_dir}/config/_base_/dataset/config.yaml")

data_args = dict(
    train=dict(
        type='Qwen3VL_MLLMDataset',
        cfg=dict(
            dataset_use=datasets_cfg_list,
            max_pixels=28 * 28 * 576,
            min_pixels=28 * 28 * 16,
            video_max_frames=8,
            video_min_frames=4,
            video_max_pixels=1024 * 28 * 28,
            video_min_pixels=256 * 28 * 28,
            video_fps=2,
            model_type="qwen3vl",
            data_packing=True,
            tokenizer=pretrain_checkpoint,
            model_max_length=8192,
        ),
    ),
    valid=None,
    test=None,
)
