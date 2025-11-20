# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, RandomSampler
from transformers import Seq2SeqTrainer, Trainer
from transformers.trainer import _is_peft_model
from typing_extensions import override

from src.utils.data_utils import is_transformers_version_equal_to_4_46

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._has_dummy_forwarded = False

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        # TODO: sequence_parallel modes other than 'zigzag-ring' may not need dummy forward
        if "sequence_parallel_group" in dir(model):
            if not self._has_dummy_forwarded and model.sequence_parallel_group is not None:
                model.eval()
                with torch.no_grad():
                    _ = model(**inputs)
                model.train()
                self._has_dummy_forwarded = True
        return super().training_step(model, inputs, *args, **kwargs)

    # @override
    # def _get_train_sampler(self, train_dataset=None):
    #     if train_dataset is None:
    #         train_dataset = self.train_dataset
    #     if self.model.sequence_parallel_group is not None:
    #         return SequentialSampler(train_dataset)
    #     else:
    #         return super()._get_train_sampler()
    
    @override
    def _get_train_sampler(self):
        if self.model.sequence_parallel_group is not None:
            return SequentialSampler(self.train_dataset)
        else:
            return super()._get_train_sampler()

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        return_outputs = True
        loss, outputs = super().compute_loss(model, inputs, return_outputs, **kwargs)
        
        if "aux_loss" in outputs:
            # 确保在多卡环境下正确同步损失值
            aux_loss = outputs.aux_loss
            lm_loss = outputs.lm_loss
            
            # 如果是多卡训练，对损失值进行平均
            if self.args.n_gpu > 1:
                # 使用分布式通信来同步损失值
                import torch.distributed as dist
                if dist.is_initialized():
                    # 同步 lm_loss
                    dist.all_reduce(lm_loss, op=dist.ReduceOp.SUM)
                    lm_loss = lm_loss / dist.get_world_size()
                    
                    # 同步 aux_loss  
                    dist.all_reduce(aux_loss, op=dist.ReduceOp.SUM)
                    aux_loss = aux_loss / dist.get_world_size()
            
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "loss": loss.detach().item(), 
                    "aux_loss": aux_loss.detach().item(), 
                    "lm_loss": lm_loss.detach().item()
                })
        else:
            # compute loss without shift labels, as we have already shifted labels in data processing when using sequence parallel
            _, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="sum")
            logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]
            # Get vocab_size
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                vocab_size = unwrapped_model.base_model.model.config.vocab_size
            else:
                vocab_size = unwrapped_model.config.vocab_size
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)

            # weighted reduce within sequence_parallel_group
            sp_group = model.sequence_parallel_group
            loss = dist.nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)
            label_num = (labels != loss_fct.ignore_index).sum()
            label_num = dist.nn.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)
            loss /= label_num

        # now is single-sequence loss
        # print('loss', loss.shape, loss)

        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss