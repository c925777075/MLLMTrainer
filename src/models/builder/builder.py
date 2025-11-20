import os
from torch import nn
from src.utils.common import print_trainable_params, print_rank0
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

def get_qwen_vl_model(model_args, training_args):
    if os.environ.get('USE_LIGER_KERNEL', 'false') == 'true':
        from transformers.models.qwen3_vl import modeling_qwen3_vl
        from src.models.builder.custom_qwen3_vl_liger_kernel import qwen3_vl_lce_forward
        modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = qwen3_vl_lce_forward

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.cfg.checkpoint_path,
        attn_implementation=model_args.cfg.attn_implementation,
        dtype="auto",
        # rope_scaling=model_args.cfg.rope_scaling
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    print_rank0(model)
    print_trainable_params(model)
    return model

def load_model(model_args, training_args) -> nn.Module:
    type_ = model_args.type
    if type_ in ['Qwen3-VL']:
        model = get_qwen_vl_model(model_args=model_args, training_args=training_args)
        return model
    else:
        raise NotImplementedError(f'{type_} is not implemented')