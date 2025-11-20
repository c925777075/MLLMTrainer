# import torch
# torch.autograd.set_detect_anomaly(True)
import os
import sys
import time
import logging
import pathlib
import warnings

project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))
import wandb
from transformers.integrations import WandbCallback
from transformers import Trainer, Seq2SeqTrainer, TrainerCallback
from src.engine.balanced_trainer import modify_trainer_for_balanced_sampler
from src.engine.trainer import CustomSeq2SeqTrainer
from src.utils.misc import patch_cosine_with_warmup_schedule
from src.utils.common import patch_transformer_logging
from src.config import prepare_args
from src.models import load_model
from src.dataset import load_dataset
from src.utils.data_utils import get_sequence_length, torch_dataset_to_hf_dataset, data_post_process_sequence_parallel, packing_dataset
from src.utils.common import IGNORE_INDEX
from src.engine.qwen_vl_trainer import replace_qwen2_vl_attention_class


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_format = "[%(levelname)s] %(pathname)s:%(lineno)d: %(message)s"
logging.basicConfig(
    format=logging_format,
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def entry(cfg, training_args):
    model = load_model(cfg.model_args, training_args)
    if cfg.data_args.train.cfg.data_packing:
        replace_qwen2_vl_attention_class()
    train_dataset_wrapper = load_dataset(cfg.data_args.train, training_args)
    valid_dataset_wrapper = load_dataset(cfg.data_args.train, training_args)
    test_dataset_wrapper = load_dataset(cfg.data_args.test, training_args)

    data_collator = train_dataset_wrapper.collator()
    tokenizer = train_dataset_wrapper.tokenizer

    if train_dataset_wrapper is not None:
        train_dataset = train_dataset_wrapper.get_hf_dataset()
    if valid_dataset_wrapper is not None:
        valid_dataset = valid_dataset_wrapper.get_hf_dataset()
    if test_dataset_wrapper is not None:
        test_dataset = test_dataset_wrapper.get_hf_dataset()

    callbacks = [TrainerCallback]
    if os.environ.get("WANDB_DISABLED").lower() == "false":
        callbacks.append(WandbCallback)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )
    if os.environ.get("LENGTH_BALANCE", "false").lower() == "true":
        trainer = modify_trainer_for_balanced_sampler(trainer, train_dataset)

    # Training
    start_train_time = time.time()
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)  # noqa
        trainer.save_metrics("train", train_result.metrics)  # noqa

        model_save_dir = os.path.join(trainer.args.output_dir, "model_save")
        trainer.save_model(output_dir=model_save_dir)
        trainer.save_state()  # noqa
    end_train_time = time.time()
    print(f"Trained in {end_train_time-start_train_time:.2f} seconds.")

    # save cfg to output_dir
    try:
        output_dir = training_args.output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg.dump(os.path.join(output_dir, "cfg.py"))
    except Exception as e:
        warnings.warn(f'try to save cfg to output_dir, but get exception {e.args}')

    # Evaluation
    if training_args.do_eval:
        if hasattr(trainer, '_test_collator') and hasattr(trainer, '_eval_collator') \
                and trainer._test_collator != trainer._eval_collator:  # noqa
            warnings.warn('[WARNING!!!] use different collator for eval and test. but do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.)')
        eval_results = trainer.predict(valid_dataset, metric_key_prefix="eval")
        trainer.log_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_metrics("eval", eval_results.metrics)  # noqa

    # Predict
    if training_args.do_predict and test_dataset is not None:
        predict_results = trainer.predict(test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", predict_results.metrics)  # noqa
        trainer.save_metrics("test", predict_results.metrics)  # noqa


def main():
    patch_transformer_logging()
    cfg, training_args = prepare_args()
    # from transformers import AutoModel; AutoModel.from_pretrained("/mnt/cxzx/share/model_checkpoints/Qwen3-0.6B")
    patch_cosine_with_warmup_schedule(getattr(training_args, 'minimal_learning_rate', 0.0))
    if os.environ.get("WANDB_DISABLED").lower() == "false":
        training_args.wandb_key = os.environ.get("WANDB_KEY", "")
        wandb.login(key=training_args.wandb_key)
        training_args.report_to = "wandb"
        training_args.run_name=os.environ.get("WANDB_RUN_NAME", "demo")
    elif os.environ.get("USE_TENSORBOARD", "false").lower() == "true":
        training_args.report_to = "tensorboard"
        exp_name=os.environ.get("WANDB_RUN_NAME", "demo")
        training_args.logging_dir=f"./logs/{exp_name}"

    os.makedirs(training_args.output_dir, exist_ok=True)

    sequence_parallel_size = int(os.environ.get("SEQUENCE_PARALLEL_SIZE", "1"))
    sequence_parallel_mode = os.environ.get("SEQUENCE_PARALLEL_MODE", "ulysses")
    cfg.model_args.cfg.sequence_parallel_size = sequence_parallel_size
    cfg.model_args.cfg.sequence_parallel_mode = sequence_parallel_mode
    entry(cfg, training_args)

if __name__ == "__main__":
    main()
