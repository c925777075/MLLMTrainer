export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_DISABLED="true"  # 1, on, yes, true  close wandb
export WANDB_KEY="your wandb key"
export WANDB_PROJECT="training"  # name your W&B project
export WANDB_RUN_NAME="demo"

export TOKENIZERS_PARALLELISM="false"
export USE_LIGER_KERNEL="true"

NNODES=1
GPUS_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=25003

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/pipeline/train.py \
    --config config/mllm/qwen3_vl.py