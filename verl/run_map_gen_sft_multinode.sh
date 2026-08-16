#!/usr/bin/env bash
# ============================================================================
# VibeWorlding-Gym — SFT training (multi-node, default 3 nodes x 8 GPUs = 24 GPUs)
#
# Same prerequisites as the single-node script (packed SFT parquet under
# DATA_DIR), plus IB connectivity across nodes and a shared mount.
# Default parallelism is tuned for the 30B MoE (Megatron TP=4, EP=2).
#
# Usage (run on every node; only NODE_RANK differs):
#   NODE_RANK=0 MASTER_ADDR=<master-ip> DATA_DIR=/path/to/sft_parquet bash run_map_gen_sft_multinode.sh
#   NODE_RANK=1 MASTER_ADDR=<master-ip> DATA_DIR=/path/to/sft_parquet bash run_map_gen_sft_multinode.sh
#   NODE_RANK=2 MASTER_ADDR=<master-ip> DATA_DIR=/path/to/sft_parquet bash run_map_gen_sft_multinode.sh
#
# Every knob is overridable via environment variables; no need to edit this file.
# ============================================================================
set -xeuo pipefail

# ---- Repo root / model home ----
VIBEWORLD_ROOT="${VIBEWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_HOME="${MODEL_HOME:-${VIBEWORLD_ROOT}/models}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${MODEL_HOME}}"

# ---- NCCL / IB ----
# HCA list below matches an 8-card H20 host; adjust to your NIC names
# (`ibv_devinfo`). Comment the whole block out on non-IB clusters.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}"
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1

# ---- Runtime env ----
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: wandb reporting
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"

# ---- Multi-node config ----
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-3}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ---- Model / data paths ----
# Base 30B MoE. We use the model's native chat template with no substitution.
MODEL_ID=${MODEL_ID:-"${MODEL_HOME}/Qwen3-VL-30B-A3B-Thinking"}

DATA_DIR=${DATA_DIR:-"${VIBEWORLD_ROOT}/data/sft"}
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/val.parquet"

if [ ! -f "${TRAIN_FILES}" ]; then
    echo "Training file not found: ${TRAIN_FILES}"
    echo "data/sft holds raw case dirs; pack them into parquet first and point DATA_DIR at it."
    exit 1
fi
if [ ! -f "${VAL_FILES}" ]; then
    VAL_FILES="null"
    echo "Validation set missing; skipping validation."
fi

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

# ---- Training engine / parallelism (30B MoE) ----
backend=${BACKEND:-megatron}

# Megatron requires world_size % (TP*EP*PP) == 0: 24 / (4*2*1) = 3.
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-2}
# FSDP fallback (BACKEND=fsdp)
SP_SIZE=${SP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:--1}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}

# ---- Hyperparameters ----
NUM_TRAINERS=${GPUS_PER_NODE}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
LR=${LR:-5e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
MAX_LENGTH=${MAX_LENGTH:-122880}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-122880}
PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

# ---- Checkpointing ----
project_name=${PROJECT_NAME:-"map_gen_sft"}
exp_name=${EXP_NAME:-"map_gen_sft_30b_a3b_multinode"}
RESUME_MODE=${RESUME_MODE:-auto}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:--1}

CKPT_HOME=${CKPT_HOME:-"${MODEL_HOME}/ckpt/${project_name}/${exp_name}"}
mkdir -p "${CKPT_HOME}"

# ---- Engine config ----
FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"

MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True \
    engine.param_offload=True \
    engine.optimizer_offload=True \
    engine.grad_offload=True \
    +engine.override_transformer_config.moe_router_dtype=fp32 \
    +engine.override_transformer_config.moe_permute_fusion=True \
    +engine.override_transformer_config.recompute_method=uniform \
    +engine.override_transformer_config.recompute_granularity=full \
    +engine.override_transformer_config.recompute_num_layers=1 \
    +engine.override_transformer_config.gradient_accumulation_fusion=True"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "FSDP (SP=${SP_SIZE}, FSDP_SIZE=${FSDP_SIZE}, Strategy=${FSDP_STRATEGY})"
    exp_name="${exp_name}-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}"
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Megatron (TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, CP=${CP_SIZE})"
    exp_name="${exp_name}-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}-cp${CP_SIZE}"
fi

# ---- Launch ----
echo "============================================="
echo "  SFT multi-node training (${NNODES} x ${GPUS_PER_NODE} GPUs)"
echo "  Model:   ${MODEL_ID}"
echo "  Data:    ${TRAIN_FILES}"
echo "  Nodes:   ${NNODES} (current NODE_RANK=${NODE_RANK})"
echo "  Master:  ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Epochs:  ${TOTAL_EPOCHS}   LR: ${LR}   BS: ${TRAIN_BATCH_SIZE}"
echo "  Max len: ${MAX_LENGTH}"
echo "  Parallel: TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, CP=${CP_SIZE}"
echo "  Ckpt:    ${CKPT_HOME}"
echo "============================================="

HYDRA_FULL_ERROR=1 WANDB_MODE=online \
torchrun \
    --nnodes=${NNODES} \
    --nproc-per-node=${NUM_TRAINERS} \
    --node-rank=${NODE_RANK} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=${PAD_MODE} \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN} \
    data.messages_key=messages \
    data.tools_key=tools \
    data.ignore_input_ids_mismatch=True \
    data.num_workers=8 \
    model.path="${MODEL_ID}" \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=3 \
    checkpoint.save_contents="[model,hf_model,optimizer,extra]" \
    "$@"

echo "SFT finished, checkpoint at ${CKPT_HOME}"
echo "Next: HF_MODEL_PATH=${CKPT_HOME}/global_step_N/actor/huggingface bash run_map_gen_grpo_multinode.sh"
