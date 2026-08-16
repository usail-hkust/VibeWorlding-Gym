#!/usr/bin/env bash
# ============================================================================
# VibeWorlding-Gym — SFT training (single node, 8 GPUs)
#
# Prerequisites:
#   1. Base model under MODEL_HOME (defaults to Qwen3-VL-8B-Thinking)
#   2. Packed SFT parquet under DATA_DIR (train.parquet / val.parquet with
#      messages / tools / reasoning_content / tool_calls columns)
#      -- produced by sampling with main.py, scoring with eval.py, and
#      packing the verifier-passing trajectories.
#
# Usage:
#   DATA_DIR=/path/to/sft_parquet bash run_map_gen_sft.sh
# Every knob is overridable via environment variables; no need to edit this file.
# ============================================================================
set -xeuo pipefail

# ---- Repo root / model home ----
VIBEWORLD_ROOT="${VIBEWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_HOME="${MODEL_HOME:-${VIBEWORLD_ROOT}/models}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${MODEL_HOME}}"

# ---- Runtime env ----
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Optional: wandb reporting (falls back to console logs if unset)
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"

# ---- Model / data paths ----
# Base model. We use the model's native chat template with no substitution.
MODEL_ID=${MODEL_ID:-"${LOCAL_MODEL_DIR}/Qwen3-VL-8B-Thinking"}

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

# ---- Training engine / parallelism ----
backend=${BACKEND:-fsdp}

# FSDP
SP_SIZE=${SP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:--1}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}
# Megatron (used when BACKEND=megatron)
TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-8}

# ---- Hyperparameters ----
NUM_TRAINERS=${NUM_TRAINERS:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
LR=${LR:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-122880}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-122880}
PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

# ---- Checkpointing ----
project_name=${PROJECT_NAME:-"map_gen_sft"}
exp_name=${EXP_NAME:-"map_gen_sft_8b"}
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
    optim.min_lr=2e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "FSDP (SP=${SP_SIZE}, FSDP_SIZE=${FSDP_SIZE}, Strategy=${FSDP_STRATEGY})"
    exp_name="${exp_name}-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}"
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Megatron (TP=${TP_SIZE}, PP=${PP_SIZE}, VPP=${VPP_SIZE}, CP=${CP_SIZE})"
    exp_name="${exp_name}-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}"
fi

# ---- Launch ----
echo "============================================="
echo "  SFT single-node training (Qwen3 native tool-calling format)"
echo "  Model:   ${MODEL_ID}"
echo "  Data:    ${TRAIN_FILES}"
echo "  GPUs:    ${NUM_TRAINERS}"
echo "  Epochs:  ${TOTAL_EPOCHS}   LR: ${LR}   BS: ${TRAIN_BATCH_SIZE}"
echo "  Max len: ${MAX_LENGTH}"
echo "  Ckpt:    ${CKPT_HOME}"
echo "============================================="

HYDRA_FULL_ERROR=1 WANDB_MODE=online \
torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS} \
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
echo "Next: HF_MODEL_PATH=${CKPT_HOME}/global_step_N/actor/huggingface bash run_map_gen_grpo.sh"
