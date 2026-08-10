#!/usr/bin/env bash
# ============================================================================
# VibeWorlding-Gym — SFT训练（单机 8卡）
#
# 前置条件：
#   1. 基座权重在 MODEL_HOME 下（默认 Qwen3-VL-8B-Thinking）
#   2. 打包好的 SFT parquet 在 DATA_DIR 下（train.parquet / val.parquet，
#      含 messages / tools / reasoning_content / tool_calls 列）
#      —— 由 main.py 采样 + eval.py 打分后，把通过 verifier 的轨迹打包得到
#
# 用法：
#   DATA_DIR=/path/to/sft_parquet bash run_map_gen_sft.sh
# 所有配置均可用环境变量覆盖，无需修改本文件。
# ============================================================================
set -xeuo pipefail

# ==================== 仓库根目录 / 模型目录 ====================
VIBEWORLD_ROOT="${VIBEWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_HOME="${MODEL_HOME:-${VIBEWORLD_ROOT}/models}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${MODEL_HOME}}"

# ==================== 环境配置 ====================
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 可选：wandb 上报（未设置则仅 console 日志）
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"

# ==================== 模型 / 数据路径 ====================
# 基座模型。使用模型原生 chat template，不做任何替换。
MODEL_ID=${MODEL_ID:-"${LOCAL_MODEL_DIR}/Qwen3-VL-8B-Thinking"}

DATA_DIR=${DATA_DIR:-"${VIBEWORLD_ROOT}/data/sft"}
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/val.parquet"

if [ ! -f "${TRAIN_FILES}" ]; then
    echo "找不到训练集 ${TRAIN_FILES}"
    echo "data/sft 下是原始 case 目录，需先打包成 parquet，再用 DATA_DIR 指过来。"
    exit 1
fi
if [ ! -f "${VAL_FILES}" ]; then
    VAL_FILES="null"
    echo "验证集不存在，跳过验证"
fi

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

# ==================== 训练引擎 / 并行配置 ====================
backend=${BACKEND:-fsdp}

# FSDP
SP_SIZE=${SP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:--1}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}
# Megatron（BACKEND=megatron 时使用）
TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-8}

# ==================== 训练超参数 ====================
NUM_TRAINERS=${NUM_TRAINERS:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
LR=${LR:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-122880}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-122880}
PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

# ==================== 检查点配置 ====================
project_name=${PROJECT_NAME:-"map_gen_sft"}
exp_name=${EXP_NAME:-"map_gen_sft_8b"}
RESUME_MODE=${RESUME_MODE:-auto}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:--1}

CKPT_HOME=${CKPT_HOME:-"${MODEL_HOME}/ckpt/${project_name}/${exp_name}"}
mkdir -p "${CKPT_HOME}"

# ==================== 引擎配置构建 ====================
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

# ==================== 启动训练 ====================
echo "============================================="
echo "  SFT 单机训练（Qwen3 原生 tool calling 格式）"
echo "  模型: ${MODEL_ID}"
echo "  数据: ${TRAIN_FILES}"
echo "  GPU: ${NUM_TRAINERS}"
echo "  Epochs: ${TOTAL_EPOCHS}   LR: ${LR}   BS: ${TRAIN_BATCH_SIZE}"
echo "  Max Length: ${MAX_LENGTH}"
echo "  Checkpoint: ${CKPT_HOME}"
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

echo "SFT 完成，checkpoint 在 ${CKPT_HOME}"
echo "接着做 RL：HF_MODEL_PATH=${CKPT_HOME}/global_step_N/actor/huggingface bash run_map_gen_grpo.sh"
