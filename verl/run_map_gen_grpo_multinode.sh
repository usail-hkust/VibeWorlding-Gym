#!/bin/bash
# ============================================================================
# VibeWorlding-Gym — GRPO 训练（多机版，默认 3机×8卡 = 24 GPUs）
#
# 前置条件与单机版一致（检索服务 :8081 / 渲染服务 :8080 / data/rl 数据），
# 另需各节点间 IB 网络互通、共享同一挂载。
#
# 用法（每个节点都要执行，仅 NODE_RANK 不同）：
#   NODE_RANK=0 MASTER_ADDR=<head-ip> bash run_map_gen_grpo_multinode.sh
#   NODE_RANK=1 MASTER_ADDR=<head-ip> bash run_map_gen_grpo_multinode.sh
#   NODE_RANK=2 MASTER_ADDR=<head-ip> bash run_map_gen_grpo_multinode.sh
#
# 默认并行配置针对 30B MoE（TP=4, EP=2）；换模型时调整 TP/EP/GEN_TP。
# 所有配置均可用环境变量覆盖，无需修改本文件。
# ============================================================================
set -x

# ==================== 仓库根目录 / 模型目录 ====================
VIBEWORLD_ROOT="${VIBEWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_HOME="${MODEL_HOME:-${VIBEWORLD_ROOT}/models}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${MODEL_HOME}}"

# ==================== NCCL / IB 通信 ====================
# 下面的 HCA 列表按8 卡 H20 机型给出，请按自己集群的网卡名调整
# （`ibv_devinfo` 查看；若非 IB 环境可整段注释掉）。
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}"
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DIST_INIT_BARRIER_TIMEOUT=7200
export VERL_CHECKPOINT_TIMEOUT_MINUTES=120

# ==================== 文件描述符上限 ====================
# 大核数机器上 raylet 会开海量 socket/eventfd，默认 nofile 上限过低时会以
# `eventfd_select_interrupter: Too many open files` abort（表现为 driver 端
# "Failed to register worker to Raylet"）。必须在 ray start 之前抬高。
# 若报 operation not permitted，需在容器启动时加 --ulimit nofile=1048576:1048576。
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[fd-limit] ulimit -n = $(ulimit -n) (hard=$(ulimit -Hn))"

# ==================== 环境配置 ====================
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 注意：不要在 ray start 前手动 pin CUDA_VISIBLE_DEVICES，会与 Ray 的 per-actor
# GPU 隔离冲突，导致 NCCL "Duplicate GPU detected"（同节点多个 rank 抢同一张卡）。

# 可选：wandb 上报
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"

# ==================== Verifier LLM 配置 ====================
# 说明同单机版 run_map_gen_grpo.sh。多机场景下 filerpc 尤其合适：一次 rollout 会
# 产生大量 verify 请求，broker 用线程池并发打分。
export VIBEWORLD_LLM_TRANSPORT="${VIBEWORLD_LLM_TRANSPORT:-direct}"
export VERIFY_MODEL_TYPE="${VERIFY_MODEL_TYPE:-gemini}"
export VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-gemini-2.5-flash}"
export VIBEWORLD_QUERY_DIR="${VIBEWORLD_QUERY_DIR:-${VIBEWORLD_ROOT}/verifier/query}"
export VIBEWORLD_RPC_TIMEOUT="${VIBEWORLD_RPC_TIMEOUT:-600}"
export VIBEWORLD_RPC_POLL_INTERVAL="${VIBEWORLD_RPC_POLL_INTERVAL:-0.5}"

# ==================== 服务地址 ====================
export RETRIEVE_SERVER_URL="${RETRIEVE_SERVER_URL:-http://localhost:8081}"
export PCG_GRADIO_SERVER="${PCG_GRADIO_SERVER:-http://localhost:8080}"
export RETRIEVE_WHITELIST_PATH="${VIBEWORLD_ROOT}/render_in_blender/assets/item_infos.json"
TOOL_CONFIG_PATH="${SCRIPT_DIR}/verl/tools/configs/map_gen_tool_config.yaml"

# ==================== 多节点配置 ====================
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-6379}
NNODES=${NNODES:-3}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ==================== 模型 / 数据 / 输出路径 ====================
# 默认从基座直接 RL（cold start）。若已跑过 SFT，把 HF_MODEL_PATH 指向 SFT ckpt，
# 例如 ${MODEL_HOME}/ckpt/map_gen_sft/<exp>/global_step_N/actor/huggingface
HF_MODEL_PATH=${HF_MODEL_PATH:-"${MODEL_HOME}/Qwen3-VL-30B-A3B-Thinking"}
HF_MODEL_PATH="${HF_MODEL_PATH%/}"

DATA_DIR=${DATA_DIR:-"${VIBEWORLD_ROOT}/data/rl"}
train_path="${DATA_DIR}/train.parquet"
test_path="${DATA_DIR}/test.parquet"

EXP_NAME=${EXP_NAME:-"map_gen_grpo_30b_a3b_multinode"}
SAVE_PATH=${SAVE_PATH:-"${MODEL_HOME}/${EXP_NAME}"}
LOG_DIR=${LOG_DIR:-"${VIBEWORLD_ROOT}/log/rl/${EXP_NAME}"}

# ==================== Anti-Hacking Reward 配置 ====================
# 说明同单机版。多机时各 worker 均需继承 VERIFIED_STRICT_SCOPE。
REWARD_ALPHA=${REWARD_ALPHA:-0.3}
REWARD_BETA=${REWARD_BETA:-0.1}
MAX_TURNS_BY_TYPE=${MAX_TURNS_BY_TYPE:-"type1:3,type3:5"}
VERIFIED_STRICT_SCOPE=${VERIFIED_STRICT_SCOPE:-1}
export VERIFIED_STRICT_SCOPE

# ==================== 并行配置 (30B MoE, 3×8 GPUs) ====================
# Megatron 要求 world_size 能被 TP*EP*PP 整除：24 / (4*2*1) = 3 ✓
ENGINE=${1:-vllm}
GEN_TP=${GEN_TP:-8}
CP=${CP:-1}
TP=${TP:-4}
PP=${PP:-1}
EP=${EP:-2}

# ==================== Ray 集群初始化 ====================
echo "============================================="
echo "  多节点 Ray 集群初始化"
echo "  NODE_RANK=${NODE_RANK}, MASTER_ADDR=${MASTER_ADDR}"
echo "  NNODES=${NNODES}, GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "============================================="

ray stop --force 2>/dev/null || true
sleep 2

if [ ${NODE_RANK} -eq 0 ]; then
    echo "[Node ${NODE_RANK}] 启动 Ray Head (port=${MASTER_PORT})..."
    ray start --head \
        --port=${MASTER_PORT} \
        --num-cpus=$(nproc) \
        --num-gpus=${GPUS_PER_NODE} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --disable-usage-stats

    echo "[Node ${NODE_RANK}] 等待 ${NNODES} 个节点全部加入 Ray 集群..."
    retry_count=0
    max_retries=60
    while [ $retry_count -lt $max_retries ]; do
        connected_nodes=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "  已连接节点: ${connected_nodes}/${NNODES} (attempt $((retry_count+1))/${max_retries})"
        if [ "$connected_nodes" -ge "$NNODES" ]; then
            echo "  所有节点已加入 Ray 集群"
            break
        fi
        retry_count=$((retry_count + 1))
        sleep 10
    done

    if [ "$connected_nodes" -lt "$NNODES" ]; then
        echo "  警告: 仅 ${connected_nodes}/${NNODES} 节点连接，继续启动训练..."
    fi
else
    echo "[Node ${NODE_RANK}] 加入 Ray 集群 (head=${MASTER_ADDR}:${MASTER_PORT})..."
    sleep 10
    ray start \
        --address=${MASTER_ADDR}:${MASTER_PORT} \
        --num-cpus=$(nproc) \
        --num-gpus=${GPUS_PER_NODE} \
        --disable-usage-stats

    echo "[Node ${NODE_RANK}] 已加入 Ray 集群，等待 head 节点启动训练..."
    sleep infinity
    exit 0
fi

# ==================== 以下仅在 Head 节点 (NODE_RANK=0) 执行 ====================

WANDB_LOG_DIR="${LOG_DIR}/wandb_log"
mkdir -p "${WANDB_LOG_DIR}"

echo "============================================="
echo "  GRPO 多机训练 (${NNODES}×${GPUS_PER_NODE} GPUs)"
echo "  模型: ${HF_MODEL_PATH}"
echo "  数据: ${train_path}"
echo "  保存: ${SAVE_PATH}"
echo "  日志: ${LOG_DIR}"
echo "  并行: TP=${TP}, EP=${EP}, PP=${PP}, CP=${CP}, GEN_TP=${GEN_TP}"
echo "  服务: retrieve=${RETRIEVE_SERVER_URL}  pcg=${PCG_GRADIO_SERVER}"
echo "  Verifier: transport=${VIBEWORLD_LLM_TRANSPORT}  ${VERIFY_MODEL_TYPE}/${VERIFY_MODEL_NAME}"
echo "  Reward: alpha=${REWARD_ALPHA} beta=${REWARD_BETA} strict_scope=${VERIFIED_STRICT_SCOPE}"
echo "============================================="

# ============================================================================
# Verifier 预检：说明同单机版。设 SKIP_VERIFY_PRECHECK=1 可跳过。
# ============================================================================
if [ "${SKIP_VERIFY_PRECHECK:-0}" != "1" ]; then
  VIBE_UTILS_DIR="${VIBEWORLD_ROOT}/utils"
  echo "[preflight] 探测 verifier LLM (transport=${VIBEWORLD_LLM_TRANSPORT}, ${VERIFY_MODEL_TYPE}/${VERIFY_MODEL_NAME}) ..."
  VIBEWORLD_RPC_TIMEOUT="${VERIFY_PRECHECK_TIMEOUT:-60}" \
  python3 - "$VIBE_UTILS_DIR" <<'PYEOF'
import os, sys
sys.path.insert(0, sys.argv[1])
from llm import MODEL_TYPE_MAP          # filerpc 时已被重绑为 FileRPCChat 工厂
mt = os.environ.get("VERIFY_MODEL_TYPE", "gemini")
mn = os.environ.get("VERIFY_MODEL_NAME", "gemini-2.5-flash")
if mt not in MODEL_TYPE_MAP:
    print(f"[preflight] X 未知 VERIFY_MODEL_TYPE={mt!r}，可选：{list(MODEL_TYPE_MAP)}")
    sys.exit(1)
try:
    bot = MODEL_TYPE_MAP[mt](model_name=mn, system_instruction="你是简洁助手。")
    bot.reset()
    bot.mllm("只回复两个字：正常", [])
    content = bot.history[-1].get("content") if bot.history else ""
except Exception as e:
    print(f"[preflight] X 调用失败: {type(e).__name__}: {e}")
    sys.exit(1)
if content:
    print(f"[preflight] OK verifier 连通，返回: {content!r}")
    sys.exit(0)
print("[preflight] X verifier 返回空")
sys.exit(1)
PYEOF
  if [ $? -ne 0 ]; then
    echo "============================================="
    echo "X Verifier 预检失败 (transport=${VIBEWORLD_LLM_TRANSPORT})。"
    if [ "${VIBEWORLD_LLM_TRANSPORT}" = "filerpc" ]; then
      echo "   1) 确认已在【有网机器】上启动 broker："
      echo "        cd ${VIBE_UTILS_DIR} && ./start_broker.sh"
      echo "      查看状态/日志：./start_broker.sh status | ./start_broker.sh log"
      echo "   2) 确认两端共享同一挂载且 query 目录一致："
      echo "        ${VIBEWORLD_QUERY_DIR}"
      echo "   3) 确认 broker 机器上已导出对应 API key"
    else
      echo "   1) VERIFY_MODEL_TYPE / VERIFY_MODEL_NAME 是否有效"
      echo "      当前：${VERIFY_MODEL_TYPE} / ${VERIFY_MODEL_NAME}"
      echo "   2) 对应 API key 是否已导出（GEMINI_API_KEY / OPENAI_API_KEY /"
      echo "      DASHSCOPE_API_KEY），且训练节点能访问该 provider"
      echo "   3) 训练节点无外网时，改用 broker："
      echo "        VIBEWORLD_LLM_TRANSPORT=filerpc bash $0"
    fi
    echo "   如需跳过本预检：SKIP_VERIFY_PRECHECK=1 bash $0"
    echo "============================================="
    exit 1
  fi
fi

# ============================================================================
# 外层 while-true：训练若因 OOM 被 kill，自动从最新 ckpt 续跑
# ============================================================================
MAX_RETRIES=${MAX_RETRIES:-30}
RETRY_WAIT=${RETRY_WAIT:-60}
retry_count=0

while true; do
  echo "============================================="
  echo "  [$(date)] 启动训练  (重试次数: ${retry_count}/${MAX_RETRIES})"
  echo "============================================="

HYDRA_FULL_ERROR=1 \
WANDB_MODE=online \
WANDB_DIR="${WANDB_LOG_DIR}" \
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_path}" \
    data.val_files="${test_path}" \
    data.train_batch_size=6 \
    data.val_batch_size=8 \
    data.max_prompt_length=28000 \
    data.max_response_length=98304 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key='images' \
    data.prompt_key='prompt' \
    +data.custom_dataset.path=verl.utils.dataset.map_gen_dataset \
    +data.custom_dataset.name=MapGenRLDataset \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=131072 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    ++actor_rollout_ref.rollout.multi_turn.enable=True \
    ++actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    ++actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    ++actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    ++actor_rollout_ref.rollout.multi_turn.format=qwen3_vl \
    ++actor_rollout_ref.rollout.multi_turn.auto_render=true \
    ++actor_rollout_ref.rollout.multi_turn.pcg_mode=gradio \
    ++actor_rollout_ref.rollout.multi_turn.pcg_gradio_server="${PCG_GRADIO_SERVER}" \
    ++actor_rollout_ref.rollout.multi_turn.retrieve_url="${RETRIEVE_SERVER_URL}" \
    ++actor_rollout_ref.rollout.multi_turn.pcg_item_infos_path="${RETRIEVE_WHITELIST_PATH}" \
    ++actor_rollout_ref.rollout.multi_turn.max_parallel_calls=16 \
    ++actor_rollout_ref.rollout.multi_turn.enable_verify=true \
    ++actor_rollout_ref.rollout.multi_turn.per_turn_max_tokens=8192 \
    ++actor_rollout_ref.rollout.multi_turn.pcg_max_concurrency=16 \
    ++actor_rollout_ref.rollout.multi_turn.log_dir="${LOG_DIR}" \
    ++actor_rollout_ref.rollout.multi_turn.reward_efficiency_alpha=${REWARD_ALPHA} \
    ++actor_rollout_ref.rollout.multi_turn.reward_efficiency_beta=${REWARD_BETA} \
    ++actor_rollout_ref.rollout.multi_turn.verified_strict_scope=${VERIFIED_STRICT_SCOPE} \
    "++actor_rollout_ref.rollout.multi_turn.max_turns_by_type='${MAX_TURNS_BY_TYPE}'" \
    ++actor_rollout_ref.rollout.agent.default_agent_loop=map_gen_agent \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.actor.checkpoint.async_save=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${SCRIPT_DIR}/verl/utils/reward_score/map_gen_reward.py" \
    reward.custom_reward_function.name=map_gen_compute_score \
    +reward.custom_reward_function.reward_kwargs.reward_strategy=hard \
    +reward.custom_reward_function.reward_kwargs.hard_weight=1.0 \
    +reward.custom_reward_function.reward_kwargs.soft_weight=0.0 \
    global_profiler.save_path="${SAVE_PATH}" \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name="${PROJECT_NAME:-MapGenRL}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.resume_mode=auto \
    +trainer.ckpt_reward_topk=3 \
    ++trainer.val_before_train=True \
    trainer.default_local_dir="${SAVE_PATH}" \
    "$@"

  EXIT_CODE=$?
  echo "[$(date)] 训练退出 (exit_code=${EXIT_CODE})"

  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "训练正常完成，权重在 ${SAVE_PATH}"
    break
  fi

  retry_count=$((retry_count + 1))
  if [ ${retry_count} -ge ${MAX_RETRIES} ]; then
    echo "已达最大重试次数 ${MAX_RETRIES}，放弃重试"
    exit 1
  fi

  echo "训练异常退出，${RETRY_WAIT}s 后从最新 ckpt 续跑 (trainer.resume_mode=auto)..."
  sleep ${RETRY_WAIT}
done
