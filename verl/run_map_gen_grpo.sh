#!/bin/bash
# ============================================================================
# VibeWorlding-Gym — GRPO 训练（单机 8卡）
#
# 前置条件：
#   1. 资产检索服务已启动        -> assets_retrieval/README.md   (默认 :8081)
#   2. PCG渲染服务已启动        -> render_in_blender/README.md  (默认 :8080)
#   3. RL 数据在 data/rl/        -> train.parquet / test.parquet
#   4. 基座或 SFT 权重在 MODEL_HOME 下
#
# 用法：
#   bash run_map_gen_grpo.sh
#   HF_MODEL_PATH=/path/to/ckpt bash run_map_gen_grpo.sh      # 从 SFT ckpt 继续
# 所有配置均可用环境变量覆盖，无需修改本文件。
# ============================================================================
set -x

# ==================== 仓库根目录 / 模型目录 ====================
# 本脚本位于 <repo-root>/verl/ 下。
VIBEWORLD_ROOT="${VIBEWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MODEL_HOME 存放基座 / SFT / RL 权重；LOCAL_MODEL_DIR 为训练期本地快盘（可指向 SSD）。
MODEL_HOME="${MODEL_HOME:-${VIBEWORLD_ROOT}/models}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${MODEL_HOME}}"

# ==================== 环境配置 ====================
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 注意：不要在此pin CUDA_VISIBLE_DEVICES。Ray 会自动探测本机 GPU 并为每个 actor
# 单独设置隔离；预先 pin 会与之冲突，导致同一 TP 组的两个 rank 落到同一张物理卡。

# 可选：wandb 上报（未设置则仅 console 日志）
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"

# ==================== Verifier LLM 配置 ====================
# unverified query 的 reward 由 verifier 调 MLLM 打分；verified query 走规则匹配
# gt_map，不需要 LLM。两种 transport：
#   direct （默认）训练节点直连 provider，最简单。
#   filerpc       训练节点把请求写到共享磁盘，由有网机器上的 utils/broker.py 代行
#                 调用。适合训练节点无外网，或希望大批 rollout 的 verify 并发打分
#                 （broker 用线程池并发处理不同 session）。
#                 用法：先在有网机器上 `bash utils/start_broker.sh`，再启动训练。
# API key 从环境变量读（direct 在训练节点，filerpc 在 broker 机器）：
#   gemini -> GEMINI_API_KEY | openai -> OPENAI_API_KEY
#   qwen3 / bailian -> DASHSCOPE_API_KEY
export VIBEWORLD_LLM_TRANSPORT="${VIBEWORLD_LLM_TRANSPORT:-direct}"
export VERIFY_MODEL_TYPE="${VERIFY_MODEL_TYPE:-gemini}"
export VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-gemini-2.5-flash}"

# filerpc 专用：共享 query 目录（两端必须一致）+ RPC 超时/轮询间隔（秒）
export VIBEWORLD_QUERY_DIR="${VIBEWORLD_QUERY_DIR:-${VIBEWORLD_ROOT}/verifier/query}"
export VIBEWORLD_RPC_TIMEOUT="${VIBEWORLD_RPC_TIMEOUT:-600}"
export VIBEWORLD_RPC_POLL_INTERVAL="${VIBEWORLD_RPC_POLL_INTERVAL:-0.5}"

# ==================== 服务地址 ====================
# 检索服务与资产白名单必须同一 id 体系（5 位 typeId），否则检索结果会被全部过滤。
export RETRIEVE_SERVER_URL="${RETRIEVE_SERVER_URL:-http://localhost:8081}"
export PCG_GRADIO_SERVER="${PCG_GRADIO_SERVER:-http://localhost:8080}"
export RETRIEVE_WHITELIST_PATH="${VIBEWORLD_ROOT}/render_in_blender/assets/item_infos.json"
TOOL_CONFIG_PATH="${SCRIPT_DIR}/verl/tools/configs/map_gen_tool_config.yaml"

# ==================== 模型 / 数据 / 输出路径 ====================
# 默认从基座直接 RL（cold start）。若已跑过 SFT，把 HF_MODEL_PATH 指向 SFT ckpt。
HF_MODEL_PATH=${HF_MODEL_PATH:-"${LOCAL_MODEL_DIR}/Qwen3-VL-8B-Thinking"}
HF_MODEL_PATH="${HF_MODEL_PATH%/}"    # verl copy_to_local 要求末尾无 /

DATA_DIR=${DATA_DIR:-"${VIBEWORLD_ROOT}/data/rl"}
train_path="${DATA_DIR}/train.parquet"
test_path="${DATA_DIR}/test.parquet"

EXP_NAME=${EXP_NAME:-"map_gen_grpo_8b"}
SAVE_PATH=${SAVE_PATH:-"${LOCAL_MODEL_DIR}/${EXP_NAME}"}
LOG_DIR=${LOG_DIR:-"${VIBEWORLD_ROOT}/log/rl/${EXP_NAME}"}
WANDB_LOG_DIR="${LOG_DIR}/wandb_log"
mkdir -p "${WANDB_LOG_DIR}"

# ==================== Anti-Hacking Reward 配置 ====================
# 效率折扣系数：verified query 按首次做对的轮次折扣 reward（0=禁用）
REWARD_ALPHA=${REWARD_ALPHA:-0.3}
# 多余轮次惩罚系数：全部做对后继续刷轮的惩罚（0=禁用）
REWARD_BETA=${REWARD_BETA:-0.1}
# 按query_type 动态限制 max_turns，格式 "type1:3,type3:5"（空=禁用）
MAX_TURNS_BY_TYPE=${MAX_TURNS_BY_TYPE:-"type1:3,type3:5"}
# verified 严格作用域校验：criteria 决定"授权改动集合"，改动越界（多删/误删/多加/
# 擅移）则 reward 清零，防止"本可精准改一处却大改地图"的 reward hacking。
# 仅基于 pos 做diff；rotate/Extend 因渲染归一化不可靠，不纳入越界检测。
VERIFIED_STRICT_SCOPE=${VERIFIED_STRICT_SCOPE:-1}
export VERIFIED_STRICT_SCOPE

# ==================== 并行配置 ====================
ENGINE=${1:-vllm}
GEN_TP=${GEN_TP:-4}
CP=${CP:-2}
TP=${TP:-4}
PP=${PP:-1}

echo "============================================="
echo "  GRPO 单机训练"
echo "  模型: ${HF_MODEL_PATH}"
echo "  数据: ${train_path}"
echo "  保存: ${SAVE_PATH}"
echo "  日志: ${LOG_DIR}"
echo "  服务: retrieve=${RETRIEVE_SERVER_URL}  pcg=${PCG_GRADIO_SERVER}"
echo "  Verifier: transport=${VIBEWORLD_LLM_TRANSPORT}  ${VERIFY_MODEL_TYPE}/${VERIFY_MODEL_NAME}"
echo "  Reward: alpha=${REWARD_ALPHA} beta=${REWARD_BETA} strict_scope=${VERIFIED_STRICT_SCOPE}"
echo "============================================="

# ============================================================================
# Verifier 预检：unverified reward 需要真实调用 LLM。若 key 缺失 / broker 未启动，
# 训练会跑到首次 verify 才失败、白白浪费算力，故先做一次真实往返探活。
# 设 SKIP_VERIFY_PRECHECK=1 可跳过。
# ============================================================================
if [ "${SKIP_VERIFY_PRECHECK:-0}" != "1" ]; then
  VIBE_UTILS_DIR="${VIBEWORLD_ROOT}/utils"
  echo "[preflight] 探测 verifier LLM (transport=${VIBEWORLD_LLM_TRANSPORT}, ${VERIFY_MODEL_TYPE}/${VERIFY_MODEL_NAME}) ..."
  # 预检用短超时，避免 broker 未启动时干等 VIBEWORLD_RPC_TIMEOUT
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
# 外层 while-true：训练若因 node-level OOM 被 kill，自动从最新 ckpt 续跑
# （配合 trainer.resume_mode=auto）。正常结束（exit 0）时跳出。
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
    data.train_batch_size=2 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
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
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=122880 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.nccl_timeout=7200 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
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
    trainer.n_gpus_per_node=${GPUS_PER_NODE:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
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
