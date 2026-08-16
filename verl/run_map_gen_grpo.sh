#!/bin/bash
# ============================================================================
# VibeWorlding-Gym — GRPO training (single node, 8 GPUs)
#
# Prerequisites:
#   1. Asset retrieval service up  -> assets_retrieval/README.md   (default :8081)
#   2. PCG rendering service up    -> render_in_blender/README.md  (default :8080)
#   3. RL data at data/rl/         -> train.parquet / test.parquet
#   4. Base or SFT weights under MODEL_HOME
#
# Usage:
#   bash run_map_gen_grpo.sh
#   HF_MODEL_PATH=/path/to/ckpt bash run_map_gen_grpo.sh      # warm-start from an SFT ckpt
# Every knob is overridable via environment variables; no need to edit this file.
# ============================================================================
set -x

# ---- Repo root / model home ----
# This script lives at <repo-root>/verl/.
VIBEWORLD_ROOT="${VIBEWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MODEL_HOME holds base / SFT / RL weights; LOCAL_MODEL_DIR is the fast disk used
# during training (point it at a local SSD if you have one).
MODEL_HOME="${MODEL_HOME:-${VIBEWORLD_ROOT}/models}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${MODEL_HOME}}"

# ---- Runtime env ----
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Do NOT pin CUDA_VISIBLE_DEVICES here. Ray auto-detects the local GPUs and
# isolates each actor on its own device; pre-pinning conflicts with that and
# lets two ranks of the same TP group land on the same physical GPU.

# Optional: wandb reporting (falls back to console logs if unset)
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"

# ---- Verifier LLM ----
# Rewards for unverified queries come from an MLLM judge; verified queries do
# rule-based matching against gt_map and need no LLM. Two transports:
#   direct   (default) training node talks to the provider directly. Simplest.
#   filerpc  training node writes requests to a shared disk; utils/broker.py on
#            a networked host executes them. Use when the training node has no
#            outbound network, OR when you want batched rollouts to be judged
#            concurrently (broker uses a thread pool across sessions).
#            Bring up the broker with `bash utils/start_broker.sh` first.
# API keys are read from the environment (direct: training node; filerpc: broker host):
#   gemini -> GEMINI_API_KEY | openai -> OPENAI_API_KEY
#   qwen3 / bailian -> DASHSCOPE_API_KEY
export VIBEWORLD_LLM_TRANSPORT="${VIBEWORLD_LLM_TRANSPORT:-direct}"
export VERIFY_MODEL_TYPE="${VERIFY_MODEL_TYPE:-gemini}"
export VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-gemini-3.5-flash}"

# filerpc only: shared query dir (must match on both ends) + RPC timeout/poll interval (seconds)
export VIBEWORLD_QUERY_DIR="${VIBEWORLD_QUERY_DIR:-${VIBEWORLD_ROOT}/verifier/query}"
export VIBEWORLD_RPC_TIMEOUT="${VIBEWORLD_RPC_TIMEOUT:-600}"
export VIBEWORLD_RPC_POLL_INTERVAL="${VIBEWORLD_RPC_POLL_INTERVAL:-0.5}"

# ---- Service endpoints ----
# Retrieval service and asset whitelist must share the same 5-digit typeId
# scheme, otherwise every retrieved asset gets filtered out.
export RETRIEVE_SERVER_URL="${RETRIEVE_SERVER_URL:-http://localhost:8081}"
export PCG_GRADIO_SERVER="${PCG_GRADIO_SERVER:-http://localhost:8080}"
export RETRIEVE_WHITELIST_PATH="${VIBEWORLD_ROOT}/render_in_blender/assets/item_infos.json"
TOOL_CONFIG_PATH="${SCRIPT_DIR}/verl/tools/configs/map_gen_tool_config.yaml"

# ---- Model / data / output paths ----
# Defaults to RL from the base model (cold start). To warm-start from SFT,
# point HF_MODEL_PATH at the SFT ckpt.
HF_MODEL_PATH=${HF_MODEL_PATH:-"${LOCAL_MODEL_DIR}/Qwen3-VL-8B-Thinking"}
HF_MODEL_PATH="${HF_MODEL_PATH%/}"    # verl copy_to_local rejects a trailing slash

DATA_DIR=${DATA_DIR:-"${VIBEWORLD_ROOT}/data/rl"}
train_path="${DATA_DIR}/train.parquet"
test_path="${DATA_DIR}/test.parquet"

EXP_NAME=${EXP_NAME:-"map_gen_grpo_8b"}
SAVE_PATH=${SAVE_PATH:-"${LOCAL_MODEL_DIR}/${EXP_NAME}"}
LOG_DIR=${LOG_DIR:-"${VIBEWORLD_ROOT}/log/rl/${EXP_NAME}"}
WANDB_LOG_DIR="${LOG_DIR}/wandb_log"
mkdir -p "${WANDB_LOG_DIR}"

# ---- Anti-hacking reward ----
# Efficiency discount: verified queries are discounted by the turn at which they
# first succeed (0 = disabled).
REWARD_ALPHA=${REWARD_ALPHA:-0.3}
# Extra-turn penalty: penalty for continuing to spin turns after the task is
# already solved (0 = disabled).
REWARD_BETA=${REWARD_BETA:-0.1}
# Per-query-type max_turns override, e.g. "type1:3,type3:5" (empty = disabled).
MAX_TURNS_BY_TYPE=${MAX_TURNS_BY_TYPE:-"type1:3,type3:5"}
# Verified strict-scope check: the criteria define the "authorized edit set";
# any edit outside it (extra deletes, wrong deletes, extra adds, unauthorized
# moves) zeros out the reward. Prevents the "could have edited one spot but
# rewrote the whole map" reward-hacking pattern. Uses pos-only diff; rotate /
# extend are excluded because render normalization makes them unreliable.
VERIFIED_STRICT_SCOPE=${VERIFIED_STRICT_SCOPE:-1}
export VERIFIED_STRICT_SCOPE

# ---- Parallelism ----
ENGINE=${1:-vllm}
GEN_TP=${GEN_TP:-4}
CP=${CP:-2}
TP=${TP:-4}
PP=${PP:-1}

echo "============================================="
echo "  GRPO single-node training"
echo "  Model:    ${HF_MODEL_PATH}"
echo "  Data:     ${train_path}"
echo "  Save:     ${SAVE_PATH}"
echo "  Log:      ${LOG_DIR}"
echo "  Services: retrieve=${RETRIEVE_SERVER_URL}  pcg=${PCG_GRADIO_SERVER}"
echo "  Verifier: transport=${VIBEWORLD_LLM_TRANSPORT}  ${VERIFY_MODEL_TYPE}/${VERIFY_MODEL_NAME}"
echo "  Reward:   alpha=${REWARD_ALPHA} beta=${REWARD_BETA} strict_scope=${VERIFIED_STRICT_SCOPE}"
echo "============================================="

# ============================================================================
# Verifier preflight: unverified rewards need a real LLM call. If the API key
# is missing / broker is down, training would only fail at the first verify
# call and burn compute until then. Do a real round-trip up front instead.
# Set SKIP_VERIFY_PRECHECK=1 to skip.
# ============================================================================
if [ "${SKIP_VERIFY_PRECHECK:-0}" != "1" ]; then
  VIBE_UTILS_DIR="${VIBEWORLD_ROOT}/utils"
  echo "[preflight] probing verifier LLM (transport=${VIBEWORLD_LLM_TRANSPORT}, ${VERIFY_MODEL_TYPE}/${VERIFY_MODEL_NAME}) ..."
  # Short timeout for preflight so we don't idle for VIBEWORLD_RPC_TIMEOUT when
  # the broker is not running.
  VIBEWORLD_RPC_TIMEOUT="${VERIFY_PRECHECK_TIMEOUT:-60}" \
  python3 - "$VIBE_UTILS_DIR" <<'PYEOF'
import os, sys
sys.path.insert(0, sys.argv[1])
from llm import MODEL_TYPE_MAP          # rebound to FileRPCChat factory under filerpc
mt = os.environ.get("VERIFY_MODEL_TYPE", "gemini")
mn = os.environ.get("VERIFY_MODEL_NAME", "gemini-2.5-flash")
if mt not in MODEL_TYPE_MAP:
    print(f"[preflight] X unknown VERIFY_MODEL_TYPE={mt!r}, choices: {list(MODEL_TYPE_MAP)}")
    sys.exit(1)
try:
    bot = MODEL_TYPE_MAP[mt](model_name=mn, system_instruction="You are a terse assistant.")
    bot.reset()
    bot.mllm("Reply with a single word: ok", [])
    content = bot.history[-1].get("content") if bot.history else ""
except Exception as e:
    print(f"[preflight] X call failed: {type(e).__name__}: {e}")
    sys.exit(1)
if content:
    print(f"[preflight] OK verifier reachable, got: {content!r}")
    sys.exit(0)
print("[preflight] X verifier returned empty")
sys.exit(1)
PYEOF
  if [ $? -ne 0 ]; then
    echo "============================================="
    echo "X Verifier preflight failed (transport=${VIBEWORLD_LLM_TRANSPORT})."
    if [ "${VIBEWORLD_LLM_TRANSPORT}" = "filerpc" ]; then
      echo "   1) Make sure the broker is running on a networked host:"
      echo "        cd ${VIBE_UTILS_DIR} && ./start_broker.sh"
      echo "      Status/logs: ./start_broker.sh status | ./start_broker.sh log"
      echo "   2) Both ends must share the same mount and query dir:"
      echo "        ${VIBEWORLD_QUERY_DIR}"
      echo "   3) Broker host must export the matching API key"
    else
      echo "   1) Is VERIFY_MODEL_TYPE / VERIFY_MODEL_NAME valid?"
      echo "      Currently: ${VERIFY_MODEL_TYPE} / ${VERIFY_MODEL_NAME}"
      echo "   2) Is the matching API key exported (GEMINI_API_KEY / OPENAI_API_KEY /"
      echo "      DASHSCOPE_API_KEY), and can the training node reach the provider?"
      echo "   3) If the training node has no network, switch to broker:"
      echo "        VIBEWORLD_LLM_TRANSPORT=filerpc bash $0"
    fi
    echo "   To skip this preflight: SKIP_VERIFY_PRECHECK=1 bash $0"
    echo "============================================="
    exit 1
  fi
fi

# ============================================================================
# Outer while-true: if training gets killed by a node-level OOM, auto-resume
# from the latest ckpt (works together with trainer.resume_mode=auto). Breaks
# out on a clean exit (exit 0).
# ============================================================================
MAX_RETRIES=${MAX_RETRIES:-30}
RETRY_WAIT=${RETRY_WAIT:-60}
retry_count=0

while true; do
  echo "============================================="
  echo "  [$(date)] starting training  (retry ${retry_count}/${MAX_RETRIES})"
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
  echo "[$(date)] training exited (exit_code=${EXIT_CODE})"

  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training finished cleanly, weights at ${SAVE_PATH}"
    break
  fi

  retry_count=$((retry_count + 1))
  if [ ${retry_count} -ge ${MAX_RETRIES} ]; then
    echo "Hit max retries ${MAX_RETRIES}, giving up"
    exit 1
  fi

  echo "Abnormal exit, resuming from latest ckpt in ${RETRY_WAIT}s (trainer.resume_mode=auto)..."
  sleep ${RETRY_WAIT}
done
