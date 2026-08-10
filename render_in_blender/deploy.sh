#!/usr/bin/env bash
###############################################################################
# deploy.sh — PCG (Blender) rendering service
#
# Starts the Gradio rendering service. Optionally starts N Gradio workers behind
# a session-sticky reverse proxy for higher throughput, which matters when many
# agent rollouts render concurrently during sampling / RL training.
#
# Usage:
#   bash deploy.sh# 1 worker per GPU
#   PORT=8080 bash deploy.sh
#   WORKERS_PER_GPU=8 bash deploy.sh        # 8 workers per GPU (RL training)
#   WORKERS=1 bash deploy.sh                # exactly 1 worker, foreground (smoke test)
#   bash deploy.sh --stop                   # stop workers/proxy started from here
#
# With more than one worker, all of them sit behind session_proxy.py on $PORT, so
# clients keep using a single address.
#
# Requirements:
#   - Blender 4.2.x -> set BLENDER_EXE
#   - GLB assets in place -> assets/models/clone/  (see README.md)
###############################################################################
set -uo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${THIS_DIR}"

PY="${PYTHON:-python3}"
PORT="${PORT:-8080}"
BASE_PORT="${BASE_PORT:-7000}"
BLENDER_EXE="${BLENDER_EXE:-/opt/blender-4.2.0-linux-x64/blender}"
MODELS_DIR="${VIBEWORLD_MODELS_DIR:-${THIS_DIR}/assets/models/clone}"

# ── worker sizing ────────────────────────────────────────────────────────────
# Rendering is GPU-bound, so the pool is sized per GPU. A single Blender render
# does not saturate a GPU, so several workers per GPU raises throughput a lot.
#   WORKERS_PER_GPU=1  -> one worker per GPU
#   WORKERS_PER_GPU=8  -> what we use for RL training / large-scale sampling
# WORKERS overrides the total directly (WORKERS=1 keeps the old single-worker
# foreground behaviour, handy for a quick smoke test).
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"

if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
    NUM_GPUS=0
fi
[[ "${NUM_GPUS}" -eq 0 ]] && NUM_GPUS=1     # no GPU -> CPU rendering, single "device"

if [[ -n "${WORKERS:-}" ]]; then
    TOTAL_WORKERS="${WORKERS}"
    # respect an explicit total: lay workers out over the GPUs as evenly as possible
    WORKERS_PER_GPU=$(( (TOTAL_WORKERS + NUM_GPUS - 1) / NUM_GPUS ))
    [[ "${TOTAL_WORKERS}" -lt "${NUM_GPUS}" ]] && NUM_GPUS="${TOTAL_WORKERS}" && WORKERS_PER_GPU=1
else
    TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
fi

LOG_DIR="${THIS_DIR}/logs"
PID_FILE="${THIS_DIR}/.worker_pids"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── stop ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    if [[ -f "${PID_FILE}" ]]; then
        while read -r pid; do
            [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null && info "killed ${pid}"
        done < "${PID_FILE}"
        rm -f "${PID_FILE}"
    else
        warn "no ${PID_FILE}; nothing to stop"
    fi
    exit 0
fi

# ── preflight ────────────────────────────────────────────────────────────────
if [[ ! -x "${BLENDER_EXE}" ]]; then
    error "Blender not found at ${BLENDER_EXE}"
    error "Install Blender 4.2.x and set BLENDER_EXE, e.g."
    error "  wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz"
    error "  tar -xf blender-4.2.0-linux-x64.tar.xz -C /opt"
    error "  export BLENDER_EXE=/opt/blender-4.2.0-linux-x64/blender"
    exit 1
fi

if [[ -z "$(ls -A "${MODELS_DIR}" 2>/dev/null | grep -v '^\.gitkeep$')" ]]; then
    warn "GLB asset directory looks empty: ${MODELS_DIR}"
    warn "Download the 3D assets from https://huggingface.co/datasets/usail-hkust/VWE-Bench"
    warn "Scenes will render without assets until they are in place."
fi

mkdir -p "${LOG_DIR}"
export BLENDER_EXE
export VIBEWORLD_MODELS_DIR="${MODELS_DIR}"
export PYTHONUNBUFFERED=1

info "Blender : ${BLENDER_EXE}"
info "Assets  : ${MODELS_DIR}"

# ── single worker ────────────────────────────────────────────────────────────
if [[ "${TOTAL_WORKERS}" -le 1 ]]; then
    info "starting a single Gradio worker on :${PORT} (foreground)"
    info "for RL / large-scale sampling use multiple workers, e.g. WORKERS_PER_GPU=8"
    PORT="${PORT}" exec "${PY}" gradio_app.py
fi

# ── multi worker + session-sticky proxy ──────────────────────────────────────
# Rendering is GPU-bound (Cycles CUDA/OptiX), so workers are spread over the
# visible GPUs and each one is pinned to a single GPU via CUDA_VISIBLE_DEVICES.
# Every worker gets its own port and its own output dir (see gradio_app.py), and
# session_proxy.py fronts them all on${PORT}, keeping each Gradio session stuck
# to one worker (required by Gradio's queue/SSE model).
info "starting ${TOTAL_WORKERS} workers = ${NUM_GPUS} GPU(s) x ${WORKERS_PER_GPU}/GPU"
info "worker ports: ${BASE_PORT}..$((BASE_PORT + TOTAL_WORKERS - 1))"
: > "${PID_FILE}"
for((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    for ((slot = 0; slot < WORKERS_PER_GPU; slot++)); do
        wport=$((BASE_PORT + gpu * WORKERS_PER_GPU + slot))
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PORT="${wport}" RENDER_HOST=127.0.0.1 WORKER_ID="gpu${gpu}_${slot}" \
            nohup "${PY}" gradio_app.py > "${LOG_DIR}/worker_${wport}.log" 2>&1 &
        echo $! >> "${PID_FILE}"
    done
done

info "waiting for workers to come up ..."
for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${BASE_PORT}/" 2>/dev/null || true)
    [[ "${code}" == "200" ]] && break
    sleep 2
done
ready=0
for ((i = 0; i < TOTAL_WORKERS; i++)); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$((BASE_PORT + i))/" 2>/dev/null || true)
    [[ "${code}" == "200" ]] && ready=$((ready + 1))
done
info "workers up: ${ready}/${TOTAL_WORKERS}"
if [[ "${ready}" -eq 0 ]]; then
    error "no worker came up; see ${LOG_DIR}/worker_${BASE_PORT}.log"
    exit 1
fi

info "starting session-sticky proxy on :${PORT}"
nohup "${PY}" session_proxy.py --port "${PORT}" --base_port "${BASE_PORT}" \
    --workers "${TOTAL_WORKERS}" > "${LOG_DIR}/proxy.log" 2>&1 &
echo $! >> "${PID_FILE}"
sleep 3

code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/" 2>/dev/null || true)
if [[ "${code}" == "200" ]]; then
    info "service ready: http://0.0.0.0:${PORT}  (${ready} workers behind a sticky proxy)"
else
    warn "proxy on :${PORT} not answering yet (http=${code}); see ${LOG_DIR}/proxy.log"
fi
info "logs: ${LOG_DIR}   stop: bash deploy.sh --stop"
