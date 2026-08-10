#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# deploy.sh — 资产检索服务最简部署（D1 single-slot 检索）
#
# 启动一个 FastAPI 检索服务，暴露 POST /recommend/single_slot（5 位 type_id 体系）。
# 与 PCG 渲染服务 + 5 位白名单 item_infos.json 对齐，供 RL / 蒸馏的 retrieve_assets 使用。
#
# 用法：
#   bash deploy.sh                 # 默认 8080 端口，前台启动
#   PORT=8081 bash deploy.sh       # 指定端口
#   D1_CKPT_DIR=/path/to/VibeWorlder-Embedding-4B PORT=8081 bash deploy.sh
#
# 首次启动会全量预编码资产 embedding（GPU 约 1-3 分钟）；之后走 cache/ 秒起。
# 验证：
#   curl -s -X POST "http://127.0.0.1:${PORT:-8080}/recommend/single_slot" \
#     -H "Content-Type: application/json" \
#     -d '{"entity_name":"探险木屋","top_k":3}' | python3 -m json.tool
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 定位：本脚本在 assets_retrieval/ 下，D1_deploy 是其子包 ──────────────
# 需从 assets_retrieval/ 作为 CWD 运行，才能 `python -m D1_deploy.main`。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── 服务参数（可被外部 env 覆盖）─────────────────────────────────────
export PORT="${PORT:-8080}"          # 监听端口
export D1_HOST="${D1_HOST:-0.0.0.0}" # 监听地址
export D1_LOG_LEVEL="${D1_LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

# ── 检索模型 ckpt：VibeWorlder-Embedding-4B ──────────────────────────
# 从 https://huggingface.co/collections/usail-hkust/vibeworlder 下载。
export D1_CKPT_DIR="${D1_CKPT_DIR:-./models/VibeWorlder-Embedding-4B}"
if [[ ! -d "${D1_CKPT_DIR}" ]]; then
  echo "[deploy] 错误：找不到 embedding 模型目录 ${D1_CKPT_DIR}" >&2
  echo "[deploy] 请先下载 VibeWorlder-Embedding-4B，或用 D1_CKPT_DIR 指定路径。" >&2
  echo "[deploy]   huggingface-cli download usail-hkust/VibeWorlder-Embedding-4B \\" >&2
  echo "[deploy]     --local-dir ./models/VibeWorlder-Embedding-4B" >&2
  exit 1
fi

# ── 离线，避免联网探测 HF 卡启动 ─────────────────────────────────────
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PY="${PYTHON:-python3}"
echo "[deploy] 检索服务启动中：host=${D1_HOST} port=${PORT} cwd=${SCRIPT_DIR}"
echo "[deploy] 模型：${D1_CKPT_DIR}"
echo "[deploy] 端点：POST http://${D1_HOST}:${PORT}/recommend/single_slot"

# ── 启动（config 从 PORT / D1_CKPT_DIR 等环境变量读取配置）─────────────
exec "${PY}" -m D1_deploy.main
