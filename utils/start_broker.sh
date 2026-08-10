#!/bin/bash
# start_broker.sh —— 在有网络的机器上常驻 LLM broker，崩溃自动重启。
#
# 用法：
#   ./start_broker.sh                 # 后台启动（nohup），崩溃自动拉起
#   ./start_broker.sh --echo          # echo 模式（疏通链路用）
#   ./start_broker.sh stop            # 停止
#   ./start_broker.sh status          # 查看状态
#   ./start_broker.sh log             # 跟踪日志
#
# 环境变量可覆盖默认：
#   BROKER_WORKERS   线程池大小（默认 8）
#   BROKER_QUERY_DIR query 目录（默认 .../verifier/query）
#   BROKER_POLL      轮询间隔秒（默认 0.3）

set -u

# 抬高文件描述符上限：broker 在高并发下会积累大量 fd（HTTP keepalive
# socket + per-session 客户端对象 + 轮询 query 目录），默认 soft limit 1024 远
# 不够用，会抛 `OSError: [Errno 24] Too many open files` 无法写 .resp.json →
# 训练侧 FileRPCChat 轮询不到响应 → verify 超时 → verifier_call_success=0。
# 训练脚本已对自身 ulimit -n 1048576，但 broker 由独立 shell 启动，需在此单独抬高。
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[start_broker] ulimit -n = $(ulimit -n) (hard=$(ulimit -Hn))"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BROKER_PY="$DIR/broker.py"
PID_FILE="$DIR/.broker.pid"
LOG_FILE="${BROKER_LOG:-$DIR/broker.log}"
QUERY_DIR="${BROKER_QUERY_DIR:-$DIR/../verifier/query}"
WORKERS="${BROKER_WORKERS:-8}"
POLL="${BROKER_POLL:-0.3}"

EXTRA_ARGS=""
case "${1:-start}" in
  stop)
    if [[ -f "$PID_FILE" ]]; then
      PID=$(cat "$PID_FILE")
      kill "$PID" 2>/dev/null && echo "已停止守护进程 (pid=$PID)"
      # 顺带杀掉可能残留的 broker.py 子进程
      pkill -f "broker.py --query-dir $QUERY_DIR" 2>/dev/null
      rm -f "$PID_FILE"
    else
      echo "未找到 pid 文件，尝试按命令行匹配停止"
      pkill -f "broker.py --query-dir $QUERY_DIR" 2>/dev/null && echo "已停止"
    fi
    exit 0
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "守护进程运行中 (pid=$(cat "$PID_FILE"))"
    else
      echo "守护进程未运行"
    fi
    echo "--- broker.py 进程 ---"
    pgrep -af "broker.py --query-dir $QUERY_DIR" || echo "(无)"
    exit 0
    ;;
  log)
    tail -f "$LOG_FILE"
    exit 0
    ;;
  --echo)
    EXTRA_ARGS="--echo"
    ;;
esac

# 已在运行则不重复启动
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "守护进程已在运行 (pid=$(cat "$PID_FILE"))，如需重启请先 stop"
  exit 1
fi

mkdir -p "$QUERY_DIR"

# 守护循环：broker.py 退出后 3 秒自动重启。
_supervise() {
  while true; do
    echo "[supervisor $(date '+%F %T')] 启动 broker.py workers=$WORKERS query_dir=$QUERY_DIR $EXTRA_ARGS" >> "$LOG_FILE"
    python -u "$BROKER_PY" --query-dir "$QUERY_DIR" --workers "$WORKERS" \
      --poll-interval "$POLL" $EXTRA_ARGS >> "$LOG_FILE" 2>&1
    echo "[supervisor $(date '+%F %T')] broker.py 退出 (code=$?)，3 秒后重启" >> "$LOG_FILE"
    sleep 3
  done
}

nohup bash -c "$(declare -f _supervise); BROKER_PY='$BROKER_PY' QUERY_DIR='$QUERY_DIR' WORKERS='$WORKERS' POLL='$POLL' EXTRA_ARGS='$EXTRA_ARGS' LOG_FILE='$LOG_FILE' _supervise" >/dev/null 2>&1 &

echo $! > "$PID_FILE"
echo "broker 守护进程已启动 (pid=$(cat "$PID_FILE"))"
echo "日志: $LOG_FILE"
echo "query 目录: $QUERY_DIR"
echo "查看状态: $0 status    停止: $0 stop    看日志: $0 log"
