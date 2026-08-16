"""
usage：
    python broker.py                       # 真实模式
    python broker.py --echo                # echo 模式（不接 LLM，仅回显，用于疏通链路）
    python broker.py --query-dir /path     # 覆盖 query 目录
    python broker.py --workers 8           # 线程池大小


"""

import os
import sys
import time
import argparse
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm  # noqa: E402  同目录 llm.py


# ---- 共享状态（多线程访问，用 _STATE_LOCK 保护 _SESSIONS / _TICK）----
_SESSIONS = {}          # session_id -> {"client","last_used","lock"}
_TICK = 0               # 单调计数，代替 wall-clock 做 TTL
_STATE_LOCK = threading.Lock()
_INFLIGHT = set()       # 正在处理的 req_id，避免主循环重复派发
_INFLIGHT_LOCK = threading.Lock()


def _get_session(payload):
    """按 session_id 取或建 session 条目（含真实客户端与 per-session 锁）。
    model_name/system_instruction 变化时重建客户端。线程安全。"""
    sid = payload["session_id"]
    backend = payload.get("backend", "gemini")
    model_name = payload.get("model_name", "") or ""
    system_instruction = payload.get("system_instruction", "") or ""
    tools = payload.get("tools")

    with _STATE_LOCK:
        entry = _SESSIONS.get(sid)
        need_new = True
        if entry is not None:
            c = entry["client"]
            if (getattr(c, "model_name", None) == model_name and
                    getattr(c, "system_instruction", None) == system_instruction):
                entry["last_used"] = _TICK
                need_new = False
        if need_new:
            cls = llm.REAL_MODEL_TYPE_MAP[backend]
            kwargs = {"system_instruction": system_instruction, "tools": tools}
            if model_name:
                kwargs["model_name"] = model_name
            client = cls(**kwargs)
            # 保留旧 lock（若有），避免同 session 并发换锁
            lock = entry["lock"] if entry is not None else threading.Lock()
            entry = {"client": client, "last_used": _TICK, "lock": lock}
            _SESSIONS[sid] = entry
        return entry


def _handle(payload, echo=False):
    op = payload.get("op", "mllm")

    if echo:
        if op == "reset":
            return {"ok": True}
        prompt = payload.get("prompt", "")
        n_img = len(payload.get("image_list") or [])
        content = f"[ECHO] backend={payload.get('backend')} imgs={n_img} prompt={prompt[:200]}"
        return {
            "ok": True,
            "reasoning_text": "",
            "function_calls": None,
            "assistant_msg": {"role": "assistant", "content": content},
        }

    entry = _get_session(payload)
    op = payload.get("op", "mllm")

    # 同一 session 串行执行，保证 history 顺序一致。
    with entry["lock"]:
        client = entry["client"]

        if op == "reset":
            client.reset()
            return {"ok": True}

        # op == "mllm"
        prompt = payload.get("prompt", "")
        image_list = payload.get("image_list") or []
        role = payload.get("role", "user")

        # 各客户端 mllm 签名差异：Gemini/Qwen/Offline 无 role 参数，OpenAI 有。
        try:
            reasoning_text, function_calls = client.mllm(prompt, image_list, role=role)
        except TypeError:
            reasoning_text, function_calls = client.mllm(prompt, image_list)

        return {
            "ok": True,
            "reasoning_text": reasoning_text or "",
            "function_calls": function_calls,
            "assistant_msg": llm.serialize_history_tail(client),
        }


def _process_one(query_dir, name, echo):
    """处理单个请求文件（在线程池里跑）。"""
    req_path = os.path.join(query_dir, name)
    req_id = name[:-len(".req.json")]
    resp_path = os.path.join(query_dir, f"{req_id}.resp.json")

    try:
        payload = llm._read_json_safe(req_path)
        if payload is None:
            # 还没完整落盘，从 inflight 摘除，下一轮再看
            return
        try:
            os.remove(req_path)
        except OSError:
            pass

        try:
            result = _handle(payload, echo=echo)
        except Exception as e:
            traceback.print_exc()
            result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                      "reasoning_text": "", "function_calls": None}

        result.setdefault("req_id", req_id)
        try:
            llm._atomic_write_json(resp_path, result)
        except Exception:
            traceback.print_exc()

        print(f"[broker] 处理 {req_id} op={payload.get('op')} "
              f"backend={payload.get('backend')} ok={result.get('ok')}", flush=True)
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT.discard(name)


def _cleanup_sessions(ttl_ticks):
    with _STATE_LOCK:
        stale = [sid for sid, e in _SESSIONS.items() if _TICK - e["last_used"] > ttl_ticks]
        for sid in stale:
            del _SESSIONS[sid]
        alive = len(_SESSIONS)
    if stale:
        print(f"[broker] 清理空闲 session {len(stale)} 个，当前存活 {alive}", flush=True)


def _gc_orphan_files(query_dir, ttl_secs):
    """删除 query 目录里的孤儿文件（陈旧的 .req/.resp/.tmp）。

    正常协议是自清理的：broker 处理完删 .req.json，训练侧 FileRPCChat 读完删
    .resp.json。但当某一端进程崩溃/被 kill/RPC 超时，会留下没人认领的孤儿
    文件（尤其旧训练进程的 .resp.json），长期大规模训练会无限累积、拖慢目录
    IO。这里按 mtime 兜底清理：只删“足够老”的文件（ttl_secs 远大于单次 RPC
    超时 VIBEWORLD_RPC_TIMEOUT，默认取其 2 倍以上），避免误删在途请求。

    正在处理中的 .req.json（在 _INFLIGHT 里）一律跳过，绝不误删活跃请求。
    """
    now = time.time()
    removed = 0
    try:
        entries = os.listdir(query_dir)
    except FileNotFoundError:
        return
    with _INFLIGHT_LOCK:
        inflight = set(_INFLIGHT)
    for name in entries:
        if not (name.endswith(".req.json") or name.endswith(".resp.json")
                or ".tmp" in name):
            continue
        if name in inflight:
            continue
        path = os.path.join(query_dir, name)
        try:
            if now - os.path.getmtime(path) < ttl_secs:
                continue
            os.remove(path)
            removed += 1
        except OSError:
            # 文件可能已被对端删除或正在写，忽略
            pass
    if removed:
        print(f"[broker] GC 清理孤儿文件 {removed} 个（mtime>{ttl_secs}s）", flush=True)


def main():
    global _TICK
    parser = argparse.ArgumentParser()
    parser.add_argument("--echo", action="store_true", help="echo 模式，不接真实 LLM")
    parser.add_argument("--query-dir", default=None, help="覆盖 query 目录")
    parser.add_argument("--poll-interval", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=8, help="线程池大小")
    parser.add_argument("--session-ttl", type=int, default=200000,
                        help="session 空闲多少个 tick 后清理")
    parser.add_argument("--gc-ttl", type=float, default=None,
                        help="孤儿文件 mtime 超过此秒数则清理；默认取 RPC 超时的 3 倍（下限 1800s）")
    parser.add_argument("--gc-interval", type=float, default=300.0,
                        help="孤儿文件 GC 最小间隔（秒），避免每 tick 都扫目录")
    args = parser.parse_args()

    if args.query_dir:
        os.environ["VIBEWORLD_QUERY_DIR"] = args.query_dir
    query_dir = llm.get_query_dir()
    os.makedirs(query_dir, exist_ok=True)

    # 孤儿文件 GC 的 TTL：默认取 RPC 超时的 3 倍，且不低于 1800s，确保远大于
    # 任何在途请求的存活窗口，绝不误删活跃 RPC。
    if args.gc_ttl is not None:
        gc_ttl = args.gc_ttl
    else:
        try:
            rpc_timeout = float(os.environ.get("VIBEWORLD_RPC_TIMEOUT", "600"))
        except ValueError:
            rpc_timeout = 600.0
        gc_ttl = max(1800.0, rpc_timeout * 3)
    _last_gc = 0.0

    print(f"[broker] 启动 mode={'echo' if args.echo else 'real'} "
          f"workers={args.workers} query_dir={query_dir}", flush=True)
    print(f"[broker] 轮询间隔={args.poll_interval}s，等待请求…", flush=True)
    print(f"[broker] 孤儿文件 GC: ttl={gc_ttl}s interval={args.gc_interval}s", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)

    while True:
        _TICK += 1
        try:
            names = sorted(n for n in os.listdir(query_dir) if n.endswith(".req.json"))
        except FileNotFoundError:
            os.makedirs(query_dir, exist_ok=True)
            names = []

        dispatched = 0
        for name in names:
            with _INFLIGHT_LOCK:
                if name in _INFLIGHT:
                    continue
                _INFLIGHT.add(name)
            pool.submit(_process_one, query_dir, name, args.echo)
            dispatched += 1

        if dispatched == 0:
            _cleanup_sessions(args.session_ttl)

        now = time.time()
        if now - _last_gc >= args.gc_interval:
            _gc_orphan_files(query_dir, gc_ttl)
            _last_gc = now

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
