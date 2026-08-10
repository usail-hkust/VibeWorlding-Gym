"""
启动 vLLM 推理服务（OpenAI 兼容 API）。

直接调用 `vllm serve` 命令，暴露 /v1/chat/completions 接口，
可被 llm.py 中的 OfflineLLM 直接调用。

用法：
    python start_verl_server.py \
        --model_path /path/to/model \
        --tp_size 4 \
        --port 8000

然后在另一个终端中：
    export VERL_SERVER_URL=http://localhost:8000
    export VERL_MODEL_PATH=/path/to/model
    python main_2.py ...
"""

import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="启动 vLLM 推理服务")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor Parallel 大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU 显存利用率")
    parser.add_argument("--max_model_len", type=int, default=32768, help="最大模型长度")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="最大并发序列数")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="模型精度")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--limit_mm_per_prompt_image", type=int, default=None,
                        help="每个 prompt 最大图片数（VLM 模型使用）")
    parser.add_argument("--enable_auto_tool_choice", action="store_true", default=False,
                        help="启用自动工具选择（v2 原生 tool calling 需要）")
    parser.add_argument("--tool_call_parser", type=str, default=None,
                        help="工具调用解析器，如 hermes, qwen25 等")
    parser.add_argument("--reasoning_parser", type=str, default=None,
                        help="推理内容解析器，Qwen3-VL-Thinking 用 deepseek_r1")
    args = parser.parse_args()

    # 构建 vllm serve 命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--tensor-parallel-size", str(args.tp_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--dtype", args.dtype,
        "--port", str(args.port),
        "--host", args.host,
    ]

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if args.limit_mm_per_prompt_image is not None:
        # 新版 vLLM 要求 JSON 格式；旧版的 "image=N" key=value 已废弃，会报
        # argument --limit-mm-per-prompt: Value image=N cannot be converted to <function loads>
        cmd.extend(["--limit-mm-per-prompt", json.dumps({"image": args.limit_mm_per_prompt_image})])

    if args.enable_auto_tool_choice:
        cmd.append("--enable-auto-tool-choice")

    if args.tool_call_parser:
        cmd.extend(["--tool-call-parser", args.tool_call_parser])

    if args.reasoning_parser:
        cmd.extend(["--reasoning-parser", args.reasoning_parser])

    print("=" * 60)
    print("[VerlServer] 启动 vLLM 推理服务")
    print(f"[VerlServer] 模型: {args.model_path}")
    print(f"[VerlServer] TP: {args.tp_size}, 端口: {args.port}")
    print(f"[VerlServer] 命令: {' '.join(cmd)}")
    print()
    print(f"[VerlServer] 服务就绪后，在另一个终端中运行:")
    print(f"    export VERL_SERVER_URL=http://localhost:{args.port}")
    print(f"    export VERL_MODEL_PATH={args.model_path}")
    print(f"    python main_2.py ...")
    print("=" * 60)

    # 直接执行，stdout/stderr 实时输出
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
