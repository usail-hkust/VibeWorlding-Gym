"""
start_verl_server_CLI.py — 为 vibeworld CLI_Demo 启动本地 vLLM 推理服务

模型：Qwen3-VL-MoE 30B-A3B（SFT 微调版 VibeWorlder）
默认端口：8080（CLI_Demo 里配置的 LOCAL_VLLM_SERVER 指向这里）

用法（在 GPU 服务器上）：
    python start_verl_server_CLI.py                   # 全用默认值
    python start_verl_server_CLI.py --tp_size 8       # 8 卡并行
    python start_verl_server_CLI.py --port 9090       # 换端口

服务就绪后，在 CLI_Demo/setup.py 里把 LOCAL_VLLM_SERVER 设为：
    http://<server-ip>:8080/v1   （对应默认端口 8080）

关键 vLLM 参数说明：
  --enable-auto-tool-choice + --tool-call-parser qwen25
      Qwen3 系列的 native function calling，vLLM 自动解析 JSON tool_call。
  --reasoning-parser deepseek_r1
      把模型输出的 <think>…</think> 段解析进 reasoning_content 字段，
      供 streaming_bailian.py 的 _consume() 读取并实时打印。
  --limit-mm-per-prompt '{"image": 5}'
      每轮最多 5 张图（vibeworld 5 视角渲染图），超出则截断而非报错。
  --served-model-name vibeworlder-30B-A3B
      API 调用时用这个短名字（LocalStreamingVLLMMultiChat 里也是这个名字）。

GPU 需求（估算）：
  30B MoE 参数 bf16 ≈ 60GB；4×80GB GPU 跑 --gpu-memory-utilization 0.85 足够。
  如果 VRAM 紧张可以加 --enforce_eager 关掉 CUDAGraph。
"""

import argparse
import json
import os
import subprocess
import sys

# 我们训练的 VibeWorlder 模型目录。下载：
#   https://huggingface.co/collections/usail-hkust/vibeworlder
# 可用 --model_path 或 VIBEWORLD_MODEL_PATH 覆盖。
MODEL_PATH = os.environ.get("VIBEWORLD_MODEL_PATH", "./models/VibeWorlder-30B-A3B")

SERVED_MODEL_NAME = "vibeworlder-30B-A3B"


def main():
    parser = argparse.ArgumentParser(
        description="启动 VibeWorlder 30B-A3B 本地 vLLM 推理服务（CLI_Demo 专用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="模型 checkpoint 路径")
    parser.add_argument("--served_model_name", type=str, default=SERVED_MODEL_NAME,
                        help="API 调用时使用的模型名（客户端 model= 参数）")
    parser.add_argument("--tp_size", type=int, default=4,
                        help="Tensor Parallel 大小（4 卡 80GB 可跑，8 卡更宽松）")
    parser.add_argument("--port", type=int, default=8080,
                        help="监听端口（CLI_Demo 里 LOCAL_VLLM_SERVER 要对应）")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="最大 context 长度（vibeworld 多轮 + 5 张图约 15k，32k 够用）")
    parser.add_argument("--max_num_seqs", type=int, default=32,
                        help="最大并发序列数（CLI 单用户改小可减显存占用）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU 显存利用率")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--limit_mm_per_prompt_image", type=int, default=5,
                        help="每 prompt 最多图片数（5 视角渲染图）")
    parser.add_argument("--enforce_eager", action="store_true", default=False,
                        help="关闭 CUDAGraph（VRAM 紧张时加这个）")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True,
                        help="开启 prefix caching（system prompt 共享，省显存）")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                      args.model_path,
        "--served-model-name",          args.served_model_name,
        "--tensor-parallel-size",       str(args.tp_size),
        "--gpu-memory-utilization",     str(args.gpu_memory_utilization),
        "--max-model-len",              str(args.max_model_len),
        "--max-num-seqs",               str(args.max_num_seqs),
        "--dtype",                      args.dtype,
        "--port",                       str(args.port),
        "--host",                       args.host,
        "--trust-remote-code",
        # ── 工具调用（Qwen3 native function calling）────────────────────────
        "--enable-auto-tool-choice",
        "--tool-call-parser",           "qwen3_xml",
        # ── 推理内容（<think>…</think> → reasoning_content 字段）───────────
        "--reasoning-parser",           "deepseek_r1",
        # ── 视觉：每 prompt 最多 5 张图 ────────────────────────────────────
        "--limit-mm-per-prompt",        json.dumps({"image": args.limit_mm_per_prompt_image}),
    ]

    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    base_url = f"http://localhost:{args.port}/v1"

    print("=" * 70)
    print("[VibeWorlder Server] 启动本地 vLLM 推理服务")
    print(f"  模型    : {args.model_path}")
    print(f"  API 名  : {args.served_model_name}")
    print(f"  TP      : {args.tp_size}  |  端口: {args.port}  |  ctx: {args.max_model_len}")
    print(f"  base_url: {base_url}")
    print()
    print("  启动命令:")
    print("    " + " \\\n        ".join(cmd))
    print()
    print("  服务就绪后，在 CLI_Demo/setup.py 里设置:")
    print(f'    LOCAL_VLLM_SERVER = "{base_url}"')
    print()
    print("  然后直接运行 vibeworld，默认即走本地模型（无需 /model 切换）。")
    print("=" * 70)
    print()

    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
