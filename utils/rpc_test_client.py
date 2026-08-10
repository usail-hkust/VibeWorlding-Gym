"""
rpc_test_client.py —— FileRPC 链路测试客户端（GPU 侧运行）。

用法：
    # echo 疏通（配合 broker.py --echo）
    python rpc_test_client.py --echo

    # 真实模型单次往返（配合 broker.py 真实模式）
    python rpc_test_client.py --backend gemini

    # 带图片
    python rpc_test_client.py --backend gemini --image /path/to/a.jpg
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import FileRPCChat  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gemini")
    parser.add_argument("--model", default="", help="model_name，留空用 backend 默认")
    parser.add_argument("--echo", action="store_true",
                        help="仅测链路（broker 需 --echo 启动），prompt 会被原样回显")
    parser.add_argument("--prompt", default="用一句话说明天空为什么是蓝色的。")
    parser.add_argument("--image", action="append", default=[], help="可重复")
    parser.add_argument("--query-dir", default=None)
    args = parser.parse_args()

    if args.query_dir:
        os.environ["VIBEWORLD_QUERY_DIR"] = args.query_dir

    bot = FileRPCChat(model_name=args.model, system_instruction="你是一个简洁的助手。",
                      backend=args.backend)
    print(f"[test] session_id={bot.session_id} query_dir={bot.query_dir}")

    print("[test] 发送 reset …")
    bot.reset()

    print(f"[test] 发送 mllm prompt={args.prompt!r} images={args.image}")
    reasoning, fcs = bot.mllm(args.prompt, args.image)

    print("=" * 60)
    print(f"reasoning ({len(reasoning)}c): {reasoning[:500]}")
    print(f"function_calls: {fcs}")
    content = bot.history[-1].get("content") if bot.history else None
    print(f"content: {content!r}")
    print("=" * 60)

    if content:
        print("[test] ✅ 往返成功")
    else:
        print("[test] ⚠️ 未取到 content，检查 broker 日志")


if __name__ == "__main__":
    main()
