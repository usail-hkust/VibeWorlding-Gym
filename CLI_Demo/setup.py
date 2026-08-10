"""
setup.py — vibeworld 安装入口 + 服务地址集中配置

服务地址在这里集中配置（不再启动时交互询问）。要改渲染 / 检索服务，
直接改下面两个常量，或用环境变量覆盖；cli.py 启动时会读取它们。

优先级（cli.py 内）：命令行 --server/--retrieve-server > 环境变量
VIBEWORLD_RENDER_SERVER / VIBEWORLD_RETRIEVE_SERVER > 这里的常量。

API key 一律不写在代码里，请用环境变量：
    export GEMINI_API_KEY=your_gemini_api_key
    export OPENAI_API_KEY=your_openai_api_key
    export DASHSCOPE_API_KEY=your_dashscope_api_key
"""

import os

# ── 服务地址（集中配置，改这里即可）────────────────────────────────────────────
# 部署见 assets_retrieval/README.md 与 render_in_blender/README.md。
RENDER_SERVER = os.environ.get("VIBEWORLD_RENDER_SERVER", "http://localhost:8080")
RETRIEVE_SERVER = os.environ.get("VIBEWORLD_RETRIEVE_SERVER", "http://localhost:8081")

# 阿里云百炼 / DashScope（/model qwen 用）
# cli.py 启动时把它注入 BAILIAN_API_KEY（已有同名环境变量时不覆盖）。
BAILIAN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "your_dashscope_api_key")

# 本地 vLLM 推理服务（我们训练的 VibeWorlder 模型）
# 先用 start_verl_server_CLI.py 把模型跑起来，再把地址填在这里（或设
# VIBEWORLD_LOCAL_VLLM_URL）。模型下载：
#   https://huggingface.co/collections/usail-hkust/vibeworlder
LOCAL_VLLM_SERVER = os.environ.get("VIBEWORLD_LOCAL_VLLM_URL", "http://localhost:8000/v1")


if __name__ == "__main__":
    from setuptools import setup, find_packages

    setup(
        name="vibeworld",
        version="1.0.0",
        packages=find_packages(),
        install_requires=[
            "rich",
            "prompt_toolkit",
            "openai",
            "google-genai",
            "gradio-client",
            "httpx",
            "trimesh",
            "numpy",
            "pillow",
        ],
        entry_points={
            "console_scripts": [
                "vibeworld=vibeworld.cli:main",
            ],
        },
    )
