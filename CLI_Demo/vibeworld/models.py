"""
models.py — 模型注册表

把 `/model` 命令可切换的模型集中在这里。所有 client 都是**流式**的
（见 streaming_bailian.py），统一实现 `mllm(prompt, image_list)` 接口。

四类 provider：
    本地 vLLM          我们训练的 VibeWorlder 模型（先用 start_verl_server_CLI.py 起服务）
    Gemini 官方        gemini-3.5-flash / gemini-3.1-pro
    OpenAI 官方        gpt-5.5
    阿里云百炼         qwen3.8-max / K3

API key 全部读环境变量：GEMINI_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY。
本地 vLLM 地址读 VIBEWORLD_LOCAL_VLLM_URL（默认 http://localhost:8000/v1）。

切换模型只换 client，不重置场景状态（由 session.py 负责续接上下文）。

默认入口模型是 qwen（Bailian 的 Qwen3.8-Max），开箱即用、无需自建服务。
"""

import importlib.util as _ilu
import os

# ── 显式加载仓库根目录的 utils/llm.py（复用其 helper，避免被同名模块覆盖）──────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "../../utils"))

_llm_spec = _ilu.spec_from_file_location("vibe_llm", os.path.join(_UTILS_DIR, "llm.py"))
_llm_mod = _ilu.module_from_spec(_llm_spec)
_llm_spec.loader.exec_module(_llm_mod)

llm_mod = _llm_mod          # 供 streaming_bailian.py 复用 helper（不重复加载 llm.py）

from .streaming_bailian import StreamingBailianMultiChat      # noqa: E402
from .streaming_bailian import StreamingGeminiMultiChat       # noqa: E402
from .streaming_bailian import StreamingOpenAIMultiChat       # noqa: E402
from .streaming_bailian import LocalStreamingVLLMMultiChat    # noqa: E402


# ── 注册表：key -> (client_class, 真实 model_name, 展示名, 一句话描述) ──────────────
# 真实 model_name 是原样传给各家 API 的 `model` 字段，绝不能加 "<provider>/" 前缀
# （加了会404，因为 provider 端根本没有叫这个的模型）。展示名才是 "<provider>/<model>"
# 这种带前缀的形式，只用于 welcome banner / HTML viewer，不会被发进请求体。
MODEL_REGISTRY = {
    # ── 我们训练的模型：本地 vLLM ──────────────────────────────────────────────
    "vibeworlder": (LocalStreamingVLLMMultiChat, "vibeworlder-30B-A3B",
                    "vibeworlder/vibeworlder-30B-A3B",
                    "VibeWorlder 30B-A3B · local vLLM (ours)"),
    # ── Gemini 官方（https://ai.google.dev/gemini-api/docs/models）─────────────
    "gemini-flash": (StreamingGeminiMultiChat, "gemini-3.5-flash",
                     "gemini/gemini-3.5-flash",
                     "Gemini 3.5 Flash · Google official API (faster)"),
    "gemini-pro": (StreamingGeminiMultiChat, "gemini-3.1-pro",
                   "gemini/gemini-3.1-pro",
                   "Gemini 3.1 Pro · Google official API"),
    # ── OpenAI 官方（https://developers.openai.com/api/docs/models）─────────────
    "gpt5": (StreamingOpenAIMultiChat, "gpt-5.5",
             "openai/gpt-5.5",
             "GPT-5.5 · OpenAI official API"),
    # ── 阿里云百炼 / DashScope（https://bailian.console.aliyun.com）──────────────
    "qwen": (StreamingBailianMultiChat, "qwen3.8-max",
             "bailian/qwen3.8-max",
             "Qwen3.8-Max · Bailian / DashScope"),
    "k3": (StreamingBailianMultiChat, "K3",
           "bailian/K3",
           "K3 · Bailian / DashScope"),
}

DEFAULT_MODEL_KEY = "qwen"


def display_name(model_name: str) -> str:
    """UI 里该显示的模型名：真实 model_name -> 带provider 前缀的展示名。

    找不到（比如已经是展示名，或未注册的自定义 model_name）时原样返回。
    """
    for _key, (_cls, real_name, disp_name, _desc) in MODEL_REGISTRY.items():
        if model_name == real_name:
            return disp_name
    return model_name


def list_models() -> list:
    """返回 [(key, 展示名, description), ...]，供 `/model` 无参时列出。"""
    return [(k, v[2], v[3]) for k, v in MODEL_REGISTRY.items()]


def resolve_model(name: str):
    """把 `/model <name>` 的参数解析成 (key, client_class, 真实 model_name)。

    支持三种写法：
      1. 注册表 key，如 "vibeworlder" / "gemini-flash" / "qwen" / "gpt5"
      2. 真实 model_name，如 "qwen3.8-max"
      3. 展示名，如 "bailian/qwen3.8-max"

    找不到时抛 KeyError，由调用方提示可用列表。
    """
    name = (name or "").strip()
    if not name:
        raise KeyError("模型名为空")

    # 1. 注册表 key
    if name in MODEL_REGISTRY:
        cls, real_name, _disp, _desc = MODEL_REGISTRY[name]
        return name, cls, real_name

    # 2/3. 真实 model_name 或展示名命中注册表
    for key, (cls, real_name, disp_name, _desc) in MODEL_REGISTRY.items():
        if name in (real_name, disp_name):
            return key, cls, real_name

    raise KeyError(f"未知模型 {name!r}。可用：{', '.join(MODEL_REGISTRY)}")


def build_bot(name: str, system_instruction: str, tools=None, on_delta=None):
    """根据模型名构造 client 实例。返回 (key, 真实 model_name, bot)。

    所有 client 都是流式的（streaming = True），故 on_delta 直接透传。
    """
    key, cls, model_name = resolve_model(name)
    kwargs = {}
    if on_delta is not None:
        kwargs["on_delta"] = on_delta
    bot = cls(model_name=model_name, system_instruction=system_instruction,
              tools=tools, **kwargs)
    return key, model_name, bot
