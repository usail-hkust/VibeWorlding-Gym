"""
llm.py — VibeWorld-Gym 统一 LLM 客户端

所有多模态 LLM 客户端都实现相同接口：
    __init__(model_name, system_instruction, tools=None)
    reset()
    mllm(prompt, image_list) -> (reasoning_text: str, function_calls: list[dict] | None)

可用的客户端类：
    GeminiMultiChat          — Google Gemini 官方 API（thinking 模式）
    OpenAIMultiChat          — OpenAI 官方 API
    QwenMultiChat            — 阿里云百炼 / DashScope（Qwen-VL）
    OfflineLLM               — 本地 vLLM（OpenAI 兼容），用于我们自己训练的模型

API key 全部从环境变量读取，不在代码中硬编码：
    GEMINI_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY
本地 vLLM 服务地址读VERL_SERVER_URL（默认 http://localhost:8000）。

MODEL_TYPE_MAP — 按字符串 key 索引客户端类，供 main.py / verifier 使用：
    {'gemini': GeminiMultiChat, 'openai': OpenAIMultiChat,
     'qwen3': QwenMultiChat, 'bailian': BailianMultiChat,
     'offline-llm': OfflineLLM}

两种 transport（实现见文件末尾的 FileRPC 段）：
    direct （默认）  本进程直接调 provider。
    filerpc设 VIBEWORLD_LLM_TRANSPORT=filerpc 后，MODEL_TYPE_MAP 被就地
                     重绑为 FileRPCChat 工厂，请求经共享磁盘转发给另一台机器上的
                     broker.py 执行 —— 既解决训练节点无外网，也让大批 verify 任务
                     能并发打分。REAL_MODEL_TYPE_MAP 始终指向真实类，供 broker 侧
                     实例化。
"""

import os
import json
import base64
import re
import time
import threading
from traceback import print_exc

import requests
from google import genai
from google.genai import types
from openai import OpenAI


# ============================================================
# 工具函数
# ============================================================

def _strip_markdown_json(raw: str) -> str:
    s = raw.strip()
    m = re.match(r'^```(?:json)?\s*\n?(.*?)\n?\s*```$', s, re.DOTALL)
    return m.group(1).strip() if m else s


def _fix_malformed_json(s: str) -> str:
    s = re.sub(r'\}\}(\s*[,\]])', r'}\1', s)
    s = re.sub(r'\}\}(\s*\])', r'}\1', s)
    s = re.sub(r',\s*\}', '}', s)
    s = re.sub(r',\s*\]', ']', s)
    return s


def parse_tool_calls_from_text(text: str) -> list:
    """从文本中提取 <tool_call>...</tool_call>，解析为 {"name":..., "arguments":{...}} 列表。"""
    function_calls = []
    for raw_block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        cleaned = _strip_markdown_json(raw_block)
        if not cleaned:
            continue
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            fixed = _fix_malformed_json(cleaned)
            try:
                parsed = json.loads(fixed)
            except json.JSONDecodeError as e:
                print(f"解析 Tool Call JSON 失败: {e} -> {cleaned[:200]}")
                continue
        if isinstance(parsed, list):
            function_calls.extend(parsed)
        else:
            function_calls.append(parsed)
    return function_calls


def parse_vllm_tool_calls(vllm_tool_calls: list) -> list:
    """将 vLLM/OpenAI 格式的 tool_calls 转换为统一的 {"name":..., "arguments":{...}} 列表。"""
    function_calls = []
    for tc in (vllm_tool_calls or []):
        try:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                name = func.get("name", "")
                raw_args = func.get("arguments", "{}")
            else:
                name = tc.function.name
                raw_args = tc.function.arguments
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            function_calls.append({"name": name, "arguments": args})
        except Exception as e:
            print(f"[解析 tool_call 失败] {e}: {tc}")
    return function_calls


def extract_reasoning_from_text(text: str) -> str:
    """从文本中提取 <think>...</think> 内容。"""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def build_openai_tool_calls(function_calls: list) -> list:
    """将 {"name":..., "arguments":{...}} 转换为 OpenAI 格式 tool_calls。"""
    tool_calls = []
    for i, fc in enumerate(function_calls):
        name = fc.get("name", "")
        arguments = fc.get("arguments", {})
        args_str = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args_str}
        })
    return tool_calls


def parse_response_message(response_message: dict) -> tuple:
    """
    统一解析 vLLM/OpenAI response message，兼容以下情况：
      A  原生 Qwen3 + hermes parser 成功：content=None, reasoning="思考", tool_calls=[...]
      B  tool_call 在 content 文本里：<tool_call>{...}</tool_call>
      B2 hermes parser 把 tool_call 放进 reasoning 字段
      C  最终回复，无工具调用

    返回: (reasoning_text, function_calls, content_text)
    """
    reasoning_raw = (
        response_message.get("reasoning_content") or
        response_message.get("reasoning") or ""
    )
    raw_content = response_message.get("content") or ""
    vllm_tool_calls = response_message.get("tool_calls") or []

    if vllm_tool_calls:
        function_calls = parse_vllm_tool_calls(vllm_tool_calls)
        if reasoning_raw.strip():
            reasoning_text = reasoning_raw.strip()
            content_text = raw_content or None
        else:
            reasoning_text = (raw_content or "").strip()
            content_text = None
    elif raw_content and ("<tool_call>" in raw_content or "</tool_call>" in raw_content):
        function_calls = parse_tool_calls_from_text(raw_content)
        reasoning_text = reasoning_raw.strip() or extract_reasoning_from_text(raw_content)
        content_text = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
        content_text = re.sub(r'<tool_call>.*?</tool_call>', '', content_text, flags=re.DOTALL).strip() or None
    elif reasoning_raw and ("<tool_call>" in reasoning_raw or "</tool_call>" in reasoning_raw):
        function_calls = parse_tool_calls_from_text(reasoning_raw)
        reasoning_text = re.sub(r'<tool_call>.*?</tool_call>', '', reasoning_raw, flags=re.DOTALL).strip()
        content_text = raw_content or None
        if function_calls:
            print(f"[parse_response_message] B2: 从 reasoning 字段提取到 {len(function_calls)} 个 tool_calls")
    else:
        function_calls = []
        reasoning_text = reasoning_raw.strip() or extract_reasoning_from_text(raw_content)
        content_text = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip() or raw_content

    return reasoning_text, function_calls, content_text


def read_image_from_path(image_path: str) -> bytes:
    if not os.path.exists(image_path):
        print(f"图片不存在：{image_path}")
        return b""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"读取图片出错：{e}")
        return b""


def detect_image_mime(path: str) -> str:
    """按 magic bytes 判断真实图片 MIME。渲染出的 *.jpg 实际常是 PNG 字节，
    Claude(anthropic) 会严格校验 media type，声明错误会 400。"""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except Exception:
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def make_image_url_part(path: str) -> dict:
    """构造 OpenAI 格式 image_url part，MIME 按 magic bytes 自动判定。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url",
            "image_url": {"url": f"data:{detect_image_mime(path)};base64,{b64}"}}


# ============================================================
# Gemini
# ============================================================

class GeminiMultiChat:
    def __init__(self, model_name: str = "gemini-3-pro-preview", system_instruction="", tools=None):
        self.client = genai.Client(http_options={'api_version': 'v1alpha'})
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.history = []
        self.mcp_tools = None
        self._init_tools()

    def _init_tools(self):
        self.mcp_tools = None
        if self.tools:
            function_declarations = [
                t["function"] for t in self.tools if t.get("type") == "function"
            ]
            if function_declarations:
                self.mcp_tools = types.Tool(function_declarations=function_declarations)

    def reset(self):
        self.history = []
        self._init_tools()

    def mllm(self, prompt, image_list):
        img_data = []
        for image in image_list:
            image_bytes = read_image_from_path(image)
            if image_bytes:
                img_data.append(types.Part(
                    inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes),
                    media_resolution={"level": "media_resolution_low"}
                ))
        self.history.append(types.Content(
            role="user",
            parts=[types.Part(text=prompt)] + img_data
        ))

        config_kwargs = {"top_p": 1, "temperature": 0.2}
        if len(self.history) == 1 and self.system_instruction:
            config_kwargs["system_instruction"] = self.system_instruction
        if self.mcp_tools:
            config_kwargs["tools"] = [self.mcp_tools]
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=8192, include_thoughts=True,
        )

        response = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=self.history,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                break
            except Exception as e:
                print(f"Gemini 请求失败 ({attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(10)

        if response is None:
            print_exc()
            return "", None

        self.history.append(response.candidates[0].content)

        function_calls = []
        text_parts = []
        thought_parts = []
        has_function_call = False

        for part in (response.candidates[0].content.parts or []):
            if getattr(part, "thought", False) and part.text:
                thought_parts.append(part.text)
            elif part.text:
                text_parts.append(part.text)
            if part.function_call:
                has_function_call = True
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args) if part.function_call.args else {}
                })

        content_text = "".join(text_parts)
        function_calls.extend(parse_tool_calls_from_text(content_text))

        if thought_parts:
            reasoning_text = "\n".join(thought_parts).strip()
        elif has_function_call and text_parts:
            reasoning_text = content_text.strip()
        else:
            reasoning_text = extract_reasoning_from_text(content_text)

        thoughts_tokens = getattr(getattr(response, 'usage_metadata', None), 'thoughts_token_count', 0) or 0
        print(f"=== Gemini Response (thoughts_tokens={thoughts_tokens}) ===")
        print(f"  Reasoning ({len(reasoning_text)}c): {reasoning_text[:300]}")
        print(f"  Tool Calls ({len(function_calls)}): {[fc['name'] for fc in function_calls]}")
        if not has_function_call:
            print(f"  Content ({len(content_text)}c): {content_text[:200]}")
        print(f"===")

        return reasoning_text, function_calls if function_calls else None


# ============================================================
# Qwen (DashScope)
# ============================================================

class QwenMultiChat:
    def __init__(self, model_name: str = "qwen3-vl-235b-a22b-thinking", system_instruction="", tools=None):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.history = []
        if self.system_instruction:
            self.history.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_instruction}]
            })

    def reset(self):
        self.history = []
        if self.system_instruction:
            self.history.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_instruction}]
            })

    def mllm(self, prompt, image_list):
        user_content = [{"type": "text", "text": prompt}]
        for image_path in image_list:
            if not os.path.exists(image_path):
                continue
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        self.history.append({"role": "user", "content": user_content})

        response = None
        for attempt in range(3):
            try:
                kwargs = {"model": self.model_name, "messages": self.history, "temperature": 0.2, "top_p": 1}
                if self.tools:
                    kwargs["tools"] = self.tools
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                print(f"Qwen 请求失败 ({attempt + 1}/3): {e}")
                time.sleep(5 if attempt < 2 else 15)

        if response is None:
            print_exc()
            return "", None

        msg = response.choices[0].message
        response_dict = {
            "content": msg.content,
            "reasoning": getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None),
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ]
        }

        print(f"[DEBUG QwenMultiChat] model={self.model_name}")
        print(f"[DEBUG QwenMultiChat] content preview={str(msg.content)[:200] if msg.content else 'NONE'}")
        print(f"[DEBUG QwenMultiChat] reasoning_content={str(getattr(msg, 'reasoning_content', 'NO_ATTR'))[:200]}")
        print(f"[DEBUG QwenMultiChat] tool_calls={msg.tool_calls}")

        reasoning_text, function_calls, content_text = parse_response_message(response_dict)

        assistant_msg = {"role": "assistant", "content": content_text}
        if reasoning_text:
            assistant_msg["reasoning_content"] = reasoning_text
        if function_calls:
            assistant_msg["tool_calls"] = build_openai_tool_calls(function_calls)
        self.history.append(assistant_msg)

        return reasoning_text, function_calls if function_calls else None


# ============================================================
# 阿里云百炼 Bailian（DashScope Model Studio，OpenAI 兼容 /compatible-mode/v1）
# ============================================================

class BailianMultiChat:
    """阿里云百炼 (Bailian / DashScope Model Studio) 客户端。

    百炼对外暴露的正是 DashScope 的 OpenAI 兼容端点
    (https://dashscope.aliyuncs.com/compatible-mode/v1)，一个 key 即可访问平台上
    的多家模型（qwen 系列、moonshot/kimi 系列 等），模型以 model id 区分：
        qwen3.7-max            通义千问 3.7-max
        kimi/kimi-k3           Moonshot Kimi-K3（需在百炼控制台「开通」该产品）

    与其它客户端同接口：__init__/reset/mllm。回包为标准 OpenAI 格式
    (content / reasoning_content / tool_calls)，复用 parse_response_message。

    视觉：默认尝试以 base64 image_url 传图；若模型不支持视觉而报错，自动降级为
    纯文本并记住状态（self.vision_ok=False），后续不再传图 —— 适配 kimi-k3 等
    可能的纯文本模型。

    环境变量：
      BAILIAN_API_KEY   百炼 API key（默认用 DASHSCOPE_API_KEY，两者同源）
      BAILIAN_BASE_URL  默认 DashScope compatible-mode 端点
      BAILIAN_MAX_TOKENS 默认 8192
      BAILIAN_SEND_IMAGES "0"/"false" 强制纯文本
    """

    def __init__(self, model_name: str = "qwen3.7-max", system_instruction="", tools=None):
        self.client = OpenAI(
            api_key=(os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")),
            base_url=os.getenv("BAILIAN_BASE_URL",
                               "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.max_tokens = int(os.environ.get("BAILIAN_MAX_TOKENS", "8192"))
        self.vision_ok = os.environ.get("BAILIAN_SEND_IMAGES", "true").lower() in ("1", "true", "yes")
        self.history = []
        if self.system_instruction:
            self.history.append({"role": "system", "content": self.system_instruction})

    def reset(self):
        self.history = []
        if self.system_instruction:
            self.history.append({"role": "system", "content": self.system_instruction})

    def _build_user_content(self, prompt, image_list, with_images):
        if not with_images or not image_list:
            return prompt
        user_content = [{"type": "text", "text": prompt}]
        for image_path in image_list:
            if os.path.exists(image_path):
                user_content.append(make_image_url_part(image_path))
        return user_content

    def mllm(self, prompt, image_list, role="user"):
        # 与 OpenAI 客户端一致的 role=tool 处理：上一条 assistant 若带 tool_calls，
        # 本轮以 role=tool 回应每个 tool_call_id，图片放到随后的 user 消息。
        prev = self.history[-1] if self.history else {}
        prev_tool_calls = (prev.get("tool_calls") or []) if isinstance(prev, dict) and prev.get("role") == "assistant" else []
        image_parts = ([make_image_url_part(p) for p in (image_list or []) if os.path.exists(p)]
                       if self.vision_ok else [])

        if prev_tool_calls:
            for i, tc in enumerate(prev_tool_calls):
                tc_id = (tc.get("id", f"call_{i}") if isinstance(tc, dict)
                         else getattr(tc, "id", f"call_{i}"))
                self.history.append({"role": "tool", "tool_call_id": tc_id,
                                     "content": prompt if i == 0 else "(done)"})
            if image_parts:
                self.history.append({"role": "user",
                                     "content": [{"type": "text", "text": "（本轮场景渲染图）"}] + image_parts})
        else:
            self.history.append({"role": "user",
                                 "content": self._build_user_content(prompt, image_list, self.vision_ok)})

        response = None
        for attempt in range(4):
            try:
                kwargs = {"model": self.model_name, "messages": self.history,
                          "temperature": 0.2, "top_p": 1, "max_tokens": self.max_tokens}
                if self.tools:
                    kwargs["tools"] = self.tools
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                emsg = str(e)
                print(f"Bailian 请求失败 ({attempt + 1}/4) model={self.model_name}: {emsg[:180]}")
                # 视觉不支持 → 降级纯文本并重构本轮 user 消息后重试
                vision_err = any(k in emsg.lower() for k in ("image", "vision", "multimodal")) or \
                             any(k in emsg for k in ("图片", "图像", "多模态", "不支持"))
                if self.vision_ok and image_list and vision_err and not prev_tool_calls:
                    print("[Bailian] 检测到视觉不支持 → 降级纯文本(text-only)并重试")
                    self.vision_ok = False
                    self.history[-1]["content"] = self._build_user_content(prompt, image_list, False)
                    continue
                if attempt < 3:
                    time.sleep(5 if attempt == 0 else 15)

        if response is None:
            print_exc()
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            return "", None

        msg = response.choices[0].message
        response_dict = {
            "content": msg.content,
            "reasoning": getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None),
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] if msg.tool_calls else []
        }

        reasoning_text, function_calls, content_text = parse_response_message(response_dict)

        assistant_msg = {"role": "assistant", "content": content_text}
        if reasoning_text:
            assistant_msg["reasoning_content"] = reasoning_text
        if function_calls:
            assistant_msg["tool_calls"] = build_openai_tool_calls(function_calls)
        self.history.append(assistant_msg)

        print(f"=== Bailian ({self.model_name}) vision={self.vision_ok} ===")
        print(f"  Reasoning ({len(reasoning_text)}c): {reasoning_text[:200]}")
        print(f"  Tool Calls ({len(function_calls)}): {[fc['name'] for fc in function_calls]}")

        return reasoning_text, function_calls if function_calls else None


# 别名（用户要求的类名）
bailian_LLM = BailianMultiChat


# ============================================================
# OpenAI 官方 API
# ============================================================

class OpenAIMultiChat:
    """OpenAI 官方 API（api.openai.com），多模态 + tool calling。

    需要设置环境变量 OPENAI_API_KEY；如需走兼容端点可设 OPENAI_BASE_URL。
    """

    def __init__(self, model_name: str = "gpt-4o", system_instruction="", tools=None):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "your_openai_api_key"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.history = []
        if self.system_instruction:
            self.history.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_instruction}]
            })

    def reset(self):
        self.history = []
        if self.system_instruction:
            self.history.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_instruction}]
            })

    def mllm(self, prompt, image_list, role="user"):
        user_content = [{"type": "text", "text": prompt}]
        for image_path in image_list:
            if not os.path.exists(image_path):
                continue
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
                })

        if role == "tool":
            prev = self.history[-1] if self.history else {}
            prev_tool_calls = (prev.get("tool_calls") or []) if prev.get("role") == "assistant" else []
            if prev_tool_calls:
                for i, tc in enumerate(prev_tool_calls):
                    tc_id = (tc.get("id", f"call_{i}") if isinstance(tc, dict)
                             else getattr(tc, "id", f"call_{i}"))
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": prompt if i == 0 else "(done)",
                    })
            else:
                self.history.append({"role": "user", "content": user_content})
        else:
            self.history.append({"role": "user", "content": user_content})

        response = None
        for attempt in range(3):
            try:
                kwargs = {
                    "model": self.model_name, "messages": self.history,
                    "temperature": 0.2, "top_p": 1, "max_tokens": 8000,
                }
                if self.tools:
                    kwargs["tools"] = self.tools
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                print(f"OpenAI 请求失败 ({attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(5 if attempt == 0 else 15)

        if response is None:
            print_exc()
            return "", None

        msg = response.choices[0].message
        response_dict = {
            "content": msg.content,
            "reasoning": getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None),
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] if msg.tool_calls else []
        }

        reasoning_text, function_calls, content_text = parse_response_message(response_dict)

        assistant_msg = {"role": "assistant", "content": content_text}
        if reasoning_text:
            assistant_msg["reasoning_content"] = reasoning_text
        if function_calls:
            assistant_msg["tool_calls"] = build_openai_tool_calls(function_calls)
        self.history.append(assistant_msg)

        print(f"=== OpenAI Response ===")
        print(f"  Content ({len(content_text or '')}c): {(content_text or '')[:200]}")
        print(f"  Tool Calls ({len(function_calls)}): {[fc['name'] for fc in function_calls]}")
        print(f"===")

        return reasoning_text, function_calls if function_calls else None


# ============================================================
# OfflineLLM（verl 框架本地 vLLM）
# ============================================================

class OfflineLLM:
    """连接本地 vLLM 推理服务（verl 框架），支持 Qwen3 原生 tool calling。"""

    def __init__(self, model_name: str = "offline-model", system_instruction="", tools=None):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.history = []

        server_base = os.environ.get("VERL_SERVER_URL", "http://localhost:8000").rstrip("/")
        self.api_url = f"{server_base}/v1/chat/completions"
        self.model_path = os.environ.get("VERL_MODEL_PATH", model_name)
        self.max_tokens = int(os.environ.get("VERL_MAX_TOKENS", "16000"))
        self.temperature = float(os.environ.get("VERL_TEMPERATURE", "0.3"))
        self.top_p = float(os.environ.get("VERL_TOP_P", "1.0"))
        self.max_model_len = int(os.environ.get("VERL_MAX_MODEL_LEN", "128000"))
        self.min_output_tokens = int(os.environ.get("VERL_MIN_OUTPUT_TOKENS", "512"))

        if self.system_instruction:
            self.history.append({"role": "system", "content": self.system_instruction})

        print(f"[OfflineLLM] api={self.api_url}, model={self.model_path}")

    def reset(self):
        self.history = []
        if self.system_instruction:
            self.history.append({"role": "system", "content": self.system_instruction})

    def _estimate_input_tokens(self) -> int:
        total_chars = 0
        image_count = 0
        for msg in self.history:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_chars += len(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            image_count += 1
            rc = msg.get("reasoning_content", "")
            if rc:
                total_chars += len(rc)
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    total_chars += len(func.get("arguments", "")) + len(func.get("name", ""))
        return int(total_chars / 3) + image_count * 1000

    def mllm(self, prompt, image_list):
        if not image_list:
            self.history.append({"role": "user", "content": prompt})
        else:
            user_content = [{"type": "text", "text": prompt}]
            for image_path in image_list:
                if not os.path.exists(image_path):
                    continue
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            self.history.append({"role": "user", "content": user_content})

        base_payload = {
            "model": self.model_path, "messages": self.history,
            "temperature": self.temperature, "top_p": self.top_p,
        }
        if self.tools:
            base_payload["tools"] = self.tools

        prompt_tokens = self._estimate_input_tokens()
        available_tokens = self.max_model_len - prompt_tokens
        actual_max_tokens = min(self.max_tokens, available_tokens)

        if actual_max_tokens < self.min_output_tokens:
            print(f"[OfflineLLM] 上下文超限 prompt_tokens={prompt_tokens}，跳过本轮调用")
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            return "", None

        payload = {**base_payload, "max_tokens": actual_max_tokens}

        response_data = None
        for attempt in range(3):
            try:
                response = requests.post(self.api_url, json=payload, timeout=600,
                                         headers={"Content-Type": "application/json"})
                if response.status_code != 200:
                    print(f"[OfflineLLM] HTTP {response.status_code}: {response.text[:500]}")
                response.raise_for_status()
                response_data = response.json()
                break
            except Exception as e:
                print(f"[OfflineLLM] 请求失败 ({attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    print_exc()

        if response_data is None:
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            return "", None

        response_message = response_data.get("choices", [{}])[0].get("message", {})
        reasoning_text, function_calls, content_text = parse_response_message(response_message)

        assistant_msg = {"role": "assistant", "content": content_text}
        if reasoning_text:
            assistant_msg["reasoning_content"] = reasoning_text
        if function_calls:
            assistant_msg["tool_calls"] = build_openai_tool_calls(function_calls)
        self.history.append(assistant_msg)

        print(f"[OfflineLLM] reasoning={len(reasoning_text)}c, calls={len(function_calls)}, content={str(content_text)[:100]}")
        return reasoning_text, function_calls if function_calls else None


# ============================================================
# MODEL_TYPE_MAP — main.py / verifier 按 key 索引客户端类
# ============================================================

MODEL_TYPE_MAP = {
    'gemini': GeminiMultiChat,      # Google Gemini 官方 API
    'openai': OpenAIMultiChat,      # OpenAI 官方 API
    'qwen3': QwenMultiChat,         # DashScope Qwen-VL 兼容模式
    'bailian': BailianMultiChat,    # 阿里云百炼
    'offline-llm': OfflineLLM,      # 本地 vLLM（读 VERL_SERVER_URL），我们训练的模型
}

REAL_MODEL_TYPE_MAP = dict(MODEL_TYPE_MAP)



# ============================================================
# 共享磁盘文件 RPC —— 训练节点（无外网）↔ 有网机器
# ============================================================
#
# 背景：GPU 训练节点常常访问不了外网，但能通过共享 mnt 磁盘与一台有网的机器通信。
# 方案：训练侧用 FileRPCChat（与其它客户端同接口）把每次 mllm/reset 请求写成
#       query/<req_id>.req.json，有网机器上的 broker.py 轮询目录、用真实客户端执行、
#       把结果写成 query/<req_id>.resp.json；训练侧轮询该响应文件取回结果。
#
# 并发：broker 用线程池并发处理不同 session 的请求，因此大批 rollout 的 verify
#       任务可以并发提交、并发打分（同一 session 内部仍串行，保证 history 顺序）。
#
# 有状态：按 session_id 复用真实客户端实例，从而完整复用现有各客户端的 history
#         累积 / 工具解析逻辑，无需在 RPC 层重写。
#
# 原子性：所有写入先写 <name>.tmp 再 os.replace 原子重命名，避免 ceph/NFS 下读到半包。

DEFAULT_QUERY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verifier", "query",
)


def get_query_dir() -> str:
    return os.environ.get("VIBEWORLD_QUERY_DIR", DEFAULT_QUERY_DIR)


def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json_safe(path: str):
    """读取 JSON；若文件尚未完整落盘（跨机可见性延迟）返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        if not data.strip():
            return None
        return json.loads(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def serialize_history_tail(client) -> dict:
    """
    把真实客户端 history[-1]（assistant 消息）统一序列化为标准 dict，
    使训练侧 append 后 verifier 的 _extract_content_from_history 能直接读取。

    - 普通客户端：history[-1] 已是 {"role":"assistant","content":...,...}，直接返回。
    - Gemini：history[-1] 是 types.Content，转成 {"role":"assistant","content":<非thought文本>,
      "reasoning_content":<thought文本>}。
    """
    if not getattr(client, "history", None):
        return {"role": "assistant", "content": ""}
    last = client.history[-1]
    if isinstance(last, dict):
        # 调用失败时不会追加 assistant 消息，末尾仍是 user；此时返回空，
        # 避免把用户 prompt 误当成模型回答。
        if last.get("role") != "assistant":
            return {"role": "assistant", "content": ""}
        # content 可能是 list（多模态）；verifier 支持 list，但 RPC 里统一成纯文本更稳。
        c = last.get("content", "")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out = {"role": "assistant", "content": c}
        if last.get("reasoning_content"):
            out["reasoning_content"] = last["reasoning_content"]
        if last.get("tool_calls"):
            out["tool_calls"] = last["tool_calls"]
        return out
    # Gemini types.Content
    if hasattr(last, "parts"):
        text_parts, thought_parts = [], []
        for p in (last.parts or []):
            if getattr(p, "thought", False) and getattr(p, "text", None):
                thought_parts.append(p.text)
            elif getattr(p, "text", None):
                text_parts.append(p.text)
        out = {"role": "assistant", "content": "".join(text_parts)}
        if thought_parts:
            out["reasoning_content"] = "".join(thought_parts).strip()
        return out
    return {"role": "assistant", "content": ""}


class FileRPCChat:
    """
    训练侧客户端：与GeminiMultiChat / OpenAIMultiChat 等同接口，
    通过共享磁盘把调用代理到有网机器上的 broker。

    构造参数与其它客户端一致：model_name / system_instruction / tools。
    额外通过 backend 指明 broker 侧应实例化哪个真实客户端
    （MODEL_TYPE_MAP 的 key：gemini / openai / qwen3 / bailian / offline-llm）。
    """

    def __init__(self, model_name: str = "", system_instruction: str = "", tools=None,
                 backend: str = "gemini"):
        self.backend = backend
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.query_dir = get_query_dir()
        os.makedirs(self.query_dir, exist_ok=True)
        self.poll_interval = float(os.environ.get("VIBEWORLD_RPC_POLL_INTERVAL", "0.5"))
        self.timeout = float(os.environ.get("VIBEWORLD_RPC_TIMEOUT", "600"))
        # session_id 唯一标识一条对话，broker 据此复用真实客户端实例。
        self._seq = 0
        self.session_id = self._new_session_id()
        self.history = []  # 本地镜像，供 verifier 的 _extract_*_from_history 读取

    def _new_session_id(self) -> str:
        # 不用 uuid/random，用 pid + 对象 id + 计数拼接，跨进程也不会撞。
        return f"{self.backend}-{os.getpid()}-{id(self)}-{self._seq}"

    def _next_req_id(self) -> str:
        self._seq += 1
        return f"{self.session_id}--req{self._seq}"

    def _rpc(self, payload: dict) -> dict:
        req_id = payload["req_id"]
        req_path = os.path.join(self.query_dir, f"{req_id}.req.json")
        resp_path = os.path.join(self.query_dir, f"{req_id}.resp.json")
        _atomic_write_json(req_path, payload)

        waited = 0.0
        while waited < self.timeout:
            resp = _read_json_safe(resp_path)
            if resp is not None:
                try:
                    os.remove(resp_path)
                except OSError:
                    pass
                return resp
            time.sleep(self.poll_interval)
            waited += self.poll_interval

        print(f"[FileRPCChat] 超时未收到响应 req_id={req_id} (>{self.timeout}s)")
        return {"ok": False, "error": "rpc_timeout", "reasoning_text": "", "function_calls": None}

    def reset(self):
        self.history = []
        self._rpc({
            "req_id": self._next_req_id(),
            "session_id": self.session_id,
            "op": "reset",
            "backend": self.backend,
            "model_name": self.model_name,
            "system_instruction": self.system_instruction,
            "tools": self.tools,
        })

    def mllm(self, prompt, image_list, role="user"):
        resp = self._rpc({
            "req_id": self._next_req_id(),
            "session_id": self.session_id,
            "op": "mllm",
            "backend": self.backend,
            "model_name": self.model_name,
            "system_instruction": self.system_instruction,
            "tools": self.tools,
            "prompt": prompt,
            "image_list": list(image_list or []),
            "role": role,
        })
        if not resp.get("ok"):
            print(f"[FileRPCChat] mllm 失败: {resp.get('error')}")
            return "", None

        assistant_msg = resp.get("assistant_msg")
        if assistant_msg:
            self.history.append(assistant_msg)

        reasoning_text = resp.get("reasoning_text", "") or ""
        function_calls = resp.get("function_calls")
        return reasoning_text, function_calls if function_calls else None


def enable_filerpc_transport():
    """
    把 MODEL_TYPE_MAP 每个 key 就地替换为绑定对应 backend 的 FileRPCChat 工厂，
    使 verifier / main.py 代码零改动即可走文件 RPC。
    由环境变量 VIBEWORLD_LLM_TRANSPORT=filerpc 触发（见模块底部）。

    注意：REAL_MODEL_TYPE_MAP 保持指向真实客户端类，broker 侧据此实例化。
    """
    def _make_factory(backend_key):
        def _factory(model_name="", system_instruction="", tools=None):
            return FileRPCChat(model_name=model_name, system_instruction=system_instruction,
                               tools=tools, backend=backend_key)
        return _factory

    for key in list(MODEL_TYPE_MAP.keys()):
        MODEL_TYPE_MAP[key] = _make_factory(key)
    print(f"[llm] FileRPC transport 已启用，query_dir={get_query_dir()}")


if os.environ.get("VIBEWORLD_LLM_TRANSPORT", "").lower() == "filerpc":
    enable_filerpc_transport()
