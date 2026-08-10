"""streaming_bailian.py — CLI 用的流式 LLM client（全部 OpenAI 兼容协议）

本文件提供 4 个流式 client，全部继承同一个基类：

  StreamingBailianMultiChat    阿里云百炼 / DashScope（基类）
  StreamingGeminiMultiChat     Gemini 官方（OpenAI 兼容端点）
  StreamingOpenAIMultiChat     OpenAI 官方
  LocalStreamingVLLMMultiChat  本地 vLLM（我们训练的 VibeWorlder 模型）

与 utils/llm.py 里其它 client **完全同接口**（`__init__` / `reset` / `mllm` / `history`），
所以 session.py / models.py 不需要特殊分支就能用；区别只有两点：

  1. `stream=True`，reasoning / content 逐 token 通过 `self.on_delta(kind, text)`
     回调实时吐出（kind 为 "reasoning" | "content"）。
  2. 类属性 `streaming = True` —— session.py 靠它判断「这个 bot 会自己实时输出」，
     从而跳过 `_suppress_llm_noise()`（那个噪音过滤器是按行缓冲的，会把逐 token
     输出攒着不放，流式效果全没了）。

为什么单独一个文件而不改 utils/llm.py：llm.py 是 RL / eval / verifier 共用的核心文件，
里面所有 client 都是阻塞式；流式只有 CLI demo 需要，放这里改动面最小。

踩坑（实测 qwen3.8-max）
------------------------
* **tool_call 的空首片**：每个 tool_call 的第一个 delta 形如
  `(index=0, name='retrieve_assets', arguments='')`，真参数在后续 delta 才分片到达
  （`'{"entity_name": '` / `'"木屋'` / `'"'` / `', "top_k": '` …）。必须按 index 累积
  成完整 JSON 再解析；若把空 arguments 当成「完整的 {}」立刻定案，会得到参数全空的
  工具调用（表现为 `🔍 retrieve: ?`，模型随后空转好几轮）。
* **name 只出现一次**：后续 delta 的 `function.name` 是 None，不能覆盖已存的 name。
"""

import json
import os
import time


def _llm():
    """拿到已被 models.py 加载好的 utils/llm.py 模块（复用其 helper，不重复加载）。

    延迟到调用时才 import，避免与 models.py 形成 import 期循环依赖。
    """
    from vibeworld import models
    return models.llm_mod


class StreamingBailianMultiChat:
    """百炼 / DashScope 流式 client（默认 qwen3.8-max）。

    on_delta: callable(kind, text) —— kind ∈ {"reasoning", "content"}。
              由 session.py 注入；未设置时静默（只是不实时打印，功能不受影响）。
    """

    streaming = True          # session.py 据此跳过按行缓冲的噪音过滤器

    def __init__(self, model_name: str = "qwen3.8-max", system_instruction="", tools=None,
                 on_delta=None, base_url: str = None, api_key: str = None):
        from openai import OpenAI
        resolved_key = (api_key
                        or os.getenv("BAILIAN_API_KEY")
                        or os.getenv("DASHSCOPE_API_KEY")
                        or "token")
        resolved_url = (base_url
                        or os.getenv("BAILIAN_BASE_URL",
                                     "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_url)
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.on_delta = on_delta
        self.max_tokens = int(os.environ.get("BAILIAN_MAX_TOKENS", "8192"))
        self.vision_ok = os.environ.get("BAILIAN_SEND_IMAGES", "true").lower() in ("1", "true", "yes")
        self.history = []
        if self.system_instruction:
            self.history.append({"role": "system", "content": self.system_instruction})

    def reset(self):
        self.history = []
        if self.system_instruction:
            self.history.append({"role": "system", "content": self.system_instruction})

    # ── 内部 ────────────────────────────────────────────────────────────────────
    def _emit(self, kind, text):
        if self.on_delta and text:
            try:
                self.on_delta(kind, text)
            except Exception:
                pass          # 展示层出错绝不能带崩 agent loop

    def _image_parts(self, image_list):
        if not self.vision_ok:
            return []
        mk = _llm().make_image_url_part
        return [mk(p) for p in (image_list or []) if os.path.exists(p)]

    def _build_user_content(self, prompt, image_list, with_images):
        if not with_images or not image_list:
            return prompt
        parts = [{"type": "text", "text": prompt}]
        parts.extend(self._image_parts(image_list))
        return parts

    def _push_turn(self, prompt, image_list, prev_tool_calls):
        """把本轮输入按 OpenAI 规范追加进 history。

        OpenAI/DashScope 严格要求：上一条 assistant 若带 tool_calls，下一条必须是
        role="tool" 且每个 tool_call_id 都有对应回应；图片只能放在随后的 user 消息里。
        """
        if prev_tool_calls:
            for i, tc in enumerate(prev_tool_calls):
                tc_id = (tc.get("id", f"call_{i}") if isinstance(tc, dict)
                         else getattr(tc, "id", f"call_{i}"))
                self.history.append({"role": "tool", "tool_call_id": tc_id,
                                     "content": prompt if i == 0 else "(done)"})
            parts = self._image_parts(image_list)
            if parts:
                self.history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "（本轮场景渲染图）"}] + parts})
        else:
            self.history.append({
                "role": "user",
                "content": self._build_user_content(prompt, image_list, self.vision_ok)})

    def _consume(self, stream):
        """吃掉整个 stream，返回 (reasoning, content, tool_calls_raw, emitted)。

        tool_calls_raw 为 OpenAI 格式 [{id,type,function:{name,arguments(str)}}]，
        交给 parse_response_message 统一解析，与阻塞版行为一致。
        """
        r_parts, c_parts = [], []
        acc, order = {}, []          # index -> {"id","name","args"}
        emitted = False

        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta

            rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
            if rc:
                r_parts.append(rc)
                self._emit("reasoning", rc)
                emitted = True

            if getattr(d, "content", None):
                c_parts.append(d.content)
                self._emit("content", d.content)
                emitted = True

            for tc in (getattr(d, "tool_calls", None) or []):
                idx = tc.index if tc.index is not None else 0
                if idx not in acc:
                    acc[idx] = {"id": None, "name": None, "args": ""}
                    order.append(idx)
                slot = acc[idx]
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):      # name 只在首片出现，别被 None 覆盖
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments   # 参数分片累积

        tool_calls_raw = []
        for i, idx in enumerate(order):
            slot = acc[idx]
            if not slot["name"]:
                continue
            tool_calls_raw.append({
                "id": slot["id"] or f"call_{i}", "type": "function",
                "function": {"name": slot["name"], "arguments": slot["args"] or "{}"},
            })
        return "".join(r_parts), "".join(c_parts), tool_calls_raw, emitted

    # ── 对外：与其它 client 同签名 ──────────────────────────────────────────────
    def mllm(self, prompt, image_list, role="user"):
        prev = self.history[-1] if self.history else {}
        prev_tool_calls = ((prev.get("tool_calls") or [])
                           if isinstance(prev, dict) and prev.get("role") == "assistant" else [])
        self._push_turn(prompt, image_list, prev_tool_calls)

        reasoning = content = ""
        tool_calls_raw = []
        ok = False
        for attempt in range(3):
            emitted = False
            try:
                kwargs = {"model": self.model_name, "messages": self.history,
                          "temperature": 0.2, "top_p": 1,
                          "max_tokens": self.max_tokens, "stream": True}
                if self.tools:
                    kwargs["tools"] = self.tools
                stream = self.client.chat.completions.create(**kwargs)
                reasoning, content, tool_calls_raw, emitted = self._consume(stream)
                ok = True
                break
            except Exception as e:
                emsg = str(e)
                # 视觉不支持 → 降级纯文本重试（与 BailianMultiChat 行为一致）
                vision_err = (any(k in emsg.lower() for k in ("image", "vision", "multimodal"))
                              or any(k in emsg for k in ("图片", "图像", "多模态", "不支持")))
                if self.vision_ok and image_list and vision_err and not prev_tool_calls:
                    self.vision_ok = False
                    self.history[-1]["content"] = self._build_user_content(
                        prompt, image_list, False)
                    continue
                # 已经吐出过内容就别重试了：重试会把同一轮 reasoning 打第二遍
                if emitted or attempt == 2:
                    self._emit("content", f"\n[流式请求失败: {type(e).__name__}: {emsg[:160]}]\n")
                    break
                time.sleep(5 if attempt == 0 else 15)

        if not ok and not (reasoning or content or tool_calls_raw):
            # 整轮失败：回滚本轮输入，避免污染 history
            while self.history and self.history[-1].get("role") in ("user", "tool"):
                self.history.pop()
            return "", None

        reasoning_text, function_calls, content_text = _llm().parse_response_message({
            "content": content or None,
            "reasoning": reasoning or None,
            "tool_calls": tool_calls_raw,
        })

        assistant_msg = {"role": "assistant", "content": content_text}
        if reasoning_text:
            assistant_msg["reasoning_content"] = reasoning_text
        if function_calls:
            assistant_msg["tool_calls"] = _llm().build_openai_tool_calls(function_calls)
        self.history.append(assistant_msg)

        return reasoning_text, (function_calls if function_calls else None)


class StreamingGeminiMultiChat(StreamingBailianMultiChat):
    """Gemini 官方 API 流式 client。

    走 Google 官方的 OpenAI 兼容端点
    (https://generativelanguage.googleapis.com/v1beta/openai/)，因此可以直接复用
    父类的流式 / tool-call 累积逻辑。

    api_key 来源（优先级从高到低）：
      1. 构造函数参数
      2. 环境变量 GEMINI_API_KEY
    base_url 可用 GEMINI_BASE_URL 覆盖。
    """

    streaming = True

    def __init__(self, model_name: str = "gemini-2.5-flash", system_instruction="",
                 tools=None, on_delta=None, base_url: str = None, api_key: str = None):
        resolved_url = (base_url
                        or os.getenv("GEMINI_BASE_URL",
                                     "https://generativelanguage.googleapis.com/v1beta/openai/"))
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or "your_gemini_api_key"
        super().__init__(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools,
            on_delta=on_delta,
            base_url=resolved_url,
            api_key=resolved_key,
        )


class StreamingOpenAIMultiChat(StreamingBailianMultiChat):
    """OpenAI 官方 API 流式 client。

    api_key 来源：构造函数参数 > 环境变量 OPENAI_API_KEY。
    base_url 默认走官方 https://api.openai.com/v1，可用 OPENAI_BASE_URL 覆盖。
    """

    streaming = True

    def __init__(self, model_name: str = "gpt-4o", system_instruction="",
                 tools=None, on_delta=None, base_url: str = None, api_key: str = None):
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "your_openai_api_key"
        super().__init__(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools,
            on_delta=on_delta,
            base_url=resolved_url,
            api_key=resolved_key,
        )


class LocalStreamingVLLMMultiChat(StreamingBailianMultiChat):
    """本地 vLLM 推理服务 client（我们训练的 VibeWorlder 模型）。

    与 StreamingBailianMultiChat 完全相同，只是把 base_url / api_key 指向本地服务。
    base_url 来源（优先级从高到低）：
      1. 环境变量 VIBEWORLD_LOCAL_VLLM_URL
      2. setup.py 里的 LOCAL_VLLM_SERVER（由 cli.py 的 _load_service_config() 注入）
      3. 默认值 http://localhost:8000/v1
    model_name 默认与 start_verl_server_CLI.py 里的 --served-model-name 对应。
    """

    streaming = True

    def __init__(self, model_name: str = "vibeworlder-30B-A3B", system_instruction="",
                 tools=None, on_delta=None,
                 base_url: str = None, api_key: str = None):
        resolved_url = (base_url
                        or os.getenv("VIBEWORLD_LOCAL_VLLM_URL",
                                     "http://localhost:8000/v1"))
        resolved_key = api_key or os.getenv("VIBEWORLD_LOCAL_VLLM_KEY", "token")
        super().__init__(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools,
            on_delta=on_delta,
            base_url=resolved_url,
            api_key=resolved_key,
        )
