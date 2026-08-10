"""
session.py — VibeWorlding 有状态交互会话

把原 agent.py 的 run_interactive 重构成 Session 类，支持 REPL 多轮续聊：
    s = Session(session_dir, viewer, model_name="vibeworlder")
    s.run_query("搭建日式庭院")     # 一个 query 内部跑多轮 agent loop 直到不再调用工具
    s.run_query("再加一个石灯笼")    # 续接上一轮上下文继续
    s.clear()                       # 重置场景与对话
    s.compact()                     # 压缩历史
    s.switch_model("gpt5")          # 换模型，保留场景状态

复用 VibeWorlding/main.py 的全部地图/检索/渲染逻辑，不重写。
每轮渲染后额外调用 glb_builder 生成交互式 3D GLB，并刷新 viewer。
"""

import contextlib
import copy
import importlib.util as _ilu
import json
import logging
import os
import shutil
import sys
import threading

logger = logging.getLogger("vibeworld.session")

# ── 路径注入：仓库根目录与 utils/（utils 必须优先，避免同名模块被覆盖）──────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VWE_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "../../"))      # repo root
_UTILS_DIR = os.path.join(_VWE_ROOT, "utils")

for _p in [_VWE_ROOT, _UTILS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if sys.path[0] != _UTILS_DIR:
    sys.path.insert(0, _UTILS_DIR)

# ── 从 main.py 导入所有逻辑 ───────────────────────────────────────────────────────
_main_spec = _ilu.spec_from_file_location("vibe_main", os.path.join(_VWE_ROOT, "main.py"))
_main_mod = _ilu.module_from_spec(_main_spec)
_main_spec.loader.exec_module(_main_mod)

TOOLS_SCHEMA_GENERATE = _main_mod.TOOLS_SCHEMA_GENERATE

# ── 强制对齐资产体系 ──────────────────────────────────────────────────────────────
# main.py 默认加载的是旧库（8 位 ID，render_in_blender_clone）。
# 当前 demo 用的是 VibeWorlding/render_in_blender 新库：5 位 ID，
# 与检索服务返回的 type_id 及 glb_builder 的 GLB 资产对齐。
_NEW_ITEM_INFOS_PATH = os.path.join(
    _VWE_ROOT, "render_in_blender", "assets", "item_infos.json"
)
try:
    _PCG_ITEM_INFOS = _main_mod.load_item_infos(path=_NEW_ITEM_INFOS_PATH, force_reload=True)
    _PCG_WHITELIST = set(_PCG_ITEM_INFOS.keys())
    logger.info(f"item_infos 已对齐新库: {len(_PCG_ITEM_INFOS)} 条 ← {_NEW_ITEM_INFOS_PATH}")
except Exception as _e:
    logger.warning(f"新库 item_infos 加载失败，回退 main.py 默认: {_e}")
    _PCG_ITEM_INFOS = _main_mod._PCG_ITEM_INFOS
    _PCG_WHITELIST = _main_mod._PCG_WHITELIST

split_function_calls = _main_mod.split_function_calls
call_retrieve_for_fc = _main_mod.call_retrieve_for_fc
apply_scene_calls_to_llm_output = _main_mod.apply_scene_calls_to_llm_output
format_retrieve_responses_for_user = _main_mod.format_retrieve_responses_for_user
enrich_component_info_for_generate = _main_mod.enrich_component_info_for_generate
llm_output_to_actors = _main_mod.llm_output_to_actors
gradio_render = _main_mod.gradio_render
fc_to_sft_dict = _main_mod.fc_to_sft_dict
get_system_prompt = _main_mod.get_system_prompt
FORMAT_PROMPT_GENERATE_TURN1 = _main_mod.FORMAT_PROMPT_GENERATE_TURN1
AssetRetrievalClient = _main_mod.AssetRetrievalClient

# refine 专用
TOOLS_SCHEMA_REFINE   = _main_mod.TOOLS_SCHEMA_REFINE
TOOLS_MAP_REFINE      = _main_mod.TOOLS_MAP_REFINE
FORMAT_PROMPT_REFINE  = _main_mod.FORMAT_PROMPT_REFINE
GRADIO_SERVER_REFINE  = _main_mod.GRADIO_SERVER_REFINE
normalize_tool_call   = _main_mod.normalize_tool_call
fix_flat_args         = _main_mod.fix_flat_args

from . import models
from . import glb_builder

# ── 服务地址（可被环境变量覆盖）──────────────────────────────────────────────────
# 渲染服务（5 视角拍照图）
RENDER_SERVER = os.environ.get("VIBEWORLD_RENDER_SERVER", "http://localhost:8080")
# 资产检索服务（2026-07 新部署）
RETRIEVE_SERVER = os.environ.get("VIBEWORLD_RETRIEVE_SERVER", "http://localhost:8081")


def _count_actors(llm_output: dict) -> int:
    return sum(
        len(v) for cat in llm_output.values() if isinstance(cat, dict)
        for v in cat.values() if isinstance(v, list)
    )


class _NoiseFilterStream:
    """过滤 utils/llm.py 内部 debug print 的噪音行，其余内容原样透传。

    noise 块以 _NOISE_START 前缀开头，以 _NOISE_END 单独行（"==="）结束。
    块内的所有连续行（如多行 reasoning 文本）也全部抑制，避免与
    on_reasoning 回调的显示重复。
    """
    _NOISE_START = (
        "  Reasoning (", "  Tool Calls (", "  Content (",
        "=== Gemini", "=== OpenAI", "[DEBUG",
    )
    _NOISE_END = "==="

    def __init__(self, real):
        self._real = real
        self._buf = ""
        self._in_noise = False   # 正在 noise 块内，抑制所有后续行直到块结束

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._in_noise:
                # 块结束标志：遇到独立的 "===" 行
                if line.strip() == self._NOISE_END.strip():
                    self._in_noise = False
                # 否则继续抑制（不输出 line）
            elif any(line.startswith(p) for p in self._NOISE_START):
                # 进入 noise 块
                self._in_noise = True
            elif line.strip() == self._NOISE_END.strip():
                # 孤立的 "===" 也是噪音（客户端打印的结束行）
                pass
            else:
                self._real.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf:
            is_noise = (self._in_noise or
                        any(self._buf.startswith(p) for p in self._NOISE_START) or
                        self._buf.strip() == self._NOISE_END.strip())
            if not is_noise:
                self._real.write(self._buf)
        self._buf = ""
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def _suppress_llm_noise():
    """临时把 stdout 套上噪音过滤器（仅包住 mllm 调用）。"""
    real = sys.stdout
    sys.stdout = _NoiseFilterStream(real)
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = real


class Session:
    """一个有状态的 VibeWorlding 会话。"""

    def __init__(
        self,
        session_dir: str,
        viewer,
        model_name: str = models.DEFAULT_MODEL_KEY,
        quality: str = "低质量 (快速预览)",
        max_turns_per_query: int = 8,
        log_dir: str = None,
        callbacks: dict = None,
    ):
        self.session_dir = session_dir
        self.viewer = viewer
        self.quality = quality
        self.max_turns_per_query = max_turns_per_query
        self.callbacks = callbacks or {}
        os.makedirs(session_dir, exist_ok=True)

        # trajectory 目录
        if log_dir is None:
            session_name = os.path.basename(session_dir.rstrip("/"))
            _demo_root = os.path.normpath(os.path.join(_THIS_DIR, "../../data/demo"))
            log_dir = os.path.join(_demo_root, "log", session_name)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # system prompt + bot
        self.sys_prompt = get_system_prompt("generate")
        self.model_key, self.model_name, self.bot = models.build_bot(
            model_name, self.sys_prompt, TOOLS_SCHEMA_GENERATE,
            on_delta=self._on_delta,
        )
        self.bot.reset()

        self.retrieve_client = AssetRetrievalClient(base_url=RETRIEVE_SERVER)

        # 场景状态
        self.llm_output: dict = {}
        self.component_info: dict = {}
        self.images: list = []
        self.pending_retrieves: list = []
        self.turn = 0                 # 跨 query 累计轮数（viewer 展示用）
        self.started = False          # 是否已发过首个 query
        self.last_response = ""       # 本 query 的最终 assistant 回复（用于裁剪历史）
        self.sft = {"system_instruction": self.sys_prompt, "task_setting": "generate",
                    "query": None, "conversations": []}

        # 协作式取消：Ctrl+C 时主线程 set，worker 在每轮检查点优雅退出
        self._cancel = threading.Event()

    # ── 协作式取消 ─────────────────────────────────────────────────────────────
    def request_cancel(self):
        """主线程调用：请求中断当前 run_query。"""
        self._cancel.set()

    def reset_cancel(self):
        """开始新一轮 query 前清除取消标志。"""
        self._cancel.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ── 回调小工具 ────────────────────────────────────────────────────────────
    def _cb(self, name, *args):
        fn = self.callbacks.get(name)
        if fn:
            fn(*args)

    @property
    def is_streaming(self) -> bool:
        """当前 bot 是否自己实时输出（流式 client 声明 streaming = True）。"""
        return getattr(self.bot, "streaming", False)

    def _on_delta(self, kind: str, text: str):
        """流式 client 的逐 token 回调 → 交给 on_stream_delta 展示回调。

        注意 `self._streamed`：流式轮里 reasoning 已经实时打完了，on_reasoning
        回调就不能再整段打一遍（否则同一段 reasoning 出现两次）。
        """
        self._streamed = True
        self._cb("on_stream_delta", kind, text)

    def _reasoning_for_cb(self, reasoning: str) -> str:
        """给 on_reasoning 用的 reasoning：本轮已流式打过则返回空串。

        on_reasoning 同时负责打印工具调用行，所以不能整个跳过它 —— 只把
        reasoning 置空，让它只打 tool_calls。
        """
        if getattr(self, "_streamed", False):
            return ""
        return reasoning or ""

    def _full_reasoning(self, reasoning_returned: str) -> str:
        """合并 mllm 返回的 reasoning + history 里被丢弃的 content。

        gemini 走 OpenAI 兼容端点时文字永远在 content 字段：
        - 有 tool_call 的轮：parse_response_message 把 content 搬进 reasoning → 正常
        - 无 tool_call 的总结轮：reasoning 来源全空，文本留在 content 被丢弃 → 这里捞回

        优先级：reasoning > reasoning_content > content（去重合并）。
        tool_call 的 reason 字段不在此处展示，以免与模型真实推理混淆。
        """
        parts = []
        seen = set()

        def _add(txt):
            if isinstance(txt, str):
                t = txt.strip()
                if t and t not in seen:
                    seen.add(t)
                    parts.append(t)

        _add(reasoning_returned)

        last = self.bot.history[-1] if getattr(self.bot, "history", None) else None
        if isinstance(last, dict) and last.get("role") == "assistant":
            _add(last.get("reasoning_content"))
            c = last.get("content")
            if isinstance(c, str):
                _add(c)

        return "\n\n".join(parts)

    # ── 主入口：跑一个 query 的完整 agent loop ─────────────────────────────────
    def run_query(self, query: str):
        """运行一个 query，内部多轮直到模型不再调用工具。返回 (llm_output, images, turns)。"""
        query = (query or "").strip()
        if not query:
            return self.llm_output, self.images, self.turn
        if self.sft["query"] is None:
            self.sft["query"] = query
        self._current_query = query
        self.last_response = ""

        # 本 query 首条消息
        if not self.started:
            next_message = FORMAT_PROMPT_GENERATE_TURN1.format(user_query=query)
            self.started = True
        else:
            current_map_str = (json.dumps(self.llm_output, ensure_ascii=False, indent=2)
                               if self.llm_output else "(场景为空)")
            next_message = (
                f"# 用户追加需求\n\n{query}\n\n"
                f"当前场景元件信息:\n{current_map_str}\n"
                f"请基于现有场景继续（可 retrieve_assets / add / rotation_and_translation / delete）。"
            )
        next_role = "user"

        error_info = ""
        is_terminate = False
        query_turns = 0

        while not is_terminate and query_turns < self.max_turns_per_query:
            # ── 取消检查点①：进入新一轮前 ──
            if self.cancelled:
                logger.info("run_query: 已取消，停止开新轮")
                break

            self.turn += 1
            query_turns += 1
            turn = self.turn
            turn_dir = os.path.join(self.session_dir, f"turn_{turn}")
            os.makedirs(os.path.join(turn_dir, "image"), exist_ok=True)

            self._cb("on_turn_start", turn)

            # 续轮：构造 tool_response 消息
            if query_turns > 1:
                next_message, next_role = self._build_followup_message(error_info)
                error_info = ""

            # sft: user/tool 消息
            self.sft["conversations"].append({
                "role": next_role, "content": next_message,
                "images": list(self.images), "turn_n": turn,
            })

            # ── 调 LLM ────────────────────────────────────────────────────────
            # 阻塞式 client：套上噪音过滤器，屏蔽 llm.py 内部的 debug print。
            # 流式 client：**不能**套 —— _NoiseFilterStream 按行缓冲（只在遇到 \n
            # 才输出），会把逐 token 的增量攒住，流式效果全没了。
            self.viewer.set_status("thinking")
            self._streamed = False
            if self.is_streaming:
                reasoning, fcs = self.bot.mllm(next_message, self.images,
                                                role=next_role)
            else:
                with _suppress_llm_noise():
                    reasoning, fcs = self.bot.mllm(next_message, self.images,
                                                    role=next_role)
            fc_list = (fcs if isinstance(fcs, list) else [fcs]) if fcs else []

            # 合并 reasoning + 被丢弃的 content
            reasoning = self._full_reasoning(reasoning or "")
            if reasoning.strip():
                self.last_response = reasoning.strip()

            # ── 取消检查点②：LLM 返回后、执行工具前 ──
            if self.cancelled:
                self._cb("on_reasoning", turn, self._reasoning_for_cb(reasoning), fc_list)
                logger.info("run_query: LLM 返回后检测到取消，丢弃本轮工具执行")
                break

            if fc_list:
                retrieve_calls, scene_calls = split_function_calls(fc_list)
                sft_fc = [x for x in (fc_to_sft_dict(fc) for fc in retrieve_calls + scene_calls) if x]
            else:
                retrieve_calls, scene_calls, sft_fc = [], [], []

            self._cb("on_reasoning", turn, self._reasoning_for_cb(reasoning), fc_list)

            self.sft["conversations"].append({
                "role": "assistant", "content": reasoning or "",
                "function_calls": sft_fc or None, "turn_n": turn,
            })

            # viewer：先显示 reasoning + tool_calls
            self.viewer.update(
                turn_idx=turn, reasoning=reasoning or "", tool_calls=fc_list,
                images=[], glb=None, is_current=True,
                status="rendering" if scene_calls else ("thinking" if retrieve_calls else "done"),
                actor_count=_count_actors(self.llm_output),
            )

            # 无工具调用 → 本 query 结束
            if not fc_list:
                is_terminate = True
                break

            # ── retrieve ────────────────────────────────────────────────────
            if retrieve_calls:
                for fc in retrieve_calls:
                    resp = call_retrieve_for_fc(
                        fc, self.retrieve_client,
                        item_infos=_PCG_ITEM_INFOS or None,
                        pcg_whitelist=_PCG_WHITELIST or None,
                    )
                    self.pending_retrieves.append(resp)
                    for item in resp.get("response", {}).get("results", []):
                        name_r = item.get("name", "")
                        if name_r:
                            self.component_info[name_r] = {
                                "typeId": item.get("type_id", ""),
                                "native_bbox_m": item.get("native_bbox_m"),
                                "category": item.get("category_minor", ""),
                            }
                with open(os.path.join(turn_dir, "retrieve_responses.json"), "w", encoding="utf-8") as f:
                    json.dump(self.pending_retrieves, f, ensure_ascii=False, indent=2)

            # ── scene_calls ─────────────────────────────────────────────────
            if scene_calls:
                try:
                    new_output, scene_err = apply_scene_calls_to_llm_output(
                        scene_calls, self.llm_output, _PCG_ITEM_INFOS or {},
                    )
                    if new_output:
                        self.llm_output = new_output
                    if scene_err:
                        error_info += scene_err
                except Exception as e:
                    error_info += f"scene_call 异常: {e} "
                    logger.warning(f"scene_call 异常: {e}")

            with open(os.path.join(turn_dir, "map.json"), "w", encoding="utf-8") as f:
                json.dump(self.llm_output, f, ensure_ascii=False, indent=4)

            # ── 取消检查点③：渲染（最慢步骤）前 ──
            if self.cancelled:
                logger.info("run_query: 渲染前检测到取消，跳过本轮渲染")
                break

            # ── 渲染 5 视角 + 拼 GLB ──────────────────────────────────────────
            new_images, actors = self._render_turn(turn_dir, scene_calls)
            if new_images:
                self.images = new_images
                self._cb("on_rendered", turn, self.images)

            glb_url = None
            if actors:
                glb_url = self._build_glb(turn_dir, turn, actors)

            self.viewer.update(
                turn_idx=turn, reasoning=reasoning or "", tool_calls=fc_list,
                images=self.images, glb=glb_url, is_current=True,
                status="waiting", actor_count=_count_actors(self.llm_output),
            )

        self._save_outputs()
        self.viewer.set_status("done", actor_count=_count_actors(self.llm_output))
        # 裁剪 bot 历史：下一轮对话只保留 system + 最新场景&query + 最终回复，
        # 清空中间的 tool_call / tool_response（检索结果、多轮图片等），防止上下文膨胀。
        self._trim_history_for_next_query()
        return self.llm_output, self.images, self.turn

    # ── 裁剪历史：为下一轮对话只保留精炼上下文 ──────────────────────────────────
    def _trim_history_for_next_query(self):
        """把 bot.history 压缩为：[system?] + 最新场景&query(user) + 最终回复(assistant)。

        中间所有 tool_call 消息、tool_response（含检索结果与图片占位）全部丢弃，
        既保住「当前场景 + 上一轮意图 + 模型结论」的连续性，又避免历史无限增长。
        """
        hist = getattr(self.bot, "history", None)
        if hist is None:
            return

        # 保留原有的 system 消息（若有）
        system_msgs = [m for m in hist if isinstance(m, dict) and m.get("role") == "system"]

        scene_str = (json.dumps(self.llm_output, ensure_ascii=False, indent=2)
                     if self.llm_output else "(场景为空)")
        recap = (
            f"[上一轮回顾] 用户需求：{getattr(self, '_current_query', '') or '(无)'}\n\n"
            f"当前已构建场景元件信息：\n{scene_str}"
        )

        new_hist = list(system_msgs)
        new_hist.append({"role": "user", "content": recap})
        new_hist.append({"role": "assistant",
                         "content": self.last_response or "已完成本轮场景构建。"})
        self.bot.history = new_hist
        logger.info(f"history 已裁剪 → {len(new_hist)} 条（system={len(system_msgs)}）")

    # ── 续轮 tool_response 消息（照搬 main.py 逻辑）────────────────────────────
    def _build_followup_message(self, error_info: str):
        parts = []
        has_retrieve = bool(self.pending_retrieves)
        if self.pending_retrieves:
            parts.append(format_retrieve_responses_for_user(self.pending_retrieves))
            self.pending_retrieves = []

        tools_hint = (
            "当前可用 tools: retrieve_assets / add / rotation_and_translation / delete。"
            "add 必须传 type_id（来自之前 retrieve_assets 返回过的）。"
        )
        current_map_str = (json.dumps(self.llm_output, ensure_ascii=False, indent=2)
                           if self.llm_output else "(场景为空，还没有摆放任何元件)")

        if self.images:
            parts.append(
                f"<tool_response>本轮场景已渲染，当前元件信息:\n{current_map_str}\n\n"
                f"5视角图(左/右/前/后/俯):<image><image><image><image><image>。\n"
                f"{tools_hint}</tool_response>"
            )
        elif self.llm_output:
            parts.append(
                f"<tool_response>当前元件信息:\n{current_map_str}\n\n"
                f"(渲染失败或未生成。)\n{tools_hint}</tool_response>"
            )
        elif has_retrieve:
            parts.append(f"<tool_response>{tools_hint}\n请基于以上检索结果继续。</tool_response>")
        else:
            parts.append(f"<tool_response>(本轮工具调用未产生有效结果。)\n{tools_hint}</tool_response>")

        if error_info:
            parts.append(f"提示：{error_info}")
        return "\n".join(parts), "tool"

    # ── 渲染 5 视角，返回 (images, actors) ─────────────────────────────────────
    def _render_turn(self, turn_dir: str, scene_calls: list):
        if not (scene_calls and self.llm_output):
            return [], []

        if _PCG_ITEM_INFOS:
            render_ci = enrich_component_info_for_generate(
                base_component_info={}, llm_output=self.llm_output, item_infos=_PCG_ITEM_INFOS,
            )
        else:
            render_ci = self.component_info

        actors, parse_err = llm_output_to_actors(self.llm_output, render_ci)
        if parse_err or not actors:
            return [], (actors or [])

        with open(os.path.join(turn_dir, "pcg_render.json"), "w", encoding="utf-8") as f:
            json.dump([{"actors": actors}], f, ensure_ascii=False, indent=2)

        try:
            new_images, pcg_err = gradio_render(
                None, actors, os.path.join(turn_dir, "image"),
                quality=self.quality, lens=31, pcg_timeout=120,
                server_url=RENDER_SERVER,
            )
            if pcg_err:
                logger.warning(f"渲染告警: {pcg_err}")
            return new_images, actors
        except Exception as e:
            logger.warning(f"渲染异常: {e}")
            return [], actors

    # ── 拼装交互式 GLB，返回供 viewer 用的相对 URL（失败 None）─────────────────
    def _build_glb(self, turn_dir: str, turn: int, actors: list):
        try:
            glb_path = os.path.join(turn_dir, f"scene_{turn}.glb")
            out = glb_builder.build_scene_glb(actors, glb_path)
            if out:
                # viewer HTTP server 根目录是 session_dir，返回相对路径
                return os.path.relpath(out, self.session_dir)
        except Exception as e:
            logger.warning(f"GLB 构建异常: {e}")
        return None

    # ── 保存最终结果 + trajectory ──────────────────────────────────────────────
    def _save_outputs(self):
        final_dir = os.path.join(self.session_dir, "final_image")
        os.makedirs(final_dir, exist_ok=True)
        for img in self.images:
            try:
                shutil.copy(img, final_dir)
            except Exception:
                pass
        with open(os.path.join(self.session_dir, "final_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.llm_output, f, ensure_ascii=False, indent=4)

        with open(os.path.join(self.log_dir, "sft_trajectory.json"), "w", encoding="utf-8") as f:
            json.dump(self.sft, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.log_dir, "final_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.llm_output, f, ensure_ascii=False, indent=4)
        if self.images:
            log_img_dir = os.path.join(self.log_dir, "final_image")
            os.makedirs(log_img_dir, exist_ok=True)
            for img in self.images:
                try:
                    shutil.copy(img, log_img_dir)
                except Exception:
                    pass

    # ── /clear：重置场景与对话 ────────────────────────────────────────────────
    def clear(self):
        self.bot.reset()
        self.llm_output = {}
        self.component_info = {}
        self.images = []
        self.pending_retrieves = []
        self.turn = 0
        self.started = False
        self.sft = {"system_instruction": self.sys_prompt, "task_setting": "generate",
                    "query": None, "conversations": []}
        self.viewer.reset()
        self.viewer.set_status("waiting")

    # ── /compact：压缩历史，保留场景状态 ──────────────────────────────────────
    def compact(self):
        """把当前 bot 历史压缩成一段场景状态摘要，重置 bot 后注入。

        保留 llm_output / images / turn 计数；丢弃逐轮对话细节。
        """
        summary = self._scene_summary()
        self.bot.reset()
        # 以一条 user 消息把场景状态喂回新历史（不触发工具调用，仅占位上下文）
        self.bot.history.append({
            "role": "user",
            "content": [{"type": "text", "text":
                         f"[历史已压缩] 这是当前已构建场景的状态摘要：\n{summary}\n"
                         f"后续请在此基础上继续。"}],
        })
        self.bot.history.append({
            "role": "assistant",
            "content": "已了解当前场景状态，准备继续。",
        })
        return summary

    def _scene_summary(self) -> str:
        n = _count_actors(self.llm_output)
        cats = []
        for cat, v in self.llm_output.items():
            if isinstance(v, dict):
                for sub, items in v.items():
                    if isinstance(items, list) and items:
                        cats.append(f"{cat}/{sub}×{len(items)}")
        cat_str = "，".join(cats) if cats else "（空）"
        return (f"已完成 {self.turn} 轮，共 {n} 个元件。分类：{cat_str}。\n"
                f"当前地图 JSON:\n{json.dumps(self.llm_output, ensure_ascii=False, indent=2)}")

    # ── /model：切换模型，保留场景状态 ────────────────────────────────────────
    def switch_model(self, name: str):
        """换 client。把当前场景状态摘要注入新 bot 以延续上下文。"""
        old_history = list(getattr(self.bot, "history", []))
        self.model_key, self.model_name, self.bot = models.build_bot(
            name, self.sys_prompt, TOOLS_SCHEMA_GENERATE,
            on_delta=self._on_delta,
        )
        self.bot.reset()
        if self.started and self.llm_output:
            summary = self._scene_summary()
            self.bot.history.append({
                "role": "user",
                "content": [{"type": "text", "text":
                             f"[切换模型，承接已有进度] 当前场景状态：\n{summary}\n后续在此基础上继续。"}],
            })
            self.bot.history.append({
                "role": "assistant",
                "content": "已接管当前场景，准备继续。",
            })
        return self.model_key, self.model_name


# ══════════════════════════════════════════════════════════════════════════════
# RefineSession — 基于已有场景数据的交互式修改
# ══════════════════════════════════════════════════════════════════════════════

class RefineSession:
    """从 VWE-Bench 格式的数据目录加载初始场景，然后通过对话逐步修改。

    数据目录结构（/path/to/<case>/）：
        init_map.json       初始场景 JSON
        component_info.json 各元件元数据（pos/rot/bbox 等）
        query.json          scene description / theme / gt_map 等
        image/              5 视角初始渲染图（*.jpg）
        camera_params.json  固定相机参数（可选）
        scatter_cache.json  撒点缓存（可选）

    用法：
        /refine /path/to/case        → 加载并展示初始场景，提示 description
        接下来 REPL 输入即为 query   → run_query() 执行修改
        /refine /path/to/other_case  → 切换到新案例（重置状态）
    """

    def __init__(
        self,
        data_dir: str,
        session_dir: str,
        viewer,
        model_name: str = models.DEFAULT_MODEL_KEY,
        quality: str = "低质量 (快速预览)",
        max_turns_per_query: int = 8,
        callbacks: dict = None,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.session_dir = session_dir
        self.viewer = viewer
        self.quality = quality
        self.max_turns_per_query = max_turns_per_query
        self.callbacks = callbacks or {}
        os.makedirs(session_dir, exist_ok=True)

        # ── 加载数据目录 ──────────────────────────────────────────────────────
        self._load_data()

        # ── bot（refine 用不同 system prompt + TOOLS_SCHEMA_REFINE）─────────
        self.sys_prompt = get_system_prompt("refine")
        self.model_key, self.model_name, self.bot = models.build_bot(
            model_name, self.sys_prompt, TOOLS_SCHEMA_REFINE,
            on_delta=self._on_delta,
        )
        self.bot.reset()

        # ── 运行状态 ─────────────────────────────────────────────────────────
        self.turn = 0
        self.started = False
        self.last_response = ""
        self._current_query = ""
        self._cancel = threading.Event()
        self.sft = {
            "system_instruction": self.sys_prompt,
            "task_setting": "refine",
            "data_dir": self.data_dir,
            "conversations": [],
        }

        # 把初始图片推入 viewer
        self._show_initial()

    # ── 加载数据目录 ──────────────────────────────────────────────────────────
    def _load_data(self):
        import glob as _glob
        d = self.data_dir
        with open(os.path.join(d, "init_map.json"), encoding="utf-8") as f:
            self.llm_output = json.load(f)
        with open(os.path.join(d, "component_info.json"), encoding="utf-8") as f:
            self.component_info = json.load(f)
        with open(os.path.join(d, "query.json"), encoding="utf-8") as f:
            self.query_info = json.load(f)

        # 去掉地图信息元数据（不参与工具调用）
        self.llm_output.pop("地图信息", None)
        self.component_info.pop("地图信息", None)

        self.theme = self.query_info.get("theme", "")
        self.description = self.query_info.get("description", "")

        # 初始渲染图
        self.images = (
            sorted(_glob.glob(os.path.join(d, "image", "*.jpg"))) or
            sorted(_glob.glob(os.path.join(d, "*.jpg")))
        )

        # 可选：固定相机参数
        self.fixed_cam = None
        for search_dir in [d, os.path.dirname(d)]:
            cam_path = os.path.join(search_dir, "camera_params.json")
            if os.path.exists(cam_path):
                with open(cam_path, encoding="utf-8") as f:
                    self.fixed_cam = json.load(f)
                break

        # 可选：撒点缓存
        self.scatter_cache = {}
        for search_dir in [d, os.path.dirname(d)]:
            cache_path = os.path.join(search_dir, "scatter_cache.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, encoding="utf-8") as f:
                        self.scatter_cache = json.load(f)
                except Exception:
                    pass
                break

    # ── 把初始图片推入 viewer（初始场景展示）────────────────────────────────
    def _show_initial(self):
        # 干净的英文历史：模型单独设，首条历史用场景 description（refine 的初始意图）
        self.viewer.set_model(models.display_name(self.model_name))
        if self.description:
            self.viewer.add_query(self.description)
        if self.images:
            # 用 turn_idx=0 展示初始状态
            self.viewer.update(
                turn_idx=1, images=self.images, glb=None,
                is_current=True, status="waiting", actor_count=_count_actors(self.llm_output),
            )
        # 尝试从初始 actors 生成初始 GLB
        try:
            actors, _ = llm_output_to_actors(self.llm_output, self.component_info, self.scatter_cache)
            if actors:
                glb_path = os.path.join(self.session_dir, "init_scene.glb")
                glb = glb_builder.build_scene_glb(actors, glb_path)
                if glb:
                    self.viewer.update(
                        turn_idx=1, images=self.images,
                        glb=os.path.relpath(glb, self.viewer._dir),
                        is_current=True, status="waiting",
                        actor_count=_count_actors(self.llm_output),
                    )
        except Exception as e:
            logger.warning(f"[RefineSession] 初始 GLB 生成失败: {e}")

    # ── 协作式取消（与 Session 接口一致）────────────────────────────────────
    def request_cancel(self): self._cancel.set()
    def reset_cancel(self):   self._cancel.clear()
    @property
    def cancelled(self): return self._cancel.is_set()

    # ── 回调小工具 ────────────────────────────────────────────────────────────
    def _cb(self, name, *args):
        fn = self.callbacks.get(name)
        if fn: fn(*args)

    # 流式支持（与 Session 同语义；RefineSession 不继承 Session，需各自定义）
    @property
    def is_streaming(self) -> bool:
        return getattr(self.bot, "streaming", False)

    def _on_delta(self, kind: str, text: str):
        self._streamed = True
        self._cb("on_stream_delta", kind, text)

    def _reasoning_for_cb(self, reasoning: str) -> str:
        """本轮已流式打过 → 返回空串，避免 on_reasoning 再整段打一遍。"""
        if getattr(self, "_streamed", False):
            return ""
        return reasoning or ""

    # ── 主入口 ────────────────────────────────────────────────────────────────
    def run_query(self, query: str):
        """执行一次修改 query，内部多轮直到模型不再调工具。"""
        query = (query or self.description).strip()
        if not query:
            return self.llm_output, self.images, self.turn

        self._current_query = query
        self.last_response = ""

        error_info = ""
        is_terminate = False
        query_turns = 0

        while not is_terminate and query_turns < self.max_turns_per_query:
            if self.cancelled:
                break

            self.turn += 1
            query_turns += 1
            turn = self.turn
            turn_dir = os.path.join(self.session_dir, f"turn_{turn}")
            os.makedirs(os.path.join(turn_dir, "image"), exist_ok=True)

            self._cb("on_turn_start", turn)

            # ── 构造 user/tool 消息（照搬 run_sample_refine 逻辑）──────────
            if not self.started:
                user_message = FORMAT_PROMPT_REFINE.format(
                    theme=self.theme,
                    scene_description=query,
                    element_info=self.llm_output,
                    component_info=list(self.component_info.keys()),
                )
                next_role = "user"
                self.started = True
            else:
                map_str = json.dumps(self.llm_output, ensure_ascii=False, indent=4)
                tools_hint = (
                    "当前可用tools如下（rotation_and_translation(arguments: corrections), "
                    "delete(arguments: modified_data)，add(arguments: modified_data)）"
                )
                if self.images:
                    user_message = (
                        f"本轮改造后场景的基本信息如下：\n{map_str}\n\n"
                        f"下面5张图片展示了当前场景（左/右/前/后/俯）："
                        f" <image><image><image><image><image>。\n{tools_hint}"
                    )
                else:
                    user_message = (
                        f"本轮场景信息：\n{map_str}\n\n（本轮渲染失败，无新图片。）\n{tools_hint}"
                    )
                if error_info:
                    user_message += f"\n提示：{error_info}"
                next_role = "tool"

            error_info = ""
            self.sft["conversations"].append({
                "role": next_role, "content": user_message,
                "images": list(self.images), "turn_n": turn,
            })

            # ── 调 LLM ──────────────────────────────────────────────────────
            # 流式 client 不能套 _suppress_llm_noise（按行缓冲会吞掉逐 token 增量）
            self.viewer.set_status("thinking")
            self._streamed = False
            if self.is_streaming:
                reasoning, fcs = self.bot.mllm(user_message, self.images, role=next_role)
            else:
                with _suppress_llm_noise():
                    reasoning, fcs = self.bot.mllm(user_message, self.images, role=next_role)
            fc_list = (fcs if isinstance(fcs, list) else [fcs]) if fcs else []
            reasoning = self._full_reasoning(reasoning or "")
            if reasoning.strip():
                self.last_response = reasoning.strip()

            if self.cancelled:
                self._cb("on_reasoning", turn, self._reasoning_for_cb(reasoning), fc_list)
                break

            self._cb("on_reasoning", turn, self._reasoning_for_cb(reasoning), fc_list)
            self.sft["conversations"].append({
                "role": "assistant", "content": reasoning,
                "function_calls": [fc_to_sft_dict(fc) for fc in fc_list] if fc_list else None,
                "turn_n": turn,
            })

            self.viewer.update(
                turn_idx=turn, reasoning=reasoning, tool_calls=fc_list,
                images=[], glb=None, is_current=True,
                status="rendering" if fc_list else "done",
                actor_count=_count_actors(self.llm_output),
            )

            # 无工具调用 → 终止
            if not fc_list:
                is_terminate = True
                break

            # ── 执行工具（复刻 _agent_step_refine）──────────────────────────
            updated = copy.deepcopy(self.llm_output)
            for raw_tc in fc_list:
                name, args = normalize_tool_call(raw_tc)
                if name is None or name == "terminate":
                    if name == "terminate":
                        is_terminate = True
                    continue
                if name in TOOLS_MAP_REFINE:
                    args = fix_flat_args(name, args)
                    try:
                        updated = TOOLS_MAP_REFINE[name](llm_output=updated, **args)
                    except Exception as e:
                        error_info += f"{name}失败: {e} "
                        logger.warning(f"[refine] {name} 失败: {e}")
                else:
                    logger.warning(f"[refine] 未知工具: {name}")
            self.llm_output = updated

            with open(os.path.join(turn_dir, "map.json"), "w", encoding="utf-8") as f:
                json.dump(self.llm_output, f, ensure_ascii=False, indent=4)

            if self.cancelled:
                break

            # ── 渲染 ────────────────────────────────────────────────────────
            self.viewer.set_status("rendering")
            new_images, actors = self._render_turn(turn_dir)
            if new_images:
                self.images = new_images
                self._cb("on_rendered", turn, self.images)

            glb_url = None
            if actors:
                glb_url = self._build_glb(turn_dir, turn, actors)

            self.viewer.update(
                turn_idx=turn, reasoning=reasoning, tool_calls=fc_list,
                images=self.images, glb=glb_url, is_current=True,
                status="waiting", actor_count=_count_actors(self.llm_output),
            )

        self._save_outputs()
        self.viewer.set_status("done", actor_count=_count_actors(self.llm_output))
        self._trim_history()
        return self.llm_output, self.images, self.turn

    # ── 渲染（固定相机参数）──────────────────────────────────────────────────
    def _render_turn(self, turn_dir: str):
        actors, parse_err = llm_output_to_actors(
            self.llm_output, self.component_info, self.scatter_cache
        )
        if parse_err or not actors:
            return [], actors or []

        turn_image_dir = os.path.join(turn_dir, "image")
        lens = self.fixed_cam.get("lens", 31) if self.fixed_cam else 31
        cam_pos = self.fixed_cam.get("cam_pos") if self.fixed_cam else None
        cam_target = self.fixed_cam.get("cam_target") if self.fixed_cam else None
        render_server = RENDER_SERVER

        try:
            new_images, pcg_err = gradio_render(
                None, actors, turn_image_dir,
                quality=self.quality, lens=lens, pcg_timeout=120,
                cam_pos_override=cam_pos, cam_target_override=cam_target,
                server_url=render_server,
            )
            if pcg_err:
                logger.warning(f"[refine] 渲染告警: {pcg_err}")
            return new_images, actors
        except Exception as e:
            logger.warning(f"[refine] 渲染异常: {e}")
            return [], actors

    # ── GLB（与 Session 一致）────────────────────────────────────────────────
    def _build_glb(self, turn_dir: str, turn: int, actors: list):
        try:
            glb_path = os.path.join(turn_dir, f"scene_{turn}.glb")
            out = glb_builder.build_scene_glb(actors, glb_path)
            if out:
                # relpath 기준: HTTP server root = viewer._dir (viewer.html 위치)
                # refine_session_dir은 그 하위 디렉토리이므로 self.session_dir 기준이면 404
                return os.path.relpath(out, self.viewer._dir)
        except Exception as e:
            logger.warning(f"[refine] GLB 失败: {e}")
        return None

    # ── 保存输出 ─────────────────────────────────────────────────────────────
    def _save_outputs(self):
        final_dir = os.path.join(self.session_dir, "final_image")
        os.makedirs(final_dir, exist_ok=True)
        for img in self.images:
            try: shutil.copy(img, final_dir)
            except Exception: pass
        with open(os.path.join(self.session_dir, "final_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.llm_output, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.session_dir, "sft_trajectory.json"), "w", encoding="utf-8") as f:
            json.dump(self.sft, f, ensure_ascii=False, indent=4)

    # ── 裁剪历史（与 Session._trim_history_for_next_query 一致）────────────
    def _trim_history(self):
        hist = getattr(self.bot, "history", None)
        if not hist:
            return
        system_msgs = [m for m in hist if isinstance(m, dict) and m.get("role") == "system"]
        scene_str = json.dumps(self.llm_output, ensure_ascii=False, indent=2)
        recap = (
            f"[上一轮回顾] 用户需求：{self._current_query}\n\n"
            f"当前场景元件信息：\n{scene_str}"
        )
        new_hist = list(system_msgs)
        new_hist.append({"role": "user", "content": recap})
        new_hist.append({"role": "assistant",
                         "content": self.last_response or "已完成本轮场景修改。"})
        self.bot.history = new_hist

    # ── reasoning 合并（与 Session._full_reasoning 一致）────────────────────
    def _full_reasoning(self, reasoning_returned: str) -> str:
        parts = []; seen = set()
        def _add(t):
            if isinstance(t, str):
                t = t.strip()
                if t and t not in seen: seen.add(t); parts.append(t)
        _add(reasoning_returned)
        last = self.bot.history[-1] if getattr(self.bot, "history", None) else None
        if isinstance(last, dict) and last.get("role") == "assistant":
            _add(last.get("reasoning_content"))
            c = last.get("content")
            if isinstance(c, str): _add(c)
        return "\n\n".join(parts)

    # ── /clear：重置状态，保留数据目录 ──────────────────────────────────────
    def clear(self):
        self.bot.reset()
        self._load_data()         # 重新加载初始场景
        self.turn = 0
        self.started = False
        self.last_response = ""
        self.sft = {"system_instruction": self.sys_prompt, "task_setting": "refine",
                    "data_dir": self.data_dir, "conversations": []}
        self.viewer.reset()
        self._show_initial()

    # ── /compact ────────────────────────────────────────────────────────────
    def compact(self):
        self._trim_history()
        return f"[Refine] 历史已压缩，当前轮数: {self.turn}"

    # ── /model：切换模型 ─────────────────────────────────────────────────────
    def switch_model(self, name: str):
        self.model_key, self.model_name, self.bot = models.build_bot(
            name, self.sys_prompt, TOOLS_SCHEMA_REFINE,
            on_delta=self._on_delta,
        )
        self.bot.reset()
        if self.started:
            self._trim_history()   # 承接当前场景状态
        return self.model_key, self.model_name
