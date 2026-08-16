
import subprocess
import signal
import sys
import faulthandler
import tempfile
import shutil
import glob
import hashlib
import json
import asyncio
import logging
import math
import os
import random
import re
from typing import Any, Optional
from uuid import uuid4
from enum import Enum
from copy import deepcopy

# 所有 verifier 模块统一从仓库根目录的 verifier/ 加载
# __file__ 在 <repo-root>/verl/verl/experimental/agent_loop/
# 上 4 级到仓库根目录，再进入 verifier / utils
_VIBE_VERIFIER_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "../../../..",
    "verifier"
))
# utils 工具模块（component_info_builder、llm 等）
_VIBE_UTILS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "../../../..",
    "utils"
))

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentState,
    AgentData,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ========= Hang 诊断：SIGUSR1 触发的全线程栈 dump =========
# Ray AgentLoopWorker 是独立 python 进程，不会 import main_ppo，所以这里
# 单独注册一次。对任一 AgentLoopWorker 的 pid 跑 `kill -USR1 <pid>`，它会
# 把所有线程（主线程 + asyncio default executor 线程 + Ray worker 线程）
# 的 python 栈 dump 到 stderr——rl.log 里能看到。
try:
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(
            signal.SIGUSR1,
            file=sys.stderr,
            all_threads=True,
            chain=False,
        )
except Exception:
    pass


# ==================== v2 辅助函数 ====================

def _extract_reasoning_from_text(text: str) -> str:
    """从模型原始输出中提取思考文本。

    优先从 <think>...</think> 提取；如果没有 <think> 标签，
    则取 <tool_call> 之前的所有文本作为 reasoning（模型经常不加 <think> 标签
    但在 tool_call 前输出思考/规划内容）。
    """
    # 优先：有 <think> 标签
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 兜底：取 <tool_call> 之前的文本
    tc_idx = text.find('<tool_call>')
    if tc_idx > 0:
        return text[:tc_idx].strip()

    # 都没有就返回空
    return ""


def _extract_content_after_think(text: str) -> str:
    """提取 </think> 之后、<tool_call> 之前的纯文本内容（最后轮的最终回复）。"""
    # 去除 <think>...</think>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 去除 <tool_call>...</tool_call>
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _build_v2_assistant_message(
    response_text: str,
    tool_calls: list[FunctionCall],
    is_final_turn: bool,
) -> dict:
    """
    根据模型原始输出和解析到的 tool_calls 构建 v2 格式的 assistant 消息。

    与 SFT v2 数据格式完全对齐：
    - 中间轮（有 tool_calls）: content="" , reasoning_content="思考", tool_calls=[...]
    - 最后轮（无 tool_calls）: content="最终回复", reasoning_content="思考", tool_calls=[]
    """
    reasoning_text = _extract_reasoning_from_text(response_text)

    if tool_calls:
        # 中间轮：有工具调用
        openai_tool_calls = []
        for i, tc in enumerate(tool_calls):
            openai_tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,  # 已经是 JSON string
                }
            })
        return {
            "role": "assistant",
            "content": "",                       # v2: 中间轮 content 为空字符串
            "reasoning_content": reasoning_text,
            "tool_calls": openai_tool_calls,
        }
    else:
        # 最后轮：无工具调用
        final_content = _extract_content_after_think(response_text)
        if not final_content:
            final_content = response_text.strip()
        return {
            "role": "assistant",
            "content": final_content,            # v2: 最后轮 content 为最终回复
            "reasoning_content": reasoning_text,
            "tool_calls": [],
        }


class MapGenAgentData(AgentData):
    """扩展AgentData以支持地图生成特有数据"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 地图生成特有字段
        self.current_map: dict = {}
        self.init_map: dict = {}
        self.component_info: dict = {}
        self.user_query: dict = {}
        self.scatter_cache: dict = {}  # 撒点缓存，保证同一样本多轮交互中散布生成结果一致
        self.camera_params: dict = {}  # 固定相机参数 {cam_pos, cam_target, lens}，优先于动态计算
        # === verified/unverified 分流所需字段 ===
        self.verifier_type: str = "unverified"
        self.query_tag: str = ""
        self.query_type: str = ""
        self.verification_criteria: list = []
        self.render_history: list = []
        self.verify_result: Optional[dict] = None

        # === 逐轮 criteria 评估（anti-hacking 效率折扣） ===
        # 每轮 tool 执行后对 verified query 做 criteria 检查
        # per_turn_criteria_results[i] = [bool, bool, ...] 第 i+1 轮后每个 criterion 是否 pass
        self.per_turn_criteria_results: list[list[bool]] = []
        
        # === 逐轮渲染图记录 ===
        # turn_images[i] = [PIL.Image, ...] 第 i+1 轮（PROCESSING_TOOLS 轮）渲染的图片
        self.turn_images: list[list] = []

        # === 逐轮 retrieve 结果记录 ===
        # turn_retrieve_results[i] = {"entity_name": str, "results": [dict, ...]}
        self.turn_retrieve_results: list[list[dict]] = []
        
        # ===== 监控统计字段 =====
        # tool_call 解析统计
        self.tool_call_parse_attempts: int = 0    # 模型输出中尝试解析 tool_call 的总次数
        self.tool_call_parse_successes: int = 0   # 成功解析为合法 tool_call 的次数
        self.tool_call_parse_failures: int = 0    # 解析失败的次数
        
        # PCG sandbox 调用统计（总计，含转换失败，保持向后兼容）
        self.sandbox_call_attempts: int = 0       # sandbox 调用总次数
        self.sandbox_call_successes: int = 0      # sandbox 调用成功次数
        self.sandbox_call_failures: int = 0       # sandbox 调用失败次数

        # PCG 工具解析统计（地图格式→actors 转换阶段，即模型输出的地图能否被解析）
        self.pcg_convert_attempts: int = 0        # 尝试转换次数（= sandbox_call_attempts）
        self.pcg_convert_failures: int = 0        # 转换失败次数（"转换后无有效actors"等）

        # PCG 渲染服务调用统计（仅转换成功后实际发起的渲染服务调用）
        self.pcg_render_attempts: int = 0         # 渲染服务调用次数
        self.pcg_render_successes: int = 0        # 渲染服务返回图片成功次数
        self.pcg_render_failures: int = 0         # 渲染服务失败次数（超时/空图等）
        
        # Verifier 调用统计
        self.verifier_call_success: bool = False  # verifier 是否调用成功
        
        # 失败原因记录 (每轮的详细记录)
        self.turn_details: list[dict] = []        # 每轮的详细信息
        self.failed_reasons: list[dict] = []      # 所有失败原因列表
        
        # PCG sandbox 日志
        self.sandbox_logs: list[dict] = []        # sandbox 调用的详细日志


@register("map_gen_agent")
class MapGenAgentLoop(ToolAgentLoop):
    """
    地图生成专用Agent Loop

    主要特点：
    1. 每轮工具调用后自动进行PCG渲染
    2. 维护地图状态（current_map）
    3. 任务结束时调用verify获取reward
    4. 支持多轮交互（最多8轮）
    """

    # ===== Per-event-loop PCG 并发限流 =====
    # 每个样本各自实例化一个 MapGenAgentLoop，但 Ray 的 AgentLoopWorker 在同一
    # python 进程内可能有多个 event loop（每个 worker 一个）。原先把 Semaphore
    # 做成类属性（绑定到第一个创建它的 event loop）会导致跨 loop 使用时
    # 永远等不到 release（典型死锁现象：PCG tool 调用成功后下一步 render 永远 hang）。
    # 现在改成 "per event loop" 字典：loop id -> Semaphore，每个 loop 独立限流。
    _pcg_semaphores: dict = {}          # {loop_id: asyncio.Semaphore}
    _pcg_max_concurrency: int = 16      # 默认最大并发 PCG 请求数

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # PCG渲染配置
        self.pcg_server_url = self.rollout_config.multi_turn.get(
            "pcg_server_url", "http://localhost:8080"
        )
        self.auto_render = self.rollout_config.multi_turn.get("auto_render", True)
        self.enable_verify = self.rollout_config.multi_turn.get("enable_verify", True)

        # pcg_request_batch.py 脚本路径配置
        # 优先从配置读取绝对路径；否则自动从当前文件位置向上推导项目根目录
        default_script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "pcg_request_batch.py"
        )
        self.pcg_script_path = self.rollout_config.multi_turn.get("pcg_script_path", default_script_path)
        if not os.path.isfile(self.pcg_script_path):
            logger.warning(
                f"pcg_request_batch.py not found at '{self.pcg_script_path}'. "
                f"PCG rendering will fail. Please set 'pcg_script_path' in multi_turn config."
            )

        # 单轮生成 token 上限（不影响多轮总预算 response_length）
        self.per_turn_max_tokens = self.rollout_config.multi_turn.per_turn_max_tokens

        # 日志目录配置
        self.log_dir = self.rollout_config.multi_turn.get("log_dir", "/tmp/map_gen_rl_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # ===== Anti-Hacking: 效率折扣参数 =====
        self.reward_efficiency_alpha = float(
            self.rollout_config.multi_turn.get("reward_efficiency_alpha", 0.0)
        )
        self.reward_efficiency_beta = float(
            self.rollout_config.multi_turn.get("reward_efficiency_beta", 0.0)
        )
        max_turns_by_type_str = str(
            self.rollout_config.multi_turn.get("max_turns_by_type", "") or ""
        )
        self.max_turns_by_type = self._parse_max_turns_by_type(max_turns_by_type_str)
        if self.reward_efficiency_alpha > 0 or self.reward_efficiency_beta > 0:
            logger.info(
                f"[Anti-Hacking] efficiency_alpha={self.reward_efficiency_alpha}, "
                f"efficiency_beta={self.reward_efficiency_beta}, "
                f"max_turns_by_type={self.max_turns_by_type}"
            )

        # 预加载 verified_verifier（逐轮评估需要）
        self._verified_verifier_fn = None
        if self.reward_efficiency_alpha > 0 or self.reward_efficiency_beta > 0:
            try:
                import sys as _sys
                verifier_dir = _VIBE_VERIFIER_DIR
                if verifier_dir not in _sys.path:
                    _sys.path.insert(0, verifier_dir)
                from verified_verifier import verify_single_criterion
                self._verified_verifier_fn = verify_single_criterion
                logger.info("[Anti-Hacking] verified_verifier loaded for per-turn evaluation")
            except ImportError as e:
                logger.warning(f"[Anti-Hacking] Cannot import verified_verifier: {e}. "
                               f"Per-turn evaluation disabled.")

        # ===== Anti-Hacking: verified 严格作用域校验（越界改动即清零）=====
        # 优先级：multi_turn config > env VERIFIED_STRICT_SCOPE > 默认开启。
        _strict_cfg = self.rollout_config.multi_turn.get("verified_strict_scope", None)
        if _strict_cfg is None:
            _strict_cfg = os.environ.get("VERIFIED_STRICT_SCOPE", "1")
        self.verified_strict_scope = str(_strict_cfg).lower() not in ("0", "false", "off")
        self._scope_violation_fn = None
        if self.verified_strict_scope:
            try:
                import sys as _sys
                verifier_dir = _VIBE_VERIFIER_DIR
                if verifier_dir not in _sys.path:
                    _sys.path.insert(0, verifier_dir)
                from verified_verifier import compute_scope_violation
                self._scope_violation_fn = compute_scope_violation
                logger.info("[Anti-Hacking] verified 严格作用域校验已启用（越界改动即清零）")
            except ImportError as e:
                logger.warning(f"[Anti-Hacking] 无法导入 compute_scope_violation: {e}，作用域校验禁用")
                self.verified_strict_scope = False

        # ===== PCG 并发限流 =====
        # 从配置读取，默认 16
        self._pcg_concurrency = self.rollout_config.multi_turn.get(
            "pcg_max_concurrency", MapGenAgentLoop._pcg_max_concurrency
        )

        # ===== PCG 渲染模式 =====
        # "gradio" = 新 Gradio 服务直调, "legacy" = 老 subprocess + HTTP 服务
        self.pcg_mode = self.rollout_config.multi_turn.get("pcg_mode", "gradio")
        self.pcg_gradio_server = self.rollout_config.multi_turn.get(
            "pcg_gradio_server",
            os.environ.get("PCG_GRADIO_SERVER", "http://localhost:8080"),
        )
        self._gradio_client = None  # 懒初始化，进程级复用
        logger.info(
            f"[PCG Config] mode={self.pcg_mode}, "
            f"gradio_server={self.pcg_gradio_server}, "
            f"legacy_server={self.pcg_server_url}, "
            f"concurrency={self._pcg_concurrency}"
        )

        # ===== generate 任务：预加载 PCG item_infos 白名单（用于动态重建 component_info）=====
        # generate 任务的 component_info 初始为空 {}，需在每轮 add 后从 item_infos 重建，
        # 才能让 _convert_to_actors 找到 comp_name 对应的渲染元数据（box/typeId/rot 等）。
        # 与 main_2_v4.py 第 886 行 enrich_component_info_for_generate 的逻辑完全对齐。
        self._pcg_item_infos = {}
        self._pcg_whitelist_set = None   # None = 不过滤；set() = 全过滤；set(type_ids) = 白名单
        try:
            import sys as _sys
            if _VIBE_UTILS_DIR not in _sys.path:
                _sys.path.insert(0, _VIBE_UTILS_DIR)
            from component_info_builder import load_item_infos as _load_item_infos
            # 优先从 config 指定路径加载（作为超参数传入），再回退到内置默认路径
            _config_path = self.rollout_config.multi_turn.get("pcg_item_infos_path", "")
            _item_infos_candidates = []
            if _config_path:
                _item_infos_candidates.append(_config_path)
            _item_infos_candidates += [
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "..",
                             "render_in_blender", "assets", "item_infos.json"),
            ]
            for _p in _item_infos_candidates:
                if os.path.exists(_p):
                    self._pcg_item_infos = _load_item_infos(path=_p)
                    logger.info(f"[Generate] item_infos 加载成功: {len(self._pcg_item_infos)} 条, path={_p}")
                    break
            else:
                # 都找不到时走 component_info_builder 的默认路径
                self._pcg_item_infos = _load_item_infos()
                logger.info(f"[Generate] item_infos 加载成功(默认路径): {len(self._pcg_item_infos)} 条")
            # item_infos 的 key 即为 PCG 白名单 type_id 集合（与 main_2_v4._PCG_WHITELIST 对齐）
            if self._pcg_item_infos:
                self._pcg_whitelist_set = set(self._pcg_item_infos.keys())
                logger.info(f"[Generate] PCG 白名单构建完成: {len(self._pcg_whitelist_set)} 个 type_id")
        except Exception as _e:
            logger.warning(f"[Generate] item_infos 加载失败: {_e}，generate 任务渲染将依赖已有 component_info")

    @staticmethod
    def _parse_max_turns_by_type(config_str: str) -> dict[str, int]:
        """解析 'type1:3,type2:4,type3:5,type4:6' → {'type1': 3, 'type2': 4, ...}"""
        result = {}
        if not config_str or not config_str.strip():
            return result
        # Hydra 传入时可能带单引号，strip 掉
        config_str = config_str.strip().strip("'\"")
        if not config_str:
            return result
        for pair in config_str.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            key, val = pair.split(":", 1)
            try:
                result[key.strip()] = int(val.strip())
            except ValueError:
                logger.warning(f"[Anti-Hacking] Invalid max_turns_by_type pair: '{pair}'")
        return result

    def _evaluate_criteria_at_current_map(
        self, agent_data: 'MapGenAgentData'
    ) -> list[bool]:
        """对 verified query 的当前地图状态逐 criterion 评估。

        Returns:
            每个 criterion 的 pass/fail 列表，或空列表（非 verified / 无 verifier）
        """
        if self._verified_verifier_fn is None:
            return []
        if agent_data.verifier_type != "verified":
            return []
        verification_criteria = agent_data.verification_criteria
        if not verification_criteria:
            return []

        try:
            init_elems = self._extract_fixed_elements(agent_data.init_map)
            current_elems = self._extract_fixed_elements(agent_data.current_map)
            results = []
            for vc in verification_criteria:
                result = self._verified_verifier_fn(vc, init_elems, current_elems)
                results.append(bool(result.get("pass", False)))
            return results
        except Exception as e:
            logger.warning(f"[Anti-Hacking] Per-turn criteria eval failed: {e}")
            return []

    def _compute_efficiency_reward(
        self,
        verify_reward: float,
        per_turn_criteria_results: list[list[bool]],
        total_interaction_turns: int,
        n_criteria: int,
    ) -> tuple[float, dict]:
        """计算带效率折扣的 reward（仅 verified query）。

        两个折扣系数（目的：惩罚"本可一次改对却改多次、导致地图改动过大"）:
          α (alpha, reward_efficiency_alpha): 首次做对的轮次折扣。
            越晚首次全部做对，折扣越狠: × (1 - α)^(first_solve_turn - 1)。
            first_solve_turn=1（首轮即全对）→ 不折扣。
          β (beta, reward_efficiency_beta): 多余轮次惩罚。
            达到最佳结果后仍继续调用工具刷轮 → × (1 - β)^wasted_turns。

        计算流程:
          1. 计算每轮的 criteria pass count
          2. best_turn = pass count 达到最大值的最早轮次
          3. first_solve_turn = pass count 首次达到 n_criteria（全部做对）的轮次；
             未全部做对则不施加 α 折扣
          4. wasted_turns = best_turn 之后仍调用工具但无提升的轮次
          5. final = verify_reward × (1-α)^(first_solve-1) × (1-β)^wasted_turns

        Returns:
            (efficiency_reward, metrics_dict)
        """
        alpha = self.reward_efficiency_alpha
        beta = self.reward_efficiency_beta
        max_turns = self.max_assistant_turns or 8

        metrics = {
            "efficiency_alpha": alpha,
            "efficiency_beta": beta,
            "original_verify_reward": verify_reward,
        }

        if n_criteria <= 0 or not per_turn_criteria_results:
            metrics["efficiency_applied"] = False
            return verify_reward, metrics

        # 1. 计算每轮的 pass count
        per_turn_pass_counts = [sum(turn_results) for turn_results in per_turn_criteria_results]

        # 2. 找"最佳结果轮"：pass count 达到最大值的最早轮次 (1-indexed)
        best_pass_count = max(per_turn_pass_counts)
        best_turn = next(i + 1 for i, c in enumerate(per_turn_pass_counts) if c == best_pass_count)

        # 3. 首次全部做对的轮次（1-indexed）；未全对则为 None
        first_solve_turn = next(
            (i + 1 for i, c in enumerate(per_turn_pass_counts) if c >= n_criteria),
            None,
        )

        # 4. 计算浪费轮
        n_tool_turns = len(per_turn_criteria_results)  # 实际调用工具的轮次数

        # 检查模型是否"干净退出"：最后一个 assistant turn 没有调用工具
        total_assistant_turns = total_interaction_turns - 1  # 减去初始 prompt
        clean_exit = (total_assistant_turns > n_tool_turns)

        # wasted_turns = best_turn 之后还继续调用工具的轮次数
        wasted_turns = max(0, n_tool_turns - best_turn)

        # 5a. α 折扣：首次全部做对的轮次越晚，折扣越狠
        if first_solve_turn is not None and first_solve_turn > 1 and alpha > 0:
            alpha_factor = (1.0 - alpha) ** (first_solve_turn - 1)
        else:
            alpha_factor = 1.0

        # 5b. β 折扣：每浪费一轮乘以 (1 - β)
        #    beta=0.15 时: 1轮→85%, 2轮→72%, 3轮→61%, 5轮→44%
        if wasted_turns == 0 or beta <= 0:
            beta_factor = 1.0
        else:
            beta_factor = (1.0 - beta) ** wasted_turns

        discount_factor = alpha_factor * beta_factor
        final_reward = verify_reward * discount_factor

        metrics.update({
            "efficiency_applied": True,
            "efficiency_reward": final_reward,
            "best_pass_count": best_pass_count,
            "best_turn": best_turn,
            "first_solve_turn": first_solve_turn,
            "n_criteria": n_criteria,
            "n_tool_turns": n_tool_turns,
            "clean_exit": clean_exit,
            "wasted_turns": wasted_turns,
            "alpha_factor": alpha_factor,
            "beta_factor": beta_factor,
            "discount_factor": discount_factor,
            "per_turn_pass_counts": per_turn_pass_counts,
        })

        return final_reward, metrics

    def _get_pcg_semaphore(self) -> asyncio.Semaphore:
        """按当前 event loop 取（或创建）对应的 PCG Semaphore。

        跨 event loop 共用同一个 asyncio.Semaphore 会 hang（其内部的 Future 绑定
        在某一个 loop 上；另一个 loop 的 task await 它时 waiter 永远不会被唤醒）。
        这里以 running loop 的 id 作为 key，保证每个 loop 拿到自己那份。
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        sem = MapGenAgentLoop._pcg_semaphores.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(self._pcg_concurrency)
            MapGenAgentLoop._pcg_semaphores[loop_id] = sem
            logger.info(
                f"[PCG Semaphore] created new sem for loop_id={loop_id}, "
                f"max_concurrency={self._pcg_concurrency}, "
                f"total_loops_tracked={len(MapGenAgentLoop._pcg_semaphores)}"
            )
        return sem
    
    def _serialize_message_content(self, content):
        """序列化 message content，处理不可JSON序列化的对象（如PIL图片）"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            serialized = []
            for item in content:
                if isinstance(item, dict):
                    new_item = {}
                    for k, v in item.items():
                        if isinstance(v, Image.Image):
                            new_item[k] = f"<PIL.Image mode={v.mode} size={v.size}>"
                        else:
                            new_item[k] = v
                    serialized.append(new_item)
                else:
                    serialized.append(item)
            return serialized
        # fallback: 转字符串
        return str(content)

    def _serialize_v2_message(self, msg: dict) -> dict:
        """序列化单条消息为 JSON 可保存的格式，保留 v2 结构化字段。"""
        result = {"role": msg.get("role")}
        result["content"] = self._serialize_message_content(msg.get("content", ""))
        # v2 结构化字段
        if "reasoning_content" in msg:
            result["reasoning_content"] = msg["reasoning_content"]
        if "tool_calls" in msg:
            result["tool_calls"] = msg["tool_calls"]
        return result
    
    @staticmethod
    def _euler_to_quaternion(euler):
        """欧拉角转四元数"""
        rx, ry, rz = math.radians(euler[0]), math.radians(euler[1]), math.radians(euler[2])
        sx, cx = math.sin(rx / 2), math.cos(rx / 2)
        sy, cy = math.sin(ry / 2), math.cos(ry / 2)
        sz, cz = math.sin(rz / 2), math.cos(rz / 2)
        x = sx * cy * cz - cx * sy * sz
        y = cx * sy * cz + sx * cy * sz
        z = cx * cy * sz - sx * sy * cz
        w = cx * cy * cz + sx * sy * sz
        return [round(i, 4) for i in [x, y, z, w]]

    @staticmethod
    def _generate_loose_distribution(input_rule):
        """程序化散布生成（Best-Candidate 算法）"""
        config = {
            "name": input_rule["name"],
            "pos_range": {
                "x": [input_rule["pos"][0][0], input_rule["pos"][1][0]],
                "y": [input_rule["pos"][0][1], input_rule["pos"][1][1]],
                "z": input_rule["pos"][0][2]
            },
            "extend_range": {
                "x": [input_rule["Extend"][0][0], input_rule["Extend"][1][0]],
                "y": [input_rule["Extend"][0][1], input_rule["Extend"][1][1]],
                "z": [input_rule["Extend"][0][2], input_rule["Extend"][1][2]]
            },
            "interval_range": {
                "x": [input_rule["Interval"][0][0], input_rule["Interval"][1][0]],
                "y": [input_rule["Interval"][0][1], input_rule["Interval"][1][1]],
            },
            "num": input_rule["num"]
        }
        results = []
        candidates_per_attempt = 15
        max_attempts = config["num"] * 200
        total_attempts = 0

        def get_random_candidate():
            ext = [
                round(random.uniform(*config["extend_range"]["x"]), 3),
                round(random.uniform(*config["extend_range"]["y"]), 3),
                round(random.uniform(*config["extend_range"]["z"]), 3)
            ]
            p = [
                round(random.uniform(*config["pos_range"]["x"]), 3),
                round(random.uniform(*config["pos_range"]["y"]), 3),
                config["pos_range"]["z"]
            ]
            return {"pos": p, "Extend": ext}

        while len(results) < config["num"]:
            total_attempts += 1
            if total_attempts > max_attempts:
                break
            best_candidate = None
            max_dist = -1
            current_samples = 1 if len(results) == 0 else candidates_per_attempt
            for _ in range(current_samples):
                candidate = get_random_candidate()
                pos, extend = candidate["pos"], candidate["Extend"]
                in_bound = (
                    config["pos_range"]["x"][0] + extend[0] / 2 <= pos[0] <= config["pos_range"]["x"][1] - extend[0] / 2
                    and config["pos_range"]["y"][0] + extend[1] / 2 <= pos[1] <= config["pos_range"]["y"][1] - extend[1] / 2
                )
                if not in_bound:
                    continue
                if len(results) == 0:
                    best_candidate = candidate
                    break
                min_d = float('inf')
                for existing in results:
                    d = math.sqrt((pos[0] - existing["pos"][0]) ** 2 + (pos[1] - existing["pos"][1]) ** 2)
                    if d < min_d:
                        min_d = d
                if min_d > max_dist:
                    max_dist = min_d
                    best_candidate = candidate
            if best_candidate is None:
                continue
            fp, fe = best_candidate["pos"], best_candidate["Extend"]
            overlap = False
            for existing in results:
                dx = abs(fp[0] - existing["pos"][0])
                dy = abs(fp[1] - existing["pos"][1])
                ix = random.uniform(*config["interval_range"]["x"])
                iy = random.uniform(*config["interval_range"]["y"])
                if dx < (fe[0] / 2 + existing["Extend"][0] / 2) + ix or dy < (fe[1] / 2 + existing["Extend"][1] / 2) + iy:
                    overlap = True
            if not overlap:
                results.append({"name": config["name"], "pos": fp, "Extend": fe})
        return results

    @staticmethod
    def _make_scatter_cache_key(v2: dict) -> str:
        """生成撒点缓存键，与 data_process.py 中 _make_scatter_cache_key 逻辑保持一致"""
        key_data = json.dumps({
            "name": v2.get("name"),
            "pos": v2.get("pos"),
            "Extend": v2.get("Extend"),
            "Interval": v2.get("Interval"),
            "num": v2.get("num"),
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_data.encode()).hexdigest()

    @staticmethod
    def _build_scatter_name_num_index(scatter_cache: dict) -> dict:
        """把预构建 scatter_cache 按 (撒点组名, 实例数) 重新索引。

        与 data_process._build_scatter_name_num_index 逻辑保持一致。历史上写 cache
        用的 hash key 与现在算的不一致（键函数改过），导致 100% miss、每个 rollout
        重新随机撒点。改用 (name, len) 索引即可稳定命中预构建结果，使训练看到的世界
        与 init_image / verifier 参考世界一致、且跨 rollout 可复现。
        """
        idx = {}
        if not scatter_cache:
            return idx
        for instances in scatter_cache.values():
            if not instances:
                continue
            key = (instances[0].get("name"), len(instances))
            if key not in idx:
                idx[key] = instances
        return idx


    @staticmethod
    def _update_component_info(llm_plan_sample, component_info_sample):
        """将 LLM 的语义信息与 component_info 的渲染属性合并"""
        box_old = component_info_sample["box"]
        box_new = [x * 100 for x in llm_plan_sample["Extend"]]
        sca = [n / o for o, n in zip(box_old, box_new)]
        rot = component_info_sample["rot"]
        if "rotate" in llm_plan_sample:
            rot = MapGenAgentLoop._euler_to_quaternion(llm_plan_sample["rotate"])
        return dict(
            c=component_info_sample["c"],
            name=component_info_sample["name"],
            pos=llm_plan_sample["pos"],
            rot=rot,
            gname=component_info_sample["gname"],
            id=component_info_sample["id"],
            m=component_info_sample["m"],
            col=component_info_sample.get("col", []),
            typeId=component_info_sample["typeId"],
            sca=sca,
        )

    def _convert_to_actors(self, current_map: dict, component_info: dict, scatter_cache: dict = None) -> tuple[list, str]:
        """
        将结构化地图格式转换为 actors 列表。

        返回的 actors pos 单位与 LLM 输出一致（cm），
        可直接传给 Gradio 渲染服务或 _compute_camera_params。

        Args:
            scatter_cache: 撒点缓存字典，保证同一样本多轮交互中
                           撒点类元件（带 num 字段）的随机结果一致。

        Returns:
            (actors, error_msg): actors 为列表，error_msg 为空字符串表示成功
        """
        try:
            actors = []
            # 预构建 cache 的 (name, num) 索引：优先按此命中，避免历史 hash key 失配。
            scatter_name_num_idx = self._build_scatter_name_num_index(scatter_cache)
            for k, v in current_map.items():
                if not isinstance(v, dict):
                    continue
                for k1, v1 in v.items():
                    if not isinstance(v1, list):
                        continue
                    for v2 in v1:
                        if not isinstance(v2, dict) or "name" not in v2:
                            continue
                        comp_name = v2["name"]
                        if comp_name not in component_info:
                            # fallback: name 없으면 type_id 로 _pcg_item_infos 에서 직접 빌드
                            tid = str(v2.get("type_id") or v2.get("typeId") or "")
                            if tid and self._pcg_item_infos and tid in self._pcg_item_infos:
                                from component_info_builder import build_component_info_entry as _bce
                                built = _bce(tid, name=comp_name, item_infos=self._pcg_item_infos)
                                if built:
                                    component_info[comp_name] = built
                                    logger.info(f"[PCG Convert] Built component_info for '{comp_name}' (type_id={tid})")
                                else:
                                    logger.warning(f"[PCG Convert] type_id={tid} not in item_infos, skipping '{comp_name}'")
                                    continue
                            else:
                                logger.warning(f"[PCG Convert] Component '{comp_name}' not found in component_info, skipping")
                                continue
                        comp_info = component_info[comp_name]

                        if "num" in v2:
                            # 区域散布生成（带缓存，保证多轮一致）。解析优先级：
                            #   1) 按 (name, num) 命中预构建 cache —— 与 init_image 一致；
                            #   2) 退回历史 hash key（兼容老路径写入的条目）；
                            #   3) 都未命中才重新随机生成，并同时写回两种键。
                            try:
                                nn_key = (v2.get("name"), v2.get("num"))
                                generated = scatter_name_num_idx.get(nn_key)
                                if generated is None:
                                    cache_key = self._make_scatter_cache_key(v2)
                                    if scatter_cache is not None and cache_key in scatter_cache:
                                        generated = scatter_cache[cache_key]
                                    else:
                                        generated = self._generate_loose_distribution(v2)
                                        if scatter_cache is not None:
                                            scatter_cache[cache_key] = generated
                                            scatter_name_num_idx[nn_key] = generated
                                for item in generated:
                                    actors.append(self._update_component_info(item, comp_info))
                            except Exception as e:
                                logger.warning(f"[PCG Convert] Failed to generate distribution for '{comp_name}': {e}")
                            continue
                        
                        # 兼容 Extend 为嵌套列表
                        extend = v2.get("Extend", [])
                        if isinstance(extend, list) and len(extend) > 0 and isinstance(extend[0], list):
                            v2_copy = dict(v2)
                            v2_copy["Extend"] = extend[0]
                        else:
                            v2_copy = v2
                        
                        try:
                            actors.append(self._update_component_info(v2_copy, comp_info))
                        except Exception as e:
                            logger.warning(f"[PCG Convert] Failed to convert '{comp_name}': {e}")
            
            # 清理嵌套列表
            keys_to_flatten = ["pos", "rot", "sca", "Extend"]
            for item in actors:
                for key in keys_to_flatten:
                    if key in item:
                        val = item[key]
                        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                            item[key] = val[0]
            
            if not actors:
                return [], "转换后无有效 actors"
            
            return actors, ""
        except Exception as e:
            return [], f"地图格式转换失败: {str(e)}"

    def _convert_to_pcg_format(self, current_map: dict, component_info: dict, scatter_cache: dict = None) -> tuple[list, str]:
        """
        将结构化地图格式转换为老 PCG 服务器可识别的格式（legacy 模式使用）。

        Returns:
            (pcg_json, error_msg): pcg_json 为老 PCG 格式的列表，error_msg 为空字符串表示成功
        """
        actors, error = self._convert_to_actors(current_map, component_info, scatter_cache)
        if error:
            return [], error
        
        # 包装为老 PCG 格式
        data = {"actors": actors}
        escaped_string_ascii = json.dumps(data, ensure_ascii=True)
        pcg_json = [
            {
                "Type": "PCGGenerate",
                "Param": {
                    "AIID": "398a907a-c40f-477c-b278-734e42af1df5",
                    "bNeedGroup": False,
                    "Info": escaped_string_ascii
                }
            }
        ]
        return pcg_json, ""

    # ===== Gradio PCG 渲染（新服务）=====

    def _get_gradio_client(self):
        """每次创建新的 Gradio client（独立 session_hash），带重试。

        独立 session 确保 PCG dispatcher 能将不同请求分发到不同 GPU worker，
        实现多卡并行渲染。进程级复用单个 client 会导致所有请求被 dispatcher
        的 session 粘性路由到同一个 worker，无法并行。

        重试机制：dispatcher 偶尔瞬时不可达（重启、连接池刷新等），
        重试 3 次避免因瞬时网络抖动导致 sample 失败。
        """
        import time as _time
        import httpx as _httpx
        from gradio_client import Client as GradioClient
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 设置足够长的 httpx 超时，防止渲染耗时长（30-120s）时被 httpx 默认超时断开
                client = GradioClient(
                    self.pcg_gradio_server,
                    verbose=False,
                    httpx_kwargs={"timeout": _httpx.Timeout(600.0, connect=30.0)},
                )
                return client
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)  # 3s, 6s
                    logger.warning(
                        f"[PCG Gradio] _get_gradio_client attempt {attempt+1} failed: "
                        f"{type(e).__name__}: {e}, retry in {wait}s..."
                    )
                    _time.sleep(wait)
                    continue
                raise

    @staticmethod
    def _compute_camera_params(actors: list) -> tuple[str, str]:
        """计算 5 视角相机参数（front/back/left/right/topdown）。

        与 main_2_v4.compute_camera_params 完全对齐：
        - 边界计算时把每个 actor 的 Extend 尺寸纳入，确保大型资产不出画面
        - 28° 仰角（与 SFT 蒸馏轨迹一致）
        - dist = max(span_xy * 0.6, 8.0)（不依赖 span_z）
        - topdown_z = cz + dist * 1.1 + span_z

        注意：接收的 actors pos 单位是厘米（_pcg_render_gradio 里已乘100），
        SCALE=0.01 将 cm 还原为 Blender 米。cam_pos/cam_target 输出单位为 Blender 米。
        """
        SCALE = 0.01  # cm → m

        def _is_scalar_pos(a):
            pos = a.get("pos")
            if not isinstance(pos, list) or len(pos) < 3:
                return False
            return not isinstance(pos[0], list)

        all_x, all_y, all_z = [], [], []
        for a in actors:
            if not _is_scalar_pos(a):
                continue
            px = a["pos"][0] * SCALE
            py = -a["pos"][1] * SCALE   # Y-flip for Blender
            pz = a["pos"][2] * SCALE

            # 用 Extend（actor 已是 cm）估算半径，确保大型资产边缘不超出画面
            ext = a.get("Extend") or a.get("col")
            if isinstance(ext, list) and len(ext) >= 2:
                try:
                    half = min(max(abs(float(ext[0])), abs(float(ext[1]))) * SCALE * 0.5, 8.0)
                except (TypeError, ValueError):
                    half = 2.0
            else:
                # 无 Extend 信息时用 sca 估算（与 main_2_v4 一致）
                sca = a.get("sca") or [1, 1, 1]
                if isinstance(sca, list) and len(sca) >= 3:
                    sca_avg = sum(abs(s) for s in sca[:3]) / 3
                else:
                    sca_avg = 1.0
                half = min(sca_avg * 10.0 * 0.5, 8.0)

            all_x.extend([px - half, px + half])
            all_y.extend([py - half, py + half])
            all_z.append(pz)

        if not all_x:
            cx, cy, cz = 0.0, 0.0, 0.0
            span_xy, span_z = 10.0, 0.1
        else:
            cx = (max(all_x) + min(all_x)) / 2
            cy = (max(all_y) + min(all_y)) / 2
            cz = (max(all_z) + min(all_z)) / 2 if all_z else 0.0
            span_xy = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 0.1)
            span_z = max(max(all_z) - min(all_z), 0.1) if all_z else 0.1

        # 视距：以 xy footprint 为主，最小 8m（与 main_2_v4 一致）
        dist = max(span_xy * 0.6, 8.0)
        elev_rad = math.radians(28)
        h_offset = dist * math.sin(elev_rad)
        d_horiz = dist * math.cos(elev_rad)
        cam_z = cz + h_offset + span_z * 0.3
        target_z = cz + span_z * 0.2
        topdown_z = cz + dist * 1.1 + span_z

        # 5 views: front, back, left, right, topdown
        cameras = [
            (f"{cx:.2f},{cy - d_horiz:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
            (f"{cx:.2f},{cy + d_horiz:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
            (f"{cx - d_horiz:.2f},{cy:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
            (f"{cx + d_horiz:.2f},{cy:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
            (f"{cx:.2f},{cy:.2f},{topdown_z:.2f}", f"{cx:.2f},{cy:.2f},{cz:.2f}"),
        ]

        cam_pos_str = "\n".join(c[0] for c in cameras)
        cam_target_str = "\n".join(c[1] for c in cameras)
        return cam_pos_str, cam_target_str

    @staticmethod
    def _extract_gradio_path(item):
        """从 Gradio 返回值提取文件路径（与 main_distill_v3.py extract_path 一致）"""
        if item is None:
            return None
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if "image" in item:
                img = item["image"]
                if isinstance(img, dict):
                    return img.get("path") or img.get("url")
                return img
            return item.get("path") or item.get("name") or item.get("url")
        if isinstance(item, (list, tuple)) and item:
            return MapGenAgentLoop._extract_gradio_path(item[0])
        return None

    def _load_gradio_images(self, images_raw) -> list:
        """提取 Gradio 返回图片，加载为 PIL Image 并 resize 到 1280x720"""
        images = []
        if not images_raw:
            logger.warning("[PCG Gradio] _load_gradio_images: images_raw is empty/None")
            return images
        items = images_raw if isinstance(images_raw, list) else [images_raw]
        for i, item in enumerate(items):
            path = self._extract_gradio_path(item)
            logger.debug(f"[PCG Gradio] _load_gradio_images[{i}]: item={str(item)[:200]}, extracted_path={path}")
            if path and os.path.exists(str(path)):
                img = Image.open(str(path)).convert("RGB")
                if img.size != (1280, 720):
                    img = img.resize((1280, 720), Image.Resampling.LANCZOS)
                images.append(img)
            else:
                logger.warning(f"[PCG Gradio] _load_gradio_images[{i}]: path not found: {path}")
        return images

    async def _pcg_render_gradio(
        self, current_map: dict, component_info: dict, turn: int = 0,
        scatter_cache: dict = None, camera_params: dict = None,
    ) -> tuple[list, str, dict]:
        """通过 Gradio Client 直接调用新 PCG 渲染服务。

        严格参照 main_distill_v3.py 的 gradio_render 函数实现。
        """
        sandbox_log = {
            "turn": turn,
            "success": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error_msg": "",
            "image_count": 0,
            "pcg_mode": "gradio",
            "convert_failed": False,  # 标记是否为地图格式转换失败（vs 渲染服务失败）
        }
        try:
            # Step 1: current_map → actors 列表
            actors, convert_error = self._convert_to_actors(
                current_map, component_info, scatter_cache
            )
            if convert_error:
                sandbox_log["error_msg"] = f"地图格式转换失败: {convert_error}"
                sandbox_log["convert_failed"] = True
                logger.error(f"[PCG Gradio] turn={turn} Step1 FAIL: {convert_error}")
                return [], sandbox_log["error_msg"], sandbox_log

            # Step 1.5: pos 米→厘米（与 main_distill_v3.py actors_meter_to_cm 对齐）
            # agent_loop 的 LLM 输出 pos 单位是米，Gradio 渲染服务期望厘米
            for actor in actors:
                pos = actor.get("pos")
                if isinstance(pos, list) and len(pos) > 0 and not isinstance(pos[0], list):
                    actor["pos"] = [p * 100 for p in pos]

            logger.info(
                f"[PCG Gradio] turn={turn} Step1 OK: {len(actors)} actors, "
                f"sample_pos_cm={actors[0].get('pos') if actors else 'N/A'}"
            )

            # Step 2: 相机参数（优先使用数据自带的固定参数，否则动态计算）
            if camera_params and camera_params.get("cam_pos") and camera_params.get("cam_target"):
                cam_pos_str = camera_params["cam_pos"]
                cam_target_str = camera_params["cam_target"]
                lens = camera_params.get("lens", 31)
                logger.info(
                    f"[PCG Gradio] turn={turn} Step2 camera: FIXED from data, "
                    f"lens={lens}, cam_pos(first)={cam_pos_str.split(chr(10))[0]}"
                )
            else:
                cam_pos_str, cam_target_str = self._compute_camera_params(actors)
                lens = 31
                logger.info(
                    f"[PCG Gradio] turn={turn} Step2 camera: COMPUTED from actors, "
                    f"cam_pos(first)={cam_pos_str.split(chr(10))[0]}, "
                    f"cam_target(first)={cam_target_str.split(chr(10))[0]}"
                )

            # Step 3: 构建新格式 JSON
            pcg_json = [{"actors": actors}]
            json_text = json.dumps(pcg_json, ensure_ascii=False)
            logger.info(
                f"[PCG Gradio] turn={turn} Step3 json_text: "
                f"len={len(json_text)}, first_200={json_text[:200]}"
            )

            # Step 4: 在线程池中调用 Gradio predict（阻塞调用需要 run_in_executor）
            # client 用完后必须 close()，否则服务端 session 永远不释放，随训练步数
            # 累积导致服务端资源耗尽、pcg_sandbox_success_rate 持续下降。
            client = self._get_gradio_client()
            logger.info(f"[PCG Gradio] turn={turn} Step4 calling predict on {self.pcg_gradio_server}...")

            import time as _time
            _t0 = _time.monotonic()

            def _predict_and_close():
                """执行 predict，无论成功/超时/异常都 close client 释放服务端 session。

                所有渲染参数必须以**位置参数**传入（gradio_client 4.16
                不接受按 component label 的 keyword args），与
                main_distill_v3.py 和 render_raw_data_images.py 一致。
                """
                try:
                    return client.predict(
                        "（不使用预设）",   # scene_preset（位置参数，非 keyword！）
                        None,                # json_file
                        json_text,           # json_text
                        "自定义",            # preset
                        cam_pos_str,         # cam_pos
                        cam_target_str,      # cam_target
                        lens,                # lens
                        "低质量 (快速预览)",  # quality
                        api_name="/render",
                    )
                finally:
                    try:
                        client.close()
                        logger.debug(f"[PCG Gradio] turn={turn} client closed")
                    except Exception as _ce:
                        logger.debug(f"[PCG Gradio] turn={turn} client.close() error (ignored): {_ce}")

            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _predict_and_close),
                timeout=300.0,
            )
            _elapsed = _time.monotonic() - _t0
            images_raw, blend_file, status = result

            logger.info(
                f"[PCG Gradio] turn={turn} Step4 predict returned in {_elapsed:.1f}s, "
                f"status={str(status)[:200]}, "
                f"images_raw type={type(images_raw).__name__}, "
                f"len={len(images_raw) if isinstance(images_raw, list) else 'N/A'}, "
                f"images_raw={str(images_raw)[:300]}"
            )

            # Step 5: 提取图片路径，加载为 PIL Image
            images = self._load_gradio_images(images_raw)
            if not images:
                sandbox_log["error_msg"] = "Gradio 渲染返回空图片"
                logger.error(
                    f"[PCG Gradio] turn={turn} Step5 FAIL: 0 images loaded. "
                    f"raw={str(images_raw)[:500]}"
                )
                return [], sandbox_log["error_msg"], sandbox_log

            logger.info(
                f"[PCG Gradio] turn={turn} Step5 OK: {len(images)} images loaded, "
                f"sizes={[img.size for img in images]}"
            )

            sandbox_log["success"] = True
            sandbox_log["image_count"] = len(images)
            sandbox_log["stdout"] = str(status)[:500]
            return images, "", sandbox_log

        except asyncio.TimeoutError:
            # asyncio.wait_for 超时时，executor 线程里的 _predict_and_close 仍在跑，
            # finally 块会在线程结束时自动执行 client.close()，session 最终会释放。
            sandbox_log["error_msg"] = "PCG渲染超时(300s)"
            logger.error(f"[PCG Gradio] turn={turn} TIMEOUT after 300s")
            return [], "PCG渲染超时(300s)", sandbox_log
        except Exception as e:
            sandbox_log["error_msg"] = str(e)
            logger.error(f"[PCG Gradio] turn={turn} EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
            return [], str(e), sandbox_log

    async def _pcg_render_with_retry(
        self, current_map: dict, component_info: dict, turn: int = 0,
        scatter_cache: dict = None, camera_params: dict = None,
        max_retries: int = 1,
    ) -> tuple[list[Image.Image], str, dict]:
        """带并发限流 + 超时重试的 PCG 渲染包装器。

        并发控制：通过 per-event-loop asyncio.Semaphore 限制同一 worker 进程内
        最大 PCG 并发数（默认 16），超出的请求排队等待。
        超时重试：仅对超时错误重试，等 5s 后重试 1 次（总共最多 2 次尝试）。
        """
        sem = self._get_pcg_semaphore()
        loop_id = id(asyncio.get_running_loop())
        request_tag = f"loop={loop_id} turn={turn} pid={os.getpid()}"
        for attempt in range(max_retries + 1):
            # Semaphore 诊断日志：acquire 前/acquired/release，配合 waiters 队列长度
            waiters_before = len(sem._waiters) if sem._waiters else 0
            logger.info(
                f"[PCG Sem Before] {request_tag} attempt={attempt} "
                f"locked={sem.locked()} waiters={waiters_before}"
            )
            sem_wait_t0 = asyncio.get_running_loop().time()
            async with sem:
                sem_wait_elapsed = asyncio.get_running_loop().time() - sem_wait_t0
                logger.info(
                    f"[PCG Sem Acquired] {request_tag} attempt={attempt} "
                    f"sem_wait_s={sem_wait_elapsed:.2f}"
                )
                render_t0 = asyncio.get_running_loop().time()
                if self.pcg_mode == "gradio":
                    images, error_msg, sandbox_log = await self._pcg_render_gradio(
                        current_map, component_info, turn, scatter_cache, camera_params
                    )
                else:
                    images, error_msg, sandbox_log = await self._pcg_render(
                        current_map, component_info, turn, scatter_cache
                    )
                render_elapsed = asyncio.get_running_loop().time() - render_t0
                logger.info(
                    f"[PCG Sem Release] {request_tag} attempt={attempt} "
                    f"render_s={render_elapsed:.2f} got_images={bool(images)} "
                    f"err={error_msg[:80] if error_msg else ''}"
                )
            if images or "超时" not in error_msg:
                # 成功或非超时错误，直接返回
                sandbox_log["retry_count"] = attempt
                return images, error_msg, sandbox_log
            if attempt < max_retries:
                logger.warning(
                    f"[PCG Retry] {request_tag} attempt {attempt + 1}/{max_retries + 1} "
                    f"timed out, retrying in 5s..."
                )
                await asyncio.sleep(5)
        # 所有重试均失败
        sandbox_log["retry_count"] = max_retries
        return images, error_msg, sandbox_log

    async def _pcg_render(self, current_map: dict, component_info: dict, turn: int = 0, scatter_cache: dict = None) -> tuple[list[Image.Image], str, dict]:
        """
        执行PCG渲染，设置30秒超时
        
        将当前地图渲染为5张视角图
        
        Returns:
            (images, error_msg, sandbox_log)
            sandbox_log 包含 stdout/stderr/returncode 等信息
        """
        sandbox_log = {
            "turn": turn,
            "success": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error_msg": "",
            "image_count": 0,
        }
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="pcg_render_")
            json_dir = os.path.join(temp_dir, "pcg_json")
            output_image_dir = os.path.join(temp_dir, "images")
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(output_image_dir, exist_ok=True)
            
            # 将结构化地图转换为 PCG 格式
            pcg_json, convert_error = self._convert_to_pcg_format(current_map, component_info, scatter_cache=scatter_cache)
            if convert_error:
                sandbox_log["error_msg"] = f"地图格式转换失败: {convert_error}"
                return [], f"地图格式转换失败: {convert_error}", sandbox_log
            
            # 保存PCG格式 JSON
            result_path = os.path.join(json_dir, "result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(pcg_json, f, indent=2, ensure_ascii=False)
            
            # 调用PCG服务器
            command = [
                "python", self.pcg_script_path,
                "--server", self.pcg_server_url,
                "--local_folder", json_dir,
                "--batch_mode", "cmd_folder",
                "--out_dir", output_image_dir,
                "--stream"
            ]
            
            # 以脚本所在目录作为工作目录
            pcg_script_dir = os.path.dirname(os.path.abspath(self.pcg_script_path))
            
            # 使用 asyncio.wait_for 包裹整个子进程生命周期（创建+执行）实现超时
            # start_new_session=True: 让子进程成为新 process group 的 leader，
            # 这样超时时可以 os.killpg 一次性杀掉整棵进程树（pcg_request_batch.py
            # 若 fork 了孙进程，单 proc.kill() 只能杀主子进程，孙进程残留会
            # 让 proc.wait() 永远等——这是上次 rollout hang 的怀疑根因之一）。
            proc = None
            try:
                async def _run_pcg_subprocess():
                    p = await asyncio.create_subprocess_exec(
                        *command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=pcg_script_dir,
                        start_new_session=True,
                    )
                    stdout, stderr = await p.communicate()
                    return p, stdout, stderr

                proc, stdout, stderr = await asyncio.wait_for(
                    _run_pcg_subprocess(),
                    timeout=60.0  # PCG渲染给60秒超时（30s超时率较高，影响训练）
                )

                # 记录 sandbox 日志
                sandbox_log["stdout"] = stdout.decode("utf-8", errors="replace")[:2000] if stdout else ""
                sandbox_log["stderr"] = stderr.decode("utf-8", errors="replace")[:2000] if stderr else ""
                sandbox_log["returncode"] = proc.returncode

                if proc.returncode != 0:
                    sandbox_log["error_msg"] = f"PCG服务返回非零退出码: {proc.returncode}"
                    return [], "PCG服务出小差~", sandbox_log

            except asyncio.TimeoutError:
                # 关键：超时时必须保证 proc.wait() 不会再挂——否则整个 rollout 死锁。
                # 步骤：
                #   1) 通过 process group 发 SIGKILL，杀掉主子进程 + 所有孙进程
                #   2) proc.wait() 外面再套一层 asyncio.wait_for，给 5s 硬上限
                #   3) 仍挂住也只是 leak 一个 zombie，event loop 可以继续
                logger.warning(
                    f"[PCG Subproc] timeout at turn={turn}, killing process group..."
                )
                if proc is not None:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError) as kill_err:
                        logger.warning(f"[PCG Subproc] killpg failed: {kill_err}, falling back to proc.kill()")
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    except Exception as kill_err:
                        logger.error(f"[PCG Subproc] unexpected killpg error: {kill_err}")

                    # 等子进程真正退出，但最多等 5 秒
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                        logger.info(f"[PCG Subproc] process group reaped at turn={turn}")
                    except asyncio.TimeoutError:
                        logger.error(
                            f"[PCG Subproc] proc.wait() still hung 5s after SIGKILL "
                            f"at turn={turn}, leaking pid={proc.pid}"
                        )
                    except Exception as wait_err:
                        logger.error(f"[PCG Subproc] proc.wait() error: {wait_err}")
                sandbox_log["error_msg"] = "PCG渲染超时(60s)"
                return [], "PCG渲染超时(60s)", sandbox_log
            
            # 获取生成的图片
            image_paths = sorted(glob.glob(os.path.join(output_image_dir, "*.jpg")))
            
            if not image_paths:
                sandbox_log["error_msg"] = "渲染失败：未生成图片"
                return [], "渲染失败：未生成图片", sandbox_log
            
            # 读取图片为PIL Image，统一 resize 到 1280×720 与 SFT 数据对齐
            images = []
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                if img.size != (1280, 720):
                    img = img.resize((1280, 720), Image.Resampling.LANCZOS)
                images.append(img)
            
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            sandbox_log["success"] = True
            sandbox_log["image_count"] = len(images)
            return images, "", sandbox_log
            
        except Exception as e:
            sandbox_log["error_msg"] = str(e)
            return [], str(e), sandbox_log
    
    async def _call_verify(self, agent_data: MapGenAgentData) -> dict:
        """
        调用 verify 获取 reward — 根据 verifier_type 自动路由。

        - verified:   rule-based 对比 verification_criteria（不需要 LLM，极快）
        - unverified: 保持原有 Qwen-VL LLM 评估（兼容旧流程）

        返回统一格式:
            {
                "verifier_type": str,
                "total_reward": float,
                "hard_reward": float,
                "soft_reward": float,
                # verified 专用
                "pass_count": int, "total_count": int, "criteria_results": list,
                # unverified 专用
                "hard_pass": bool, "hard_result": dict, "soft_result": dict,
                # 兼容旧接口
                "reward": float, "reason": str,
            }
        """
        verifier_type = getattr(agent_data, 'verifier_type', 'unverified')

        if verifier_type == "verified":
            return self._call_verify_verified(agent_data)
        elif verifier_type == "onestep_scene":
            # generate (from-scratch) 任务：H1-H5 rubric-based LLM 评估
            return await self._call_verify_onestep_scene(agent_data)
        else:
            # "unverified" 及其他旧数据类型：保持原有 Hard+Soft LLM 评估
            return await self._call_verify_unverified(agent_data)

    # ==================== Onestep Scene: H1-H5 Rubric-based LLM ====================

    @staticmethod
    def _extract_retrieve_and_usage_from_messages(messages: list) -> dict:
        """
        从 agent_data.messages（内存格式）中提取 H5 所需的"召回 vs 使用"信息。

        与 unverified_verifier_onestep_scene.extract_retrieve_and_usage() 逻辑对齐，
        但从内存 messages 读取，而非从磁盘 sft_trajectory.json 读取。

        messages 格式（v2）:
          - role=user / role=tool:  tool_response 文本，含 "资产检索结果" 的检索结果块
          - role=assistant:         tool_calls 列表（含 name/function/arguments）
        """
        import re as _re

        _RECALL_HEAD_RE = _re.compile(r"type_id=(\d+)\s+name=(\S+)\s+score=([\d.]+)")
        _INTENT_HEAD_RE = _re.compile(r"\[retrieve_assets\(([^)]*)\)\]")

        def _parse_retrieve_block(text: str) -> list:
            if "资产检索结果" not in text:
                return []
            intents = []
            parts = _re.split(r"(\[retrieve_assets\([^)]*\)\])", text)
            i = 1
            while i < len(parts):
                head = parts[i]
                body = parts[i + 1] if i + 1 < len(parts) else ""
                m = _INTENT_HEAD_RE.search(head)
                entity = m.group(1) if m else "?"
                recalled = []
                cand_blocks = _re.split(r"(?=type_id=\d+\s+name=)", body)
                for cb in cand_blocks:
                    head_m = _RECALL_HEAD_RE.search(cb)
                    if not head_m:
                        continue
                    desc_m = _re.search(r"description=(.*?)(?:\s*color=|\Z)", cb, flags=_re.DOTALL)
                    color_m = _re.search(r"color=(\[[^\]]*\])", cb)
                    recalled.append({
                        "type_id": head_m.group(1),
                        "name": head_m.group(2),
                        "score": head_m.group(3),
                        "description": (desc_m.group(1).strip()[:120] if desc_m else ""),
                        "color": (color_m.group(1) if color_m else ""),
                    })
                if recalled:
                    intents.append({"entity": entity, "recalled": recalled})
                i += 2
            return intents

        intents = []
        used = []
        seen_used = set()

        for msg in messages:
            role = msg.get("role", "")
            # tool / user 消息可能含 retrieve 结果
            if role in ("tool", "user"):
                content = msg.get("content", "") or ""
                if isinstance(content, list):
                    # 多模态格式：取所有 text 片段
                    text_parts = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                    content = text_parts
                if isinstance(content, str):
                    intents.extend(_parse_retrieve_block(content))

            elif role == "assistant":
                tcs = msg.get("tool_calls", []) or []
                for tc in tcs:
                    func = tc.get("function", {})
                    if func.get("name") != "add":
                        continue
                    args_raw = func.get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except Exception:
                            continue
                    else:
                        args = args_raw
                    md = args.get("modified_data")
                    if isinstance(md, dict):
                        md = [md]
                    if not isinstance(md, list):
                        continue
                    for item in md:
                        if not isinstance(item, dict):
                            continue
                        tid = str(item.get("type_id", ""))
                        name = item.get("name", "?")
                        key = (tid, name)
                        if key in seen_used:
                            continue
                        seen_used.add(key)
                        used.append({
                            "type_id": tid,
                            "name": name,
                            "reason": str(item.get("reason", ""))[:120],
                        })

        return {"intents": intents, "used": used}

    async def _call_verify_onestep_scene(self, agent_data: 'MapGenAgentData') -> dict:
        """
        onestep_scene (generate) 任务：H1-H5 Rubric-based LLM 评估。

        与 verify_mix_onestep_scene.py 逻辑对齐，但以内存数据调用，
        不读磁盘文件（final_map.json / sft_trajectory.json）。

        返回统一格式 dict（兼容 _build_detailed_verify_reason 和 map_gen_reward.py）：
          {
            "verifier_type": "onestep_scene",
            "total_reward": float,
            "hard_reward": float,
            "soft_reward": 0.0,
            "hard_pass": bool,
            "hard_result": { H1/H2/H3/H4/H5 ... },
            "soft_result": None,
            "reward": float,
            "reason": str,
          }
        """
        # === Step 1: 导入 onestep_scene verifier 模块 ===
        try:
            import sys as _sys
            verifier_dir = _VIBE_VERIFIER_DIR
            if verifier_dir not in _sys.path:
                _sys.path.insert(0, verifier_dir)

            from unverified_verifier_onestep_scene import (
                MODEL_TYPE_MAP as OS_MODEL_TYPE_MAP,
                call_hard_h12_scene,
                call_h3_scene,
                call_h5_scene,
                compute_onestep_reward,
                _has_any_element,
            )
            from unverified_verifier import (
                rule_based_collision_check,
                rule_based_height_check,
            )
            from prompts import (
                HARD_H12_SCENE_SYSTEM_PROMPT,
                H3_SCENE_SYSTEM_PROMPT,
                H5_SCENE_SYSTEM_PROMPT,
            )
            from unverified_verifier import format_agent_turns_text
        except ImportError as e:
            logger.warning(f"[OnestepScene Verify] 无法导入 verifier 模块 ({e})，回退到 unverified 流程")
            return await self._call_verify_unverified(agent_data)

        # === Step 2: 准备评估所需的内存数据 ===
        query_info = agent_data.user_query or {}
        final_map = agent_data.current_map or {}
        messages = agent_data.messages if hasattr(agent_data, 'messages') else []

        # === 空场景直接判 0：agent 未真正生成/摆放任何元件时，H1/H2/H4/H5 会
        #     因“场景无元件即无高度/生态/碰撞问题”而空虚为真(vacuously true)拿到
        #     pass=1，compute_onestep_reward 累加出非 0 分（如 0.16）——这是奖励
        #     漏洞。离线 evaluate_onestep_scene_case 已有此 gate，但在线 RL 路径
        #     直接调子函数绕过了它，故在此补上。空场景 = 需求完全未满足 = reward 0。
        if not _has_any_element(final_map):
            logger.warning(
                "[OnestepScene Verify] final_map 无任何元件（agent 未生成场景），"
                "直接判 reward=0（跳过 LLM 评估）"
            )
            empty_hard_result = {
                "H1": {"pass": 0, "issues": ["场景为空"]},
                "H2": {"pass": 0, "issues": ["场景为空"]},
                "H3": {"pass": 0, "VU": {"score": 0, "evidence": "场景为空"},
                       "VR": {"score": 0, "evidence": "场景为空"},
                       "Response": {"pass": 0}, "pass_reason": "场景为空"},
                "H4": {"pass": 0, "issues": ["场景为空"]},
                "H5": {"pass": 0, "worst_tier": 4, "intents": []},
                "hard_pass": False,
                "summary": "场景为空，agent 未生成任何元件，reward=0",
                "H3_VU": {"score": 0, "evidence": "场景为空"},
                "H3_VR": {"score": 0, "evidence": "场景为空"},
            }
            return {
                "verifier_type": "onestep_scene",
                "total_reward": 0.0,
                "hard_reward": 0.0,
                "soft_reward": 0.0,
                "hard_pass": False,
                "hard_result": empty_hard_result,
                "soft_result": None,
                "reward": 0.0,
                "reason": "onestep_scene: 场景为空（agent 未生成任何元件），reward=0",
            }

        # 最终 agent response（无 tool_calls 的最后一个 assistant turn）
        agent_final_response = self._extract_agent_final_response(messages)

        # agent turns（用于 H3 VU/VR/Response 评估）
        agent_turns = self._build_agent_turns_for_verify(messages)

        # 检索使用信息（用于 H5 评估）
        rinfo = self._extract_retrieve_and_usage_from_messages(messages)

        # 最终渲染图（PIL Image 列表）
        final_images_pil = []
        if agent_data.image_data:
            imgs = agent_data.image_data
            if not isinstance(imgs, list):
                imgs = [imgs]
            for img in imgs:
                if isinstance(img, Image.Image):
                    final_images_pil.append(img)

        # === Step 3: 保存图片到临时目录（LLM verifier 需要文件路径） ===
        temp_dir = tempfile.mkdtemp(prefix="verify_onestep_")
        final_image_paths = []
        try:
            if final_images_pil:
                img_dir = os.path.join(temp_dir, "final_image")
                os.makedirs(img_dir, exist_ok=True)
                for i, img in enumerate(final_images_pil):
                    p = os.path.join(img_dir, f"{i}.jpg")
                    img.save(p)
                    final_image_paths.append(p)

            logger.info(
                f"[OnestepScene Verify] 准备评估: final_map_keys={len(final_map)}, "
                f"final_images={len(final_image_paths)}, "
                f"agent_turns={len(agent_turns)}, intents={len(rinfo.get('intents', []))}"
            )

            # === Step 4: 构建 LLM bots ===
            model_type = os.environ.get("VERIFY_MODEL_TYPE", "gemini")
            model_name = os.environ.get("VERIFY_MODEL_NAME", "gemini-2.5-flash")
            logger.info(f"[OnestepScene Verify] 使用 model={model_type}/{model_name}")

            h12_bot = OS_MODEL_TYPE_MAP[model_type](
                model_name=model_name, system_instruction=HARD_H12_SCENE_SYSTEM_PROMPT
            )
            h3_bot = OS_MODEL_TYPE_MAP[model_type](
                model_name=model_name, system_instruction=H3_SCENE_SYSTEM_PROMPT
            )
            h5_bot = OS_MODEL_TYPE_MAP[model_type](
                model_name=model_name, system_instruction=H5_SCENE_SYSTEM_PROMPT
            )

            # === Step 5: H4 碰撞检测（rule-based，同步，极快） ===
            h4 = rule_based_collision_check(final_map)
            h4p = h4.get("pass", 0)
            logger.info(f"[OnestepScene Verify] H4 rule: pass={h4p}, collisions={h4.get('collision_count', 0)}")

            # === Step 6: H1/H2/H3/H5 LLM 评估（在线程池中运行，避免阻塞 event loop） ===
            loop = asyncio.get_event_loop()

            def _run_llm_evals():
                # H1/H2
                h12 = call_hard_h12_scene(h12_bot, query_info, final_map, final_image_paths)
                # H3
                h3 = call_h3_scene(
                    h3_bot, query_info, final_map, final_image_paths,
                    agent_final_response, agent_turns,
                )
                # H5
                h5 = call_h5_scene(h5_bot, query_info, rinfo, final_image_paths)
                return h12, h3, h5

            h12, h3, h5 = await asyncio.wait_for(
                loop.run_in_executor(None, _run_llm_evals),
                timeout=240,
            )

            # === Step 7: 汇总 ===
            h1p = h12["H1"]["pass"]
            h2p = h12["H2"]["pass"]
            h3p = h3["H3_pass"]
            h5p = h5["H5_pass"]
            hard_pass = bool(h1p == 1 and h2p == 1 and h3p == 1 and h4p == 1 and h5p == 1)
            total_reward = compute_onestep_reward(h1p, h2p, h3p, h4p, h5p)

            logger.info(
                f"[OnestepScene Verify] H1={h1p} H2={h2p} "
                f"H3={h3p}(VU={h3['H3_VU']['score']},VR={h3['H3_VR']['score']},"
                f"Resp={h3['H3_Response']['pass']}) H4={h4p} "
                f"H5={h5p}(tier={h5['worst_tier']}) → hard_pass={hard_pass} reward={total_reward}"
            )

            hard_result = {
                "H1": h12["H1"],
                "H2": h12["H2"],
                "H3": {
                    "pass": h3p,
                    "VU": h3["H3_VU"],
                    "VR": h3["H3_VR"],
                    "Response": h3["H3_Response"],
                    "pass_reason": h3.get("H3_pass_reason", ""),
                },
                "H4": {"pass": h4p, "issues": h4.get("issues", [])},
                "H5": {
                    "pass": h5p,
                    "worst_tier": h5["worst_tier"],
                    "intents": h5.get("intents", []),
                },
                "hard_pass": hard_pass,
                "summary": h12.get("summary", ""),
                # H3 子维度（兼容 _build_detailed_verify_reason 中的 H3_VU/H3_VR 读取）
                "H3_VU": h3["H3_VU"],
                "H3_VR": h3["H3_VR"],
            }

            reason_str = (
                f"onestep_scene: hard_pass={hard_pass}, "
                f"H1={h1p} H2={h2p} H3={h3p}(VU={h3['H3_VU']['score']},"
                f"VR={h3['H3_VR']['score']}) H4={h4p} H5={h5p}(tier{h5['worst_tier']}), "
                f"total={total_reward}"
            )

            return {
                "verifier_type": "onestep_scene",
                "total_reward": total_reward,
                "hard_reward": total_reward,
                "soft_reward": 0.0,
                "hard_pass": hard_pass,
                "hard_result": hard_result,
                "soft_result": None,
                "reward": total_reward,
                "reason": reason_str,
            }

        except asyncio.TimeoutError:
            logger.error("[OnestepScene Verify] LLM 评估超时 (240s)")
            return {
                "verifier_type": "onestep_scene",
                "total_reward": 0.0, "hard_reward": 0.0, "soft_reward": 0.0,
                "hard_pass": False, "hard_result": {}, "soft_result": None,
                "reward": 0.0, "reason": "OnestepScene 评估超时 (240s)",
                "error": "timeout",
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[OnestepScene Verify] 评估异常: {e}\n{tb}")
            return {
                "verifier_type": "onestep_scene",
                "total_reward": 0.0, "hard_reward": 0.0, "soft_reward": 0.0,
                "hard_pass": False, "hard_result": {}, "soft_result": None,
                "reward": 0.0, "reason": f"OnestepScene error: {str(e)}",
                "error": str(e),
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ==================== 构建完整 verify_reason ====================

    @staticmethod
    def _build_detailed_verify_reason(verify_result: dict) -> dict:
        """
        从 verify 返回的完整结果中构建详细的 verify_reason（与 verify_mix.py._build_reward_info 对齐）。

        返回结构化 dict，包含完整的评估细节：
        - verified: criteria_results (每条规则的 pass/score/reason/detail)
        - unverified: H1/H2/H3 (pass + issues) + S1/S2/S3 (score + reason) + summaries
        """
        info = {
            "verifier_type": verify_result.get("verifier_type", "unknown"),
            "query_tag": verify_result.get("query_tag", "?"),
            "query_type": verify_result.get("query_type", "?"),
        }

        vtype = verify_result.get("verifier_type", "unknown")

        if vtype == "verified":
            total = verify_result.get("total_reward", 0.0)
            info["total_reward"] = total
            info["hard_reward"] = total
            info["soft_reward"] = 0.0
            info["pass_count"] = verify_result.get("pass_count", 0)
            info["total_count"] = verify_result.get("total_count", 0)
            # 完整 criteria_results：保留 type/pass/score/reason/detail
            info["criteria_results"] = [
                {
                    "type": cr.get("criterion_type", cr.get("type", "?")),
                    "pass": cr.get("pass", False),
                    "score": cr.get("score", 0.0),
                    "reason": cr.get("reason", ""),
                    "detail": cr.get("detail", {}),
                }
                for cr in verify_result.get("criteria_results", [])
            ]

        elif vtype == "unverified":
            hard = verify_result.get("hard_result", {})
            hard_pass = hard.get("hard_pass", False)
            hard_reward = 1.0 if hard_pass else 0.0

            # soft 已移除：unverified reward = hard_pass ? 1.0 : 0.0
            info["total_reward"] = verify_result.get("total_reward", hard_reward)
            info["hard_reward"] = hard_reward
            info["soft_reward"] = 0.0
            info["hard_pass"] = hard_pass
            info["hard_summary"] = hard.get("summary", "")

            # H1/H2/H3/H4: pass + issues
            for hk in ["H1", "H2", "H3", "H4"]:
                h = hard.get(hk, {})
                info[hk] = {
                    "pass": h.get("pass", 0),
                    "issues": h.get("issues", []),
                }

            # v2: H3 子维度 (H3_VU / H3_VR)
            h3_vu = hard.get("H3_VU", {})
            h3_vr = hard.get("H3_VR", {})
            info["H3_VU"] = {"score": h3_vu.get("score", 0), "evidence": h3_vu.get("evidence", "")}
            info["H3_VR"] = {"score": h3_vr.get("score", 0), "evidence": h3_vr.get("evidence", "")}

        elif vtype == "onestep_scene":
            # onestep_scene: H1-H5 rubric，reward = 1.0 if all pass else 0.2*(passed/5)
            hard = verify_result.get("hard_result", {})
            hard_pass = hard.get("hard_pass", False)
            total_reward = verify_result.get("total_reward", 0.0)

            info["total_reward"] = total_reward
            info["hard_reward"] = total_reward
            info["soft_reward"] = 0.0
            info["hard_pass"] = hard_pass
            info["hard_summary"] = hard.get("summary", "")

            # H1/H2/H4: pass + issues
            for hk in ["H1", "H2", "H4"]:
                h = hard.get(hk, {})
                info[hk] = {
                    "pass": h.get("pass", 0),
                    "issues": h.get("issues", []),
                }

            # H3: pass + VU/VR/Response 子维度
            h3 = hard.get("H3", {})
            info["H3"] = {
                "pass": h3.get("pass", 0),
                "VU": h3.get("VU", h3.get("H3_VU", {})),
                "VR": h3.get("VR", h3.get("H3_VR", {})),
                "Response": h3.get("Response", h3.get("H3_Response", {})),
                "pass_reason": h3.get("pass_reason", ""),
            }
            # 同时暴露 H3_VU / H3_VR（供 reward 函数读取）
            h3_vu = hard.get("H3_VU", h3.get("VU", {}))
            h3_vr = hard.get("H3_VR", h3.get("VR", {}))
            info["H3_VU"] = {"score": h3_vu.get("score", 0), "evidence": h3_vu.get("evidence", "")}
            info["H3_VR"] = {"score": h3_vr.get("score", 0), "evidence": h3_vr.get("evidence", "")}

            # H5: pass + worst_tier + intents
            h5 = hard.get("H5", {})
            info["H5"] = {
                "pass": h5.get("pass", 0),
                "worst_tier": h5.get("worst_tier", 4),
                "intents": h5.get("intents", []),
            }

        else:
            # legacy fallback
            info["total_reward"] = verify_result.get("total_reward", verify_result.get("reward", 0.0))
            info["reason"] = verify_result.get("reason", "")

        if verify_result.get("error"):
            info["error"] = verify_result["error"]

        return info

    # ==================== Verified: Rule-based ====================

    def _call_verify_verified(self, agent_data: MapGenAgentData) -> dict:
        """Verified query: rule-based 对比 verification_criteria，不需要 LLM。"""
        try:
            verification_criteria = getattr(agent_data, 'verification_criteria', [])
            if not verification_criteria:
                return {
                    "verifier_type": "verified",
                    "total_reward": 0.0, "hard_reward": 0.0, "soft_reward": 0.0,
                    "pass_count": 0, "total_count": 0, "criteria_results": [],
                    "reward": 0.0, "reason": "无 verification_criteria",
                }

            init_map = agent_data.init_map
            final_map = agent_data.current_map

            # 提取固定放置的元件
            init_elems = self._extract_fixed_elements(init_map)
            final_elems = self._extract_fixed_elements(final_map)

            # 导入 verified_verifier 的验证函数
            try:
                import sys as _sys
                verifier_dir = _VIBE_VERIFIER_DIR
                if verifier_dir not in _sys.path:
                    _sys.path.insert(0, verifier_dir)
                from verified_verifier import verify_single_criterion
            except ImportError:
                logger.warning("无法导入 verified_verifier，使用内置简化版本")
                verify_single_criterion = self._verify_single_criterion_builtin

            # 逐条验证
            criteria_results = []
            for vc in verification_criteria:
                result = verify_single_criterion(vc, init_elems, final_elems)
                result["criterion_type"] = vc.get("type", "unknown")
                criteria_results.append(result)

            pass_count = sum(1 for r in criteria_results if r.get("pass", False))
            total_count = len(criteria_results)
            total_reward = round(pass_count / total_count, 4) if total_count > 0 else 0.0

            # ===== Anti-Hacking: 严格作用域校验 —— 越界改动即清零 =====
            # criteria 决定"授权改动集合"；凡改了 criteria 未指定的资产(多删/误删/
            # 多加/擅移)即越界 -> reward 清零。防"本可精准改一处却大改地图"的 hack。
            scope_detail = {}
            scope_violation = False
            if self.verified_strict_scope and self._scope_violation_fn is not None:
                try:
                    scope_violation, scope_detail = self._scope_violation_fn(
                        verification_criteria, init_elems, final_elems
                    )
                    if scope_violation:
                        total_reward = 0.0
                        logger.info(
                            f"[Anti-Hacking] Scope violation -> reward=0 | "
                            f"illegal_del={scope_detail.get('illegal_delete_names')} "
                            f"illegal_add={scope_detail.get('illegal_add_names')} "
                            f"(criteria pass_count was {pass_count}/{total_count})"
                        )
                except Exception as _se:
                    logger.warning(f"[Anti-Hacking] scope check failed: {_se}")

            return {
                "verifier_type": "verified",
                "total_reward": total_reward,
                "hard_reward": total_reward,
                "soft_reward": 0.0,
                "pass_count": pass_count,
                "total_count": total_count,
                "criteria_results": criteria_results,
                "scope_violation": scope_violation,
                "scope_detail": scope_detail,
                "reward": total_reward,
                "reason": (
                    f"verified: {pass_count}/{total_count} criteria passed"
                    + (" | SCOPE_VIOLATION->0" if scope_violation else "")
                ),
            }
        except Exception as e:
            logger.error(f"Verified verify error: {e}")
            return {
                "verifier_type": "verified",
                "total_reward": 0.0, "hard_reward": 0.0, "soft_reward": 0.0,
                "pass_count": 0, "total_count": 0, "criteria_results": [],
                "reward": 0.0, "reason": f"Verified verify error: {str(e)}",
            }

    @staticmethod
    def _extract_fixed_elements(map_json: dict) -> list:
        """提取地图中所有固定放置的元件（非散布区域）"""
        elems = []
        for cat_key, cat_val in map_json.items():
            if cat_key == "地图信息" or not isinstance(cat_val, dict):
                continue
            for sub_key, sub_val in cat_val.items():
                if not isinstance(sub_val, list):
                    continue
                for elem in sub_val:
                    pos = elem.get("pos")
                    if isinstance(pos, list) and len(pos) > 0 and not isinstance(pos[0], list):
                        elems.append(elem)
        return elems

    @staticmethod
    def _verify_single_criterion_builtin(vc: dict, init_elems: list, final_elems: list) -> dict:
        """内置简化版验证（当无法导入 verified_verifier 时的 fallback）"""
        vc_type = vc.get("type", "")
        # 简化处理：只检查元件是否存在
        if vc_type == "proximity":
            expected_name = vc.get("expected_element_name", "")
            found = any(e.get("name") == expected_name for e in final_elems
                        if not any(ie.get("name") == expected_name and ie.get("pos") == e.get("pos")
                                   for ie in init_elems))
            return {"pass": found, "score": 1.0 if found else 0.0,
                    "reason": f"{'找到' if found else '未找到'}新增的 {expected_name}"}
        elif vc_type == "exact_match":
            expected_name = vc.get("expected_name", "")
            expected_pos = vc.get("expected_pos", [])
            still_exists = any(e.get("name") == expected_name and e.get("pos") == expected_pos
                               for e in final_elems)
            return {"pass": not still_exists, "score": 1.0 if not still_exists else 0.0,
                    "reason": f"{expected_name} {'已删除' if not still_exists else '仍存在'}"}
        return {"pass": False, "score": 0.0, "reason": f"不支持的验证类型: {vc_type}"}

    # ==================== Verify 辅助方法 ====================

    @staticmethod
    def _build_agent_turns_for_verify(messages: list, max_tool_arg_preview: int = 600) -> list:
        """从 agent_data.messages 构造 v2 verifier 需要的 agent_turns 列表。

        与 unverified_verifier_v2.extract_agent_turns_with_toolcalls() 输出格式对齐：
        [{"turn_idx": 1, "content": "...", "tool_calls": [{"name":..,"arguments_preview":..}], "is_final": bool}, ...]
        """
        assistant_turns = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            # content: 优先用 reasoning_content + content 拼接
            reasoning = msg.get("reasoning_content", "") or ""
            content = msg.get("content", "") or ""
            full_content = f"{reasoning}\n{content}".strip() if reasoning else content

            # tool_calls: 从 v2 结构化 tool_calls 字段提取
            tcs = msg.get("tool_calls", [])
            tool_calls = []
            for tc in tcs:
                func = tc.get("function", {})
                args_raw = func.get("arguments", "{}")
                # arguments 可能是 str 或 dict
                if isinstance(args_raw, dict):
                    arg_preview = json.dumps(args_raw, ensure_ascii=False)[:max_tool_arg_preview]
                else:
                    arg_preview = str(args_raw)[:max_tool_arg_preview]
                tool_calls.append({
                    "name": func.get("name", "?"),
                    "arguments_preview": arg_preview,
                })
            assistant_turns.append({"content": full_content, "tool_calls": tool_calls})

        # 标记 turn_idx 和 is_final
        for i, t in enumerate(assistant_turns):
            t["turn_idx"] = i + 1
            t["is_final"] = (i == len(assistant_turns) - 1)
        return assistant_turns

    @staticmethod
    def _extract_agent_final_response(messages: list) -> str:
        """从 messages 中提取最后一个无 tool_calls 的 assistant turn 的 content。

        与 verifier 侧 extract_agent_final_response() 逻辑对齐：
        找最后一个 role=assistant 且 tool_calls 为空的 turn，提取 </think> 之后的文本。
        """
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            tc = msg.get("tool_calls", [])
            if tc:
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content else ""
            content = content.strip()
            if not content:
                continue
            # 提取 </think> 之后的文本（与 verifier 侧一致）
            if "</think>" in content:
                final_text = content.split("</think>", 1)[1].strip()
                if final_text:
                    return final_text
            return content
        return ""

    @staticmethod
    def _build_available_assets(component_info: dict) -> str:
        """从 component_info 构建可用资产描述文本。

        与 verifier 侧 extract_available_assets() 逻辑对齐。
        """
        if not component_info:
            return "（无可用资产信息）"
        asset_names = sorted(component_info.keys()) if isinstance(component_info, dict) else []
        if not asset_names:
            return "（资产列表为空）"
        return f"可用资产共 {len(asset_names)} 种：{', '.join(asset_names)}"

    # ==================== Unverified: LLM-as-Judge ====================

    async def _call_verify_unverified(self, agent_data: MapGenAgentData) -> dict:
        """
        Unverified query: Hard-only LLM-as-Judge 评估（soft reward 已移除）。

        导入策略：
        - 从仓库根目录的 verifier/ 导入 unverified_verifier
        - 仅在 ImportError 时回退到 legacy（表示 verifier 代码不存在）
        - 运行时异常不回退，直接报错返回 0 分（避免静默降级到旧评估）

        reward: hard_pass → 1.0, 否则 0.0（不再有 soft，最大值 1.0）。
        返回统一格式的 dict。
        """
        # === Step 1: 尝试导入 verifier 模块 ===
        try:
            import sys as _sys
            verifier_dir = _VIBE_VERIFIER_DIR
            if verifier_dir not in _sys.path:
                _sys.path.insert(0, verifier_dir)

            from unverified_verifier import (
                evaluate_unverified_case_v2, MODEL_TYPE_MAP as UV_MODEL_TYPE_MAP,
            )
            from prompts import HARD_H12_SYSTEM_PROMPT, H3_VURR_SYSTEM_PROMPT
        except ImportError as e:
            logger.error(f"无法导入 unverified_verifier ({e})，返回 0 分")
            return {
                "verifier_type": "unverified",
                "total_reward": 0.0, "hard_reward": 0.0, "soft_reward": 0.0,
                "hard_pass": False, "hard_result": {}, "soft_result": None,
                "reward": 0, "reason": f"ImportError: {e}",
                "error": str(e),
            }

        # === Step 1.5: 提取 agent_final_response 和 available_assets ===
        agent_final_response = self._extract_agent_final_response(
            agent_data.messages if hasattr(agent_data, 'messages') else []
        )
        available_assets = self._build_available_assets(
            agent_data.component_info if hasattr(agent_data, 'component_info') else {}
        )

        # === Step 2: 准备临时数据目录并运行 Hard+Soft 评估 ===
        temp_dir = tempfile.mkdtemp(prefix="verify_unverified_")
        try:
            # 写入必要文件
            with open(os.path.join(temp_dir, "query.json"), 'w', encoding='utf-8') as f:
                json.dump(agent_data.user_query, f, ensure_ascii=False)
            with open(os.path.join(temp_dir, "init_map.json"), 'w', encoding='utf-8') as f:
                json.dump(agent_data.init_map, f, ensure_ascii=False)
            with open(os.path.join(temp_dir, "final_map.json"), 'w', encoding='utf-8') as f:
                json.dump(agent_data.current_map, f, ensure_ascii=False)

            # 保存 init 图片
            init_images = agent_data.extra_fields.get("init_images", [])
            # [DEBUG] 打印 extra_fields 中 init_images 的情况
            print(f"[DEBUG VERIFY] extra_fields keys: {list(agent_data.extra_fields.keys())}")
            print(f"[DEBUG VERIFY] init_images from extra_fields: count={len(init_images)}, types={[type(x).__name__ for x in init_images[:3]]}")
            print(f"[DEBUG VERIFY] agent_data.image_data: type={type(agent_data.image_data)}, "
                  f"count={len(agent_data.image_data) if isinstance(agent_data.image_data, list) else 'N/A'}")
            if init_images:
                init_img_dir = os.path.join(temp_dir, "init_image")
                os.makedirs(init_img_dir, exist_ok=True)
                for i, img in enumerate(init_images):
                    if isinstance(img, Image.Image):
                        img.save(os.path.join(init_img_dir, f"{i}.jpg"))

            # 保存 final 图片
            final_images = agent_data.image_data if agent_data.image_data else []
            if final_images:
                final_img_dir = os.path.join(temp_dir, "final_image")
                os.makedirs(final_img_dir, exist_ok=True)
                for i, img in enumerate(final_images):
                    if isinstance(img, Image.Image):
                        img.save(os.path.join(final_img_dir, f"{i}.jpg"))

            # [DEBUG] 打印 temp_dir 中实际保存的图片
            import glob as _glob
            init_saved = sorted(_glob.glob(os.path.join(temp_dir, "init_image", "*.jpg")))
            final_saved = sorted(_glob.glob(os.path.join(temp_dir, "final_image", "*.jpg")))
            print(f"[DEBUG VERIFY] temp_dir={temp_dir}")
            print(f"[DEBUG VERIFY] saved init_images: {len(init_saved)}, final_images: {len(final_saved)}")
            print(f"[DEBUG VERIFY] init_map == current_map: {agent_data.init_map == agent_data.current_map}")
            print(f"[DEBUG VERIFY] interaction_turns: user={agent_data.user_turns}, assistant={agent_data.assistant_turns}")

            # 初始化 LLM bots
            model_type = os.environ.get("VERIFY_MODEL_TYPE", "gemini")
            model_name = os.environ.get("VERIFY_MODEL_NAME", "gemini-2.5-flash")
            logger.info(f"[Unverified Verify] 使用 Hard+Soft 流程 (v2: H3-VU/VR), model={model_type}/{model_name}")

            hard_h12_bot = UV_MODEL_TYPE_MAP[model_type](
                model_name=model_name, system_instruction=HARD_H12_SYSTEM_PROMPT
            )
            h3_bot = UV_MODEL_TYPE_MAP[model_type](
                model_name=model_name, system_instruction=H3_VURR_SYSTEM_PROMPT
            )

            # v2: 在 agent_loop 侧构造 agent_turns（直接传参，不写文件）
            agent_turns = self._build_agent_turns_for_verify(
                agent_data.messages if hasattr(agent_data, 'messages') else []
            )

            # 在线程池中运行同步的 evaluate_unverified_case_v2（soft 已移除，不再传 soft_bot）
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: evaluate_unverified_case_v2(
                        temp_dir, hard_h12_bot, h3_bot,
                        agent_final_response=agent_final_response,
                        agent_turns=agent_turns,
                        available_assets=available_assets,
                    )
                ),
                timeout=240,  # v2 多一个 H3 LLM call，给更多时间
            )

            # 构建统一返回格式
            hard_result = result.get("hard_result", {})

            # 检测 verifier 输出解析失败
            hard_parse_ok = hard_result.get("llm_raw_output", {}).get("parse_success", True)
            parse_failed = not hard_parse_ok

            if parse_failed:
                hard_raw = hard_result.get("llm_raw_output", {})
                logger.warning(
                    f"[Unverified Verify] 输出解析失败 | "
                    f"hard_parse_ok={hard_parse_ok} | "
                    f"hard_raw_content={str(hard_raw.get('content', ''))[:500]} | "
                    f"hard_raw_reasoning={str(hard_raw.get('reasoning', ''))[:500]}"
                )

            hard_pass = hard_result.get("hard_pass", False)
            hard_reward = 1.0 if hard_pass else 0.0
            soft_reward = 0.0
            # soft 已移除：unverified reward = hard_pass ? 1.0 : 0.0（最大 1.0）
            total_reward = result.get("total_reward", hard_reward)

            # 构建简洁的 reason 摘要（详细信息在 verify_reason_detail 中）
            reason_parts = [f"hard_pass={hard_pass}"]
            if parse_failed:
                reason_parts.append("parse_failed=True")
            for hk in ["H1", "H2", "H3", "H4"]:
                h = hard_result.get(hk, {})
                reason_parts.append(f"{hk}={'✓' if h.get('pass', 0) == 1 else '✗'}")
            # v2: H3 子维度 scores
            h3_vu_score = hard_result.get("H3_VU", {}).get("score", "?")
            h3_vr_score = hard_result.get("H3_VR", {}).get("score", "?")
            reason_parts.append(f"H3_VU={h3_vu_score}")
            reason_parts.append(f"H3_VR={h3_vr_score}")
            reason_str = f"unverified: {', '.join(reason_parts)}, total={total_reward}"

            logger.info(f"[Unverified Verify] {reason_str}")

            return {
                "verifier_type": "unverified",
                "total_reward": total_reward,
                "hard_reward": hard_reward,
                "soft_reward": soft_reward,
                "hard_pass": hard_pass,
                "hard_result": hard_result,
                "soft_result": None,
                "reward": total_reward,
                "reason": reason_str,
                "parse_failed": parse_failed,
            }
        except asyncio.TimeoutError:
            logger.error("[Unverified Verify] Hard+Soft 评估超时 (180s)")
            return {
                "verifier_type": "unverified",
                "total_reward": 0.0, "hard_reward": 0.0, "soft_reward": 0.0,
                "hard_pass": False, "hard_result": {}, "soft_result": None,
                "reward": 0, "reason": "Unverified Hard+Soft 评估超时 (180s)",
                "error": "timeout",
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[Unverified Verify] Hard+Soft 评估异常: {e}\n{tb}")
            return {
                "verifier_type": "unverified",
                "total_reward": 0.0, "hard_reward": 0.0, "soft_reward": 0.0,
                "hard_pass": False, "hard_result": {}, "soft_result": None,
                "reward": 0, "reason": f"Unverified Hard+Soft error: {str(e)}",
                "error": str(e),
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _handle_generating_state(
        self, agent_data: MapGenAgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls.

        v2: 构建结构化 assistant 消息（content / reasoning_content / tool_calls），
        与 SFT v2 数据格式和 main_2_v2.py 推理格式完全对齐。

        Anti-Hacking: 支持 max_turns_by_type 动态 max_turns。
        """
        turn = agent_data.assistant_turns + 1

        # ===== Anti-Hacking: 动态 max_turns 检查 =====
        if self.max_turns_by_type:
            query_tag = getattr(agent_data, 'query_tag', '')
            dynamic_max = self.max_turns_by_type.get(query_tag)
            if dynamic_max is not None and agent_data.assistant_turns >= dynamic_max:
                logger.info(
                    f"[Anti-Hacking] Dynamic max_turns reached: "
                    f"query_tag={query_tag}, max={dynamic_max}, "
                    f"assistant_turns={agent_data.assistant_turns}"
                )
                return AgentState.TERMINATED

        # 限制单轮生成的 max_tokens，避免单轮输出过长占满整个 response_length 预算
        if self.per_turn_max_tokens is not None:
            per_turn_params = dict(sampling_params)
            per_turn_params["max_tokens"] = self.per_turn_max_tokens
        else:
            per_turn_params = sampling_params

        # 调用父类的生成方法（会执行 tool_parser.extract_tool_calls 并填充 agent_data.tool_calls）
        state = await super()._handle_generating_state(agent_data, per_turn_params, ignore_termination)

        # 解码 LLM 完整输出
        response_text = ""
        if agent_data.response_ids:
            response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)

        print(f"LLM Output: Turn={turn}, State={state.name}, Length={len(agent_data.response_ids) if agent_data.response_ids else 0}, Text={response_text[:500].replace(chr(10), ' ')}")

        # ===== v2: 构建结构化 assistant 消息 =====
        is_final_turn = (state == AgentState.TERMINATED and not agent_data.tool_calls)

        v2_msg = _build_v2_assistant_message(
            response_text=response_text,
            tool_calls=agent_data.tool_calls,
            is_final_turn=is_final_turn,
        )
        agent_data.messages.append(v2_msg)

        # 日志：显示结构化消息摘要
        reasoning_len = len(v2_msg.get("reasoning_content", "") or "")
        tc_names = [tc["function"]["name"] for tc in v2_msg.get("tool_calls", [])]
        content_preview = str(v2_msg.get("content", ""))[:100]
        logger.info(
            f"V2 Assistant Msg: Turn={turn}, reasoning={reasoning_len}c, "
            f"tool_calls={tc_names}, content={content_preview!r}"
        )

        # ===== 统计 tool_call 解析情况 =====
        if not is_final_turn:
            # 非最后一轮才计入 tool_call 解析统计
            agent_data.tool_call_parse_attempts += 1

        turn_detail = {
            "turn": turn,
            "llm_response_length": len(response_text),
            "reasoning_length": reasoning_len,
            "tool_call_parsed": False,
            "tool_call_count": 0,
            "tool_names": [],
            "parse_failed_reason": None,
        }

        if agent_data.tool_calls:
            # 成功解析出 tool_call
            if not is_final_turn:
                agent_data.tool_call_parse_successes += 1
            turn_detail["tool_call_parsed"] = True
            turn_detail["tool_call_count"] = len(agent_data.tool_calls)

            tool_calls_info = []
            for tc in agent_data.tool_calls:
                tool_calls_info.append(f"{tc.name}({tc.arguments})")
                turn_detail["tool_names"].append(tc.name)
            logger.info(f"Tool Calls: Turn={turn}, Count={len(agent_data.tool_calls)}, Tools=[{', '.join(tool_calls_info)}]")
        else:
            # 模型未输出合法的 tool_call —— 细分失败原因
            has_tool_call_tag = "<tool_call>" in response_text or "</tool_call>" in response_text
            has_json_block = "```json" in response_text
            has_json_like = "[{" in response_text or '{"name"' in response_text
            # v2: 也检查 XML 格式 <function=
            has_xml_function = "<function=" in response_text

            if is_final_turn:
                # 模型正常结束（最后一轮总结），不算解析失败，不计入统计
                fail_type = "no_tool_call_intended"
                fail_reason = "模型正常结束，未输出 tool_call（最后一轮总结）"
            elif has_tool_call_tag or has_json_block or has_json_like or has_xml_function:
                # 有 tool_call 相关内容但解析失败 → 格式问题
                agent_data.tool_call_parse_failures += 1
                fail_type = "tool_call_parse_error"
                fail_reason = (
                    f"模型输出了 tool_call 相关内容但解析失败 "
                    f"(tag={has_tool_call_tag}, json_block={has_json_block}, "
                    f"json_like={has_json_like}, xml_func={has_xml_function})"
                )
            else:
                # 模型应该调用工具但完全没有输出 tool_call 内容
                agent_data.tool_call_parse_failures += 1
                fail_type = "no_tool_call_generated"
                fail_reason = "模型回复中完全没有 tool_call 相关内容"

            failed_reason = {
                "type": fail_type,
                "turn": turn,
                "reason": fail_reason,
            }
            if not is_final_turn:
                agent_data.failed_reasons.append(failed_reason)
            turn_detail["parse_failed_reason"] = fail_reason

            if fail_type != "no_tool_call_intended":
                logger.warning(f"Tool Call Issue: Turn={turn}, Type={fail_type}, Response={response_text[:300]}")

            logger.info(f"Tool Calls: Turn={turn}, Count=0, FailType={fail_type}")

        agent_data.turn_details.append(turn_detail)
        return state

    async def _handle_processing_tools_state(self, agent_data: MapGenAgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses.

        v2: 与 SFT v2 数据格式对齐：
        - tool_response 使用 role=tool（而非 role=user）
        - 去掉 <tool_response>...</tool_response> 外层标签
        - 内容（含图片）直接作为 tool 消息的 content
        - 这样 Qwen3 chat_template 能正确识别 last_query_index，
          保证历史 assistant 轮 <think> block 被正确序列化
        """
        new_images_this_turn: list[Any] = []
        
        turn = agent_data.user_turns + 1

        # 准备 tool_context 传递给工具
        tool_context = {
            "current_map": getattr(agent_data, 'current_map', {}),
            "init_map": getattr(agent_data, 'init_map', {}),
            "component_info": getattr(agent_data, 'component_info', {}),
            # retrieve_assets 工具所需的运行时配置（由 MapGenAgentLoop.__init__ 预加载）
            "pcg_whitelist": getattr(self, '_pcg_whitelist_set', None),
            "pcg_item_infos": self._pcg_item_infos if self._pcg_item_infos else None,
            "retrieve_url": self.rollout_config.multi_turn.get(
                "retrieve_url",
                os.environ.get("RETRIEVE_SERVER_URL", "http://localhost:8081"),
            ),
        }

        # ===== 串行执行工具（不能并行！多个工具共享 current_map，并行会互相覆盖）=====
        tool_call_names = []
        responses = []
        with simple_timer("tool_calls", agent_data.metrics):
            for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
                tool_call_names.append(tool_call.name)
                resp = await self._call_tool_with_context(tool_call, agent_data.tools_kwargs, tool_context)
                responses.append(resp)

        # 从工具响应中更新地图状态
        current_map = tool_context.get("current_map", {})
        agent_data.current_map = current_map
        # retrieve_assets 工具会把检索到的资产写入 tool_context["component_info"]，
        # 这里同步回 agent_data，供后续轮次的 add 工具和渲染使用
        updated_comp_info = tool_context.get("component_info", {})
        if updated_comp_info:
            agent_data.component_info = updated_comp_info

        # 记录工具执行结果日志 + 回写到当前轮的 turn_detail
        tool_exec_results = []
        turn_retrieve_results = []   # 本轮 retrieve_assets 的结果
        for i, (tool_response, tool_reward, res) in enumerate(responses):
            success = bool(tool_response.text and "Error" not in tool_response.text[:50])
            status = "SUCCESS" if success else "FAILED"
            resp_text = tool_response.text or "empty"
            logger.info(f"Tool Exec: Turn={turn}, Tool={tool_call_names[i]}, Status={status}, Response={resp_text}")
            exec_entry = {
                "tool_name": tool_call_names[i],
                "status": status,
                "response": resp_text,
            }
            # 如果是 retrieve_assets，额外记录结构化结果
            if tool_call_names[i] == "retrieve_assets" and success:
                retrieve_detail = {
                    "entity_name": res.get("entity_name", "?"),
                    "result_count": res.get("result_count", 0),
                }
                exec_entry["retrieve_detail"] = retrieve_detail
                turn_retrieve_results.append(retrieve_detail)
            tool_exec_results.append(exec_entry)

        # 将工具执行结果写入最近一条 turn_detail
        if agent_data.turn_details:
            agent_data.turn_details[-1]["tool_exec_results"] = tool_exec_results
        if turn_retrieve_results:
            agent_data.turn_retrieve_results.append(turn_retrieve_results)

        # 收集工具奖励 + 提取 retrieve_assets 的 tool_response.text（供 tool 消息拼接）
        # 直接使用 tool_response.text，它已经是 RetrieveAssetsTool._format_retrieve_response 格式化好的文本
        retrieve_texts = []
        for i, (tool_response, tool_reward, res) in enumerate(responses):
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
            if tool_call_names[i] == "retrieve_assets" and tool_response.text:
                retrieve_texts.append(tool_response.text)
        
        # 执行PCG渲染
        render_success = False
        error_info = ""
        if self.auto_render and current_map:
            # ===== generate 任务：渲染前动态重建 component_info =====
            # generate 任务（task_setting="generate"）的 component_info 初始为空 {}，
            # agent 通过 retrieve_assets 检索 + add 工具把资产写入 current_map，
            # 但 component_info 本身不被工具更新。
            # 渲染前用 enrich_component_info_for_generate 从 current_map 中的 type_id
            # 反查 item_infos 白名单，重建完整的渲染元数据（box/typeId/rot 等）。
            # 这与 main_2_v4.py 第 885-890 行的逻辑完全对齐。
            render_component_info = agent_data.component_info  # refine/verified 路径直接用
            task_setting = agent_data.user_query.get("task_setting", "")
            if task_setting == "generate" and self._pcg_item_infos:
                try:
                    import sys as _sys
                    if _VIBE_UTILS_DIR not in _sys.path:
                        _sys.path.insert(0, _VIBE_UTILS_DIR)
                    from component_info_builder import enrich_component_info_for_generate
                    # 用空 base 强制从 item_infos 完整重建（避免 retrieve 累积的残缺 entry 干扰）
                    render_component_info = enrich_component_info_for_generate(
                        base_component_info={},
                        llm_output=current_map,
                        item_infos=self._pcg_item_infos,
                    )
                    logger.info(
                        f"[Generate] enrich_component_info: {len(render_component_info)} 条, turn={turn}"
                    )
                except Exception as _enrich_err:
                    logger.warning(
                        f"[Generate] enrich_component_info_for_generate 失败: {_enrich_err}，"
                        f"回退到原始 component_info"
                    )
                    render_component_info = agent_data.component_info

            agent_data.sandbox_call_attempts += 1
            agent_data.pcg_convert_attempts += 1
            render_images, error_msg, sandbox_log = await self._pcg_render_with_retry(current_map, render_component_info, turn, scatter_cache=agent_data.scatter_cache, camera_params=agent_data.camera_params)
            
            # 记录 sandbox 日志
            agent_data.sandbox_logs.append(sandbox_log)

            # 判断是地图格式转换失败 还是 渲染服务本身失败
            convert_failed = sandbox_log.get("convert_failed", False)

            if render_images:
                agent_data.sandbox_call_successes += 1
                agent_data.pcg_render_attempts += 1
                agent_data.pcg_render_successes += 1
                new_images_this_turn.extend(render_images)
                # 记录本轮渲染图（供 trajectory 落盘使用）
                agent_data.turn_images.append(list(render_images))
                render_success = True
                logger.info(f"PCG Render: Turn={turn}, Status=SUCCESS, Images={len(render_images)}")
            elif convert_failed:
                # 地图格式转换失败（模型输出无法解析为 actors）
                agent_data.sandbox_call_failures += 1
                agent_data.pcg_convert_failures += 1
                error_info = error_msg
                failed_reason = {
                    "type": "pcg_convert_failed",
                    "turn": turn,
                    "reason": error_msg,
                    "sandbox_stdout": sandbox_log.get("stdout", "")[:500],
                    "sandbox_stderr": sandbox_log.get("stderr", "")[:500],
                    "sandbox_returncode": sandbox_log.get("returncode"),
                }
                agent_data.failed_reasons.append(failed_reason)
                logger.warning(f"PCG Render: Turn={turn}, Status=FAILED (convert), Error={error_msg}")
            else:
                # 渲染服务本身失败（超时/空图/服务异常等）
                agent_data.sandbox_call_failures += 1
                agent_data.pcg_render_attempts += 1
                agent_data.pcg_render_failures += 1
                error_info = error_msg
                failed_reason = {
                    "type": "pcg_render_failed",
                    "turn": turn,
                    "reason": error_msg,
                    "sandbox_stdout": sandbox_log.get("stdout", "")[:500],
                    "sandbox_stderr": sandbox_log.get("stderr", "")[:500],
                    "sandbox_returncode": sandbox_log.get("returncode"),
                }
                agent_data.failed_reasons.append(failed_reason)
                logger.warning(f"PCG Render: Turn={turn}, Status=FAILED (render), Error={error_msg}")

        # ===== Anti-Hacking: 逐轮 criteria 评估 =====
        if (self.reward_efficiency_alpha > 0 or self.reward_efficiency_beta > 0) and current_map:
            criteria_results = self._evaluate_criteria_at_current_map(agent_data)
            if criteria_results:
                agent_data.per_turn_criteria_results.append(criteria_results)
                n_pass = sum(criteria_results)
                n_total = len(criteria_results)
                logger.info(
                    f"[Anti-Hacking] Turn={turn} criteria: {n_pass}/{n_total} passed "
                    f"[{''.join('✓' if p else '✗' for p in criteria_results)}]"
                )

        # ===== v2: 构造 role=tool 消息，与 SFT v2 数据格式对齐 =====
        # SFT v2 格式：role=tool，不含 <tool_response> 标签，内容（含图片）直接作为 content
        current_map_json_str = json.dumps(current_map, ensure_ascii=False, separators=(',', ':'))

        # ---- 拼接 retrieve_assets 结果到 tool 消息前缀 ----
        # retrieve_texts 中每条已经是 RetrieveAssetsTool._format_retrieve_response 格式化好的文本
        retrieve_prefix = ""
        if retrieve_texts:
            retrieve_prefix = "\n".join(retrieve_texts).rstrip('\n') + '\n\n'

        tools_hint = (
            "当前可用的tools如下（rotation_and_translation(arguments: corrections), "
            "delete(arguments: modified_data)，add(arguments: modified_data)）"
        )

        if render_success:
            # 使用结构化多模态格式，让 chat template 和 processor 能正确处理图片
            tool_message_content = [
                {
                    "type": "text",
                    "text": (
                        f"{retrieve_prefix}"
                        f"本轮改造后场景的基本信息，地图中元件的详细位置、尺寸信息如下："
                        f"{current_map_json_str}"
                        f"下面5张相机拍摄的图片分别展示了当前场景的左视图、右视图、前视图、后视图以及俯视图，"
                        f"展示了各元件在当前场景地图中的位置: "
                    )
                },
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": f" 。{tools_hint}"
                }
            ]
            add_messages = [{"role": "tool", "content": tool_message_content}]
        else:
            tool_message_text = (
                f"{retrieve_prefix}"
                f"本轮改造后场景的基本信息，地图中元件的详细位置、尺寸信息如下："
                f"{current_map_json_str}"
                f"由于环境元件不合理，本轮未能渲染出新的场景图片。"
                f"{tools_hint}"
            )
            if error_info:
                tool_message_text += f"提示：{error_info}"
            add_messages = [{"role": "tool", "content": tool_message_text}]
        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        # 统一将本轮新图片追加到 agent_data.image_data（不覆盖历史图片）
        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        
        return AgentState.GENERATING
    
    async def _call_tool_with_context(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], tool_context: dict
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool with context and return tool response."""
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, tool_context=tool_context
            )
        except Exception as e:
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        tool_response_kwargs = {"text": tool_response_text}

        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        运行地图生成agent loop
        """
        messages = list(kwargs["raw_prompt"])
        
        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        
        # 从extra_info中提取地图生成特有数据
        extra_info_raw = kwargs.get("extra_info", "{}")
        if isinstance(extra_info_raw, str):
            try:
                extra_info = json.loads(extra_info_raw)
            except json.JSONDecodeError:
                extra_info = {}
        else:
            extra_info = extra_info_raw
        
        init_map = extra_info.get("init_map", {})
        component_info = extra_info.get("component_info", {})
        user_query = extra_info.get("user_query", {})
        scatter_cache = extra_info.get("scatter_cache", {})
        camera_params = extra_info.get("camera_params", {})
        # === verified/unverified 分流字段 ===
        verifier_type = extra_info.get("verifier_type", "unverified")
        query_tag = extra_info.get("query_tag", "")
        query_type = extra_info.get("query_type", "")
        verification_criteria = extra_info.get("verification_criteria", [])
        
        # 打印 user query
        user_query_text = user_query.get("description", "")
        # 获取当前训练步数（由 agent_loop.py 注入）
        global_steps = kwargs.get("global_steps", -1)
        logger.info(f"User Query: {user_query_text} | global_steps={global_steps}")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        
        # Create AgentData instance
        agent_data = MapGenAgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        
        # 设置地图生成特有数据
        agent_data.init_map = deepcopy(init_map)
        agent_data.current_map = deepcopy(init_map)
        agent_data.component_info = component_info
        agent_data.user_query = user_query
        agent_data.scatter_cache = scatter_cache if scatter_cache else {}
        agent_data.camera_params = camera_params if camera_params else {}
        # 保存初始图片引用，供 verify 时对比使用
        # images 是从 messages 中提取的初始场景渲染图（通常 5 张）
        # 注意：即使 images 为空（PCG 渲染服务失败等原因），也必须写入该 key，
        # 否则 DataProto.concat 时不同 worker 的 extra_fields key 集合不一致，
        # 会导致 non_tensor_batch 里 init_images 长度 < batch_size，触发
        # check_consistency AssertionError（历史曾因此连续失败 3 次）。
        agent_data.extra_fields["init_images"] = list(images) if images else []
        # === verified/unverified 分流字段 ===
        agent_data.verifier_type = verifier_type
        agent_data.query_tag = query_tag
        agent_data.query_type = query_type
        agent_data.verification_criteria = verification_criteria
        
        # State machine loop
        state = AgentState.PENDING
        total_turns = 0
        
        while state != AgentState.TERMINATED:
            total_turns += 1
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
        
        # 任务结束，调用verify获取reward
        reward_score = None
        verify_reason = None           # 简短摘要字符串（兼容旧日志格式）
        verify_reason_detail = {}      # 完整结构化 verify 结果（与 mix_reward_report.json 对齐）
        verify_result_full = {}        # 原始 verify 返回（传递给 reward 函数）
        if self.enable_verify:
            try:
                verify_result = await asyncio.wait_for(
                    self._call_verify(agent_data),
                    timeout=300
                )
                verify_result_full = verify_result
                reward_score = verify_result.get("reward", verify_result.get("total_reward", 0))
                verify_reason = verify_result.get("reason", verify_result.get("verify_reason", ""))
                # 构建完整结构化的 verify_reason_detail（与 verify_mix.py._build_reward_info 对齐）
                verify_reason_detail = self._build_detailed_verify_reason(verify_result)
                agent_data.verifier_call_success = True
            except asyncio.TimeoutError:
                reward_score = 0
                verify_reason = "Verify timeout"
                verify_reason_detail = {"error": "timeout", "total_reward": 0.0}
                agent_data.verifier_call_success = False
                agent_data.failed_reasons.append({
                    "type": "verifier_call_failed",
                    "turn": total_turns,
                    "reason": "Verify API 调用超时 (300s)",
                })
            except Exception as e:
                reward_score = 0
                verify_reason = f"Verify error: {str(e)}"
                verify_reason_detail = {"error": str(e), "total_reward": 0.0}
                agent_data.verifier_call_success = False
                agent_data.failed_reasons.append({
                    "type": "verifier_call_failed",
                    "turn": total_turns,
                    "reason": f"Verify error: {str(e)}",
                })

        # interaction_turns = user-assistant 交互轮数（初始 prompt 算第1轮 user，每次 tool_response 算后续 user）
        interaction_turns = agent_data.user_turns + 1  # +1 是初始 prompt

        # ===== Anti-Hacking: 效率折扣 reward（仅 verified query） =====
        efficiency_metrics = {}
        if (
            (self.reward_efficiency_alpha > 0 or self.reward_efficiency_beta > 0)
            and agent_data.verifier_type == "verified"
            and agent_data.per_turn_criteria_results
        ):
            n_criteria = len(agent_data.verification_criteria)
            original_reward = reward_score if reward_score is not None else 0.0
            reward_score, efficiency_metrics = self._compute_efficiency_reward(
                verify_reward=original_reward,
                per_turn_criteria_results=agent_data.per_turn_criteria_results,
                total_interaction_turns=interaction_turns,
                n_criteria=n_criteria,
            )
            logger.info(
                f"[Anti-Hacking] Efficiency reward: {original_reward:.3f} → {reward_score:.3f} "
                f"(beta={self.reward_efficiency_beta}, best_turn={efficiency_metrics.get('best_turn')}, "
                f"wasted={efficiency_metrics.get('wasted_turns', 0)}, clean_exit={efficiency_metrics.get('clean_exit')})"
            )
            # 让折扣后的 reward 同步反映到 total_reward（指标 reward_by_type/verified/mean
            # 取的是 verify_result 的 total_reward，否则 wandb 上看不到 α/β 折扣效果）。
            if efficiency_metrics.get("efficiency_applied"):
                verify_result_full["total_reward"] = reward_score
                verify_result_full["hard_reward"] = reward_score
                if isinstance(verify_reason_detail, dict):
                    verify_reason_detail["total_reward"] = reward_score
                    verify_reason_detail["hard_reward"] = reward_score
                    verify_reason_detail["efficiency_metrics"] = efficiency_metrics

        # ===== Bad Sample 统一检测与丢弃 =====
        # 以下情况 reward 不可信，统一设 -100 让 veRL 框架丢弃（不参与梯度）：
        #   1. verifier 调用失败（超时/异常）
        #   2. verifier 输出解析失败（JSON 格式问题）
        #   3. PCG 渲染全部失败（有调用但 0 成功，verifier 评估基于无图片/错误图片的地图）
        is_bad_sample = False
        bad_sample_reason = ""

        if not agent_data.verifier_call_success:
            is_bad_sample = True
            bad_sample_reason = "verifier_call_failed"
        elif verify_result_full.get("parse_failed"):
            is_bad_sample = True
            bad_sample_reason = "verifier_parse_failed"
        elif (agent_data.sandbox_call_attempts > 0 and agent_data.sandbox_call_successes == 0):
            is_bad_sample = True
            bad_sample_reason = "pcg_all_failed"

        if is_bad_sample:
            reward_score = -100
            logger.warning(
                f"[Bad Sample] reason={bad_sample_reason}, setting reward=-100 | "
                f"verifier_ok={agent_data.verifier_call_success}, "
                f"parse_failed={verify_result_full.get('parse_failed', False)}, "
                f"sandbox={agent_data.sandbox_call_successes}/{agent_data.sandbox_call_attempts}"
            )

        # 记录最终评估结果到日志
        # 提取三个 reward 分量
        total_reward = verify_reason_detail.get("total_reward", reward_score if reward_score is not None else 0.0)
        hard_reward = verify_reason_detail.get("hard_reward", 0.0)
        soft_reward = verify_reason_detail.get("soft_reward", 0.0)
        logger.info(
            f"Verify Result: Reward={reward_score}, "
            f"total_reward={total_reward}, hard_reward={hard_reward}, soft_reward={soft_reward}, "
            f"Reason={verify_reason[:200] if verify_reason else ''}, "
            f"InteractionTurns={interaction_turns}"
        )
        if verify_reason_detail:
            logger.info(f"Verify Detail: {json.dumps(verify_reason_detail, ensure_ascii=False, default=str)[:1000]}")
        
        # ===== 计算监控指标 =====
        tool_call_parse_rate = (
            agent_data.tool_call_parse_successes / agent_data.tool_call_parse_attempts
            if agent_data.tool_call_parse_attempts > 0 else 0.0
        )
        sandbox_success_rate = (
            agent_data.sandbox_call_successes / agent_data.sandbox_call_attempts
            if agent_data.sandbox_call_attempts > 0 else 0.0
        )
        verifier_success = 1.0 if agent_data.verifier_call_success else 0.0
        
        monitoring_stats = {
            "interaction_turns": interaction_turns,
            "tool_call_parse_attempts": agent_data.tool_call_parse_attempts,
            "tool_call_parse_successes": agent_data.tool_call_parse_successes,
            "tool_call_parse_failures": agent_data.tool_call_parse_failures,
            "tool_call_parse_rate": tool_call_parse_rate,
            "sandbox_call_attempts": agent_data.sandbox_call_attempts,
            "sandbox_call_successes": agent_data.sandbox_call_successes,
            "sandbox_call_failures": agent_data.sandbox_call_failures,
            "sandbox_success_rate": sandbox_success_rate,
            "verifier_call_success": verifier_success,
        }
        
        logger.info(
            f"Monitoring Stats: interaction_turns={interaction_turns}, "
            f"tool_call_parse_rate={tool_call_parse_rate:.2%} "
            f"({agent_data.tool_call_parse_successes}/{agent_data.tool_call_parse_attempts}, excl. final turn), "
            f"sandbox_success_rate={sandbox_success_rate:.2%}, "
            f"verifier_success={verifier_success}, "
            f"failed_reasons_count={len(agent_data.failed_reasons)}"
        )
        
        # 保存 trajectory 到日志目录
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_id = agent_data.user_query.get("sample_id", request_id[:8])
            # 从 run() 入口获取的 global_steps
            rl_step = global_steps
            log_subdir = os.path.join(self.log_dir, f"step{rl_step}_{sample_id}_{timestamp}")
            os.makedirs(log_subdir, exist_ok=True)
            
            # 保存 trajectory JSON（增强版，含完整监控信息）
            trajectory_data = {
                "request_id": request_id,
                "sample_id": sample_id,
                "rl_step": rl_step,
                "user_query": agent_data.user_query,
                "init_map": agent_data.init_map,
                "final_map": agent_data.current_map,
                "component_info": agent_data.component_info,
                "messages": [
                    self._serialize_v2_message(msg)
                    for msg in agent_data.messages
                ],
                "interaction_turns": interaction_turns,
                "reward": reward_score,
                "total_reward": total_reward,
                "hard_reward": hard_reward,
                "soft_reward": soft_reward,
                "verify_reason": verify_reason,
                # ===== 完整结构化 verify 详情（与 mix_reward_report.json 对齐） =====
                "verify_reason_detail": verify_reason_detail,
                # ===== 新增监控信息 =====
                "monitoring_stats": monitoring_stats,
                "failed_reasons": agent_data.failed_reasons,
                "turn_details": agent_data.turn_details,
                "sandbox_logs": agent_data.sandbox_logs,
                # ===== Anti-Hacking: 逐轮评估和效率折扣 =====
                "per_turn_criteria_results": agent_data.per_turn_criteria_results,
                "efficiency_metrics": efficiency_metrics,
                # ===== 逐轮 retrieve 检索结果 =====
                "turn_retrieve_results": agent_data.turn_retrieve_results,
            }
            
            trajectory_path = os.path.join(log_subdir, "trajectory.json")
            with open(trajectory_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, ensure_ascii=False, indent=2)

            # 保存逐轮渲染图（images/turn_N/）
            if agent_data.turn_images:
                image_root = os.path.join(log_subdir, "images")
                for turn_idx, turn_imgs in enumerate(agent_data.turn_images):
                    turn_dir = os.path.join(image_root, f"turn_{turn_idx + 1}")
                    os.makedirs(turn_dir, exist_ok=True)
                    for img_i, img in enumerate(turn_imgs):
                        if isinstance(img, Image.Image):
                            img.save(os.path.join(turn_dir, f"{img_i}.jpg"))
                logger.info(f"  保存 {len(agent_data.turn_images)} 轮渲染图 -> {image_root}")

            # 也保留旧的合并图片（兼容，后续可删除）
            if agent_data.image_data and not agent_data.turn_images:
                image_dir = os.path.join(log_subdir, "images")
                os.makedirs(image_dir, exist_ok=True)
                for i, img in enumerate(agent_data.image_data):
                    if isinstance(img, Image.Image):
                        img.save(os.path.join(image_dir, f"image_{i}.jpg"))
                        
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")
        
        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        
        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=agent_data.extra_fields,
            reward_score=reward_score,
        )
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "final_map": agent_data.current_map,
            "verify_reason": verify_reason,
            # === 完整结构化 verify 详情（与 mix_reward_report.json 对齐） ===
            "verify_reason_detail": verify_reason_detail,
            # === reward 三分量（便于 val/critic 日志展示） ===
            "total_reward": total_reward,
            "hard_reward": hard_reward,
            "soft_reward": soft_reward,
            "render_history": agent_data.render_history,
            "log_dir": log_subdir if 'log_subdir' in locals() else None,
            # === verify_result 完整结果（传递给 reward 函数） ===
            "verify_result": verify_result_full,
            # === query 元数据（便于 reward 分流和监控） ===
            "verifier_type": agent_data.verifier_type,
            "query_tag": agent_data.query_tag,
            "query_type": agent_data.query_type,
            # ===== 3+1 核心监控指标（传递到 wandb） =====
            # a. verifier_call_success: verifier API 调用成功（非超时/异常）
            "map_gen_verifier_call_success_rate": 1.0 if agent_data.verifier_call_success else 0.0,
            # b. verifier_parse_success: verifier 输出成功解析为有效 JSON
            #    对 verified (rule-based) 类型始终为 1.0，仅 unverified 可能为 0.0
            "map_gen_verifier_parse_success_rate": (
                0.0 if verify_result_full.get("parse_failed") else 1.0
            ),
            # b. instruction_following: 模型输出了 tool_call 内容时的解析成功率
            #    分子 = 成功解析的次数，分母 = 成功 + 失败（排除最后一轮没有 tool_call 的情况）
            #    如果分母为 0（模型从未尝试输出 tool_call），设为 None 不参与统计
            "map_gen_instruction_following_rate": (
                agent_data.tool_call_parse_successes / (agent_data.tool_call_parse_successes + agent_data.tool_call_parse_failures)
                if (agent_data.tool_call_parse_successes + agent_data.tool_call_parse_failures) > 0
                else None
            ),
            # c. pcg_sandbox 成功率（所有调用，含转换失败，保持原语义不变）
            "map_gen_pcg_sandbox_success_rate": (
                agent_data.sandbox_call_successes / agent_data.sandbox_call_attempts
                if agent_data.sandbox_call_attempts > 0
                else None
            ),
            # c2. PCG 工具解析成功率：地图格式→actors 转换成功的比例
            #     反映模型输出的地图结构是否能被正确解析（component 匹配等）
            "map_gen_pcg_convert_success_rate": (
                (agent_data.pcg_convert_attempts - agent_data.pcg_convert_failures) / agent_data.pcg_convert_attempts
                if agent_data.pcg_convert_attempts > 0
                else None
            ),
            # c3. PCG 渲染服务成功率：仅统计转换成功后实际调用渲染服务的成功率
            #     反映渲染服务本身的稳定性（超时/空图/服务异常等）
            "map_gen_pcg_render_success_rate": (
                agent_data.pcg_render_successes / agent_data.pcg_render_attempts
                if agent_data.pcg_render_attempts > 0
                else None
            ),
            # 交互轮次（辅助指标）
            "map_gen_interaction_turns": interaction_turns,
            # ===== Bad Sample 标记（传递到 wandb） =====
            "map_gen_bad_sample": 1.0 if is_bad_sample else 0.0,
            "map_gen_bad_sample_reason": bad_sample_reason if is_bad_sample else "",
            # ===== Anti-Hacking: 效率折扣指标 =====
            "map_gen_efficiency_applied": 1.0 if efficiency_metrics.get("efficiency_applied") else 0.0,
            "map_gen_wasted_turns": float(efficiency_metrics.get("wasted_turns", 0)),
            "map_gen_best_turn": float(efficiency_metrics.get("best_turn", -1)),
            "map_gen_best_pass_count": float(efficiency_metrics.get("best_pass_count", 0)),
            "map_gen_clean_exit": 1.0 if efficiency_metrics.get("clean_exit") else 0.0,
            "map_gen_efficiency_reward": float(efficiency_metrics.get("efficiency_reward", reward_score if reward_score else 0)),
            "map_gen_original_verify_reward": float(efficiency_metrics.get("original_verify_reward", reward_score if reward_score else 0)),
        })
        return output
