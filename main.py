"""
main.py — 3D 世界生成 / 修改的agent 采样主程序

支持根据 task_setting 自动切换：
  - task_setting == "generate" → 从零搭建场景（retrieve → add → 渲染）
  - 其他（refine 等）         → 基于初始场景修改（观察 → 调元件 → 渲染）

前置条件：先按 assets_retrieval/README.md 与 render_in_blender/README.md
把检索服务(默认 8081)与 PCG 渲染服务(默认 8080)跑起来。

用法：
  # refine 任务
  python main.py \
    --base_data_dir data/test \
    --log_dir log/eval_refine \
    --model_type gemini \
    --model_name gemini-2.5-pro \
    --server http://localhost:8080 \
    --retrieve_server http://localhost:8081

  # generate 任务
  python main.py \
    --base_data_dir data/test \
    --log_dir log/eval_generate \
    --model_type gemini \
    --model_name gemini-2.5-pro \
    --server http://localhost:8080 \
    --task_setting generate

  # 自动检测（从 query.json 的 task_setting 字段）
  python main.py \
    --base_data_dir data/test \
    --log_dir log/eval_mix \
    --model_type gemini \
    --model_name gemini-2.5-pro
"""

import os
import sys
import json
import glob
import copy
import math
import shutil
import logging
import argparse

# ── utils 路径注入（必须在最前）──────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.join(_SCRIPT_DIR, "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

# ── 工具模块 ───────────────────────────────────────────────────────────────────
from llm import MODEL_TYPE_MAP
from tools import rotation_and_translation, delete, add
from pcg_render import (
    gradio_render,
    llm_output_to_actors,
    normalize_tool_call,
    fix_flat_args,
    fc_to_sft_dict,
)
# pcg_render 可能改动 sys.path，重新确保 utils/ 优先
if sys.path[0] != _UTILS_DIR:
    sys.path.insert(0, _UTILS_DIR)
from scene_utils import (
    call_retrieve_for_fc,
    split_function_calls,
    apply_scene_calls_to_llm_output,
    format_retrieve_responses_for_user,
    enrich_component_info_for_generate,
)
from component_info_builder import load_item_infos
from asset_retrieval_client import AssetRetrievalClient
from prompt import (
    get_system_prompt,
    FORMAT_PROMPT_REFINE,
    FORMAT_PROMPT_GENERATE_TURN1,
)

# ══════════════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════════════

GRADIO_SERVER_REFINE   = os.environ.get("VIBEWORLD_RENDER_SERVER", "http://localhost:8080")
GRADIO_SERVER_GENERATE = os.environ.get("VIBEWORLD_RENDER_SERVER", "http://localhost:8080")
RETRIEVE_SERVER_DEFAULT = os.environ.get("VIBEWORLD_RETRIEVE_SERVER", "http://localhost:8081")

# PCG item_infos 加载（generate 渲染 & retrieve 白名单用）
# ⚠️ 必须与检索服务(8081)同一 id 体系(5位新id)。component_info_builder 的
# DEFAULT_ITEM_INFOS_PATH 是旧的 8位id(6803条 item_infos_dream_creator)，会把检索服务
# 返回的 5位id 结果在 call_retrieve_for_fc 白名单处全部过滤成空 → 场景空/actors=0。
# 显式指向 render_in_blender/assets/item_infos.json(2617条5位id，与服务/main_distill.sh 一致)。
_PCG_ITEM_INFOS_PATH = os.environ.get(
    "PCG_ITEM_INFOS",
    os.path.join(_SCRIPT_DIR, "render_in_blender", "assets", "item_infos.json"),
)
try:
    _PCG_ITEM_INFOS: dict = load_item_infos(path=_PCG_ITEM_INFOS_PATH)
    _PCG_WHITELIST: set = set(_PCG_ITEM_INFOS.keys())
    logging.info(f"PCG item_infos: {len(_PCG_ITEM_INFOS)} 条 from {_PCG_ITEM_INFOS_PATH}")
except Exception as _e:
    _PCG_ITEM_INFOS = {}
    _PCG_WHITELIST = set()
    logging.warning(f"PCG item_infos 加载失败: {_e}")

# ══════════════════════════════════════════════════════════════════════════════
# Tools Schema
# ══════════════════════════════════════════════════════════════════════════════

TOOLS_SCHEMA_REFINE = [
    {
        "type": "function",
        "function": {
            "name": "rotation_and_translation",
            "description": "用于旋转以及平移场景中已有的元件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "corrections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_data": {"type": "object"},
                                "modified_data":  {"type": "object"},
                            },
                            "required": ["original_data", "modified_data"],
                        },
                    }
                },
                "required": ["corrections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": "用于删除场景中不合理的元件。",
            "parameters": {
                "type": "object",
                "properties": {"modified_data": {"type": "array", "items": {"type": "object"}}},
                "required": ["modified_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "用于在场景中添加新元件（name 和 type_id 均必填）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "modified_data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":    {"type": "string"},
                                "type_id": {"type": "string",
                                            "description": "来自 retrieve_assets 的 8 位资产 ID"},
                            },
                            "required": ["name", "type_id"],
                        },
                    }
                },
                "required": ["modified_data"],
            },
        },
    },
]

TOOLS_SCHEMA_GENERATE = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_assets",
            "description": "从资产库检索 top-K 候选资产。必填 entity_name（中文）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "top_k":       {"type": "integer"},
                    "size_class":  {"type": "string"},
                    "scene_limit": {"type": "string"},
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotation_and_translation",
            "description": "用于旋转以及平移场景中已有的元件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "corrections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_data": {"type": "object"},
                                "modified_data":  {"type": "object"},
                            },
                            "required": ["original_data", "modified_data"],
                        },
                    }
                },
                "required": ["corrections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": "用于删除场景中不合理的元件。",
            "parameters": {
                "type": "object",
                "properties": {"modified_data": {"type": "array", "items": {"type": "object"}}},
                "required": ["modified_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "添加新元件（必须传 type_id，来自 retrieve_assets 返回）。",
            "parameters": {
                "type": "object",
                "properties": {"modified_data": {"type": "array", "items": {"type": "object"}}},
                "required": ["modified_data"],
            },
        },
    },
]

TOOLS_MAP_REFINE = {
    "rotation_and_translation": rotation_and_translation,
    "delete": delete,
    "add": add,
}

# ══════════════════════════════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logger(log_dir: str) -> logging.Logger:
    log_file = os.path.join(log_dir, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Refine 分支
# ══════════════════════════════════════════════════════════════════════════════

def _agent_step_refine(bot, message, images, llm_output, debug=False):
    """一轮 LLM 调用 → 执行工具 → 更新场景。"""
    reasoning, fcs = bot.mllm(message, images)
    if debug:
        print(f"[reasoning] {str(reasoning)[:300]}")
        print(f"[fcs] {fcs}")

    updated = copy.deepcopy(llm_output)
    is_terminate = False
    error_msg = ""
    final_content = None

    if fcs:
        for raw_tc in (fcs if isinstance(fcs, list) else [fcs]):
            name, args = normalize_tool_call(raw_tc)
            if name is None:
                continue
            if name == "terminate":
                is_terminate = True
                continue
            if name in TOOLS_MAP_REFINE:
                args = fix_flat_args(name, args)
                try:
                    updated = TOOLS_MAP_REFINE[name](llm_output=updated, **args)
                except Exception as e:
                    logging.error(f"{name} 执行失败: {e}")
                    error_msg += f"{name}调用失败 "
            else:
                logging.warning(f"未知工具: {name}")
    else:
        is_terminate = True
        # 从 history 提取最后一轮 assistant 文本
        for msg in reversed(bot.history or []):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                final_content = msg.get("content") or None
                break

    return updated, is_terminate, reasoning, fcs, final_content, error_msg


def run_sample_refine(
    sample_dir, log_dir, bot, max_turns=8,
    quality="低质量 (快速预览)", debug=False,
    sys_prompt=None, server_url=None,
):
    """处理一个 refine sample。"""
    logger = _setup_logger(log_dir)
    logger.info(f"=== [refine] {os.path.basename(sample_dir)} ===")

    try:
        with open(os.path.join(sample_dir, "init_map.json"), encoding="utf-8") as f:
            llm_output = json.load(f)
        with open(os.path.join(sample_dir, "component_info.json"), encoding="utf-8") as f:
            component_info = json.load(f)
        with open(os.path.join(sample_dir, "query.json"), encoding="utf-8") as f:
            query_info = json.load(f)
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return False

    llm_output.pop("地图信息", None)
    component_info.pop("地图信息", None)
    theme = query_info.get("theme", "")
    description = query_info.get("description", "")

    # 初始图片
    images = sorted(glob.glob(os.path.join(sample_dir, "image", "*.jpg"))) or \
             sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))

    # 输出目录准备
    final_image_dir = os.path.join(log_dir, "final_image")
    os.makedirs(final_image_dir, exist_ok=True)
    for src_name in ("init_map.json", "component_info.json", "query.json"):
        src = os.path.join(sample_dir, src_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(log_dir, src_name))
    init_image_dir = os.path.join(log_dir, "init_image")
    os.makedirs(init_image_dir, exist_ok=True)
    for img in images:
        shutil.copy(img, init_image_dir)

    # 撒点缓存
    scatter_cache = {}
    for depth_dir in [sample_dir, os.path.dirname(sample_dir)]:
        cache_path = os.path.join(depth_dir, "scatter_cache.json")
        if os.path.exists(cache_path):
            try:
                scatter_cache = json.load(open(cache_path, encoding="utf-8"))
                logger.info(f"已加载撒点缓存 ({len(scatter_cache)} 条)")
            except Exception:
                pass
            break

    # 固定相机参数
    fixed_cam = None
    for depth_dir in [sample_dir, os.path.dirname(sample_dir)]:
        cam_path = os.path.join(depth_dir, "camera_params.json")
        if os.path.exists(cam_path):
            try:
                fixed_cam = json.load(open(cam_path, encoding="utf-8"))
            except Exception:
                pass
            break

    _sys = sys_prompt or get_system_prompt("refine")
    sft = {"system_instruction": _sys, "task_setting": "refine", "conversations": []}
    bot.reset()
    is_terminate = False
    turn = 1
    error_info = ""

    while not is_terminate and turn <= max_turns:
        logger.info(f"--- 第 {turn} 轮 ---")
        turn_dir = os.path.join(log_dir, f"turn_{turn}")
        turn_image_dir = os.path.join(turn_dir, "image")
        os.makedirs(turn_image_dir, exist_ok=True)

        if turn == 1:
            user_message = FORMAT_PROMPT_REFINE.format(
                theme=theme, scene_description=description,
                element_info=llm_output,
                component_info=list(component_info.keys()),
            )
        else:
            map_str = json.dumps(llm_output, ensure_ascii=False, indent=4)
            tools_hint = (
                "当前可用tools如下（rotation_and_translation(arguments: corrections), "
                "delete(arguments: modified_data)，add(arguments: modified_data)）"
            )
            if images:
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

        error_info = ""
        role = "user" if turn == 1 else "tool"
        sft["conversations"].append({
            "role": role, "content": user_message,
            "images": list(images), "turn_n": turn,
        })

        llm_output, is_terminate, reasoning, fcs, final_content, tool_err = _agent_step_refine(
            bot=bot, message=user_message, images=images,
            llm_output=llm_output, debug=debug,
        )
        if tool_err:
            error_info += tool_err

        if fcs:
            sft_fc = [fc_to_sft_dict(fc) for fc in (fcs if isinstance(fcs, list) else [fcs])]
            sft_fc = [x for x in sft_fc if x]
            sft["conversations"].append({
                "role": "assistant", "content": reasoning,
                "function_calls": sft_fc or None, "turn_n": turn,
            })
        else:
            sft["conversations"].append({
                "role": "assistant", "content": reasoning,
                "final_content": final_content,
                "function_calls": None, "turn_n": turn,
            })

        with open(os.path.join(turn_dir, "map.json"), "w", encoding="utf-8") as f:
            json.dump(llm_output, f, ensure_ascii=False, indent=4)

        if is_terminate:
            for img in images:
                shutil.copy(img, final_image_dir)
            break

        # 渲染
        actors, parse_err = llm_output_to_actors(llm_output, component_info, scatter_cache)
        if parse_err:
            error_info += parse_err
            images = []
        else:
            pcg_path = os.path.join(turn_dir, "pcg_render.json")
            with open(pcg_path, "w", encoding="utf-8") as f:
                json.dump([{"actors": actors}], f, ensure_ascii=False, indent=2)
            lens = fixed_cam.get("lens", 31) if fixed_cam else 31
            cam_pos = fixed_cam.get("cam_pos") if fixed_cam else None
            cam_target = fixed_cam.get("cam_target") if fixed_cam else None
            try:
                images, pcg_err = gradio_render(
                    None, actors, turn_image_dir,
                    quality=quality, lens=lens, pcg_timeout=120,
                    cam_pos_override=cam_pos, cam_target_override=cam_target,
                    server_url=server_url or GRADIO_SERVER_REFINE,
                )
                if pcg_err:
                    error_info += pcg_err
            except Exception as e:
                logging.error(f"渲染异常: {e}")
                error_info += f"渲染异常: {e} "
                images = []

        turn += 1

    if images:
        for img in images:
            shutil.copy(img, final_image_dir)

    with open(os.path.join(log_dir, "final_map.json"), "w", encoding="utf-8") as f:
        json.dump(llm_output, f, ensure_ascii=False, indent=4)
    with open(os.path.join(log_dir, "sft_trajectory.json"), "w", encoding="utf-8") as f:
        json.dump(sft, f, ensure_ascii=False, indent=4)

    logger.info(f"[{os.path.basename(sample_dir)}] refine 完成")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Generate 分支
# ══════════════════════════════════════════════════════════════════════════════

def run_sample_generate(
    sample_dir, log_dir, bot, retrieve_client,
    max_turns=8, quality="低质量 (快速预览)", debug=False,
    sys_prompt=None, server_url=None,
):
    """处理一个 generate (from-scratch) sample。"""
    logger = logging.getLogger(__name__)
    logger.info(f"=== [generate] {os.path.basename(sample_dir)} ===")

    try:
        with open(os.path.join(sample_dir, "query.json"), encoding="utf-8") as f:
            query_info = json.load(f)
    except Exception as e:
        logger.error(f"加载 query.json 失败: {e}")
        return False

    user_query = query_info.get("user_query", "")
    theme = query_info.get("theme", "")
    description = query_info.get("description") or query_info.get("scene_description", "")

    _setup_logger(log_dir)
    shutil.copy(os.path.join(sample_dir, "query.json"), os.path.join(log_dir, "query.json"))
    os.makedirs(os.path.join(log_dir, "final_image"), exist_ok=True)

    _sys = sys_prompt or get_system_prompt("generate")
    sft = {"system_instruction": _sys, "task_setting": "generate", "conversations": []}

    llm_output: dict = {}
    component_info: dict = {}
    images: list = []
    bot.reset()

    is_terminate = False
    turn = 1
    error_info = ""
    pending_retrieves = []

    while not is_terminate and turn <= max_turns:
        logger.info(f"--- 第 {turn} 轮 ---")
        turn_dir = os.path.join(log_dir, f"turn_{turn}")
        os.makedirs(os.path.join(turn_dir, "image"), exist_ok=True)

        # 构造 user message
        if turn == 1:
            if user_query:
                user_message = FORMAT_PROMPT_GENERATE_TURN1.format(user_query=user_query)
            else:
                user_message = f"# 用户输入\n\n主题：{theme}\n描述：{description}"
            role = "user"
        else:
            parts = []
            has_retrieve = bool(pending_retrieves)
            if pending_retrieves:
                parts.append(format_retrieve_responses_for_user(pending_retrieves))
                pending_retrieves = []

            tools_hint = (
                "当前可用 tools: retrieve_assets / add / rotation_and_translation / delete。"
                "add 必须传 type_id（来自之前 retrieve_assets 返回过的）。"
            )
            current_map_str = (json.dumps(llm_output, ensure_ascii=False, indent=2)
                               if llm_output else "(场景为空，还没有摆放任何元件)")

            if images:
                parts.append(
                    f"<tool_response>本轮场景已渲染，当前元件信息:\n{current_map_str}\n\n"
                    f"5视角图(左/右/前/后/俯):<image><image><image><image><image>。\n"
                    f"{tools_hint}</tool_response>"
                )
            elif llm_output:
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
            user_message = "\n".join(parts)
            role = "tool"

        error_info = ""
        sft["conversations"].append({
            "role": role, "content": user_message,
            "images": list(images), "turn_n": turn,
        })

        # 调用 LLM
        reasoning, fcs = bot.mllm(user_message, images)
        if debug:
            print(f"[Turn {turn}] reasoning: {str(reasoning)[:200]}")
            print(f"[Turn {turn}] fcs: {len(fcs) if fcs else 0}")

        fc_list = (fcs if isinstance(fcs, list) else [fcs]) if fcs else []

        if fc_list:
            retrieve_calls, scene_calls = split_function_calls(fc_list)
            sft_fc = [fc_to_sft_dict(fc) for fc in retrieve_calls + scene_calls]
            sft_fc = [x for x in sft_fc if x]
        else:
            retrieve_calls, scene_calls, sft_fc = [], [], []

        sft["conversations"].append({
            "role": "assistant", "content": reasoning or "",
            "function_calls": sft_fc or None, "turn_n": turn,
        })

        if not fc_list:
            is_terminate = True
            for img in images:
                shutil.copy(img, os.path.join(log_dir, "final_image"))
            break

        # 处理 retrieve
        if retrieve_calls:
            for fc in retrieve_calls:
                resp = call_retrieve_for_fc(
                    fc, retrieve_client,
                    item_infos=_PCG_ITEM_INFOS or None,
                    pcg_whitelist=_PCG_WHITELIST or None,
                )
                pending_retrieves.append(resp)
                for item in resp.get("response", {}).get("results", []):
                    name_r = item.get("name", "")
                    if name_r:
                        component_info[name_r] = {
                            "typeId": item.get("type_id", ""),
                            "native_bbox_m": item.get("native_bbox_m"),
                            "category": item.get("category_minor", ""),
                        }
            with open(os.path.join(turn_dir, "retrieve_responses.json"), "w", encoding="utf-8") as f:
                json.dump(pending_retrieves, f, ensure_ascii=False, indent=2)

        # 处理 scene_calls
        if scene_calls:
            try:
                new_output, scene_err = apply_scene_calls_to_llm_output(
                    scene_calls, llm_output, _PCG_ITEM_INFOS or {},
                )
                if new_output:
                    llm_output = new_output
                if scene_err:
                    error_info += scene_err
            except Exception as e:
                error_info += f"scene_call 异常: {e} "
                logger.warning(f"scene_call 异常: {e}")

        with open(os.path.join(turn_dir, "map.json"), "w", encoding="utf-8") as f:
            json.dump(llm_output, f, ensure_ascii=False, indent=4)

        # 渲染（只在有 scene_calls 且地图非空时）
        if scene_calls and llm_output:
            if _PCG_ITEM_INFOS:
                render_ci = enrich_component_info_for_generate(
                    base_component_info={},
                    llm_output=llm_output,
                    item_infos=_PCG_ITEM_INFOS,
                )
            else:
                render_ci = component_info

            actors, parse_err = llm_output_to_actors(llm_output, render_ci)
            if parse_err:
                error_info += parse_err
                images = []
            elif actors:
                pcg_path = os.path.join(turn_dir, "pcg_render.json")
                with open(pcg_path, "w", encoding="utf-8") as f:
                    json.dump([{"actors": actors}], f, ensure_ascii=False, indent=2)
                try:
                    images, pcg_err = gradio_render(
                        None, actors, os.path.join(turn_dir, "image"),
                        quality=quality, lens=31, pcg_timeout=120,
                        server_url=server_url or GRADIO_SERVER_GENERATE,
                    )
                    if pcg_err:
                        error_info += pcg_err
                except Exception as e:
                    error_info += f"渲染异常:{e} "
                    images = []
            else:
                images = []

        turn += 1

    if images:
        for img in images:
            shutil.copy(img, os.path.join(log_dir, "final_image"))

    with open(os.path.join(log_dir, "final_map.json"), "w", encoding="utf-8") as f:
        json.dump(llm_output, f, ensure_ascii=False, indent=4)
    with open(os.path.join(log_dir, "final_component_info.json"), "w", encoding="utf-8") as f:
        json.dump(component_info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(log_dir, "sft_trajectory.json"), "w", encoding="utf-8") as f:
        json.dump(sft, f, ensure_ascii=False, indent=4)

    n_actors = sum(
        len(v) for cat in llm_output.values() if isinstance(cat, dict)
        for v in cat.values() if isinstance(v, list)
    )
    logger.info(f"[generate] 完成 turns={turn - 1} actors={n_actors}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Main：根据 task_setting 自动分发
# ══════════════════════════════════════════════════════════════════════════════

def main(
    base_data_dir, log_dir,
    model_type="gemini", model_name="gemini-3.1-pro-preview",
    max_turns=8, debug=False,
    server_url=None, quality="低质量 (快速预览)",
    max_cases=0, case_filter=None,
    task_setting=None,
    retrieve_server=RETRIEVE_SERVER_DEFAULT,
):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info(f"task_setting 检测: {'自动' if task_setting is None else task_setting}")
    logging.info(f"渲染服务: {server_url or '自动按 task_setting 选择'}")

    retrieve_client = AssetRetrievalClient(base_url=retrieve_server)
    generate_sys = get_system_prompt("generate")
    refine_sys   = get_system_prompt("refine")

    n_done = 0
    for folder_name in sorted(os.listdir(base_data_dir)):
        sample_dir = os.path.join(base_data_dir, folder_name)
        if not os.path.isdir(sample_dir):
            continue
        if case_filter and folder_name not in case_filter:
            continue
        if max_cases > 0 and n_done >= max_cases:
            break

        case_log_dir = os.path.join(log_dir, folder_name)
        if os.path.exists(os.path.join(case_log_dir, "final_map.json")):
            logging.info(f"⏩ 已处理，跳过: {folder_name}")
            continue

        os.makedirs(case_log_dir, exist_ok=True)
        logging.info(f"\n{'='*50}\n处理: {folder_name}\n{'='*50}")

        # 检测 task_setting
        q_path = os.path.join(sample_dir, "query.json")
        detected = task_setting
        if detected is None and os.path.exists(q_path):
            try:
                detected = json.load(open(q_path, encoding="utf-8")).get("task_setting", "refine")
            except Exception:
                detected = "refine"
        logging.info(f"  → task_setting = {detected}")

        try:
            if detected == "generate":
                bot = MODEL_TYPE_MAP[model_type](
                    model_name=model_name,
                    system_instruction=generate_sys,
                    tools=TOOLS_SCHEMA_GENERATE,
                )
                ok = run_sample_generate(
                    sample_dir, case_log_dir, bot=bot,
                    retrieve_client=retrieve_client,
                    max_turns=max_turns, quality=quality, debug=debug,
                    sys_prompt=generate_sys, server_url=server_url,
                )
            else:
                bot = MODEL_TYPE_MAP[model_type](
                    model_name=model_name,
                    system_instruction=refine_sys,
                    tools=TOOLS_SCHEMA_REFINE,
                )
                ok = run_sample_refine(
                    sample_dir, case_log_dir, bot=bot,
                    max_turns=max_turns, quality=quality, debug=debug,
                    sys_prompt=refine_sys, server_url=server_url,
                )
        except Exception as e:
            logging.error(f"❌ {folder_name} 处理异常: {e}", exc_info=True)
            continue

        if ok:
            n_done += 1
            logging.info(f"✅ [{n_done}] {folder_name} ({detected}) done")

    logging.info(f"\n全部完成: {n_done} 个 case")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Gen 模型推理采样评估")
    parser.add_argument("--base_data_dir", type=str, required=True)
    parser.add_argument("--log_dir",       type=str, required=True)
    parser.add_argument("--model_type",    type=str, default="gemini",
                        choices=list(MODEL_TYPE_MAP.keys()))
    parser.add_argument("--model_name",    type=str, default="gemini-2.5-pro")
    parser.add_argument("--max_turns",     type=int, default=8)
    parser.add_argument("--debug",         action="store_true")
    parser.add_argument("--server",        type=str, default=None, help="Gradio 渲染服务地址")
    parser.add_argument("--quality",       type=str, default="低质量 (快速预览)")
    parser.add_argument("--max_cases",     type=int, default=0, help="0=全量")
    parser.add_argument("--cases",         type=str, default=None,
                        help="只处理指定 case，逗号分隔")
    parser.add_argument("--task_setting",  type=str, default=None,
                        choices=["refine", "generate"],
                        help="强制指定任务类型（None=从 query.json 自动检测）")
    parser.add_argument("--retrieve_server", type=str, default=RETRIEVE_SERVER_DEFAULT)
    args = parser.parse_args()

    main(
        base_data_dir=args.base_data_dir,
        log_dir=args.log_dir,
        model_type=args.model_type,
        model_name=args.model_name,
        max_turns=args.max_turns,
        debug=args.debug,
        server_url=args.server,
        quality=args.quality,
        max_cases=args.max_cases,
        case_filter=[c.strip() for c in args.cases.split(",")] if args.cases else None,
        task_setting=args.task_setting,
        retrieve_server=args.retrieve_server,
    )
