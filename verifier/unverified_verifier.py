"""
unverified_verifier.py — Unverified Reward 计算器（合并版 v1+v2）

合并说明：
  - 工具函数、rule-based check、元件摘要等来自原 v1（unverified_verifier.py）
  - Hard H1/H2 + H3-VU/VR 拆分调用来自原 v2（unverified_verifier_v2.py）
  - 主入口为 evaluate_unverified_case_v2（v2 流程，H3 拆为 VU+VR）
  - 保留 evaluate_unverified_case（v1 流程）作为兼容入口

Reward 设计（v2，soft 已移除）：
  Hard 全通过（H1==1 AND H2==1 AND H3_pass==1 AND H4==1）：reward = 1.0
  任一 Hard fail：reward = 0.0

Usage（离线批量评估）：
  python unverified_verifier.py \
    --result_dir <result_dir> \
    --output_file <output.json>
"""

import os
import sys
import json
import glob
import re
import logging
import argparse
from typing import List, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../utils"))
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from llm import (
    GeminiMultiChat, QwenMultiChat, OpenAIMultiChat,
    BailianMultiChat, OfflineLLM, MODEL_TYPE_MAP,
)

from prompts import (
    HARD_SYSTEM_PROMPT, HARD_USER_PROMPT,
    HARD_H12_SYSTEM_PROMPT, HARD_H12_USER_PROMPT,
    H3_VURR_SYSTEM_PROMPT, H3_VURR_USER_PROMPT,
)


# ============================================================
# JSON 解析工具
# ============================================================

def extract_and_parse_json(text: str) -> Optional[dict]:
    if not text or not text.strip():
        return None
    try:
        text = text.strip()
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None


def _extract_content_from_history(llm_bot) -> str:
    if not hasattr(llm_bot, 'history') or not llm_bot.history:
        return ""
    last_msg = llm_bot.history[-1]
    if hasattr(last_msg, 'parts'):
        texts = [p.text for p in (last_msg.parts or [])
                 if not getattr(p, 'thought', False) and getattr(p, 'text', None)]
        return "".join(texts).strip()
    if isinstance(last_msg, dict):
        c = last_msg.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _extract_reasoning_from_history(llm_bot) -> str:
    if not hasattr(llm_bot, 'history') or not llm_bot.history:
        return ""
    last_msg = llm_bot.history[-1]
    if hasattr(last_msg, 'parts'):
        texts = [p.text for p in (last_msg.parts or [])
                 if getattr(p, 'thought', False) and getattr(p, 'text', None)]
        return "".join(texts).strip()
    return ""


def call_llm_and_parse_with_raw(llm_bot, user_message: str, image_list: List[str],
                                 label: str = "",
                                 required_keys: List[str] = None) -> tuple:
    """调用 LLM + 双路径 JSON 提取。返回 (parsed_dict_or_None, raw_output_dict)。"""
    if hasattr(llm_bot, 'reset'):
        llm_bot.reset()
    else:
        llm_bot.history = []

    raw_output = {"reasoning": "", "content": "", "parse_success": False}

    try:
        print(f"[DEBUG VERIFIER call_llm_and_parse] label={label}, image_count={len(image_list)}, "
              f"user_msg_len={len(user_message)}, required_keys={required_keys}")

        reasoning_text, _ = llm_bot.mllm(user_message, image_list)

        content_text = _extract_content_from_history(llm_bot)
        reasoning_full = _extract_reasoning_from_history(llm_bot)

        raw_output["reasoning"] = reasoning_full or (reasoning_text or "")
        raw_output["content"] = content_text

        def _is_valid(parsed):
            if not parsed or not isinstance(parsed, dict):
                return False
            if required_keys:
                return any(k in parsed for k in required_keys)
            return True

        result = extract_and_parse_json(content_text) if content_text else None
        if _is_valid(result):
            raw_output["parse_success"] = True
            raw_output["parse_source"] = "content"
            return result, raw_output

        result = extract_and_parse_json(reasoning_text) if reasoning_text else None
        if _is_valid(result):
            raw_output["parse_success"] = True
            raw_output["parse_source"] = "reasoning"
            return result, raw_output

        logging.warning(f"  [{label}] LLM 返回无法解析为有效 JSON (required_keys={required_keys})")
        return None, raw_output

    except Exception as e:
        logging.error(f"  [{label}] LLM 调用异常: {e}")
        raw_output["error"] = str(e)
        return None, raw_output


# ============================================================
# 知识库（预留接口）
# ============================================================

def get_knowledge_base_context(final_map_json: dict) -> str:
    return "（知识库暂未接入，请基于你的专业知识和视觉信息进行判断）"


# ============================================================
# Rule-based 检测
# ============================================================

def rule_based_collision_check(final_map_json: dict) -> dict:
    """H4 碰撞/重叠检测（AABB rule-based，不经过 LLM）。"""
    TOLERANCE = 0.5
    VOLUME_THRESHOLD = 10.0

    SKIP_KEYWORDS = {
        "粒子光束", "辉光", "拖尾星星", "流萤", "环形光束", "金币雨",
        "香炉烟雾", "炊烟", "光芒球", "火焰", "泡泡", "气泡",
        "涟漪", "金币堆", "光柱", "烟雾", "火山爆发", "瀑布水花",
        "蝴蝶", "飞鸟", "蝙蝠", "鱼群", "水草", "夏日水草", "水母",
        "海星", "大海草",
    }
    TERRAIN_KEYWORDS = {
        "岩石山体", "天然石块", "灰岩", "沙地地块", "长条地块",
        "天然礁石", "夏日礁石", "冬日石块", "礁石", "大石头",
        "岛屿石", "草地", "地块", "国风云纹", "假山",
    }

    def _should_skip(name):
        return any(name.startswith(kw) or kw in name for kw in SKIP_KEYWORDS)

    def _is_terrain(name):
        return any(name.startswith(kw) or kw in name for kw in TERRAIN_KEYWORDS)

    elements = []
    for cat_key, cat_val in final_map_json.items():
        if cat_key == "地图信息" or not isinstance(cat_val, dict):
            continue
        for sub_key, sub_val in cat_val.items():
            if not isinstance(sub_val, list):
                continue
            for elem in sub_val:
                name = elem.get("name", "未知")
                pos = elem.get("pos")
                ext = elem.get("Extend", [0, 0, 0])
                if pos is None or _should_skip(name):
                    continue
                if isinstance(pos, list) and len(pos) == 3 and isinstance(pos[0], (int, float)):
                    x, y, z = pos
                    ext = list(map(int, ext)) if len(ext) >= 1 else ext
                    ex = ext[0] if len(ext) >= 1 else 0
                    ey = ext[1] if len(ext) >= 2 else 0
                    ez = ext[2] if len(ext) >= 3 else 0
                    elements.append({
                        'name': name,
                        'x_min': x - ex, 'x_max': x + ex,
                        'y_min': y - ey, 'y_max': y + ey,
                        'z_min': z, 'z_max': z + ez,
                        'pos': pos, 'ext': ext,
                    })

    collisions = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            a, b = elements[i], elements[j]
            x_ov = min(a['x_max'], b['x_max']) - max(a['x_min'], b['x_min']) - TOLERANCE
            y_ov = min(a['y_max'], b['y_max']) - max(a['y_min'], b['y_min']) - TOLERANCE
            z_ov = min(a['z_max'], b['z_max']) - max(a['z_min'], b['z_min']) - TOLERANCE
            if x_ov > 0 and y_ov > 0 and z_ov > 0:
                vol = x_ov * y_ov * z_ov
                if vol < 0.1:
                    continue
                if _is_terrain(a['name']) or _is_terrain(b['name']):
                    continue
                collisions.append({'a': a['name'], 'b': b['name'],
                                   'a_pos': a['pos'], 'b_pos': b['pos'], 'overlap_vol': vol})

    collision_summary = {}
    for c in collisions:
        key = tuple(sorted([c['a'], c['b']]))
        if key not in collision_summary:
            collision_summary[key] = {'count': 0, 'max_vol': 0, 'details': []}
        collision_summary[key]['count'] += 1
        collision_summary[key]['max_vol'] = max(collision_summary[key]['max_vol'], c['overlap_vol'])
        collision_summary[key]['details'].append(c)

    issues = []
    for (name_a, name_b), info in sorted(collision_summary.items(), key=lambda x: -x[1]['max_vol']):
        if info['max_vol'] < VOLUME_THRESHOLD:
            continue
        problem = (f"{info['count']}处AABB重叠，最大重叠体积≈{info['max_vol']:.1f}m³"
                   if info['count'] > 1 else f"AABB重叠体积≈{info['max_vol']:.1f}m³")
        issues.append({
            "element_a": name_a, "element_b": name_b,
            "overlap_vol": round(info['max_vol'], 1),
            "count": info['count'], "problem": problem,
        })

    lines = []
    if not issues:
        lines.append("✅ 未检测到固定元件之间的明显碰撞/重叠。")
    else:
        lines.append(f"⚠️ 检测到 {len(issues)} 组元件碰撞/重叠：")
        for iss in issues:
            lines.append(f"- {iss['element_a']} ↔ {iss['element_b']}：{iss['problem']}")

    return {
        "pass": 1 if not issues else 0,
        "issues": issues,
        "collision_count": len(issues),
        "text_summary": "\n".join(lines),
    }


def rule_based_height_check(final_map_json: dict, init_map_json: dict = None) -> str:
    """H1 规则预检，返回可注入 prompt 的文本摘要。"""
    TERRAIN_KEYWORDS = {
        "岩石山体", "天然石块", "灰岩", "沙地地块", "长条地块",
        "天然礁石", "夏日礁石", "冬日石块", "礁石", "大石头",
    }
    EFFECT_KEYWORDS = {
        "粒子光束", "辉光", "拖尾星星", "流萤", "环形光束", "金币雨",
        "香炉烟雾", "炊烟", "光芒球", "火焰", "泡泡", "气泡",
        "涟漪", "金币堆", "光柱", "烟雾", "火山爆发", "瀑布水花", "蝴蝶",
    }
    FLYING_KEYWORDS = {"飞鸟", "蝙蝠"}
    AQUATIC_KEYWORDS = {"鱼群", "水草", "夏日水草", "水母", "海星", "大海草"}
    HANGING_KEYWORDS = {"巨型藤蔓", "藤蔓树枝", "藤蔓", "苔藓球", "转角屋檐"}

    def _kw(name, kws):
        return any(name.startswith(k) or k in name for k in kws)

    init_element_set = set()
    if init_map_json:
        for cat_val in init_map_json.values():
            if not isinstance(cat_val, dict):
                continue
            for sub_val in cat_val.values():
                if not isinstance(sub_val, list):
                    continue
                for elem in sub_val:
                    name = elem.get("name", "")
                    pos = elem.get("pos", [])
                    init_element_set.add(f"{name}|{json.dumps(pos, sort_keys=True)}")

    all_elements = []
    support_structures = []

    for cat_key, cat_val in final_map_json.items():
        if cat_key == "地图信息" or not isinstance(cat_val, dict):
            continue
        for sub_key, sub_val in cat_val.items():
            if not isinstance(sub_val, list):
                continue
            for elem in sub_val:
                name = elem.get("name", "未知")
                pos = elem.get("pos")
                ext = elem.get("Extend", [0, 0, 0])
                if pos is None:
                    continue
                if isinstance(pos, list) and len(pos) == 3 and isinstance(pos[0], (int, float)):
                    x, y, z = pos
                    ex = ext[0] if len(ext) >= 1 else 0
                    ey = ext[1] if len(ext) >= 2 else 0
                    ez = ext[2] if len(ext) >= 3 else 0
                    all_elements.append({'name': name, 'x': x, 'y': y, 'z': z,
                                         'ext': ext, 'is_range': False, 'pos': pos})
                    if ez > 3:
                        support_structures.append({
                            'name': name,
                            'x_min': x - ex, 'x_max': x + ex,
                            'y_min': y - ey, 'y_max': y + ey,
                            'z_bot': z, 'z_top': z + ez,
                        })
                elif isinstance(pos, list) and len(pos) == 2 and isinstance(pos[0], list):
                    z_min = pos[0][2] if len(pos[0]) >= 3 else 0
                    z_max = pos[1][2] if len(pos[1]) >= 3 else 0
                    all_elements.append({'name': name, 'z': z_min, 'z_max': z_max,
                                         'is_range': True, 'pos': pos, 'num': elem.get('num', '?')})

    def _find_supports(x, y, z, self_name):
        TOL, ZTOL = 3, 3
        return [f"{s['name']}(顶部Z={s['z_top']:.0f})"
                for s in support_structures
                if s['name'] != self_name
                and s['x_min'] - TOL <= x <= s['x_max'] + TOL
                and s['y_min'] - TOL <= y <= s['y_max'] + TOL
                and s['z_bot'] <= z <= s['z_top'] + ZTOL]

    cleared_lines, flagged_lines, supported_lines = [], [], []
    z0_count = 0
    terrain_cleared, effect_cleared, flying_cleared = {}, {}, {}
    aquatic_cleared, preexist_cleared, hanging_cleared = {}, {}, {}

    for elem in all_elements:
        name = elem['name']
        z = elem['z']
        is_range = elem['is_range']
        fingerprint = f"{name}|{json.dumps(elem['pos'], sort_keys=True)}"

        if is_range:
            z_max = elem.get('z_max', z)
            num = elem.get('num', '?')
            if _kw(name, EFFECT_KEYWORDS):
                effect_cleared[name] = effect_cleared.get(name, 0) + 1; continue
            if _kw(name, FLYING_KEYWORDS):
                flying_cleared[name] = flying_cleared.get(name, 0) + 1; continue
            if _kw(name, AQUATIC_KEYWORDS) and z <= 0:
                aquatic_cleared[name] = aquatic_cleared.get(name, 0) + 1; continue
            if z == 0:
                z0_count += 1
                if z_max > 5:
                    supported_lines.append(
                        f"- {name} ×{num}（范围生成 Z=[{z},{z_max}]）：底部从地面开始，"
                        f"高处部分可能依附在山体/岩石斜面上，请结合截图判断")
                continue
            if z > 0:
                supported_lines.append(
                    f"- {name} ×{num}（范围生成 Z=[{z},{z_max}]）：起始高度 Z={z}，"
                    f"可能是撒在山体/岩石表面上的元件，请结合截图判断是否有支撑")
                continue
            if z < -5:
                cleared_lines.append(f"- {name}: Z={z}，不可见，不参与评判")
            continue

        if z < -5:
            cleared_lines.append(f"- {name}: Z={z}，不可见（地下/水下深处），不参与评判"); continue
        if _kw(name, EFFECT_KEYWORDS):
            effect_cleared[name] = effect_cleared.get(name, 0) + 1; continue
        if _kw(name, FLYING_KEYWORDS):
            flying_cleared[name] = flying_cleared.get(name, 0) + 1; continue
        if _kw(name, AQUATIC_KEYWORDS) and z <= 0:
            aquatic_cleared[name] = aquatic_cleared.get(name, 0) + 1; continue
        if _kw(name, TERRAIN_KEYWORDS) and z == 0:
            terrain_cleared[name] = terrain_cleared.get(name, 0) + 1; continue
        if z == 0:
            z0_count += 1; continue
        if init_map_json and fingerprint in init_element_set:
            preexist_cleared[name] = preexist_cleared.get(name, 0) + 1; continue
        if _kw(name, TERRAIN_KEYWORDS):
            terrain_cleared[name] = terrain_cleared.get(name, 0) + 1; continue
        if _kw(name, HANGING_KEYWORDS):
            hanging_cleared[name] = hanging_cleared.get(name, 0) + 1; continue

        if z > 0:
            x, y = elem.get('x', 0), elem.get('y', 0)
            supports = _find_supports(x, y, z, name)
            if supports:
                supported_lines.append(
                    f"- {name}: Z={z}，附近有大型支撑结构 {', '.join(supports[:3])}，"
                    f"很可能放置在其表面上（请结合截图确认）")
            else:
                flagged_lines.append(
                    f"- {name}: Z={z}，附近未检测到明显支撑结构，请结合截图仔细判断是否悬空")

    lines = ["以下是 Rule-based 高度预检结果，请参考：", ""]
    lines.append("✅ 已通过预检的元件（请勿将这些判为 H1 高度问题）：")
    if z0_count > 0:
        lines.append(f"- 共 {z0_count} 个 Z=0 元件：处于地面基准线，高度合理")
    for n, c in sorted(terrain_cleared.items()):
        lines.append(f"- {n} ×{c}：地形类元件，高度合理")
    for n, c in sorted(effect_cleared.items()):
        lines.append(f"- {n} ×{c}：特效/粒子类，允许悬空")
    for n, c in sorted(flying_cleared.items()):
        lines.append(f"- {n} ×{c}：飞行生物，高处合理")
    for n, c in sorted(aquatic_cleared.items()):
        lines.append(f"- {n} ×{c}：水下生物，Z≤0 合理")
    for n, c in sorted(hanging_cleared.items()):
        lines.append(f"- {n} ×{c}：悬挂/攀附类元件，依附于周围结构是正常的")
    for n, c in sorted(preexist_cleared.items()):
        lines.append(f"- {n} ×{c}：[预置元件] 初始场景已有，非本次修改产生")
    for cl in cleared_lines:
        lines.append(cl)

    if supported_lines:
        lines.append("")
        lines.append("📐 支撑分析（以下元件 Z>0，但附近有大型结构可能提供支撑，请结合截图判断）：")
        lines.append("注意：包围盒分析只是近似的，实际资产形状可能不同，请以截图为准")
        lines.extend(supported_lines)
    if flagged_lines:
        lines.append("")
        lines.append("⚠️ 需重点关注（附近未发现支撑结构，可能存在悬空问题）：")
        lines.extend(flagged_lines)
    if not supported_lines and not flagged_lines:
        lines.append("")
        lines.append("所有 Z>0 的非豁免元件均已通过预检或有支撑分析。")

    return "\n".join(lines)


# ============================================================
# 元件摘要 & 工具函数
# ============================================================

def build_element_summary(map_json: dict) -> str:
    lines = []
    total = 0
    for cat_key, cat_val in map_json.items():
        if cat_key == "地图信息" or not isinstance(cat_val, dict):
            continue
        for sub_key, sub_val in cat_val.items():
            if not isinstance(sub_val, list):
                continue
            for elem in sub_val:
                pos = elem.get("pos")
                name = elem.get("name", "未知")
                extend = elem.get("Extend", [])
                rotate = elem.get("rotate")
                if isinstance(pos, list) and len(pos) > 0 and not isinstance(pos[0], list):
                    total += 1
                    line = f"- {name} | 位置={pos} | 尺寸={extend}"
                    if rotate:
                        line += f" | 旋转={rotate}"
                    lines.append(line)
                elif isinstance(pos, list) and len(pos) > 0 and isinstance(pos[0], list):
                    num = elem.get("num", "?")
                    total += 1
                    lines.append(f"- {name} | 范围生成×{num} | 区域={pos}")
    header = f"元件总数: {total}"
    return header + "\n" + "\n".join(lines) if lines else header + "\n（无元件）"


def get_terrain_type(map_json: dict) -> str:
    return map_json.get("地图信息", {}).get("ground", "未知")


def extract_agent_final_response(case_dir: str) -> str:
    """从 sft_trajectory.json 提取 agent 最终文本回复（离线评估用）。"""
    traj_path = os.path.join(case_dir, "sft_trajectory.json")
    if not os.path.isfile(traj_path):
        return "（无 agent 回复记录）"
    try:
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
        conversations = traj.get("conversations", [])
        for conv in reversed(conversations):
            if conv.get("role") != "assistant":
                continue
            fc = conv.get("function_calls")
            if fc is not None and len(fc) > 0:
                continue
            content = conv.get("content", "")
            if "</think>" in content:
                final_text = content.split("</think>", 1)[1].strip()
                if final_text:
                    return final_text
            if content.strip():
                return content.strip()
        return "（agent 无最终文本回复）"
    except Exception as e:
        logging.warning(f"  提取 agent final response 失败: {e}")
        return "（提取失败）"


def extract_available_assets(case_dir: str) -> str:
    """从 component_info.json 提取可用资产名称列表（离线评估用）。"""
    ci_path = os.path.join(case_dir, "component_info.json")
    if not os.path.isfile(ci_path):
        return "（无可用资产信息）"
    try:
        with open(ci_path, encoding="utf-8") as f:
            ci = json.load(f)
        asset_names = sorted(ci.keys()) if isinstance(ci, dict) else []
        if not asset_names:
            return "（资产列表为空）"
        return f"可用资产共 {len(asset_names)} 种：{', '.join(asset_names)}"
    except Exception as e:
        logging.warning(f"  读取 component_info 失败: {e}")
        return "（读取失败）"


# ============================================================
# 轨迹提取（v2 新增）
# ============================================================

def extract_agent_turns_with_toolcalls(case_dir: str,
                                        max_tool_arg_preview: int = 600) -> List[Dict]:
    """从 sft_trajectory.json 提取每个 assistant turn 的 content + tool_calls（离线评估用）。"""
    traj_path = os.path.join(case_dir, "sft_trajectory.json")
    if not os.path.isfile(traj_path):
        return []
    try:
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
    except Exception:
        return []

    conversations = traj.get("conversations", [])
    assistant_turns = []
    for conv in conversations:
        if conv.get("role") != "assistant":
            continue
        content = conv.get("content", "") or ""
        fc = conv.get("function_calls") or []
        tool_calls = []
        for f in fc:
            args = f.get("arguments")
            arg_preview = json.dumps(args, ensure_ascii=False)[:max_tool_arg_preview] if args is not None else "{}"
            tool_calls.append({"name": f.get("name", "?"), "arguments_preview": arg_preview})
        assistant_turns.append({"content": content, "tool_calls": tool_calls})

    for i, t in enumerate(assistant_turns):
        t["turn_idx"] = i + 1
        t["is_final"] = (i == len(assistant_turns) - 1)
    return assistant_turns


def format_agent_turns_text(assistant_turns: List[Dict]) -> str:
    """将 assistant turn 列表渲染成紧凑文本，用于 prompt 注入。"""
    if not assistant_turns:
        return "（无 agent 交互记录）"
    blocks = []
    for t in assistant_turns:
        tag = "【最终一轮】" if t.get("is_final") else f"Turn {t.get('turn_idx')}"
        content = str(t.get("content", "")).strip()
        if len(content) > 1500:
            content = content[:1000] + f"\n... (省略 {len(content) - 1500} 字) ...\n" + content[-500:]
        tcs = t.get("tool_calls", [])
        tool_lines = [f"  - {tc['name']}({tc['arguments_preview']})" for tc in tcs] or ["  - （本轮无 tool_call）"]
        blocks.append(
            f"### {tag}\n内容（含 thinking/回复）：\n{content}\n本轮 tool_call：\n" + "\n".join(tool_lines)
        )
    return "\n\n".join(blocks)


# ============================================================
# Hard H1/H2 评估（v2）
# ============================================================

def call_hard_h12_verify(llm_bot, case_dir: str, query_info: dict,
                          init_map: dict, final_map: dict,
                          init_image_list: List[str], final_image_list: List[str],
                          available_assets: str = "") -> dict:
    theme = query_info.get("theme", "")
    query = query_info.get("query", query_info.get("description", ""))
    terrain = get_terrain_type(init_map)
    init_image_tags = "<image>" * len(init_image_list) if init_image_list else "（无初始截图）"
    final_image_tags = "<image>" * len(final_image_list) if final_image_list else "（无最终截图）"

    user_msg = HARD_H12_USER_PROMPT.format(
        theme=theme, query=query, terrain_type=terrain,
        init_map_summary=build_element_summary(init_map),
        final_map_summary=build_element_summary(final_map),
        rule_flagged_issues=rule_based_height_check(final_map, init_map),
        knowledge_base_context=get_knowledge_base_context(final_map),
        available_assets=available_assets,
        init_image_tags=init_image_tags,
        final_image_tags=final_image_tags,
    )

    all_images = init_image_list + final_image_list
    parsed, raw_output = call_llm_and_parse_with_raw(
        llm_bot, user_msg, all_images, "Hard-H12",
        required_keys=["H1", "H2", "H1_高度合理性"],
    )

    if not parsed:
        logging.warning("  Hard-H12 Verify 解析失败，默认全部不通过")
        return {
            "H1": {"pass": 0, "issues": []}, "H2": {"pass": 0, "issues": []},
            "summary": "Hard-H12 Verify LLM 输出解析失败",
            "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": all_images,
        }

    h1 = parsed.get("H1", parsed.get("H1_高度合理性", {}))
    h2 = parsed.get("H2", parsed.get("H2_生态合理性", {}))

    for h in [h1, h2]:
        h.setdefault("pass", 0)
        h["pass"] = int(h.get("pass", 0))
        if isinstance(h.get("issues"), list) and len(h["issues"]) > 0:
            h["pass"] = 0

    return {
        "H1": h1, "H2": h2,
        "summary": parsed.get("summary", ""),
        "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": all_images,
    }


# ============================================================
# H3 VU+VR 评估（v2）
# ============================================================

def call_h3_vu_vr_verify(llm_bot, case_dir: str, query_info: dict,
                          init_map: dict, final_map: dict,
                          init_image_list: List[str], final_image_list: List[str],
                          agent_final_response: str,
                          agent_turns: List[Dict],
                          available_assets: str = "") -> dict:
    theme = query_info.get("theme", "")
    query = query_info.get("query", query_info.get("description", ""))
    terrain = get_terrain_type(init_map)
    init_image_tags = "<image>" * len(init_image_list) if init_image_list else "（无初始截图）"
    final_image_tags = "<image>" * len(final_image_list) if final_image_list else "（无最终截图）"

    user_msg = H3_VURR_USER_PROMPT.format(
        theme=theme, query=query, terrain_type=terrain,
        init_map_summary=build_element_summary(init_map),
        final_map_summary=build_element_summary(final_map),
        agent_turns=format_agent_turns_text(agent_turns),
        agent_final_response=agent_final_response or "（无最终 response）",
        available_assets=available_assets,
        init_image_tags=init_image_tags,
        final_image_tags=final_image_tags,
    )

    all_images = init_image_list + final_image_list
    parsed, raw_output = call_llm_and_parse_with_raw(
        llm_bot, user_msg, all_images, "H3-VURR",
        required_keys=["H3_VU", "H3_VR"],
    )

    def _default_h3(reason):
        return {
            "H3_VU": {"score": 0, "evidence": reason},
            "H3_VR": {"score": 0, "evidence": reason},
            "H3_pass": 0,
            "H3_pass_reason": f"VU=0, VR=0（fallback: {reason}）",
            "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": all_images,
        }

    if not parsed:
        logging.warning("  H3-VURR 解析失败，VU=VR=0")
        return _default_h3("LLM 输出解析失败")

    def _norm(x):
        if isinstance(x, (int, float)):
            return {"score": int(max(0, min(5, x))), "evidence": ""}
        if not isinstance(x, dict):
            return {"score": 0, "evidence": ""}
        try:
            s = int(x.get("score", 0))
        except Exception:
            s = 0
        x["score"] = max(0, min(5, s))
        x.setdefault("evidence", "")
        return x

    vu = _norm(parsed.get("H3_VU", {}))
    vr = _norm(parsed.get("H3_VR", {}))
    h3_pass = 1 if (vu["score"] >= 4 and vr["score"] >= 4) else 0

    return {
        "H3_VU": vu, "H3_VR": vr,
        "H3_pass": h3_pass,
        "H3_pass_reason": f"VU={vu['score']}, VR={vr['score']}, pass条件: VU>=4 AND VR>=4",
        "summary": parsed.get("summary", ""),
        "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": all_images,
    }


# ============================================================
# Reward 计算
# ============================================================

def compute_unverified_reward_v2(hard_result: dict, soft_result: Optional[dict] = None) -> float:
    """Hard 全通过 → 1.0，任一 Hard fail → 0.0（soft reward 已移除，最大 1.0）。

    soft_result 参数保留仅为向后兼容旧调用签名，已不参与计算。
    """
    return 1.0 if hard_result.get("hard_pass", False) else 0.0


# 别名：兼容旧调用
compute_unverified_reward = compute_unverified_reward_v2


# ============================================================
# 图片收集（离线评估辅助）
# ============================================================

def _collect_images(case_dir: str, dir_names: List[str]) -> List[str]:
    imgs = []
    for d in dir_names:
        img_dir = os.path.join(case_dir, d)
        if os.path.isdir(img_dir):
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
                imgs.extend(sorted(glob.glob(os.path.join(img_dir, ext))))
            if imgs:
                break
    return imgs


# ============================================================
# 单 Case 评估（v2 主入口）
# ============================================================

def evaluate_unverified_case_v2(case_dir: str, hard_h12_bot, h3_bot, soft_bot=None,
                                 agent_final_response: str = None,
                                 agent_turns: Optional[List[Dict]] = None,
                                 available_assets: str = None) -> dict:
    """
    评估单个 unverified case（v2，soft reward 已移除）。

    soft_bot 参数保留仅为向后兼容旧调用签名，已不使用。
    reward = hard_pass ? 1.0 : 0.0（最大 1.0）。

    在线 RL 评估时由 agent_loop 直接传入 agent_final_response / agent_turns / available_assets；
    离线批量评估时传 None，自动从文件提取。
    """
    case_name = os.path.basename(case_dir)

    try:
        with open(os.path.join(case_dir, "query.json"), encoding="utf-8") as f:
            query_info = json.load(f)
        with open(os.path.join(case_dir, "init_map.json"), encoding="utf-8") as f:
            init_map = json.load(f)
        with open(os.path.join(case_dir, "final_map.json"), encoding="utf-8") as f:
            final_map = json.load(f)
    except Exception as e:
        logging.error(f"  加载文件失败: {e}")
        return {"case_name": case_name, "error": str(e), "total_reward": 0.0}

    init_image_list = _collect_images(case_dir, ["init_image", "image"])
    final_image_list = _collect_images(case_dir, ["final_image"])

    if agent_final_response is None:
        agent_final_response = extract_agent_final_response(case_dir)
    if agent_turns is None:
        agent_turns = extract_agent_turns_with_toolcalls(case_dir)
    if available_assets is None:
        available_assets = extract_available_assets(case_dir)

    # H4：rule-based 碰撞检测
    h4_result = rule_based_collision_check(final_map)
    logging.info(f"  [H4] 碰撞检测: pass={h4_result['pass']} collisions={h4_result['collision_count']}")

    # H1/H2
    logging.info(f"  [Hard-H12] 评估中...")
    h12_result = call_hard_h12_verify(
        hard_h12_bot, case_dir, query_info, init_map, final_map,
        init_image_list, final_image_list, available_assets=available_assets,
    )

    # H3 VU+VR
    logging.info(f"  [H3-VU/VR] 评估中...")
    h3_result = call_h3_vu_vr_verify(
        h3_bot, case_dir, query_info, init_map, final_map,
        init_image_list, final_image_list,
        agent_final_response=agent_final_response,
        agent_turns=agent_turns, available_assets=available_assets,
    )

    h1p = h12_result["H1"]["pass"]
    h2p = h12_result["H2"]["pass"]
    h3p = h3_result["H3_pass"]
    h4p = h4_result["pass"]
    hard_pass = (h1p == 1) and (h2p == 1) and (h3p == 1) and (h4p == 1)

    hard_result = {
        "H1": h12_result["H1"],
        "H2": h12_result["H2"],
        "H3": {
            "pass": h3p,
            "VU": h3_result["H3_VU"],
            "VR": h3_result["H3_VR"],
            "pass_reason": h3_result["H3_pass_reason"],
        },
        "H4": {"pass": h4p, "issues": h4_result["issues"]},
        "H3_VU": h3_result["H3_VU"],
        "H3_VR": h3_result["H3_VR"],
        "hard_pass": hard_pass,
        "summary": h12_result.get("summary", ""),
        "h12_llm_raw_output": h12_result.get("llm_raw_output"),
        "h3_llm_raw_output": h3_result.get("llm_raw_output"),
        # eval.py 重建 reward_instruction 需要 verify LLM 的输入 prompt / 图片
        "h12_user_prompt": h12_result.get("user_prompt", ""),
        "h12_image_paths": h12_result.get("image_paths", []),
        "h3_user_prompt": h3_result.get("user_prompt", ""),
        "h3_image_paths": h3_result.get("image_paths", []),
    }

    logging.info(
        f"  [Hard] H1={h1p} H2={h2p} "
        f"H3={h3p}(VU={h3_result['H3_VU']['score']},VR={h3_result['H3_VR']['score']}) "
        f"H4={h4p} → hard_pass={hard_pass}"
    )

    # soft reward 已移除：reward = hard_pass ? 1.0 : 0.0
    total_reward = compute_unverified_reward_v2(hard_result)

    return {
        "case_name": case_name,
        "query_tag": query_info.get("query_tag", ""),
        "query": query_info.get("query", query_info.get("description", "")),
        "hard_result": hard_result,
        "soft_result": None,
        "total_reward": total_reward,
        "verifier_version": "v2",
    }


# ============================================================
# 批量评估
# ============================================================

def evaluate_all_v2(result_dir: str, model_type: str, model_name: str,
                    output_file: str = None) -> dict:
    logger = logging.getLogger(__name__)

    hard_h12_bot = MODEL_TYPE_MAP[model_type](model_name=model_name, system_instruction=HARD_H12_SYSTEM_PROMPT)
    h3_bot = MODEL_TYPE_MAP[model_type](model_name=model_name, system_instruction=H3_VURR_SYSTEM_PROMPT)
    for b in [hard_h12_bot, h3_bot]:
        if hasattr(b, "reset"):
            b.reset()

    cases = []
    reward_sum = 0.0
    hard_pass_count = 0
    vu_sum, vr_sum, n_scored = 0, 0, 0

    for case_name in sorted(os.listdir(result_dir)):
        case_dir = os.path.join(result_dir, case_name)
        if not os.path.isdir(case_dir):
            continue
        if not os.path.exists(os.path.join(case_dir, "final_map.json")):
            continue
        logger.info(f"\n{'='*50}\n评估: {case_name}\n{'='*50}")
        result = evaluate_unverified_case_v2(case_dir, hard_h12_bot, h3_bot)
        cases.append(result)
        reward_sum += result.get("total_reward", 0.0)
        hr = result.get("hard_result", {})
        if hr.get("hard_pass", False):
            hard_pass_count += 1
        vu = hr.get("H3_VU", {}).get("score")
        vr = hr.get("H3_VR", {}).get("score")
        if isinstance(vu, int) and isinstance(vr, int):
            vu_sum += vu; vr_sum += vr; n_scored += 1
        r = result.get("total_reward", 0.0)
        logger.info(f"  {'✅' if r >= 1.0 else '❌'} reward={r:.4f}")

    n = len(cases)
    avg_reward = reward_sum / n if n > 0 else 0.0

    report = {
        "summary": {
            "verifier_version": "v2",
            "total_cases": n,
            "avg_reward": round(avg_reward, 4),
            "hard_pass_count": hard_pass_count,
            "hard_pass_rate": round(hard_pass_count / n, 4) if n > 0 else 0.0,
            "h3_vu_avg": round(vu_sum / n_scored, 4) if n_scored else 0.0,
            "h3_vr_avg": round(vr_sum / n_scored, 4) if n_scored else 0.0,
        },
        "cases": cases,
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info(f"\n📄 报告已保存: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Unverified Reward 计算器（v2: Hard H1/H2 + H3-VU/VR，soft 已移除）")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="gemini",
                        choices=list(MODEL_TYPE_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="gemini-2.5-pro-preview")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()], force=True)

    report = evaluate_all_v2(args.result_dir, args.model_type, args.model_name, args.output_file)
    s = report["summary"]
    print(f"\n{'='*60}\nUnverified Reward V2 评估报告\n{'='*60}")
    print(f"总 case 数:      {s['total_cases']}")
    print(f"平均 reward:     {s['avg_reward']:.4f}")
    print(f"Hard 通过率:     {s['hard_pass_count']}/{s['total_cases']} ({s['hard_pass_rate']:.1%})")
    print(f"H3-VU 均值:      {s['h3_vu_avg']:.2f} / 5")
    print(f"H3-VR 均值:      {s['h3_vr_avg']:.2f} / 5")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
