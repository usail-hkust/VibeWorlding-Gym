"""
unverified_verifier_onestep_scene.py — onestep_scene (generate/from-scratch) Reward 计算器

维度：
  H1 高度合理性 (rule+LLM)   —— rule_based_height_check(final_map, {})
  H2 生态/风格合理性 (LLM)
  H3 = VU(0-5) + VR(0-5) + Response(T4澄清, pass/fail)   H3_pass = VU>=4 AND VR>=4 AND Response_pass
  H4 碰撞 (rule)            —— rule_based_collision_check(final_map)
  H5 检索使用合理性 (LLM, 4档)  H5_pass = 无 tier3/tier4

hard_pass = H1 ∧ H2 ∧ H3 ∧ H4 ∧ H5
total_reward = 1.0 if hard_pass else 0.2 * (passed_dims/5)

Usage:
  python unverified_verifier_onestep_scene.py \
    --result_dir log/xxx \
    --output_file log/xxx/onestep_report.json \
    --model_name gemini-2.5-flash
"""

import os
import re
import sys
import json
import glob
import logging
import argparse
from typing import List, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../utils"))
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from llm import (
    GeminiMultiChat, QwenMultiChat, OpenAIMultiChat,
    BailianMultiChat, MODEL_TYPE_MAP,
)

# 复用合并版的工具函数和 rule-based check
from unverified_verifier import (
    call_llm_and_parse_with_raw,
    rule_based_collision_check,
    rule_based_height_check,
    extract_agent_final_response,
    build_element_summary,
    get_terrain_type,
    get_knowledge_base_context,
    extract_agent_turns_with_toolcalls,
    format_agent_turns_text,
)

from prompts import (
    HARD_H12_SCENE_SYSTEM_PROMPT, HARD_H12_SCENE_USER_PROMPT,
    H3_SCENE_SYSTEM_PROMPT, H3_SCENE_USER_PROMPT,
    H5_SCENE_SYSTEM_PROMPT, H5_SCENE_USER_PROMPT,
)


# ============================================================
# H5 数据提取：retrieve 召回 ↔ add 使用 配对
# ============================================================

_RECALL_HEAD_RE = re.compile(r"type_id=(\d+)\s+name=(\S+)\s+score=([\d.]+)")
_INTENT_HEAD_RE = re.compile(r"\[retrieve_assets\(([^)]*)\)\]")


def _parse_retrieve_block(text: str) -> List[Dict]:
    """从 user 消息文本中解析所有 retrieve 意图及其召回列表。"""
    if "资产检索结果" not in text:
        return []
    intents = []
    parts = re.split(r"(\[retrieve_assets\([^)]*\)\])", text)
    i = 1
    while i < len(parts):
        head = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        m = _INTENT_HEAD_RE.search(head)
        entity = m.group(1) if m else "?"
        recalled = []
        for cb in re.split(r"(?=type_id=\d+\s+name=)", body):
            head_m = _RECALL_HEAD_RE.search(cb)
            if not head_m:
                continue
            desc_m = re.search(r"description=(.*?)(?:\s*color=|\Z)", cb, flags=re.DOTALL)
            color_m = re.search(r"color=(\[[^\]]*\])", cb)
            recalled.append({
                "type_id": head_m.group(1), "name": head_m.group(2), "score": head_m.group(3),
                "description": (desc_m.group(1).strip()[:120] if desc_m else ""),
                "color": (color_m.group(1) if color_m else ""),
            })
        if recalled:
            intents.append({"entity": entity, "recalled": recalled})
        i += 2
    return intents


def extract_retrieve_and_usage(case_dir: str) -> Dict:
    """从 sft_trajectory.json 提取 H5 所需的召回 vs 使用信息（离线评估用）。"""
    traj_path = os.path.join(case_dir, "sft_trajectory.json")
    if not os.path.isfile(traj_path):
        return {"intents": [], "used": []}
    try:
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
    except Exception:
        return {"intents": [], "used": []}

    conversations = traj.get("conversations", [])
    intents, used, seen_used = [], [], set()

    for conv in conversations:
        role = conv.get("role")
        if role == "user":
            content = conv.get("content", "") or ""
            intents.extend(_parse_retrieve_block(content))
        elif role == "assistant":
            for fc in (conv.get("function_calls") or []):
                if fc.get("name") != "add":
                    continue
                args = fc.get("arguments") or {}
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
                        "type_id": tid, "name": name,
                        "reason": str(item.get("reason", ""))[:120],
                    })

    return {"intents": intents, "used": used}


def format_retrieve_usage_pairs(rinfo: Dict, max_recall_per_intent: int = 5) -> str:
    intents = rinfo.get("intents", [])
    used = rinfo.get("used", [])
    if not intents:
        return "（无检索记录）"
    used_tids = {u["type_id"] for u in used}
    blocks = []
    for idx, it in enumerate(intents, 1):
        entity = it.get("entity", "?")
        recalled = it.get("recalled", [])[:max_recall_per_intent]
        rec_lines = []
        for r in recalled:
            mark = " ★已使用" if r["type_id"] in used_tids else ""
            rec_lines.append(
                f"    - type_id={r['type_id']} name={r['name']} score={r['score']}{mark}\n"
                f"      description={r.get('description', '')} color={r.get('color', '')}"
            )
        blocks.append(
            f"### 检索意图 {idx}：entity=「{entity}」\n"
            f"  召回候选(top-{len(recalled)})：\n" + "\n".join(rec_lines)
        )
    used_summary = "、".join(
        f"{u['name']}(type_id={u['type_id']})" for u in used
    ) if used else "（未使用任何资产）"
    return "\n\n".join(blocks) + f"\n\n## Agent 实际放入场景的全部资产\n{used_summary}"


# ============================================================
# 图片收集
# ============================================================

def _collect_final_images(case_dir: str) -> List[str]:
    for d in ["final_image", "image"]:
        img_dir = os.path.join(case_dir, d)
        if os.path.isdir(img_dir):
            imgs = []
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
                imgs.extend(sorted(glob.glob(os.path.join(img_dir, ext))))
            if imgs:
                return imgs
    return []


# ============================================================
# H1/H2 LLM 评估（from-scratch）
# ============================================================

def call_hard_h12_scene(llm_bot, query_info: dict, final_map: dict,
                        final_image_list: List[str]) -> dict:
    theme = query_info.get("theme", "")
    query = query_info.get("user_query", query_info.get("query", query_info.get("description", "")))
    terrain = query_info.get("terrain_hint", get_terrain_type(final_map))
    final_image_tags = "<image>" * len(final_image_list) if final_image_list else "（无最终截图）"

    user_msg = HARD_H12_SCENE_USER_PROMPT.format(
        theme=theme, query=query, terrain_type=terrain,
        final_map_summary=build_element_summary(final_map),
        rule_flagged_issues=rule_based_height_check(final_map, {}),
        knowledge_base_context=get_knowledge_base_context(final_map),
        final_image_tags=final_image_tags,
    )
    parsed, raw_output = call_llm_and_parse_with_raw(
        llm_bot, user_msg, final_image_list, "Scene-H12",
        required_keys=["H1", "H2"],
    )
    if not parsed:
        logging.warning("  Scene-H12 解析失败，默认不通过")
        return {
            "H1": {"pass": 0, "issues": []}, "H2": {"pass": 0, "issues": []},
            "summary": "H12 解析失败",
            "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": final_image_list,
        }
    h1 = parsed.get("H1", {})
    h2 = parsed.get("H2", {})
    for h in (h1, h2):
        h["pass"] = int(h.get("pass", 0))
        if isinstance(h.get("issues"), list) and len(h["issues"]) > 0:
            h["pass"] = 0
    return {
        "H1": h1, "H2": h2, "summary": parsed.get("summary", ""),
        "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": final_image_list,
    }


# ============================================================
# H3 VU+VR+Response 评估
# ============================================================

def call_h3_scene(llm_bot, query_info: dict, final_map: dict,
                  final_image_list: List[str],
                  agent_final_response: str, agent_turns: List[Dict]) -> dict:
    theme = query_info.get("theme", "")
    query = query_info.get("user_query", query_info.get("query", query_info.get("description", "")))
    terrain = query_info.get("terrain_hint", get_terrain_type(final_map))
    final_image_tags = "<image>" * len(final_image_list) if final_image_list else "（无最终截图）"
    is_t4 = bool(query_info.get("t4_distractor_type")) or str(query_info.get("query_tag", "")).upper() == "T4"

    user_msg = H3_SCENE_USER_PROMPT.format(
        theme=theme, query=query, terrain_type=terrain,
        agent_turns=format_agent_turns_text(agent_turns),
        agent_final_response=agent_final_response or "（无最终 response）",
        final_map_summary=build_element_summary(final_map),
        final_image_tags=final_image_tags,
    )
    parsed, raw_output = call_llm_and_parse_with_raw(
        llm_bot, user_msg, final_image_list, "Scene-H3",
        required_keys=["H3_VU", "H3_VR"],
    )

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

    if not parsed:
        logging.warning("  Scene-H3 解析失败，VU=VR=0")
        return {
            "H3_VU": {"score": 0, "evidence": "解析失败"},
            "H3_VR": {"score": 0, "evidence": "解析失败"},
            "H3_Response": {"pass": 1, "evidence": "解析失败(默认不扣Response，VU/VR已因解析失败=0)"},
            "H3_pass": 0, "H3_pass_reason": "解析失败",
            "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": final_image_list,
        }

    vu = _norm(parsed.get("H3_VU", {}))
    vr = _norm(parsed.get("H3_VR", {}))

    r = parsed.get("H3_Response", {})
    if isinstance(r, dict):
        resp_pass = int(r.get("pass", 1))
        resp = {"pass": resp_pass, "evidence": r.get("evidence", "")}
    else:
        resp_pass = int(bool(r)) if r != "" else 1
        resp = {"pass": resp_pass, "evidence": ""}

    h3_pass = 1 if (vu["score"] >= 4 and vr["score"] >= 4 and resp_pass == 1) else 0
    return {
        "H3_VU": vu, "H3_VR": vr, "H3_Response": resp,
        "H3_pass": h3_pass,
        "H3_pass_reason": (f"VU={vu['score']}, VR={vr['score']}, Response={resp_pass}, "
                           f"pass条件: VU>=4 AND VR>=4 AND Response==1"),
        "is_t4": is_t4,
        "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": final_image_list,
    }


# ============================================================
# H5 检索使用合理性评估（4 档）
# ============================================================

def call_h5_scene(llm_bot, query_info: dict, rinfo: Dict,
                  final_image_list: List[str]) -> dict:
    theme = query_info.get("theme", "")
    query = query_info.get("user_query", query_info.get("query", query_info.get("description", "")))
    final_image_tags = "<image>" * len(final_image_list) if final_image_list else "（无最终截图）"

    intents = rinfo.get("intents", [])
    if not intents:
        return {
            "H5_pass": 1, "worst_tier": 1, "intents": [],
            "summary": "无检索记录，H5 默认 pass",
            "llm_raw_output": {}, "user_prompt": "", "image_paths": final_image_list,
            "skipped": True,
        }

    user_msg = H5_SCENE_USER_PROMPT.format(
        theme=theme, query=query,
        retrieve_usage_pairs=format_retrieve_usage_pairs(rinfo),
        final_image_tags=final_image_tags,
    )
    parsed, raw_output = call_llm_and_parse_with_raw(
        llm_bot, user_msg, final_image_list, "Scene-H5",
        required_keys=["H5_pass", "worst_tier"],
    )
    if not parsed:
        logging.warning("  Scene-H5 解析失败，默认不通过")
        return {
            "H5_pass": 0, "worst_tier": 4, "intents": [],
            "summary": "H5 解析失败",
            "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": final_image_list,
        }

    intents_out = parsed.get("intents", [])
    tiers = []
    for it in intents_out:
        try:
            tiers.append(int(it.get("tier", 4)))
        except Exception:
            tiers.append(4)
    worst = max(tiers) if tiers else int(parsed.get("worst_tier", 4))
    h5_pass = 1 if worst <= 2 else 0
    return {
        "H5_pass": h5_pass, "worst_tier": worst, "intents": intents_out,
        "summary": parsed.get("summary", ""),
        "llm_raw_output": raw_output, "user_prompt": user_msg, "image_paths": final_image_list,
    }


# ============================================================
# Reward
# ============================================================

def compute_onestep_reward(h1p, h2p, h3p, h4p, h5p) -> float:
    dims = [h1p, h2p, h3p, h4p, h5p]
    if all(d == 1 for d in dims):
        return 1.0
    passed = sum(1 for d in dims if d == 1)
    return round(0.2 * (passed / 5), 4)


def _has_any_element(final_map: dict) -> bool:
    """检查 final_map 中是否存在至少一个场景元素（非空 list）。"""
    if not isinstance(final_map, dict):
        return False
    for v in final_map.values():
        if not isinstance(v, dict):
            continue
        for v1 in v.values():
            if isinstance(v1, list) and len(v1) > 0:
                return True
    return False


# ============================================================
# 单 Case 评估
# ============================================================

def evaluate_onestep_scene_case(case_dir: str, h12_bot, h3_bot, h5_bot) -> dict:
    case_name = os.path.basename(case_dir)
    try:
        with open(os.path.join(case_dir, "query.json"), encoding="utf-8") as f:
            query_info = json.load(f)
        with open(os.path.join(case_dir, "final_map.json"), encoding="utf-8") as f:
            final_map = json.load(f)
    except Exception as e:
        logging.error(f"  加载文件失败: {e}")
        return {"case_name": case_name, "error": str(e), "total_reward": 0.0,
                "verifier_type": "onestep_scene"}

    query_tag = query_info.get("query_tag", "?")

    # 无场景直接返回 0 分，不调用 LLM
    if not _has_any_element(final_map):
        logging.warning(f"  [{case_name}] final_map 无任何元素，直接返回 reward=0")
        return {
            "case_name": case_name,
            "verifier_type": "onestep_scene",
            "verifier_version": "onestep_scene_v1",
            "query_tag": query_tag,
            "query": query_info.get("user_query", query_info.get("description", "")),
            "hard_result": {
                "H1": {"pass": 0, "issues": ["场景为空"]},
                "H2": {"pass": 0, "issues": ["场景为空"]},
                "H3": {"pass": 0, "VU": {"score": 0}, "VR": {"score": 0},
                       "Response": {"pass": 0}, "pass_reason": "场景为空"},
                "H4": {"pass": 0, "issues": ["场景为空"]},
                "H5": {"pass": 0, "worst_tier": 4, "intents": []},
                "hard_pass": False,
                "summary": "场景为空，reward=0",
            },
            "soft_result": None,
            "total_reward": 0.0,
        }

    final_image_list = _collect_final_images(case_dir)
    agent_final_response = extract_agent_final_response(case_dir)
    agent_turns = extract_agent_turns_with_toolcalls(case_dir)
    rinfo = extract_retrieve_and_usage(case_dir)

    h4 = rule_based_collision_check(final_map)
    logging.info(f"  [H4] pass={h4['pass']} collisions={h4['collision_count']}")

    logging.info(f"  [H1/H2] 评估中...")
    h12 = call_hard_h12_scene(h12_bot, query_info, final_map, final_image_list)

    logging.info(f"  [H3 VU/VR/Resp] 评估中...")
    h3 = call_h3_scene(h3_bot, query_info, final_map, final_image_list,
                       agent_final_response, agent_turns)

    logging.info(f"  [H5 检索使用] 评估中...")
    h5 = call_h5_scene(h5_bot, query_info, rinfo, final_image_list)

    h1p = h12["H1"]["pass"]
    h2p = h12["H2"]["pass"]
    h3p = h3["H3_pass"]
    h4p = h4["pass"]
    h5p = h5["H5_pass"]
    hard_pass = (h1p == 1 and h2p == 1 and h3p == 1 and h4p == 1 and h5p == 1)
    total_reward = compute_onestep_reward(h1p, h2p, h3p, h4p, h5p)

    logging.info(
        f"  [汇总] H1={h1p} H2={h2p} H3={h3p}(VU={h3['H3_VU']['score']},"
        f"VR={h3['H3_VR']['score']},Resp={h3['H3_Response']['pass']}) "
        f"H4={h4p} H5={h5p}(tier{h5['worst_tier']}) → hard_pass={hard_pass} reward={total_reward}"
    )

    hard_result = {
        "H1": h12["H1"], "H2": h12["H2"],
        "H3": {"pass": h3p, "VU": h3["H3_VU"], "VR": h3["H3_VR"],
               "Response": h3["H3_Response"], "pass_reason": h3["H3_pass_reason"]},
        "H4": {"pass": h4p, "issues": h4["issues"]},
        "H5": {"pass": h5p, "worst_tier": h5["worst_tier"], "intents": h5.get("intents", [])},
        "hard_pass": hard_pass,
        "summary": h12.get("summary", ""),
        "h12_llm_raw_output": h12.get("llm_raw_output"),
        "h3_llm_raw_output": h3.get("llm_raw_output"),
        "h5_llm_raw_output": h5.get("llm_raw_output"),
        # eval.py 重建 reward_instruction 需要 verify LLM 的输入 prompt / 图片
        "h12_user_prompt": h12.get("user_prompt", ""),
        "h12_image_paths": h12.get("image_paths", []),
        "h3_user_prompt": h3.get("user_prompt", ""),
        "h3_image_paths": h3.get("image_paths", []),
        "h5_user_prompt": h5.get("user_prompt", ""),
        "h5_image_paths": h5.get("image_paths", []),
    }

    return {
        "case_name": case_name,
        "verifier_type": "onestep_scene",
        "verifier_version": "onestep_scene_v1",
        "query_tag": query_tag,
        "query": query_info.get("user_query", query_info.get("description", "")),
        "hard_result": hard_result,
        "soft_result": None,
        "total_reward": total_reward,
    }


# ============================================================
# 批量
# ============================================================

def _build_report(cases: List[dict]) -> dict:
    n = len(cases)
    scored = [c for c in cases if "hard_result" in c]
    hard_pass = sum(1 for c in scored if c["hard_result"].get("hard_pass"))
    reward_sum = sum(c.get("total_reward", 0.0) for c in cases)

    def _dim_pass(dim):
        return sum(1 for c in scored if c["hard_result"].get(dim, {}).get("pass") == 1)

    h5_tiers = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in scored:
        t = c["hard_result"].get("H5", {}).get("worst_tier")
        if t in h5_tiers:
            h5_tiers[t] += 1

    return {
        "summary": {
            "verifier_version": "onestep_scene_v1",
            "total_cases": n, "scored_cases": len(scored),
            "hard_pass_count": hard_pass,
            "hard_pass_rate": round(hard_pass / len(scored), 4) if scored else 0.0,
            "avg_reward": round(reward_sum / n, 4) if n else 0.0,
            "dim_pass_rate": {
                "H1": round(_dim_pass("H1") / len(scored), 4) if scored else 0.0,
                "H2": round(_dim_pass("H2") / len(scored), 4) if scored else 0.0,
                "H3": round(_dim_pass("H3") / len(scored), 4) if scored else 0.0,
                "H4": round(_dim_pass("H4") / len(scored), 4) if scored else 0.0,
                "H5": round(_dim_pass("H5") / len(scored), 4) if scored else 0.0,
            },
            "h5_worst_tier_dist": h5_tiers,
        },
        "cases": cases,
    }


def evaluate_all_onestep_scene(result_dir: str, model_type: str, model_name: str,
                               output_file: str = None) -> dict:
    logger = logging.getLogger(__name__)
    h12_bot = MODEL_TYPE_MAP[model_type](model_name=model_name, system_instruction=HARD_H12_SCENE_SYSTEM_PROMPT)
    h3_bot = MODEL_TYPE_MAP[model_type](model_name=model_name, system_instruction=H3_SCENE_SYSTEM_PROMPT)
    h5_bot = MODEL_TYPE_MAP[model_type](model_name=model_name, system_instruction=H5_SCENE_SYSTEM_PROMPT)
    for b in (h12_bot, h3_bot, h5_bot):
        if hasattr(b, "reset"):
            b.reset()

    cases = []
    for case_name in sorted(os.listdir(result_dir)):
        case_dir = os.path.join(result_dir, case_name)
        if not os.path.isdir(case_dir):
            continue
        if not os.path.exists(os.path.join(case_dir, "final_map.json")):
            continue
        logger.info(f"\n{'='*50}\n评估: {case_name}\n{'='*50}")
        result = evaluate_onestep_scene_case(case_dir, h12_bot, h3_bot, h5_bot)
        cases.append(result)

    report = _build_report(cases)
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info(f"\n📄 报告已保存: {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description="onestep_scene (generate) Reward 计算器")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="gemini",
                        choices=list(MODEL_TYPE_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()], force=True)

    report = evaluate_all_onestep_scene(args.result_dir, args.model_type, args.model_name, args.output_file)
    s = report["summary"]
    print(f"\n{'='*60}\nonestep_scene 评估报告\n{'='*60}")
    print(f"总 case: {s['total_cases']}  打分: {s['scored_cases']}")
    print(f"hard_pass: {s['hard_pass_count']}/{s['scored_cases']} ({s['hard_pass_rate']:.1%})")
    print(f"平均 reward: {s['avg_reward']:.4f}")
    print(f"各维度通过率: {s['dim_pass_rate']}")
    print(f"H5 worst_tier 分布: {s['h5_worst_tier_dist']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
