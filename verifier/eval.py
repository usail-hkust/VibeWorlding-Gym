#!/usr/bin/env python3
"""eval.py — 三路径统一验证器(2026-07-02)

输入一个蒸馏/RL 输出的 log 目录,遍历每个 case,读 query.json 的
(task_setting, verifier_type) 自动分派到对应 verifier:

| 路径 | 判定 | verifier |
|------|------|----------|
| generate           | task_setting=="generate"                             | evaluate_onestep_scene_case (H1~H5) |
| refine-unverified  | task_setting=="refine" & verifier_type=="unverified" | evaluate_unverified_case_v2 (H1~H4, hard-only) |
| refine-verified    | task_setting=="refine" & verifier_type=="verified"   | evaluate_verified_case (规则,无需 LLM) |

★ 每个 case 目录下生成 {case}/sft_trajectory_verified.json：
  - 基底 = 该 case 的 sft_trajectory.json（agent 原始 trajectory）
  - reward_info：各维度 verify 结果 + total_reward
  - reward_instruction：每次 verify LLM 调用的 system_instruction + 输入(user+图) + 输出(assistant)
  （verified 走规则匹配无 LLM 调用，reward_instruction 为空）

可选 --output_file 额外写一份跨 case 汇总 report。

Usage:
  python eval.py \
    --result_dir log/20260702_mix \
    --model_type gemini \
    --model_name gemini-2.5-flash
"""
import os
import sys
import json
import logging
import argparse
from typing import Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../utils"))
for _p in (SCRIPT_DIR, _UTILS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 三路径 verifier 入口
from unverified_verifier_onestep_scene import evaluate_onestep_scene_case  # generate
from unverified_verifier import evaluate_unverified_case_v2                # refine-unverified
from verified_verifier import evaluate_verified_case                       # refine-verified

from llm import MODEL_TYPE_MAP
from prompts import (
    HARD_H12_SYSTEM_PROMPT, H3_VURR_SYSTEM_PROMPT,                            # unverified
    HARD_H12_SCENE_SYSTEM_PROMPT, H3_SCENE_SYSTEM_PROMPT, H5_SCENE_SYSTEM_PROMPT,  # onestep_scene
)

logger = logging.getLogger(__name__)


def _classify(query_info: Dict) -> str:
    """按 query.json 分派到 generate / unverified / verified。"""
    task_setting = query_info.get("task_setting", "refine")
    verifier_type = query_info.get("verifier_type", "unverified")
    if task_setting == "generate":
        return "generate"
    if verifier_type == "verified":
        return "verified"
    return "unverified"


def _assistant_content_from_raw(raw: Optional[Dict]) -> str:
    """把 verify LLM 的原始输出(reasoning + content)拼成 assistant content。"""
    raw = raw or {}
    reasoning = raw.get("reasoning", "") or ""
    content = raw.get("content", "") or ""
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n{content}"
    return content


def _instruction_item(system_prompt: str, user_prompt: str, raw: Optional[Dict],
                      images: Optional[list]) -> Optional[Dict]:
    """构造一条 reward_instruction（一次 verify LLM 调用的输入+输出）。"""
    if not user_prompt:
        return None
    return {
        "system_instruction": system_prompt,
        "conversations": [
            {"role": "user", "content": user_prompt, "images": images or []},
            {"role": "assistant", "content": _assistant_content_from_raw(raw),
             "function_calls": None},
        ],
    }


def _build_reward_info(result: Dict, route: str) -> Dict:
    """构造 reward_info（各维度 verify 结果，写入 sft_trajectory_verified.json）。"""
    info = {
        "verifier_type": result.get("verifier_type", route),
        "verifier_version": result.get("verifier_version", ""),
        "route": route,
        "query_tag": result.get("query_tag", "?"),
        "total_reward": result.get("total_reward", 0.0),
    }

    if route == "verified":
        info["hard_reward"] = result.get("total_reward", 0.0)
        info["soft_reward"] = 0.0
        info["pass_count"] = result.get("pass_count", 0)
        info["total_count"] = result.get("total_count", 0)
        info["criteria_results"] = [
            {"type": cr.get("criterion_type", cr.get("type", "?")),
             "pass": cr.get("pass", False),
             "reason": cr.get("reason", "")}
            for cr in result.get("criteria_results", [])
        ]
        if result.get("error"):
            info["error"] = result["error"]
        return info

    # generate / unverified：结构基本一致（generate 多 H5）
    hard = result.get("hard_result", {})
    info["hard_pass"] = hard.get("hard_pass", False)
    info["hard_summary"] = hard.get("summary", "")
    info["soft_reward"] = 0.0  # soft 已移除

    for hk in ["H1", "H2", "H4"]:
        h = hard.get(hk, {})
        info[hk] = {"pass": h.get("pass", 0), "issues": h.get("issues", [])}

    h3 = hard.get("H3", {})
    info["H3"] = {"pass": h3.get("pass", 0), "pass_reason": h3.get("pass_reason", "")}
    vu = hard.get("H3_VU", {}) or h3.get("VU", {})
    vr = hard.get("H3_VR", {}) or h3.get("VR", {})
    info["H3_VU"] = {"score": vu.get("score", 0), "evidence": vu.get("evidence", "")}
    info["H3_VR"] = {"score": vr.get("score", 0), "evidence": vr.get("evidence", "")}

    if route == "generate":
        h5 = hard.get("H5", {})
        info["H5"] = {"pass": h5.get("pass", 0),
                      "worst_tier": h5.get("worst_tier"),
                      "intents": h5.get("intents", [])}

    if result.get("error"):
        info["error"] = result["error"]
    return info


def _build_reward_instruction(result: Dict, route: str) -> list:
    """构造 reward_instruction（每次 verify LLM 调用的 system/user/assistant）。

    verified 走规则匹配无 LLM 调用 → 返回空 list。
    """
    if route == "verified":
        return []

    hard = result.get("hard_result", {})
    items = []

    if route == "generate":
        specs = [
            (HARD_H12_SCENE_SYSTEM_PROMPT, "h12"),
            (H3_SCENE_SYSTEM_PROMPT, "h3"),
            (H5_SCENE_SYSTEM_PROMPT, "h5"),
        ]
    else:  # unverified
        specs = [
            (HARD_H12_SYSTEM_PROMPT, "h12"),
            (H3_VURR_SYSTEM_PROMPT, "h3"),
        ]

    for sys_prompt, key in specs:
        item = _instruction_item(
            sys_prompt,
            hard.get(f"{key}_user_prompt", ""),
            hard.get(f"{key}_llm_raw_output", {}),
            hard.get(f"{key}_image_paths", []),
        )
        if item:
            items.append(item)
    return items


def save_reward_to_case(case_dir: str, result: Dict, route: str) -> None:
    """在 case 目录下生成 sft_trajectory_verified.json（基底 = sft_trajectory.json
    + reward_info + reward_instruction）。"""
    src = os.path.join(case_dir, "sft_trajectory.json")
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            sft = json.load(f)
    else:
        # 无 agent trajectory 基底（如反向合成 verified 也应有；兜底空壳）
        sft = {"system_instruction": "", "conversations": []}

    sft["reward_info"] = _build_reward_info(result, route)
    ri = _build_reward_instruction(result, route)
    if ri:
        sft["reward_instruction"] = ri

    dst = os.path.join(case_dir, "sft_trajectory_verified.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(sft, f, ensure_ascii=False, indent=4)


def evaluate_all_mix(result_dir: str, model_type: str, model_name: str,
                     output_file: Optional[str] = None) -> Dict:
    """遍历 result_dir,按路径分派 verify,汇总统一 report。"""
    # 惰性构造三套 LLM bot(只在遇到对应路径时才真正用到,但一次性建好省事)
    _bots: Dict[str, object] = {}

    def _make_bot(key: str, sys_prompt: str):
        if key not in _bots:
            b = MODEL_TYPE_MAP[model_type](model_name=model_name, system_instruction=sys_prompt)
            if hasattr(b, "reset"):
                b.reset()
            _bots[key] = b
        return _bots[key]

    cases = []
    stats = {
        "generate":   {"n": 0, "reward_sum": 0.0, "hard_pass": 0},
        "unverified": {"n": 0, "reward_sum": 0.0, "hard_pass": 0},
        "verified":   {"n": 0, "reward_sum": 0.0, "perfect": 0},
    }

    for case_name in sorted(os.listdir(result_dir)):
        case_dir = os.path.join(result_dir, case_name)
        if not os.path.isdir(case_dir):
            continue
        query_path = os.path.join(case_dir, "query.json")
        final_map_path = os.path.join(case_dir, "final_map.json")
        if not os.path.isfile(query_path):
            continue
        if not os.path.exists(final_map_path):
            logger.warning(f"⏩ 跳过(无 final_map.json): {case_name}")
            continue

        with open(query_path, encoding="utf-8") as f:
            query_info = json.load(f)
        route = _classify(query_info)

        logger.info(f"\n{'='*50}\n[{route}] 评估: {case_name}\n{'='*50}")

        try:
            if route == "generate":
                h12_bot = _make_bot("h12_scene", HARD_H12_SCENE_SYSTEM_PROMPT)
                h3_bot = _make_bot("h3_scene", H3_SCENE_SYSTEM_PROMPT)
                h5_bot = _make_bot("h5_scene", H5_SCENE_SYSTEM_PROMPT)
                result = evaluate_onestep_scene_case(case_dir, h12_bot, h3_bot, h5_bot)
                reward = result.get("total_reward", 0.0)
                stats["generate"]["n"] += 1
                stats["generate"]["reward_sum"] += reward
                if result.get("hard_pass"):
                    stats["generate"]["hard_pass"] += 1

            elif route == "unverified":
                hard_h12_bot = _make_bot("hard_h12", HARD_H12_SYSTEM_PROMPT)
                h3_bot = _make_bot("h3_vurr", H3_VURR_SYSTEM_PROMPT)
                result = evaluate_unverified_case_v2(case_dir, hard_h12_bot, h3_bot)
                reward = result.get("total_reward", 0.0)
                stats["unverified"]["n"] += 1
                stats["unverified"]["reward_sum"] += reward
                if result.get("hard_result", {}).get("hard_pass"):
                    stats["unverified"]["hard_pass"] += 1

            else:  # verified — 规则匹配,gt 信息与 final_map 同在 case_dir
                result = evaluate_verified_case(case_dir, case_dir)
                reward = result.get("total_reward", 0.0)
                stats["verified"]["n"] += 1
                stats["verified"]["reward_sum"] += reward
                if reward >= 0.999:
                    stats["verified"]["perfect"] += 1

            result["route"] = route
            result["case_name"] = case_name
            result["query_tag"] = query_info.get("query_tag", "?")
            cases.append(result)

            # ★ 在 case 目录下生成 sft_trajectory_verified.json
            try:
                save_reward_to_case(case_dir, result, route)
            except Exception as se:
                logger.warning(f"  写入 sft_trajectory_verified.json 失败 ({case_name}): {se}")

            logger.info(f"  → reward={reward:.4f}  (已写 {case_name}/sft_trajectory_verified.json)")

        except Exception as e:
            logger.exception(f"  ❌ {case_name} 评估失败: {e}")
            cases.append({"case_name": case_name, "route": route,
                          "total_reward": 0.0, "error": str(e)})

    # 汇总
    def _avg(key):
        n = stats[key]["n"]
        return round(stats[key]["reward_sum"] / n, 4) if n else 0.0

    n_all = sum(stats[k]["n"] for k in stats)
    reward_all = sum(stats[k]["reward_sum"] for k in stats)
    summary = {
        "total_cases": n_all,
        "avg_reward": round(reward_all / n_all, 4) if n_all else 0.0,
        "by_route": {
            "generate": {
                "n": stats["generate"]["n"],
                "avg_reward": _avg("generate"),
                "hard_pass": stats["generate"]["hard_pass"],
            },
            "unverified": {
                "n": stats["unverified"]["n"],
                "avg_reward": _avg("unverified"),
                "hard_pass": stats["unverified"]["hard_pass"],
            },
            "verified": {
                "n": stats["verified"]["n"],
                "avg_reward": _avg("verified"),
                "perfect": stats["verified"]["perfect"],
            },
        },
    }
    report = {"summary": summary, "cases": cases}

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info(f"\n📄 报告已保存: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="三路径统一验证器")
    parser.add_argument("--result_dir", type=str, required=True, help="蒸馏/RL 输出 log 目录")
    parser.add_argument("--output_file", type=str, default=None,
                        help="可选：额外写一份跨 case 汇总 report。默认只在各 case 目录写 sft_trajectory_verified.json")
    parser.add_argument("--model_type", type=str, default="gemini",
                        choices=list(MODEL_TYPE_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()], force=True)

    report = evaluate_all_mix(args.result_dir, args.model_type, args.model_name, args.output_file)
    s = report["summary"]
    print(f"\n{'='*60}\nMIX 三路径评估报告\n{'='*60}")
    print(f"总 case: {s['total_cases']}   平均 reward: {s['avg_reward']:.4f}")
    for route, r in s["by_route"].items():
        extra = f"hard_pass={r.get('hard_pass')}" if route != "verified" else f"perfect={r.get('perfect')}"
        print(f"  [{route:11s}] n={r['n']:3d}  avg_reward={r['avg_reward']:.4f}  {extra}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
