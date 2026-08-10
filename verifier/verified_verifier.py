"""
verified_verifier.py — Verified Reward 计算器

基于 verification_criteria 精确对比 agent 输出的 final_map.json 与 GT 预期。

四种验证类型：
  1. proximity:       add 操作 — 检查 final_map 中是否新增了指定元件且在 acceptance_radius 内
  2. exact_match:     delete 操作 — 检查 final_map 中指定元件是否被删除
  3. position_delta:  modify-move — 检查元件是否移动到了 expected_pos (±tolerance)
  4. rotation_z:      modify-rotate — 检查元件是否旋转到了 expected_rotate (±tolerance_degrees)

Reward 计算：
  - 每个 verification_criteria 独立评分 (0 或 1)
  - total_reward = sum(scores) / len(criteria)  ∈ [0, 1]

Usage:
  python verified_verifier.py \
    --gt_dir test_case_gen/output/verified_queries_test_v2 \
    --result_dir verifier/verified_queries_test_v2_result \
    --output_file verifier/verified_reward_report.json
"""

import os
import sys
import json
import math
import argparse
import logging
from typing import List, Dict, Optional, Tuple

# ============================================================
# 元件提取
# ============================================================

def is_fixed_element(elem: dict) -> bool:
    pos = elem.get("pos")
    if not isinstance(pos, list) or len(pos) == 0:
        return False
    return not isinstance(pos[0], list)


def extract_fixed_elements(map_json: dict) -> List[Dict]:
    """提取地图中所有固定放置的元件"""
    elems = []
    for cat_key, cat_val in map_json.items():
        if cat_key == "地图信息" or not isinstance(cat_val, dict):
            continue
        for sub_key, sub_val in cat_val.items():
            if not isinstance(sub_val, list):
                continue
            for elem in sub_val:
                if is_fixed_element(elem):
                    elems.append(elem)
    return elems


def dist_2d(pos1: list, pos2: list) -> float:
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def dist_3d(pos1: list, pos2: list) -> float:
    return math.sqrt(sum((a - b)**2 for a, b in zip(pos1[:3], pos2[:3])))


# ============================================================
# 四种验证函数
# ============================================================

def verify_proximity(vc: dict, init_elems: List[dict], final_elems: List[dict]) -> dict:
    """
    验证 add 操作：final_map 中是否新增了指定元件，且在 acceptance_radius 内。

    策略：
    1. 找到 final 中有但 init 中没有的、名为 expected_element_name 的元件
    2. 检查这些新增元件中是否有任何一个在 acceptance_radius 内
    """
    expected_name = vc["expected_element_name"]
    expected_pos = vc["expected_pos"]
    radius = vc["acceptance_radius"]

    # 找 init 中已有的同名元件（用 name+pos 做 key）
    init_keys = set()
    for e in init_elems:
        if e.get("name") == expected_name:
            init_keys.add(tuple(e["pos"]))

    # 找 final 中新增的同名元件
    new_elems = []
    for e in final_elems:
        if e.get("name") == expected_name:
            if tuple(e["pos"]) not in init_keys:
                new_elems.append(e)

    if not new_elems:
        return {
            "pass": False, "score": 0.0,
            "reason": f"未找到新增的 {expected_name}",
            "detail": {"expected_name": expected_name, "expected_pos": expected_pos,
                       "found_new": []},
        }

    # 检查是否有新增元件在 radius 内
    best_dist = float("inf")
    best_elem = None
    for e in new_elems:
        d = dist_3d(e["pos"], expected_pos)
        if d < best_dist:
            best_dist = d
            best_elem = e

    passed = best_dist <= radius
    return {
        "pass": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (f"找到 {expected_name} @ {best_elem['pos']}，距期望位置 {best_dist:.2f}m "
                   f"({'≤' if passed else '>'} radius {radius}m)"),
        "detail": {"expected_name": expected_name, "expected_pos": expected_pos,
                   "closest_new_pos": best_elem["pos"], "distance": round(best_dist, 2),
                   "radius": radius},
    }


def verify_exact_match(vc: dict, init_elems: List[dict], final_elems: List[dict]) -> dict:
    """
    验证 delete 操作：final_map 中指定元件是否被删除。

    策略：检查 init 中存在的 (name, pos) 在 final 中是否不存在了。
    """
    expected_name = vc["expected_name"]
    expected_pos = vc["expected_pos"]

    # 检查 init 中确实有这个元件
    found_in_init = any(
        e.get("name") == expected_name and e.get("pos") == expected_pos
        for e in init_elems
    )
    if not found_in_init:
        return {
            "pass": False, "score": 0.0,
            "reason": f"init 中未找到 {expected_name}@{expected_pos}（数据异常）",
            "detail": {},
        }

    # 检查 final 中是否还存在（精确 pos 匹配）
    still_exists = any(
        e.get("name") == expected_name and e.get("pos") == expected_pos
        for e in final_elems
    )

    # 也检查近似 pos 匹配（容差 0.1m，防止浮点误差）
    if still_exists:
        passed = False
    else:
        # 再验证：是否有同名但 pos 极近的（浮点精度问题）
        for e in final_elems:
            if e.get("name") == expected_name and dist_3d(e["pos"], expected_pos) < 0.1:
                still_exists = True
                break
        passed = not still_exists

    return {
        "pass": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (f"{expected_name}@{expected_pos} {'已被删除 ✓' if passed else '仍然存在 ✗'}"),
        "detail": {"expected_name": expected_name, "expected_pos": expected_pos,
                   "deleted": passed},
    }


def verify_position_delta(vc: dict, init_elems: List[dict], final_elems: List[dict]) -> dict:
    """
    验证 modify-move 操作：元件是否移动到了 expected_pos。

    策略：
    1. 在 final 中找到与 original_pos 同名、但位置已变的元件
    2. 检查其新位置是否与 expected_pos 匹配（tolerance）
    """
    element_name = vc["element_name"]
    original_pos = vc["original_pos"]
    expected_pos = vc["expected_pos"]
    tolerance = vc.get("tolerance", 0.5)

    # 方法1：在 final 中直接找 expected_pos 附近的同名元件
    found_at_expected = None
    best_dist = float("inf")
    for e in final_elems:
        if e.get("name") == element_name:
            d = dist_3d(e["pos"], expected_pos)
            if d < best_dist:
                best_dist = d
                found_at_expected = e

    if found_at_expected and best_dist <= tolerance:
        return {
            "pass": True, "score": 1.0,
            "reason": f"{element_name} 移动到 {found_at_expected['pos']}，距期望 {best_dist:.2f}m (≤{tolerance}m) ✓",
            "detail": {"element_name": element_name, "original_pos": original_pos,
                       "expected_pos": expected_pos, "actual_pos": found_at_expected["pos"],
                       "distance": round(best_dist, 2), "tolerance": tolerance},
        }

    # 方法2：检查原位置的元件是否还在（如果还在说明没移动）
    still_at_original = any(
        e.get("name") == element_name and dist_3d(e["pos"], original_pos) < 0.1
        for e in final_elems
    )

    actual_pos = found_at_expected["pos"] if found_at_expected else None
    return {
        "pass": False, "score": 0.0,
        "reason": (f"{element_name} 未正确移动到 {expected_pos}。"
                   f" 最近同名元件在 {actual_pos}（距离 {best_dist:.2f}m > {tolerance}m）"
                   if found_at_expected else
                   f"{element_name} 在 final 中未找到"),
        "detail": {"element_name": element_name, "original_pos": original_pos,
                   "expected_pos": expected_pos,
                   "actual_pos": actual_pos,
                   "distance": round(best_dist, 2) if found_at_expected else None,
                   "still_at_original": still_at_original},
    }


def verify_rotation_z(vc: dict, init_elems: List[dict], final_elems: List[dict]) -> dict:
    """
    验证 modify-rotate 操作：元件是否旋转到了 expected_rotate。

    策略：
    1. 在 final 中找到 original_pos 附近（同名、位置不变）的元件
    2. 检查其 rotate 字段的 z 分量是否匹配
    """
    element_name = vc["element_name"]
    original_pos = vc["original_pos"]
    expected_rotate = vc["expected_rotate"]
    tolerance_deg = vc.get("tolerance_degrees", 5)

    # 在 final 中找同名且位置匹配的元件
    candidates = []
    for e in final_elems:
        if e.get("name") == element_name and dist_3d(e["pos"], original_pos) < 1.0:
            candidates.append(e)

    if not candidates:
        return {
            "pass": False, "score": 0.0,
            "reason": f"{element_name}@{original_pos} 在 final 中未找到",
            "detail": {"element_name": element_name, "original_pos": original_pos},
        }

    # 取最近的
    target = min(candidates, key=lambda e: dist_3d(e["pos"], original_pos))
    actual_rotate = target.get("rotate")

    if actual_rotate is None:
        return {
            "pass": False, "score": 0.0,
            "reason": f"{element_name} 无 rotate 字段（未旋转）",
            "detail": {"element_name": element_name, "actual_rotate": None,
                       "expected_rotate": expected_rotate},
        }

    # 比较 z 轴旋转角度
    expected_z = expected_rotate[2] if len(expected_rotate) > 2 else 0
    actual_z = actual_rotate[2] if len(actual_rotate) > 2 else 0

    # 处理角度环绕（0° = 360°）
    diff = abs(actual_z - expected_z) % 360
    diff = min(diff, 360 - diff)

    passed = diff <= tolerance_deg
    return {
        "pass": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (f"{element_name} z轴旋转={actual_z}°，期望={expected_z}°，"
                   f"差={diff:.1f}° ({'≤' if passed else '>'} {tolerance_deg}°)"),
        "detail": {"element_name": element_name, "expected_rotate": expected_rotate,
                   "actual_rotate": actual_rotate, "z_diff_degrees": round(diff, 1),
                   "tolerance_degrees": tolerance_deg},
    }


# ============================================================
# 验证分发器
# ============================================================

VERIFY_DISPATCH = {
    "proximity": verify_proximity,
    "exact_match": verify_exact_match,
    "position_delta": verify_position_delta,
    "rotation_z": verify_rotation_z,
}


def verify_single_criterion(vc: dict, init_elems: List[dict],
                            final_elems: List[dict]) -> dict:
    """验证单个 criterion"""
    vc_type = vc.get("type", "")
    if vc_type not in VERIFY_DISPATCH:
        return {"pass": False, "score": 0.0, "reason": f"未知验证类型: {vc_type}", "detail": {}}
    return VERIFY_DISPATCH[vc_type](vc, init_elems, final_elems)


# ============================================================
# 严格作用域校验（Anti-Hacking）：越界改动即清零
# ============================================================
# 动机：verified reward 原本只看 criteria 是否 pass，不管"其余地图有没有被误改"。
#   实测大量 case 出现 "删1个却删3个 / 移1个却移2个 / 顺手删别的资产" 等 reward
#   hacking——criteria 满足拿满分，但地图被过度改动。
# 定义：criteria 决定"授权改动集合"，凡是 criteria 未授权的 增/删/移(pos 结构性
#   改动) 残留，即判越界 -> total_reward 清零。
# 取舍：只基于 pos 做 diff（未触碰元件渲染往返后 pos 零漂移，可靠）。in-place 元件
#   的 rotate/Extend 变化不检测——渲染往返会对其做归一化/抖动(实测 360°环绕、Extend
#   被重算)，据此判越界会误杀好样本。旋转/缩放的对错交给对应 criterion 判分。

SCOPE_POS_TOL = 0.1  # 位置匹配容差(m)


def _scope_match_index(elem: dict, pool: List[dict], used: set) -> Optional[int]:
    """在 pool 中找 name 相同、pos 在 SCOPE_POS_TOL 内且未占用的最近元件 index。"""
    best, bi = SCOPE_POS_TOL + 1e-9, None
    for i, e in enumerate(pool):
        if i in used or e.get("name") != elem.get("name"):
            continue
        d = dist_3d(e.get("pos", []), elem.get("pos", []))
        if d <= SCOPE_POS_TOL and d < best:
            best, bi = d, i
    return bi


def _scope_pop_match(lst: List[dict], name: str, pos: Optional[list] = None,
                     tol: float = SCOPE_POS_TOL) -> Optional[dict]:
    """从 lst 中弹出一个 name 匹配（pos 给定时还需在 tol 内）的元件。"""
    best, bi = tol + 1e-9, None
    for i, e in enumerate(lst):
        if e.get("name") != name:
            continue
        if pos is None:
            bi = i
            break
        d = dist_3d(e.get("pos", []), pos)
        if d <= tol and d < best:
            best, bi = d, i
    if bi is not None:
        return lst.pop(bi)
    return None


def compute_scope_violation(all_criteria: List[dict], init_elems: List[dict],
                            final_elems: List[dict]) -> Tuple[bool, dict]:
    """
    判定 agent 是否做了 criteria 未授权的改动（越界）。

    Returns:
        (violation: bool, detail: dict)

    流程：
      1. init 与 final 按 (name, pos±tol) 配对，配对成功=未改动，剔除；
         剩下 unmatched_init = 被删/移走，added = 新增/移入。
      2. 用 criteria 授权配额核销这些增删：
         - position_delta: 授权 original_pos 处删除 + expected_pos 附近新增一个同名
         - exact_match:    授权 expected_pos 处删除（最多 expected_count_removed 个）
         - proximity:      授权新增 expected_element_name（最多 expected_count 个）
         - rotation_z:     纯旋转，pos 不变，不授权任何增删
      3. 核销后仍有残留 => 越界。
    """
    unmatched_init: List[dict] = []
    used_final: set = set()
    for e in init_elems:
        idx = _scope_match_index(e, final_elems, used_final)
        if idx is None:
            unmatched_init.append(e)
        else:
            used_final.add(idx)
    added: List[dict] = [f for i, f in enumerate(final_elems) if i not in used_final]

    # 授权核销 —— 顺序：move -> delete -> add
    for vc in all_criteria:
        if vc.get("type") != "position_delta":
            continue
        nm = vc.get("element_name")
        orig = vc.get("original_pos")
        exp = vc.get("expected_pos")
        tol_move = float(vc.get("tolerance", 0.5)) + SCOPE_POS_TOL
        # 授权：原位置删除 + 期望位置附近新增一个同名
        _scope_pop_match(unmatched_init, nm, orig)
        if _scope_pop_match(added, nm, exp, tol=tol_move) is None:
            # 期望位置附近没找到新增（可能没移动/移歪），退而核销任意同名新增，
            # 移动对错由 position_delta criterion 本身判分，不在越界层重复惩罚。
            _scope_pop_match(added, nm, None)

    for vc in all_criteria:
        if vc.get("type") != "exact_match":
            continue
        nm = vc.get("expected_name")
        pos = vc.get("expected_pos")
        quota = int(vc.get("expected_count_removed", 1) or 1)
        for _ in range(quota):
            if _scope_pop_match(unmatched_init, nm, pos) is None:
                break

    for vc in all_criteria:
        if vc.get("type") != "proximity":
            continue
        nm = vc.get("expected_element_name")
        quota = int(vc.get("expected_count", 1) or 1)
        for _ in range(quota):
            if _scope_pop_match(added, nm, None) is None:
                break

    violation = bool(unmatched_init or added)
    detail = {
        "illegal_delete_count": len(unmatched_init),
        "illegal_add_count": len(added),
        "illegal_delete_names": [e.get("name") for e in unmatched_init][:10],
        "illegal_add_names": [e.get("name") for e in added][:10],
    }
    return violation, detail


# ============================================================
# 主评估函数
# ============================================================

def evaluate_verified_case(gt_case_dir: str, result_case_dir: str) -> dict:
    """
    评估单个 verified case。

    Args:
        gt_case_dir: GT 目录（含 query.json + init_map.json）
            - query.json 中内联 verification_criteria 和 gt_map
            - 兼容旧格式（operations.json + gt_map.json 独立文件）
        result_case_dir: Agent 结果目录（含 final_map.json）

    Returns:
        {
            "case_name": str,
            "query": str,
            "criteria_results": [...],
            "total_reward": float,
            "pass_count": int,
            "total_count": int,
        }
    """
    # 加载 GT 信息
    query_json = json.load(open(os.path.join(gt_case_dir, "query.json"), encoding="utf-8"))
    init_map = json.load(open(os.path.join(gt_case_dir, "init_map.json"), encoding="utf-8"))

    # 加载 agent 结果
    final_map_path = os.path.join(result_case_dir, "final_map.json")
    if not os.path.exists(final_map_path):
        return {
            "case_name": os.path.basename(gt_case_dir),
            "query": query_json.get("description", ""),
            "criteria_results": [],
            "total_reward": 0.0,
            "pass_count": 0, "total_count": 0,
            "error": "final_map.json 不存在",
        }

    final_map = json.load(open(final_map_path, encoding="utf-8"))

    # 提取元件列表
    init_elems = extract_fixed_elements(init_map)
    final_elems = extract_fixed_elements(final_map)

    # 收集 verification_criteria — 优先从 query.json 内联读取
    all_criteria = query_json.get("verification_criteria", [])

    # 兼容旧格式：从 operations.json 读取
    if not all_criteria:
        ops_path = os.path.join(gt_case_dir, "operations.json")
        if os.path.exists(ops_path):
            operations = json.load(open(ops_path, encoding="utf-8"))
            for op in operations:
                vc = op.get("verification_criteria")
                if vc:
                    all_criteria.append(vc)

    if not all_criteria:
        return {
            "case_name": os.path.basename(gt_case_dir),
            "query": query_json.get("description", ""),
            "criteria_results": [],
            "total_reward": 0.0,
            "pass_count": 0, "total_count": 0,
            "error": "无 verification_criteria",
        }

    # 逐条验证
    criteria_results = []
    for vc in all_criteria:
        result = verify_single_criterion(vc, init_elems, final_elems)
        result["criterion_type"] = vc.get("type", "unknown")
        criteria_results.append(result)

    pass_count = sum(1 for r in criteria_results if r["pass"])
    total_count = len(criteria_results)
    # Verified reward: pass_count / total_count ∈ [0, 1]
    total_reward = round(pass_count / total_count, 4) if total_count > 0 else 0.0

    # ===== Anti-Hacking: 严格作用域校验 —— 越界改动即清零 =====
    # 由 env VERIFIED_STRICT_SCOPE 控制（默认开启；设 "0"/"false" 关闭以做 A/B 对比）。
    scope_detail = None
    if os.environ.get("VERIFIED_STRICT_SCOPE", "1").lower() not in ("0", "false", "off"):
        violation, scope_detail = compute_scope_violation(all_criteria, init_elems, final_elems)
        if violation:
            total_reward = 0.0

    return {
        "case_name": os.path.basename(gt_case_dir),
        "query_tag": query_json.get("query_tag", "?"),
        "query": query_json.get("description", ""),
        "criteria_results": criteria_results,
        "total_reward": total_reward,
        "pass_count": pass_count,
        "total_count": total_count,
        "scope_violation": (scope_detail or {}),
    }


def evaluate_all(gt_dir: str, result_dir: str, output_file: str = None) -> dict:
    """
    批量评估所有 verified cases。

    Args:
        gt_dir: GT 根目录（含多个 case 子目录）
        result_dir: Agent 结果根目录（含同名子目录）
        output_file: 输出报告路径（可选）

    Returns:
        完整评估报告 dict
    """
    logger = logging.getLogger(__name__)
    cases = []
    total_reward_sum = 0.0
    pass_all_count = 0

    for case_name in sorted(os.listdir(gt_dir)):
        gt_case = os.path.join(gt_dir, case_name)
        result_case = os.path.join(result_dir, case_name)

        if not os.path.isdir(gt_case):
            continue
        if not os.path.isdir(result_case):
            logger.warning(f"结果目录缺失: {case_name}")
            cases.append({
                "case_name": case_name, "total_reward": 0.0,
                "error": "结果目录缺失", "criteria_results": [],
                "pass_count": 0, "total_count": 0,
            })
            continue

        result = evaluate_verified_case(gt_case, result_case)
        cases.append(result)
        total_reward_sum += result["total_reward"]
        if result["total_reward"] == 1.0:
            pass_all_count += 1

        # 打印详情
        status = "✅" if result["total_reward"] == 1.0 else ("⚠️" if result["total_reward"] > 0 else "❌")
        logger.info(f"{status} {case_name} [{result.get('query_tag','?')}] "
                     f"reward={result['total_reward']:.2f} "
                     f"({result['pass_count']}/{result['total_count']})")
        for cr in result.get("criteria_results", []):
            icon = "  ✓" if cr["pass"] else "  ✗"
            logger.info(f"  {icon} [{cr['criterion_type']}] {cr['reason']}")

    # 汇总
    n = len(cases)
    avg_reward = total_reward_sum / n if n > 0 else 0.0

    report = {
        "summary": {
            "total_cases": n,
            "avg_reward": round(avg_reward, 4),
            "perfect_cases": pass_all_count,
            "perfect_rate": round(pass_all_count / n, 4) if n > 0 else 0.0,
        },
        "cases": cases,
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info(f"\n📄 报告已保存: {output_file}")

    return report


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Verified Reward 计算器")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="GT 目录（含 query.json, operations.json, gt_map.json）")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Agent 结果目录（含 final_map.json）")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出报告路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )

    report = evaluate_all(args.gt_dir, args.result_dir, args.output_file)

    print(f"\n{'='*60}")
    print(f"Verified Reward 评估报告")
    print(f"{'='*60}")
    s = report["summary"]
    print(f"总 case 数: {s['total_cases']}")
    print(f"平均 reward: {s['avg_reward']:.4f}")
    print(f"完美通过:    {s['perfect_cases']}/{s['total_cases']} ({s['perfect_rate']:.1%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
