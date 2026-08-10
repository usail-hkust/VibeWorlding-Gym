"""
地图生成任务 Reward 计算模块

支持 verified / unverified / onestep_scene 三种 query 类型的 reward 计算：
  - verified:      rule-based 验证，reward = pass_count / total_count ∈ [0, 1]
  - unverified:    LLM-as-Judge 两阶段验证，支持三种 reward 策略：
      * "hard"  — 仅使用 hard_reward (0 或 1)
      * "soft"  — 仅使用 soft_reward (前提是 hard 通过)
      * "total" — hard_weight * hard + soft_weight * soft (前提是 hard 通过)
  - onestep_scene: H1-H5 rubric-based LLM 评估 (generate 任务)
      * reward = 1.0 if H1∧H2∧H3∧H4∧H5 else 0.2*(passed_dims/5)
      * 不使用 soft reward 策略

训练脚本通过 reward_kwargs 传入策略参数：
  +reward.custom_reward_function.reward_kwargs.reward_strategy=total
  +reward.custom_reward_function.reward_kwargs.hard_weight=1.0
  +reward.custom_reward_function.reward_kwargs.soft_weight=0.5
"""

import json
import logging
from typing import Any, Optional

import numpy as np

from verl import DataProto

logger = logging.getLogger(__name__)


# ============================================================
# 核心 Reward 计算
# ============================================================

def compute_map_gen_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[Any],
    extra_info: dict,
    reward_strategy: str = "total",
    hard_weight: float = 1.0,
    soft_weight: float = 0.5,
    **kwargs,
) -> tuple[float, dict]:
    """
    计算地图生成任务的 reward 分数。

    Args:
        data_source: 数据来源标识
        solution_str: 模型输出文本（本任务中不直接使用）
        ground_truth: 地面真值（本任务中不使用）
        extra_info: 额外信息，包含 verify_result 和元数据
        reward_strategy: unverified 的 reward 策略 — "hard" / "soft" / "total"
        hard_weight: total 策略下 hard 的权重（默认 1.0）
        soft_weight: total 策略下 soft 的权重（默认 0.5）

    Returns:
        (score, metrics_dict)
    """
    # 解析 extra_info
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = {}

    # 从 extra_info 中获取 verify 结果和元数据
    verify_result = extra_info.get("verify_result", {})
    verifier_type = extra_info.get("verifier_type", verify_result.get("verifier_type", "unknown"))
    query_tag = extra_info.get("query_tag", verify_result.get("query_tag", ""))
    query_type = extra_info.get("query_type", verify_result.get("query_type", ""))

    # 初始化 metrics
    metrics = {
        "verifier_type": verifier_type,
        "query_tag": query_tag,
        "query_type": query_type,
        "reward_strategy": reward_strategy,
        "num_turns": extra_info.get("num_turns", 0),
        # Anti-Hacking 效率指标（从 agent_loop extra_fields 传入）
        "efficiency_applied": extra_info.get("map_gen_efficiency_applied", 0.0),
        "wasted_turns": extra_info.get("map_gen_wasted_turns", 0.0),
        "first_all_correct_turn": extra_info.get("map_gen_first_all_correct_turn", -1.0),
        "efficiency_reward": extra_info.get("map_gen_efficiency_reward", 0.0),
        "original_verify_reward": extra_info.get("map_gen_original_verify_reward", 0.0),
    }

    # ---- Verified: rule-based reward ----
    if verifier_type == "verified":
        total_reward = verify_result.get("total_reward", 0.0)
        # 确保是数值
        total_reward = _safe_float(total_reward, 0.0)
        score = total_reward

        metrics.update({
            "final_score": score,
            "verified_reward": total_reward,
            "verified_pass_count": verify_result.get("pass_count", 0),
            "verified_total_count": verify_result.get("total_count", 0),
        })

    # ---- OnestepScene: H1-H5 rubric reward (generate 任务) ----
    elif verifier_type == "onestep_scene":
        total_reward = _safe_float(verify_result.get("total_reward", 0.0), 0.0)
        hard_pass = verify_result.get("hard_pass", total_reward == 1.0)
        score = total_reward

        hard_result = verify_result.get("hard_result", {})
        # H3 子维度：从 hard_result 中读取（agent_loop 已同时写入 hard_result.H3_VU/VR）
        h3_vu = hard_result.get("H3_VU", hard_result.get("H3", {}).get("VU", {}))
        h3_vr = hard_result.get("H3_VR", hard_result.get("H3", {}).get("VR", {}))

        def _dim_pass(key):
            return _safe_float(hard_result.get(key, {}).get("pass", 0), 0.0)

        metrics.update({
            "final_score": score,
            "onestep_scene_reward": total_reward,
            "onestep_scene_hard_pass": 1.0 if hard_pass else 0.0,
            # 各维度通过率（0/1）
            "onestep_scene_H1": _dim_pass("H1"),
            "onestep_scene_H2": _dim_pass("H2"),
            "onestep_scene_H3": _safe_float(hard_result.get("H3", {}).get("pass", 0), 0.0),
            "onestep_scene_H4": _dim_pass("H4"),
            "onestep_scene_H5": _safe_float(hard_result.get("H5", {}).get("pass", 0), 0.0),
            # H3 子维度分值（0-5）
            "onestep_scene_H3_VU": _safe_float(h3_vu.get("score", 0), 0.0),
            "onestep_scene_H3_VR": _safe_float(h3_vr.get("score", 0), 0.0),
            # H5 最差 tier（1=最好, 4=最差）
            "onestep_scene_H5_worst_tier": _safe_float(
                hard_result.get("H5", {}).get("worst_tier", 4), 4.0
            ),
        })

    # ---- Unverified: hard + soft reward ----
    elif verifier_type == "unverified":
        # 检测 verifier 输出解析失败
        # 注意: 丢弃逻辑已统一在 agent_loop 层处理（reward=-100），
        # 这里仅做兼容处理：parse_failed → score=0（不给正向 reward）
        parse_failed = verify_result.get("parse_failed", False)
        # 兼容：没有显式 flag 时，通过 llm_raw_output 判断
        if not parse_failed:
            hard_result_raw = verify_result.get("hard_result", {})
            parse_failed = hard_result_raw.get("llm_raw_output", {}).get("parse_success") is False

        if parse_failed:
            score = 0
            metrics.update({
                "final_score": 0,
                "verifier_parse_failed": 1.0,
            })
            return score, metrics

        hard_reward = _safe_float(verify_result.get("hard_reward", 0.0), 0.0)
        soft_reward = _safe_float(verify_result.get("soft_reward", 0.0), 0.0)
        hard_pass = verify_result.get("hard_pass", hard_reward == 1.0)

        # 根据策略计算最终 score
        if reward_strategy == "hard":
            score = hard_reward
        elif reward_strategy == "soft":
            score = soft_reward if hard_pass else 0.0
        elif reward_strategy == "total":
            if hard_pass:
                score = hard_weight * hard_reward + soft_weight * soft_reward
            else:
                score = 0.0
        else:
            logger.warning(f"Unknown reward_strategy '{reward_strategy}', fallback to total")
            score = (hard_weight * hard_reward + soft_weight * soft_reward) if hard_pass else 0.0

        # 提取 H1-H3, S1-S3 细粒度指标
        hard_result = verify_result.get("hard_result", {})
        soft_result = verify_result.get("soft_result", {})

        metrics.update({
            "final_score": score,
            "unverified_hard_reward": hard_reward,
            "unverified_soft_reward": soft_reward,
            "unverified_total_reward": hard_reward + soft_reward if hard_pass else 0.0,
            "unverified_hard_pass": 1.0 if hard_pass else 0.0,
            # Hard 细粒度 (H1/H2/H3/H4: 0 或 1)
            "unverified_H1": _safe_float(hard_result.get("H1", {}).get("pass", 0), 0.0),
            "unverified_H2": _safe_float(hard_result.get("H2", {}).get("pass", 0), 0.0),
            "unverified_H3": _safe_float(hard_result.get("H3", {}).get("pass", 0), 0.0),
            "unverified_H4": _safe_float(hard_result.get("H4", {}).get("pass", 0), 0.0),
            # H3 子维度 (v2 verifier, score 0-5; v1 时 get 返回 0，向后兼容)
            "unverified_H3_VU": _safe_float(hard_result.get("H3_VU", {}).get("score", 0), 0.0),
            "unverified_H3_VR": _safe_float(hard_result.get("H3_VR", {}).get("score", 0), 0.0),
            # Soft 细粒度 (S1/S2/S3: 1-5 分)
            "unverified_S1": _safe_float(soft_result.get("S1", {}).get("score", 0), 0.0),
            "unverified_S2": _safe_float(soft_result.get("S2", {}).get("score", 0), 0.0),
            "unverified_S3": _safe_float(soft_result.get("S3", {}).get("score", 0), 0.0),
        })

    # ---- Fallback: 兼容旧格式 ----
    else:
        reward = _safe_float(verify_result.get("reward", 0), 0.0)
        score = reward
        metrics.update({
            "final_score": score,
            "verify_reward_legacy": reward,
            "verify_reason": verify_result.get("reason", ""),
        })

    return score, metrics


def _safe_float(val, default: float = 0.0) -> float:
    """安全转换为 float"""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ============================================================
# verl 兼容接口
# ============================================================

def map_gen_compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict = None,
    **kwargs,
) -> tuple[float, dict]:
    """
    verl 框架注册的 custom_reward_function 入口。

    reward_kwargs 通过 **kwargs 传入（来自训练脚本配置）：
      reward_strategy, hard_weight, soft_weight
    """
    if extra_info is None:
        extra_info = {}

    return compute_map_gen_score(
        data_source=extra_info.get("data_source", "map_gen_rl"),
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def batch_compute_map_gen_score(batch: DataProto, **kwargs) -> tuple[np.ndarray, dict]:
    """批量计算 reward"""
    batch_size = len(batch)
    rewards = np.zeros(batch_size)
    all_metrics = []

    for i in range(batch_size):
        item = batch[i]
        extra_info = item.non_tensor_batch.get("extra_info", {})

        score, metrics = map_gen_compute_score(
            solution_str="",
            ground_truth=None,
            extra_info=extra_info,
            **kwargs,
        )
        rewards[i] = score
        all_metrics.append(metrics)

    return rewards, {"per_sample_metrics": all_metrics}
