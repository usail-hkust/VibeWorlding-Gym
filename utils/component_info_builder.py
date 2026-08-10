"""Generate 任务的 component_info 动态构造工具.

设计思路:
- refine 任务有静态 component_info（从 component_info.json 加载，闭集）
- generate 任务下 agent 通过 retrieve_assets 拿到 type_id，通过 add{name, type_id, ...} 摆放
- 不修改 parse2pcg（保持 refine 100% 兼容），而是在调 parse2pcg 之前
  enrichment component_info — 把 agent llm_output 里出现的 (name, type_id) 注入进去

依赖文件:
- item_infos.json（资产全集元数据）
  优先从调用方显式传入的 path 参数加载（推荐在 main.py 中指定），
  默认路径：VibeWorlding/render_in_blender/assets/item_infos.json

输出 component_info 条目格式（与 raw_data/.../component_info.json 同 schema）:
    {
      "name": "主题02松树02",
      "typeId": "20007733",
      "box": [351.28, 347.67, 785.7],   # cm，从 item_infos.BoundingBox.Extend
      "col": [[0, 0, 0, 255]],
      "scores": 1.0,
      "c": 1, "m": 0,
      "gname": "asset0", "id": 0,
      "rot": [0.0, 0.0, 0.0, 1.0],      # 默认四元数无旋转
      "size_range": []
    }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).parent
_VIBE_ASSETS = _THIS_DIR.parent / "render_in_blender" / "assets"
# 默认指向本仓库的 5位 type_id 资产库（2617 条，与检索服务 / PCG 渲染服务对齐）。
# main.py 也可通过 load_item_infos(path=...) 或 PCG_ITEM_INFOS 环境变量显式覆盖。
DEFAULT_ITEM_INFOS_PATH = _VIBE_ASSETS / "item_infos.json"

# Module-level cache，避免反复 IO
_ITEM_INFOS_CACHE: Optional[Dict[str, Dict]] = None
_ITEM_INFOS_CACHE_PATH: Optional[Path] = None


def load_item_infos(path: Optional[Path] = None,
                    force_reload: bool = False) -> Dict[str, Dict]:
    """加载 item_infos_dream_creator.json，带模块级 cache。

    Args:
        path: 指定路径（优先级最高）；None 时按 _FALLBACK_PATHS 顺序查找。
    """
    global _ITEM_INFOS_CACHE, _ITEM_INFOS_CACHE_PATH

    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [DEFAULT_ITEM_INFOS_PATH]

    # 找到第一个存在的文件
    p = None
    for c in candidates:
        if Path(c).exists():
            p = Path(c)
            break
    if p is None:
        raise FileNotFoundError(
            f"item_infos not found. Tried: {[str(c) for c in candidates]}"
        )

    if (not force_reload) and _ITEM_INFOS_CACHE is not None and _ITEM_INFOS_CACHE_PATH == p:
        return _ITEM_INFOS_CACHE

    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"item_infos must be a dict keyed by type_id, got {type(data).__name__}")

    _ITEM_INFOS_CACHE = data
    _ITEM_INFOS_CACHE_PATH = p
    logger.info(f"loaded item_infos: {len(data)} entries from {p}")
    return data


def build_component_info_entry(
    type_id: str,
    name: Optional[str] = None,
    item_infos: Optional[Dict[str, Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """根据 type_id 构造一个 component_info_sample。

    Args:
        type_id: 8 位字符串 (e.g. "20007733")
        name: 可选 — 若 agent 自己起的 name 与 item_infos 里的 NameChinese 不同，
              用 agent 的 name；若 None，自动用 item_infos 里的 NameChinese。
        item_infos: 可选，默认走 module cache。

    Returns:
        与 raw_data 的 component_info.json 同 schema 的 dict；
        若 type_id 不在 item_infos，返回 None。
    """
    if not type_id or not isinstance(type_id, str):
        logger.warning(f"build_component_info_entry: invalid type_id {type_id!r}")
        return None

    item_infos = item_infos if item_infos is not None else load_item_infos()
    meta = item_infos.get(type_id)
    if not meta:
        logger.warning(f"build_component_info_entry: type_id {type_id!r} not in item_infos")
        return None

    bbox = meta.get("BoundingBox") or {}
    ext = bbox.get("Extend") or {}
    try:
        box = [float(ext.get("X", 100.0)),
               float(ext.get("Y", 100.0)),
               float(ext.get("Z", 100.0))]
    except (TypeError, ValueError):
        box = [100.0, 100.0, 100.0]
        logger.warning(f"type_id {type_id} has bad BoundingBox.Extend: {ext}, fallback to 1m^3")

    final_name = name or meta.get("NameChinese") or f"asset_{type_id}"

    return {
        "name": final_name,
        "typeId": type_id,
        "box": box,
        "col": [[0, 0, 0, 255]],
        "scores": 1.0,
        "c": 1,
        "gname": "asset0",
        "id": 0,
        "m": 0,
        "rot": [0.0, 0.0, 0.0, 1.0],
        "size_range": [],
    }


def enrich_component_info_for_generate(
    base_component_info: Dict[str, Dict],
    llm_output: Dict,
    item_infos: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """扫描 agent 的 llm_output，把所有 (name, type_id) 注入到 base_component_info。

    Args:
        base_component_info: dict（refine 时是从 .json 读的，generate 时通常空 {}）
        llm_output: 顶层 dict，格式同 parse2pcg 期望的嵌套字典结构：
            {"自然元件": {"type1": [{"name": "...", "type_id": "...", ...}, ...], ...}, ...}
        item_infos: 可选

    Returns:
        新的 component_info dict（浅拷贝 + 新条目）。
    """
    enriched = dict(base_component_info)
    if not isinstance(llm_output, dict):
        return enriched

    item_infos = item_infos if item_infos is not None else load_item_infos()
    n_added = 0
    n_skipped_no_type_id = 0
    n_skipped_not_found = 0

    for k, v in llm_output.items():
        if not isinstance(v, dict):
            continue
        for k1, v1 in v.items():
            if not isinstance(v1, list):
                continue
            for v2 in v1:
                if not isinstance(v2, dict):
                    continue
                name = v2.get("name", "")
                type_id = v2.get("type_id") or v2.get("typeId")
                if not name:
                    continue
                if not type_id:
                    n_skipped_no_type_id += 1
                    continue
                if name in enriched:
                    continue
                entry = build_component_info_entry(str(type_id), name=name, item_infos=item_infos)
                if entry is None:
                    n_skipped_not_found += 1
                    continue
                enriched[name] = entry
                n_added += 1

    if n_added or n_skipped_not_found:
        logger.info(
            "enrich_component_info: +%d added, %d skipped (type_id not in item_infos), "
            "%d skipped (no type_id, refine path)",
            n_added, n_skipped_not_found, n_skipped_no_type_id,
        )
    return enriched


def build_component_info_from_retrieve_result(
    retrieve_result: Dict[str, Any],
    item_infos: Optional[Dict[str, Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """从 retrieve_assets 返回的单条 simplified 记录构造 component_info 条目。

    Args:
        retrieve_result: {"type_id": "20007733", "name": "主题02松树02", ...}
    """
    if not isinstance(retrieve_result, dict):
        return None
    type_id = retrieve_result.get("type_id")
    name = retrieve_result.get("name")
    return build_component_info_entry(type_id=type_id, name=name, item_infos=item_infos)
