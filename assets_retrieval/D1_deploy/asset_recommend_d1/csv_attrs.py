"""资产属性 / 过滤的**唯一数据源**：``module1b/standardized_asset_library_with_caption.csv``。

设计：
  - 启动时加载一次 CSV → ``Dict[asset_id, Dict[col, parsed_value]]``，轻量解析：
      * `labels_json` → list[str]（同时暴露为 `labels`）
      * `env_flags` → 派生 10 个 `terrain_*` bool
      * `missing_flags` → list[str]
      * `is_independent` / `need_combination` → bool
      * `poly_count` / `collision_x/y/z` → number
      * 其余空字符串 → None
  - **所有 fields 透出 / filters 匹配都直接命中 CSV 列名**（与上游 CSV 的 schema 完全 1:1）。
  - 不再做"phase1 dotted path → API key"翻译；调用方传什么 key，就在 CSV 里查什么。

请求字段命名严格按 CSV 列名（USER_GUIDE §5.2 / §6.2 已与本模块对齐）。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.logger import get_logger

log = get_logger(__name__)


# ─── CSV 列与解析规则（按 module1b CSV header） ────────────────────────
# 字符串列：直接保留（空串 → None）
_STR_COLS = {
    "asset_id", "name_clean", "category_major", "category_minor", "artifact_nature",
    "type", "subtype", "description", "image_uri", "size_class", "placement",
    "scene_limit", "landmark_type", "landmark_detail",
    "caption_basic", "caption_visual", "caption_scene",
}
# 数字列
_INT_COLS = {"poly_count"}
_FLOAT_COLS = {"collision_x", "collision_y", "collision_z"}
# bool 列（CSV 是 "True"/"False" 字符串）
_BOOL_COLS = {"is_independent", "need_combination"}
# JSON 列
_JSON_LIST_COLS = {"labels_json", "missing_flags"}
_JSON_DICT_COLS = {"env_flags"}

# 派生：env_flags 的 10 个中文 key → 派生 bool 字段名
_TERRAIN_BOOL_FROM_ENV: Dict[str, str] = {
    "草地": "terrain_grass",
    "沙地": "terrain_sand",
    "泥地": "terrain_mud",
    "雪地": "terrain_snow",
    "火山": "terrain_volcano",
    "水面": "terrain_water_surface",
    "水边": "terrain_waterside",
    "海底（水底）": "terrain_underwater",
    "人工地板": "terrain_artificial_floor",
    "水泥地面": "terrain_concrete",
}

# ─── 颜色 / 形状补充 CSV（module1b_color_shape）派生字段 ───────────────
# 来自 color_shape_detail.csv，按 asset_id left-join 进主库：
#   - colors            list[str]   顿号 `、` 切分的中文色名（OR 过滤友好）
#   - shape_description  str|None    形状文字描述
# 缺失资产：colors → []，shape_description → None。
_COLOR_LIST_FIELD = "colors"
_SHAPE_FIELD = "shape_description"
# colors 原始字符串里可能出现的多种分隔符（统一切分）
_COLOR_SEPARATORS = ("、", "，", ",", "/", "|")


def _parse_bool(s: Optional[str]) -> Optional[bool]:
    if s is None or s == "":
        return None
    s = s.strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _parse_int(s: Optional[str]) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _parse_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_json(s: Optional[str]) -> Any:
    if s is None or s == "":
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _split_colors(s: Optional[str]) -> List[str]:
    """`浅灰、深蓝、黄色` → ``["浅灰", "深蓝", "黄色"]``。

    - 兼容多种分隔符（顿号 / 全半角逗号 / 斜杠 / 竖线）
    - 去空白、去空项、按原顺序去重（保持稳定）
    - None / 空串 → []
    """
    if not s:
        return []
    norm = s
    for sep in _COLOR_SEPARATORS[1:]:
        norm = norm.replace(sep, _COLOR_SEPARATORS[0])
    out: List[str] = []
    for part in norm.split(_COLOR_SEPARATORS[0]):
        c = part.strip()
        if c and c not in out:
            out.append(c)
    return out


def _parse_row(row: Dict[str, str]) -> Dict[str, Any]:
    """CSV 一行 → 解析后的属性 dict（key 与 CSV 列名 1:1，外加派生字段）。"""
    out: Dict[str, Any] = {}

    for k in _STR_COLS:
        v = row.get(k)
        out[k] = v if (v is not None and v != "") else None
    for k in _INT_COLS:
        out[k] = _parse_int(row.get(k))
    for k in _FLOAT_COLS:
        out[k] = _parse_float(row.get(k))
    for k in _BOOL_COLS:
        out[k] = _parse_bool(row.get(k))
    # JSON 列：保留原始字段名（labels_json / env_flags / missing_flags）
    labels_raw = _parse_json(row.get("labels_json"))
    out["labels_json"] = labels_raw if isinstance(labels_raw, list) else None
    # 同时暴露 `labels` 别名（更符合调用方习惯；list[str]）
    out["labels"] = list(labels_raw) if isinstance(labels_raw, list) else []

    env_raw = _parse_json(row.get("env_flags"))
    out["env_flags"] = env_raw if isinstance(env_raw, dict) else None
    # 派生 10 个 terrain_* bool（缺省 False）
    for _, bool_key in _TERRAIN_BOOL_FROM_ENV.items():
        out[bool_key] = False
    if isinstance(env_raw, dict):
        for cn_key, bool_key in _TERRAIN_BOOL_FROM_ENV.items():
            out[bool_key] = bool(env_raw.get(cn_key, False))

    missing_raw = _parse_json(row.get("missing_flags"))
    out["missing_flags"] = missing_raw if isinstance(missing_raw, list) else []

    # 兼容别名：name = name_clean（顶层响应 item.name 也走它）
    out["name"] = out.get("name_clean")

    # 颜色 / 形状补充字段默认值（后续 left-join color_shape CSV 覆盖）
    out[_COLOR_LIST_FIELD] = []
    out[_SHAPE_FIELD] = None

    return out


# ─── 公共数据结构 ─────────────────────────────────────────────────────
@dataclass
class CsvAttrTable:
    """资产属性查询表：唯一数据源是 module1b CSV。

    Attributes:
        attrs: ``{asset_id: {col_or_derived_key: parsed_value}}``，6560 条
        allowed_fields: 透出白名单 = CSV 全部列 + 派生字段（labels / 10 个 terrain_*）
        filterable_fields: filter 白名单（同 allowed_fields，但额外接受 _min/_max 后缀）
    """
    attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allowed_fields: frozenset = field(default_factory=frozenset)
    filterable_fields: frozenset = field(default_factory=frozenset)

    @property
    def n(self) -> int:
        return len(self.attrs)

    def has(self, asset_id: str) -> bool:
        return asset_id in self.attrs

    def get(self, asset_id: str) -> Optional[Dict[str, Any]]:
        return self.attrs.get(asset_id)

    # ── fields 透出 ──────────────────────────────────────────────────
    def select_fields(
        self, asset_id: str, fields: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        """按 ``fields`` 白名单挑属性返回。

        - ``fields`` 为 None / 空 → 返回 None（item.attributes 不出现）
        - 命中白名单的字段照原样返回（值可能为 None / "" / [] / False）
        - 不在白名单的 key 静默忽略
        - **即使所有字段值都是 None，只要传入 fields 至少有一个命中白名单，
          也保留对应 key**（用户要求：subtype 即使为空也返回 key）
        """
        if not fields:
            return None
        row = self.attrs.get(asset_id)
        if row is None:
            return None
        out: Dict[str, Any] = {}
        for k in fields:
            if k in self.allowed_fields:
                out[k] = row.get(k)
        return out or None

    def get_name(self, asset_id: str) -> str:
        """item.name 优先：name_clean → asset_id（CSV 没有 ori_name 这一列）。"""
        row = self.attrs.get(asset_id) or {}
        return row.get("name_clean") or asset_id

    # ── filters 匹配（替代原 filters.filter_card） ────────────────────
    def match_filters(
        self, asset_id: str, filters: Optional[Dict[str, Any]],
    ) -> bool:
        """判断 asset_id 是否通过 filters（AND 语义；空 filters → True）。

        语义与之前 ``filters.filter_card`` 完全一致：
          - scalar: 精确等于；缺失视为不匹配
          - list[str]: OR；asset 是 list 时取交集非空
          - bool: 必须命中
          - dict {min, max} / {op, value}: 数值范围 / 算子
          - 空字符串 ""、None、空 list 不触发过滤
          - ``<key>_min`` / ``<key>_max`` 后缀 → 等价 dict {min} / {max}
        """
        if not filters:
            return True
        row = self.attrs.get(asset_id)
        if row is None:
            return False

        # 收集 _min / _max 后缀 → 同一个 base key 的 dict 形式 range
        range_groups: Dict[str, Dict[str, Any]] = {}
        for k, v in filters.items():
            if _is_empty_condition(v):
                continue
            if k.endswith("_min"):
                range_groups.setdefault(k[:-4], {})["min"] = v
            elif k.endswith("_max"):
                range_groups.setdefault(k[:-4], {})["max"] = v

        # 普通 key
        for k, cond in filters.items():
            if k.endswith("_min") or k.endswith("_max"):
                continue
            if _is_empty_condition(cond):
                continue
            if k not in self.filterable_fields:
                # 未知 key：与之前行为一致（dotted 兜底已被废除；现在直接判 False
                # 即"asset 没有这个属性 → 不匹配"），但为了向后兼容，未知 key 改
                # 为静默忽略（不参与过滤），避免老调用方传废弃 key 把所有结果过滤掉。
                log.debug("match_filters: 未知 filter key=%r 已忽略", k)
                continue
            if not _match_one(row.get(k), cond):
                return False

        # 范围 key
        for base, rng in range_groups.items():
            if base not in self.filterable_fields:
                log.debug("match_filters: 未知 range filter base=%r 已忽略", base)
                continue
            if not _match_one(row.get(base), rng):
                return False

        return True


# ─── 匹配辅助（原 filters.py 的核心逻辑，简化保留） ─────────────────
def _is_empty_condition(cond: Any) -> bool:
    if cond is None:
        return True
    if isinstance(cond, str) and cond.strip() == "":
        return True
    if isinstance(cond, list) and len(cond) == 0:
        return True
    return False


def _coerce_number(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _match_one(asset_value: Any, condition: Any) -> bool:
    """单字段硬过滤匹配（缺失即不匹配）。"""
    if _is_empty_condition(condition):
        return True

    # bool 条件
    if isinstance(condition, bool):
        if asset_value is None:
            return False
        if isinstance(asset_value, bool):
            return asset_value == condition
        # 兜底：CSV 已解析为 bool；这里再做一次容错
        if isinstance(asset_value, str):
            s = asset_value.strip().lower()
            if s in ("true", "1", "yes"):
                return condition is True
            if s in ("false", "0", "no"):
                return condition is False
        return False

    # list 条件（OR 语义）
    if isinstance(condition, list):
        if asset_value is None:
            return False
        if isinstance(asset_value, list):
            return any(x in asset_value for x in condition)
        return asset_value in condition

    # dict 条件：{min, max} 或 {op, value}
    if isinstance(condition, dict):
        has_range = ("min" in condition) or ("max" in condition)
        has_op = ("op" in condition) and ("value" in condition)
        if asset_value is None and (has_range or has_op):
            return False
        if has_range:
            n = _coerce_number(asset_value)
            if n is None:
                return False
            if "min" in condition:
                try:
                    if n < float(condition["min"]):
                        return False
                except (TypeError, ValueError):
                    return False
            if "max" in condition:
                try:
                    if n > float(condition["max"]):
                        return False
                except (TypeError, ValueError):
                    return False
        if has_op:
            op = condition["op"]
            tgt = condition["value"]
            try:
                if op == "eq": return asset_value == tgt
                if op == "ne": return asset_value != tgt
                if op == "gt":
                    n = _coerce_number(asset_value); return n is not None and n > float(tgt)
                if op == "ge":
                    n = _coerce_number(asset_value); return n is not None and n >= float(tgt)
                if op == "lt":
                    n = _coerce_number(asset_value); return n is not None and n < float(tgt)
                if op == "le":
                    n = _coerce_number(asset_value); return n is not None and n <= float(tgt)
                if op == "in":
                    return asset_value in tgt
            except (TypeError, ValueError):
                return False
        return True

    # scalar 条件：精确等于
    if asset_value is None:
        return False
    if isinstance(asset_value, list):
        return condition in asset_value
    return asset_value == condition


# ─── 加载入口 ────────────────────────────────────────────────────────
def _load_color_map(color_csv: Path) -> Dict[str, Dict[str, Any]]:
    """加载 color_shape_detail.csv → ``{asset_id: {colors: list, shape_description: str}}``。

    容错：跳过空行 / 无 asset_id 行；重复 asset_id 以**首次出现**为准；
    脏行（缺列等）尽量解析，解析失败的字段退化为 [] / None。
    """
    out: Dict[str, Dict[str, Any]] = {}
    with color_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            log.warning("[csv_attrs] 颜色 CSV 头部解析失败，跳过 join: %s", color_csv)
            return out
        for row in reader:
            aid = (row.get("asset_id") or "").strip()
            if not aid or aid in out:
                continue
            shape = row.get(_SHAPE_FIELD)
            out[aid] = {
                _COLOR_LIST_FIELD: _split_colors(row.get(_COLOR_LIST_FIELD)),
                _SHAPE_FIELD: shape if (shape is not None and shape != "") else None,
            }
    return out


def load_csv_attrs(
    csv_path: Path | str,
    color_csv: Path | str | None = None,
) -> CsvAttrTable:
    """启动时加载一次：主库 CSV (+ 可选颜色/形状补充 CSV) → CsvAttrTable。

    Args:
        csv_path:  module1b 主库 CSV（唯一基础数据源）
        color_csv: 可选；module1b_color_shape/color_shape_detail.csv。
                   传 None / 不存在 / 空字符串 → 跳过 join，仅 colors=[]、
                   shape_description=None 的默认值，白名单仍包含这两列。
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"asset CSV 不存在: {p}")

    attrs: Dict[str, Dict[str, Any]] = {}
    # encoding='utf-8-sig' 处理头部 BOM；newline='' 避免 csv 读取时乱码
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV 头部解析失败: {p}")
        for row in reader:
            aid = (row.get("asset_id") or "").strip()
            if not aid:
                continue
            attrs[aid] = _parse_row(row)

    # ── 可选：left-join 颜色 / 形状补充 CSV ──────────────────────────
    n_color_hit = 0
    if color_csv:
        cp = Path(color_csv)
        if cp.exists():
            color_map = _load_color_map(cp)
            for aid, row in attrs.items():
                extra = color_map.get(aid)
                if extra is not None:
                    row[_COLOR_LIST_FIELD] = extra[_COLOR_LIST_FIELD]
                    row[_SHAPE_FIELD] = extra[_SHAPE_FIELD]
                    n_color_hit += 1
            log.info(
                "[csv_attrs] 颜色/形状 join：%d/%d 条资产命中 color CSV (%d 条) from %s",
                n_color_hit, len(attrs), len(color_map), cp,
            )
        else:
            log.warning("[csv_attrs] 颜色 CSV 不存在，跳过 join: %s", cp)

    # 白名单 = CSV 所有列 + 派生（labels / 10 个 terrain_*）+ name 别名
    #          + 颜色/形状补充字段（colors / shape_description）
    allowed = (
        _STR_COLS | _INT_COLS | _FLOAT_COLS | _BOOL_COLS
        | _JSON_LIST_COLS | _JSON_DICT_COLS
        | {"labels", "name"}                              # 派生 / 别名
        | set(_TERRAIN_BOOL_FROM_ENV.values())            # 10 个 terrain_*
        | {_COLOR_LIST_FIELD, _SHAPE_FIELD}               # 颜色 / 形状补充
    )

    log.info(
        "[csv_attrs] 已加载 N=%d 条资产属性 from %s（白名单 %d 字段）",
        len(attrs), p, len(allowed),
    )
    return CsvAttrTable(
        attrs=attrs,
        allowed_fields=frozenset(allowed),
        filterable_fields=frozenset(allowed),
    )
