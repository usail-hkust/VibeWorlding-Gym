"""资产卡 → 文本 / query → 文本的拼装规则。

**与 phase4_training_v2/text_utils.py 完全对齐**（关键不变式：训练用什么拼接姿势编码，
推理就必须用什么；任何错位都会让训练好的 ckpt 在线上掉点）。

主要 API：
  - `serialize_asset_card(card)`：phase1 资产卡 dict → 单段 doc 文本
  - `build_query_text(view, entity_name, query_text=None)`：
      与 phase4_training_v2 的 build_query_text 对应；推理侧固定走
      view = "baseline.query_entity"（含 query_text）
      或   "baseline.entity_only"（只有实体名）
  - `wrap_with_instruction(view, q_text, instruction_table)`：
      添加 `Instruct: {task}\n{q}` 前缀（doc 端不加任何前缀）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.io_utils import iter_jsonl, load_yaml
from ..common.logger import get_logger

log = get_logger(__name__)


# ─── 常量：与训练时一致的两个 view ─────────────────────────────────────
VIEW_QUERY_ENTITY = "baseline.query_entity"   # 有 scene/query 文本时
VIEW_ENTITY_ONLY = "baseline.entity_only"     # 仅实体名


# ─── Asset Cards ────────────────────────────────────────────────────────
# 上游 phase1 偶尔会写出 asset_id 为脏值的行（pandas 把 NaN 序列化成
# 字符串 "nan" / "NaN" / "None"）；这些是错误数据，不应进入 embedding 索引。
_INVALID_ASSET_IDS = frozenset({
    "", "nan", "NaN", "NAN", "none", "None", "NONE", "null", "Null", "NULL",
})


def load_asset_cards(jsonl: Path | str) -> Dict[str, Dict[str, Any]]:
    """phase1/asset_cards.jsonl → {asset_id: card_dict}。

    自动剔除 asset_id 为空 / "nan" / "null" 等脏行（上游 pandas 把缺失的
    asset_id 序列化成字符串 "nan" 时会落到 jsonl 里），并打印 warning
    告知调用方有多少行被丢弃。
    """
    out: Dict[str, Dict[str, Any]] = {}
    n_total = 0
    n_skipped = 0
    skipped_examples: List[str] = []
    for d in iter_jsonl(jsonl):
        n_total += 1
        raw_aid = d.get("asset_id")
        # None / 非字符串 / 空白 / "nan" / "null" 全部视为非法
        if raw_aid is None:
            n_skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append("<None>")
            continue
        aid = str(raw_aid).strip()
        if aid in _INVALID_ASSET_IDS:
            n_skipped += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(repr(raw_aid))
            continue
        out[aid] = d
    if n_skipped:
        log.warning(
            "load_asset_cards: 跳过 %d/%d 行非法 asset_id（示例: %s）from %s",
            n_skipped, n_total, skipped_examples, jsonl,
        )
    log.info("loaded asset_cards n=%d from %s", len(out), jsonl)
    return out


# ─── 资产卡序列化（与 phase4_training_v2/text_utils.serialize_asset_card 严格一致） ──
def serialize_asset_card(card: Dict[str, Any]) -> str:
    """phase1 资产卡 → 单段 doc 文本（doc 端无 instruction 前缀）。

    与训练侧完全相同的字段顺序与拼接方式：
      资产名称: <ori_name 或 canonical_name>
      一级类别: <category>
      二级类别: <secondary_category>
      基础信息: <caption_basic>
      视觉描述: <caption_visual>
      场景描述: <caption_scene>
    其它字段（style / context / temporal / ...）训练侧已注释掉，
    推理侧保持注释一致。
    """
    sem = card.get("semantic", {}) or {}
    meta = card.get("meta", {}) or {}

    parts: List[str] = []
    parts.append(f"资产名称:{meta.get('ori_name', sem.get('canonical_name', ''))}")
    parts.append(f"一级类别:{sem.get('category', '')}")
    parts.append(f"二级类别:{sem.get('secondary_category', '')}")
    parts.append(f"基础信息:{str(meta.get('caption_basic'))}")
    parts.append(f"视觉描述:{str(meta.get('caption_visual'))}")
    parts.append(f"场景描述:{str(meta.get('caption_scene') or '')}")
    return "\n".join(parts)


# ─── Query 文本拼装（推理路径） ────────────────────────────────────────
def build_query_text(
    entity_name: str,
    query_text: Optional[str] = None,
) -> str:
    """与训练 phase4_training_v2/text_utils.build_query_text 对应。

    推理侧只有两类输入：
      - 单实体名（V1.2 baseline.entity_only）       → "[实体] {ue}"
      - 实体名 + 场景/原始 query（V1.1 baseline.query_entity）
                                                  → "[实体] {ue}\\n[原始 query] {qt}"
    都不加 "Instruct: ..." 前缀；那一步走 wrap_with_instruction。
    """
    ue = (entity_name or "").strip()
    qt = (query_text or "").strip() if query_text else ""
    if qt:
        return f"[实体] {ue}\n[原始 query] {qt}"
    return f"[实体] {ue}"


def infer_view(query_text: Optional[str]) -> str:
    """推理时根据是否带 query 文本路由到对应 view（取 instruction）。"""
    return VIEW_QUERY_ENTITY if (query_text and query_text.strip()) else VIEW_ENTITY_ONLY


# ─── Instruction（从 view_prefixes.yaml 读 task description） ──────────
def load_view_instructions(yaml_path: Path | str) -> Dict[str, str]:
    """读 configs/view_prefixes.yaml 中的 view_instructions。

    与 phase4_training_v2.text_utils.load_view_instructions 行为一致：
    返回 {view: task_description, "__default__": <fallback>}。
    """
    cfg = load_yaml(yaml_path) or {}
    if "view_instructions" in cfg:
        out = dict(cfg["view_instructions"])
        out.setdefault("__default__", cfg.get("default_instruction", ""))
        return out

    # 兼容旧格式：从 prefix 反推
    out: Dict[str, str] = {}
    for v, p in (cfg.get("view_prefixes") or {}).items():
        s = str(p)
        s = s.replace("Instruct: ", "").split("\n", 1)[0].strip()
        if s.endswith(":"):
            s = s[:-1].strip()
        out[v] = s
    out.setdefault("__default__", "")
    return out


def get_instruction(table: Dict[str, str], view: Optional[str]) -> str:
    if not view:
        return table.get("__default__", "")
    return table.get(view, table.get("__default__", ""))


def wrap_with_instruction(
    view: str,
    q_text: str,
    instruction_table: Dict[str, str],
) -> str:
    """`Instruct: {task}\\n{q}` 与 swift 离线训练 jsonl 对齐（不加 "Query:" 前缀）。

    instruction 为空串时直接返回 q_text 本身（与训练侧 ``Instruct: \\n{q}`` 不同，
    避免给一个空 task 反而引入噪声）。
    """
    ins = get_instruction(instruction_table, view)
    if not ins:
        return q_text
    return f"Instruct: {ins}\n{q_text}"
