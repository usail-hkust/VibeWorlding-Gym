"""从 asset_cards.jsonl 生成 D1_deploy 所需的两份 CSV。

运行一次即可，结果写到本脚本所在目录：
  standardized_asset_library_with_caption.csv   — 主属性表（csv_attrs.py 读取）
  color_shape_detail.csv                        — 颜色/形状补充表

使用：
  python3 data/build_csv_from_jsonl.py
  # 或
  python3 /abs/path/build_csv_from_jsonl.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
JSONL = DATA_DIR / "asset_cards.jsonl"
MAIN_CSV = DATA_DIR / "standardized_asset_library_with_caption.csv"
COLOR_CSV = DATA_DIR / "color_shape_detail.csv"


def _str(v) -> str:
    if v is None:
        return ""
    return str(v)


def _env_flags_dict(available_terrain: list) -> str:
    """context.available_terrain 列表 → env_flags JSON dict（与 csv_attrs 解析规则一致）。"""
    terrain_keys = [
        "草地", "沙地", "泥地", "雪地", "火山",
        "水面", "水边", "海底（水底）", "人工地板", "水泥地面",
    ]
    d = {k: (k in available_terrain) for k in terrain_keys}
    return json.dumps(d, ensure_ascii=False)


def _labels_json(card: dict) -> str:
    """合并 setting_tags / theme_tags / function_tags / style.style → JSON 数组。"""
    ctx = card.get("context") or {}
    sty = card.get("style") or {}
    tags: list[str] = []
    for src in (
        sty.get("style") or [],
        ctx.get("setting_tags") or [],
        ctx.get("theme_tags") or [],
        ctx.get("function_tags") or [],
    ):
        for t in src:
            if t and t not in tags:
                tags.append(t)
    return json.dumps(tags, ensure_ascii=False)


def build_main_csv(cards: list[dict]) -> None:
    fieldnames = [
        "asset_id", "name_clean", "category_major", "category_minor",
        "artifact_nature", "type", "subtype", "description", "image_uri",
        "size_class", "placement", "is_independent", "need_combination",
        "landmark_type", "landmark_detail",
        "poly_count", "collision_x", "collision_y", "collision_z",
        "caption_basic", "caption_visual", "caption_scene",
        "labels_json", "env_flags", "missing_flags", "scene_limit",
    ]
    with MAIN_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            meta = card.get("meta") or {}
            ctx = card.get("context") or {}
            row = {
                "asset_id":       _str(card.get("asset_id")),
                "name_clean":     _str(meta.get("ori_name_clean") or meta.get("ori_name")),
                "category_major": _str(meta.get("category_major")),
                "category_minor": _str(meta.get("category_minor")),
                "artifact_nature":_str(meta.get("artifact_nature")),
                "type":           _str(meta.get("type_label")),
                "subtype":        _str(meta.get("subtype")),
                "description":    _str(meta.get("description")),
                "image_uri":      _str(meta.get("image_uri")),
                "size_class":     _str(meta.get("size_class")),
                "placement":      _str(meta.get("placement")),
                "is_independent": _str(meta.get("is_independent")),
                "need_combination":_str(meta.get("need_combination")),
                "landmark_type":  _str(meta.get("landmark_type")),
                "landmark_detail":_str(meta.get("landmark_detail")),
                "poly_count":     _str(meta.get("poly_count")),
                "collision_x":    _str(meta.get("collision_x")),
                "collision_y":    _str(meta.get("collision_y")),
                "collision_z":    _str(meta.get("collision_z")),
                "caption_basic":  _str(meta.get("caption_basic")),
                "caption_visual": _str(meta.get("caption_visual")),
                "caption_scene":  _str(meta.get("caption_scene")),
                "labels_json":    _labels_json(card),
                "env_flags":      _env_flags_dict(ctx.get("available_terrain") or []),
                "missing_flags":  json.dumps(meta.get("missing_flags") or [], ensure_ascii=False),
                "scene_limit":    json.dumps(ctx.get("scene_limit") or [], ensure_ascii=False),
            }
            writer.writerow(row)
    print(f"[build] main CSV: {MAIN_CSV} ({len(cards)} rows)")


def build_color_csv(cards: list[dict]) -> None:
    fieldnames = ["asset_id", "status", "has_image", "colors", "shape_description", "confidence"]
    with COLOR_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            meta = card.get("meta") or {}
            sty = card.get("style") or {}
            colors_list = sty.get("colors") or []
            colors_str = "、".join(str(c) for c in colors_list if c)
            row = {
                "asset_id":         _str(card.get("asset_id")),
                "status":           "",
                "has_image":        "",
                "colors":           colors_str,
                "shape_description":_str(meta.get("shape_description")),
                "confidence":       "",
            }
            writer.writerow(row)
    print(f"[build] color CSV: {COLOR_CSV} ({len(cards)} rows)")


def main() -> None:
    if not JSONL.exists():
        print(f"[error] jsonl not found: {JSONL}", file=sys.stderr)
        sys.exit(1)

    cards = []
    with JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(json.loads(line))
    print(f"[build] loaded {len(cards)} cards from {JSONL}")

    build_main_csv(cards)
    build_color_csv(cards)
    print("[build] done.")


if __name__ == "__main__":
    main()
