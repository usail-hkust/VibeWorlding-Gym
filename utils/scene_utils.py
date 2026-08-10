"""
scene_utils.py — generate (from-scratch) 태스크 전용 유틸리티

제공 기능:
  call_retrieve_for_fc          — retrieve_assets fc 처리, 검색 결과 반환
  split_function_calls          — fc 리스트를 retrieve/scene 으로 분류
  normalize_one_add_item        — add item 필드 정규화 (Gemini 변체 대응)
  normalize_scene_call_args     — scene call args 정규화
  apply_scene_calls_to_llm_output — scene_calls 를 llm_output 에 적용
  format_retrieve_responses_for_user — 검색 결과를 user message 형식으로 포맷
  enrich_component_info_for_generate — llm_output 의 type_id 로 component_info 보강
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# pcg_render 과 공통 함수 재사용
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pcg_render import normalize_tool_call, fix_flat_args

# component_info_builder 도 이 패키지에 있음
from component_info_builder import enrich_component_info_for_generate  # noqa: F401 (re-export)

# asset_retrieval_client
try:
    from asset_retrieval_client import AssetRetrievalClient, RetrieveError
except ImportError:
    AssetRetrievalClient = None
    RetrieveError = Exception


# ============================================================
# add 필드 alias 맵 (Gemini native fc 대응)
# ============================================================

ADD_FIELD_ALIASES: Dict[str, str] = {
    "position":        "pos",
    "Position":        "pos",
    "Pos":             "pos",
    "translation":     "pos",
    "local_translation": "pos",
    "local_position":  "pos",
    "coord":           "pos",
    "coordinates":     "pos",
    "scale":           "Extend",
    "Scale":           "Extend",
    "size":            "Extend",
    "extend":          "Extend",
    "sca":             "Extend",
    "rotation":        "rotate",
    "Rotation":        "rotate",
    "rot":             "rotate",
    "local_rotation":  "rotate",
    "typeid":          "type_id",
    "TypeId":          "type_id",
}


# ============================================================
# retrieve_assets 처리
# ============================================================

def call_retrieve_for_fc(
    fc: Any,
    client,
    item_infos: Optional[Dict[str, Dict]] = None,
    theme: str = "",
    scene_description: str = "",
    pcg_whitelist: Optional[set] = None,
) -> Dict[str, Any]:
    """retrieve_assets fc 를 처리해 구조화된 응답 dict 를 반환한다.

    Returns:
        {
          "name": "retrieve_assets",
          "arguments": <원본 args>,
          "response": {"entity_name": "...", "results": [...]}  또는 {"error": "..."}
        }
    """
    name, args = normalize_tool_call(fc)
    args = fix_flat_args(name, args) if args else {}
    entity_name = args.get("entity_name", "")
    top_k = int(args.get("top_k", 5))
    size_class = args.get("size_class")
    scene_limit = args.get("scene_limit")

    if not entity_name:
        return {"name": "retrieve_assets", "arguments": args,
                "response": {"error": "missing entity_name"}}

    # 白名单 필터 시 검색 수를 확대해 필터 후 충분한 후보 확보
    fetch_k = max(top_k * 6, 30) if pcg_whitelist else top_k
    extra_theme = args.get("theme")
    extra_scene = args.get("scene_description")

    try:
        results = client.retrieve(
            entity_name=entity_name,
            top_k=fetch_k,
            theme=extra_theme,
            scene_description=extra_scene,
            size_class=size_class or None,
            scene_limit=scene_limit or None,
        )

        slim = []
        for r in results:
            tid = r.get("type_id", "")
            if pcg_whitelist and tid not in pcg_whitelist:
                continue
            cap = (r.get("caption_visual") or "")[:80]
            entry = {
                "type_id": tid,
                "name": r["name"],
                "score": round(float(r.get("score", 0.0)), 3),
                "category_minor": r.get("category_minor"),
                "type": r.get("type"),
                "size_class": r.get("size_class"),
                "placement": r.get("placement"),
                "caption_visual": cap,
                "colors": r.get("colors"),
            }
            if item_infos and tid in item_infos:
                ext = (item_infos[tid].get("BoundingBox") or {}).get("Extend") or {}
                try:
                    entry["native_bbox_m"] = [
                        round(float(ext.get("X", 100)) / 100, 2),
                        round(float(ext.get("Y", 100)) / 100, 2),
                        round(float(ext.get("Z", 100)) / 100, 2),
                    ]
                except (TypeError, ValueError):
                    entry["native_bbox_m"] = None
            slim.append(entry)
            if len(slim) >= top_k:
                break

        return {"name": "retrieve_assets", "arguments": args,
                "response": {"entity_name": entity_name, "results": slim}}
    except Exception as e:
        logging.warning(f"retrieve_assets failed for entity={entity_name!r}: {e}")
        return {"name": "retrieve_assets", "arguments": args,
                "response": {"error": str(e)}}


def split_function_calls(function_calls) -> Tuple[List, List]:
    """fc 리스트를 retrieve_calls 와 scene_calls 로 분류한다."""
    if not function_calls:
        return [], []
    fcs = function_calls if isinstance(function_calls, list) else [function_calls]
    retrieve_calls, scene_calls = [], []
    for fc in fcs:
        name, _ = normalize_tool_call(fc)
        (retrieve_calls if name == "retrieve_assets" else scene_calls).append(fc)
    return retrieve_calls, scene_calls


# ============================================================
# add item 정규화
# ============================================================

def normalize_one_add_item(item: Dict[str, Any],
                             item_infos: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
    """add item 을 표준 스키마로 정규화한다.

    입력 허용 형태:
      v1: {"type_id": 20006579, "position": [...], "scale": [...], "rotation": [...]}
      v2: {"type_id": "...", "transform": {"position": [...], ...}}
      v3: {"type_id": "...", "transform": [16-elem 4x4 matrix]}
      v4: {"type_id": "...", "pos": [...], "Extend": [...], "rotate": [...]}  ← 표준

    출력:
      {"name": "...", "type_id": "...", "pos": [...], "Extend": [...], "rotate": [...], "reason": "..."}
    """
    if not isinstance(item, dict):
        return None

    # transform 중첩 dict 전개
    flat: Dict[str, Any] = {}
    for k, v in item.items():
        if k in ("transform", "Transform") and isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v

    # 4x4 행렬 fallback
    if "transform" in flat and isinstance(flat["transform"], list) and len(flat["transform"]) == 16:
        try:
            m = [float(x) for x in flat["transform"]]
            flat["pos"] = [m[12], m[13], m[14]]
            flat["Extend"] = [abs(m[0]), abs(m[5]), abs(m[10])]
            flat["rotate"] = [0, 0, 0]
            flat.pop("transform")
        except (TypeError, ValueError) as e:
            logging.warning(f"transform 행렬 파싱 실패: {e}")

    # alias 매핑
    out: Dict[str, Any] = {}
    for k, v in flat.items():
        out[ADD_FIELD_ALIASES.get(k, k)] = v

    # type_id 정규화
    tid = out.get("type_id") or out.get("typeId")
    if tid is None:
        logging.warning(f"normalize_add_item: type_id 없음, 건너뜀: {str(item)[:200]}")
        return None
    out["type_id"] = str(tid)

    # name 없으면 item_infos 에서 보충
    if not out.get("name"):
        meta = item_infos.get(out["type_id"])
        out["name"] = (meta.get("NameChinese") if meta else None) or f"asset_{out['type_id']}"

    # pos 필수
    if "pos" not in out:
        logging.warning(f"normalize_add_item: pos 없음, 건너뜀: {str(item)[:200]}")
        return None

    # Extend 없으면 item_infos 에서 native bbox 보충
    if "Extend" not in out:
        meta = item_infos.get(out["type_id"], {})
        bbox = (meta.get("BoundingBox") or {}).get("Extend") or {}
        try:
            out["Extend"] = [
                float(bbox.get("X", 100.0)) / 100,
                float(bbox.get("Y", 100.0)) / 100,
                float(bbox.get("Z", 100.0)) / 100,
            ]
        except (TypeError, ValueError):
            out["Extend"] = [1.0, 1.0, 1.0]

    out.setdefault("rotate", [0, 0, 0])
    if not out.get("reason"):
        out["reason"] = f"摆放 {out.get('name', 'asset')} 到 {out['pos']}"

    # z >= 0 강제 (Gemini 가 종종 음수 z 를 출력)
    pos = out.get("pos")
    if isinstance(pos, list) and len(pos) >= 3:
        try:
            if float(pos[2]) < 0:
                pos[2] = 0
                out["pos"] = pos
        except (TypeError, ValueError):
            pass

    return out


def normalize_scene_call_args(name: str, args: Dict[str, Any],
                               item_infos: Dict[str, Dict]) -> Dict[str, Any]:
    """add/delete/rotation_and_translation 의 args 를 tools.py 기대 스키마로 정규화."""
    args = dict(args or {})

    if name == "add":
        items: List[Dict] = []
        for k in ("modified_data", "elements", "items", "actors", "data", "objects"):
            if k in args:
                v = args[k]
                items = v if isinstance(v, list) else [v]
                break
        else:
            if any(k in args for k in ("type_id", "typeId", "name", "pos",
                                        "position", "translation", "local_position")):
                items = [args]
            else:
                return args
        normalized = [n for item in items
                      if (n := normalize_one_add_item(item, item_infos)) is not None]
        return {"modified_data": normalized}

    if name == "delete":
        if "modified_data" in args:
            md = args["modified_data"]
            if isinstance(md, dict):
                args["modified_data"] = [md]
        elif any(k in args for k in ("name", "pos")):
            args = {"modified_data": [args]}
        return args

    if name == "rotation_and_translation":
        if "corrections" in args:
            return args
        if any(k in args for k in ("name", "pos", "position", "translation")):
            mapped = {ADD_FIELD_ALIASES.get(k, k): v for k, v in args.items()}
            original = {k: mapped[k] for k in ("name", "pos", "Extend") if k in mapped}
            modified = {k: v for k, v in mapped.items() if k not in ("pos", "Extend")}
            return {"corrections": [{"original_data": original, "modified_data": modified}]}
        return args

    return args


# ============================================================
# scene_calls 적용
# ============================================================

def apply_scene_calls_to_llm_output(
    scene_calls: List[Any],
    current_llm_output: Dict,
    item_infos: Dict[str, Dict],
) -> Tuple[Dict, str]:
    """scene_calls 를 llm_output 에 적용하고 (new_output, error_str) 를 반환한다."""
    out, err, _ = _apply_scene_calls_with_record(scene_calls, current_llm_output, item_infos)
    return out, err


def _apply_scene_calls_with_record(
    scene_calls: List[Any],
    current_llm_output: Dict,
    item_infos: Dict[str, Dict],
) -> Tuple[Dict, str, List[Dict]]:
    import copy as _copy
    from tools import rotation_and_translation, delete, add

    fn_map = {"add": add, "delete": delete, "rotation_and_translation": rotation_and_translation}
    out = _copy.deepcopy(current_llm_output) if current_llm_output else {}
    err_acc = []
    records: List[Dict] = []

    for fc in scene_calls:
        name, raw_args = normalize_tool_call(fc)
        if name not in fn_map:
            continue
        clean_args = normalize_scene_call_args(name, raw_args, item_infos)
        if name == "add" and not clean_args.get("modified_data"):
            err_acc.append("add 归一化后 modified_data 为空，已跳过")
            continue
        try:
            out = fn_map[name](llm_output=out, **clean_args)
            records.append({"name": name, "arguments": clean_args})
            n = len(clean_args.get("modified_data", clean_args.get("corrections", [])) or [])
            logging.info(f"  {name} applied OK ({n} items)")
        except Exception as e:
            err_acc.append(f"{name} 调用失败: {e}")
            logging.warning(f"  {name} failed: {e}")

    return out, " | ".join(err_acc), records


# ============================================================
# retrieve 결과 포맷 (다음 턴 user message 용)
# ============================================================

def format_retrieve_responses_for_user(retrieve_responses: List[Dict]) -> str:
    """retrieve_assets 응답 목록을 다음 턴 user message 형식으로 포맷한다."""
    if not retrieve_responses:
        return ""
    lines = ["<tool_response>资产检索结果:"]
    for resp in retrieve_responses:
        args = resp.get("arguments", {})
        result = resp.get("response", {})
        entity = result.get("entity_name", args.get("entity_name", "?"))
        if "error" in result:
            lines.append(f"  [retrieve_assets({entity})] ❌ {result['error']}")
            continue
        items = result.get("results", [])
        lines.append(f"  [retrieve_assets({entity})] top-{len(items)}:")
        for r in items:
            bbox = r.get("native_bbox_m")
            bbox_str = f" native_bbox(m)=[{bbox[0]},{bbox[1]},{bbox[2]}]" if bbox else ""
            lines.append(
                f"    type_id={r['type_id']} name={r['name']} score={r['score']:.3f}"
                f" cat={r.get('category_minor')}/{r.get('type')}"
                f" size={r.get('size_class')}{bbox_str}"
            )
            cap = r.get("caption_visual")
            colors = r.get("colors")
            if cap or colors:
                lines.append(
                    f"        {'description=' + cap if cap else ''}"
                    f"{' color=' + str(colors) if colors else ''}"
                )
    lines.append("⚠️ 提示:")
    lines.append("  1) 结合 description/color，挑选风格与色调最契合场景主题的 type_id；若都不契合可换表述再检索。")
    lines.append("  2) add 时建议把 Extend 设为 native_bbox_m 或在其 0.7~1.5x 范围内。</tool_response>")
    return "\n".join(lines) + "\n"
