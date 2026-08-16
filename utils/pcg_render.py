

import os
import sys
import json
import math
import shutil
import logging

# map_parser.parse2pcg 与本文件同目录（utils/）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from map_parser import parse2pcg
    _PARSE2PCG_AVAILABLE = True
except ImportError:
    _PARSE2PCG_AVAILABLE = False
    logging.warning("map_parser.parse2pcg 未找到，llm_output_to_actors 不可用")


class PCGRenderError(Exception):
    pass


# ============================================================
# 相机参数计算
# ============================================================

def compute_camera_params(actors: list) -> tuple:
    """从 actors 列表计算 5 视角相机参数（front/back/left/right/topdown）。

    actor.pos 单位为厘米（渲染服务输入单位），SCALE=0.01 将 cm 换算为 Blender 米。
    返回 (cam_pos_str, cam_target_str)，每个是 5 行换行的字符串。
    """
    SCALE = 0.01  # cm → m

    def _is_scalar_pos(a):
        pos = a.get("pos")
        if not isinstance(pos, list) or len(pos) < 3:
            return False
        return not isinstance(pos[0], list)

    xs = [a["pos"][0] * SCALE for a in actors if _is_scalar_pos(a)]
    ys = [-a["pos"][1] * SCALE for a in actors if _is_scalar_pos(a)]  # Y-flip for Blender
    zs = [a["pos"][2] * SCALE for a in actors if _is_scalar_pos(a)]

    if not xs:
        cx, cy, cz = 0.0, 0.0, 0.0
        span_xy, span_z = 10.0, 0.1
    else:
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        cz = (max(zs) + min(zs)) / 2
        span_xy = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
        span_z = max(max(zs) - min(zs), 0.1)

    assumed_obj_h = max(span_z, 8.0)
    dist = max(max(span_xy, assumed_obj_h) * 2.5, 20.0)
    elev_rad = math.radians(28)
    h_offset = dist * math.sin(elev_rad)
    d_horiz = dist * math.cos(elev_rad)
    cam_z = cz + max(span_z * 0.6, h_offset) + assumed_obj_h * 0.3
    target_z = cz + assumed_obj_h * 0.35
    topdown_z = cz + dist * 1.1 + assumed_obj_h

    cameras = [
        (f"{cx:.2f},{cy - d_horiz:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
        (f"{cx:.2f},{cy + d_horiz:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
        (f"{cx - d_horiz:.2f},{cy:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
        (f"{cx + d_horiz:.2f},{cy:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),
        (f"{cx:.2f},{cy:.2f},{topdown_z:.2f}", f"{cx:.2f},{cy:.2f},{cz:.2f}"),
    ]
    return "\n".join(c[0] for c in cameras), "\n".join(c[1] for c in cameras)


# ============================================================
# Gradio 渲染
# ============================================================

def gradio_render(client, actors: list, output_image_dir: str,
                  quality: str = "低质量 (快速预览)",
                  lens: int = 31, pcg_timeout: int = 120,
                  cam_pos_override: str = None,
                  cam_target_override: str = None,
                  server_url: str = None) -> tuple:
    """通过 Gradio 服务渲染场景，保存 5 视角图片。

    Args:
        client: gradio_client.Client 实例（server_url 存在时忽略）
        actors: actors 列表（pos 单位 cm）
        output_image_dir: 图片保存目录
        quality: 渲染质量字符串
        lens: 焦距 mm
        pcg_timeout: 超时秒数
        cam_pos_override: 固定相机位置字符串（优先于自动计算）
        cam_target_override: 固定相机目标字符串
        server_url: 若提供，用 requests 直调（跳过 gradio_client heartbeat）

    Returns:
        (image_list, error_msg)  — error_msg 为空字符串表示成功
    """
    os.makedirs(output_image_dir, exist_ok=True)
    json_text = json.dumps([{"actors": actors}], ensure_ascii=False)

    if cam_pos_override and cam_target_override:
        cam_pos_str, cam_target_str = cam_pos_override, cam_target_override
    else:
        cam_pos_str, cam_target_str = compute_camera_params(actors)

    if server_url:
        return _render_via_requests(
            server_url, json_text, cam_pos_str, cam_target_str,
            lens, quality, output_image_dir, pcg_timeout,
        )

    # 旧路径：gradio_client.Client.predict（保持兼容）
    try:
        images, _, _ = client.predict(
            "（不使用预设）", None, json_text, "自定义",
            cam_pos_str, cam_target_str, lens, quality,
            api_name="/render",
        )
    except Exception as e:
        logging.error(f"Gradio 渲染失败: {e}")
        return [], str(e)

    return _save_images(images, output_image_dir)


def _render_via_requests(server_url, json_text, cam_pos_str, cam_target_str,
                          lens, quality, output_image_dir, pcg_timeout=120):
    """每次渲染新建 Client，避免长进程 heartbeat 线程堆积。"""
    import logging as _log
    from gradio_client import Client

    _log.getLogger("httpx").setLevel(_log.WARNING)
    _log.getLogger("httpcore").setLevel(_log.WARNING)

    client = Client(server_url, verbose=False)
    try:
        images, _, _ = client.predict(
            "（不使用预设）", None, json_text, "自定义",
            cam_pos_str, cam_target_str, lens, quality,
            api_name="/render",
        )
    except Exception as e:
        return [], f"渲染失败: {e}"

    return _save_images(images, output_image_dir)


def _extract_path(item):
    """从 Gradio 返回值提取文件路径。"""
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "image" in item:
            img = item["image"]
            return (img.get("path") or img.get("url")) if isinstance(img, dict) else img
        return item.get("path") or item.get("name") or item.get("url")
    if isinstance(item, (list, tuple)) and item:
        return _extract_path(item[0])
    return None


def _save_images(images, output_image_dir: str) -> tuple:
    saved = []
    view_names = [f"debug_cmd_data_result_{i}.jpg" for i in range(5)]
    if images and isinstance(images, list):
        for i, item in enumerate(images):
            path = _extract_path(item)
            if path and os.path.exists(str(path)):
                dst = os.path.join(output_image_dir,
                                   view_names[i] if i < 5 else f"debug_cmd_data_result_{i}.jpg")
                shutil.copy(str(path), dst)
                saved.append(dst)
    if not saved:
        return [], "渲染服务未返回有效图片"
    logging.info(f"渲染完成: {len(saved)} 张 → {output_image_dir}")
    return saved, ""


# ============================================================
# 单位转换 & actors 提取
# ============================================================

def actors_meter_to_cm(actors: list) -> list:
    """将 actors 的 pos 从米转换为厘米（渲染服务要求 cm）。"""
    for actor in actors:
        pos = actor.get("pos")
        if isinstance(pos, list) and pos and not isinstance(pos[0], list):
            actor["pos"] = [p * 100 for p in pos]
    return actors


def llm_output_to_actors(llm_output: dict, component_info: dict,
                          scatter_cache: dict = None) -> tuple:
    """将 map JSON (llm_output) 转换为 actors 列表。

    Returns:
        (actors, error_msg) — error_msg 为空字符串表示成功
    """
    if not _PARSE2PCG_AVAILABLE:
        return None, "map_parser.parse2pcg 不可用"
    try:
        actors, _, _, _ = parse2pcg(llm_output, component_info,
                                     see_detail=[], scatter_cache=scatter_cache)
        actors = actors_meter_to_cm(actors)
        return actors, ""
    except Exception as e:
        logging.error(f"地图解析失败: {e}")
        return None, f"解析失败（{e}）"


# ============================================================
# tool_call 工具函数
# ============================================================

def normalize_tool_call(tool_call) -> tuple:
    """统一将各种格式的 tool_call 转为 (name, args)。"""
    if isinstance(tool_call, dict):
        if 'function' in tool_call:
            func = tool_call['function']
            name = func.get('name')
            if not name:
                return None, {}
            raw = func.get('arguments') or func.get('parameters') or {}
            return name, (json.loads(raw) if isinstance(raw, str) else raw)
        elif 'name' in tool_call:
            name = tool_call['name']
            raw = tool_call.get('arguments', {}) or tool_call.get('args', {}) or {}
            return name, (json.loads(raw) if isinstance(raw, str) else raw)
        return None, {}
    # 对象形式（Gemini FunctionCall 等）
    name = getattr(tool_call, 'name', None)
    if not name:
        func = getattr(tool_call, 'function', None)
        name = getattr(func, 'name', None) if func else None
    if not name:
        return None, {}
    args = getattr(tool_call, 'args', None) or getattr(tool_call, 'arguments', None) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return name, (dict(args) if not isinstance(args, dict) else args)


def fix_flat_args(tool_name: str, tool_args: dict) -> dict:
    """将展平参数还原为标准嵌套格式（兼容 Gemini native fc 变体）。"""
    if tool_name == "terminate":
        return tool_args
    if tool_name == "rotation_and_translation":
        if 'corrections' in tool_args:
            return tool_args
        if 'name' in tool_args or 'pos' in tool_args:
            original = {k: tool_args[k] for k in ('name', 'pos', 'Extend') if k in tool_args}
            modified = {k: v for k, v in tool_args.items() if k not in ('pos', 'Extend')}
            return {'corrections': [{'original_data': original, 'modified_data': modified}]}
    if tool_name in ("add", "delete"):
        if 'modified_data' in tool_args:
            if isinstance(tool_args['modified_data'], dict):
                tool_args['modified_data'] = [tool_args['modified_data']]
            return tool_args
        if 'name' in tool_args or 'pos' in tool_args:
            return {'modified_data': [tool_args]}
    return tool_args


def fc_to_sft_dict(fc) -> dict:
    """将 function_call 转为 sft_trajectory 格式的 dict。"""
    name, args = normalize_tool_call(fc)
    if not name:
        return None
    return {"name": name, "arguments": args}
