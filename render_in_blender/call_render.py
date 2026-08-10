"""
call_render.py — 调用 Blender 渲染服务 API

Usage:
    # 单视角
    python call_render.py --scene chinese_temple.json
    python call_render.py --scene my_courtyard_edited.json --output ./out.png

    # 多视角（--cam-pos 和 --cam-target 可多次传入，数量需一致）
    python call_render.py --scene snow_camp.json --preset 自定义 \
        --cam-pos "0,-50,40" --cam-pos "50,0,40" --cam-pos "0,50,40" \
        --cam-target "0,0,5" --cam-target "0,0,5" --cam-target "0,0,5" \
        --output renders/
"""

import argparse
import os
import shutil
from gradio_client import Client

# ── 配置 ───────────────────────────────────────────────────────────────────────
# 渲染服务地址；用 VIBEWORLD_RENDER_SERVER 环境变量或 --server 覆盖。
SERVER = os.environ.get("VIBEWORLD_RENDER_SERVER", "http://localhost:8080")

# ── 参数 ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="调用 Blender 渲染服务")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--scene", help="预设场景文件名，如 chinese_temple.json")
group.add_argument("--json",  help="本地 JSON 文件路径")

parser.add_argument("--preset",  default="自动 (根据场景计算)",
                    choices=["自动 (根据场景计算)", "等轴测 (斜45°俯视)",
                             "正面 (从前方平视)", "俯视 (正上方朝下)", "自定义"])
parser.add_argument("--cam-pos",    action="append", default=[],
                    help="相机位置 x,y,z（米）。可多次传入以渲染多个视角。")
parser.add_argument("--cam-target", action="append", default=[],
                    help="目标位置 x,y,z（米）。可多次传入。")
parser.add_argument("--lens",    type=float, default=28, help="焦距 mm")
parser.add_argument("--quality", default="中质量 (默认)",
                    choices=["低质量 (快速预览)", "中质量 (默认)", "高质量"])
parser.add_argument("--output",  default="render.png",
                    help="输出路径。单视角：文件名；多视角：目录（自动追加 _0.png, _1.png...）")
parser.add_argument("--save-blend", default=None, help="保存 .blend 文件路径（可选）")
parser.add_argument("--server", default=SERVER,
                    help="渲染服务地址（默认取 VIBEWORLD_RENDER_SERVER，否则 http://localhost:8080）")
args = parser.parse_args()
SERVER = args.server

# ── 读取 JSON 内容 ──────────────────────────────────────────────────────────────
scene_preset = "（不使用预设）"
json_text    = ""

if args.scene:
    scene_preset = args.scene
elif args.json:
    with open(args.json, encoding="utf-8") as f:
        json_text = f.read()

# Join multiple cam positions with newlines (server parses by newline)
cam_pos_str    = "\n".join(args.cam_pos)    if args.cam_pos    else ""
cam_target_str = "\n".join(args.cam_target) if args.cam_target else ""
n_views = max(len(args.cam_pos), len(args.cam_target), 1)

# ── 调用 API ───────────────────────────────────────────────────────────────────
print(f"连接服务: {SERVER}")
client = Client(SERVER, verbose=False)

print(f"提交渲染: scene={scene_preset or args.json}  quality={args.quality}  lens={args.lens}  views={n_views}")
image_paths, blend_path, status = client.predict(
    scene_preset=scene_preset,
    json_file=None,
    json_text=json_text,
    preset=args.preset,
    cam_pos=cam_pos_str,
    cam_target=cam_target_str,
    lens=args.lens,
    quality=args.quality,
    api_name="/render",
)

# ── 输出结果 ───────────────────────────────────────────────────────────────────
print(status)

# Gradio Gallery returns a list of dicts like [{"image": {"path": "..."}}, ...]
# Normalize to list of paths
def _extract_path(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "image" in item and isinstance(item["image"], dict):
            return item["image"].get("path")
        return item.get("path")
    if isinstance(item, (list, tuple)) and item:
        return _extract_path(item[0])
    return None

if image_paths:
    paths = [_extract_path(p) for p in image_paths] if isinstance(image_paths, list) else [_extract_path(image_paths)]
    paths = [p for p in paths if p]

    if len(paths) == 1:
        # Single view: copy to exact output path
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        shutil.copy(paths[0], args.output)
        print(f"图片已保存: {args.output}")
    else:
        # Multi view: treat --output as base
        base, ext = os.path.splitext(args.output)
        if not ext:
            ext = ".png"
            os.makedirs(args.output, exist_ok=True)
            base = os.path.join(args.output, "render")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
        for i, p in enumerate(paths):
            dst = f"{base}_{i}{ext}"
            shutil.copy(p, dst)
            print(f"图片已保存: {dst}")
else:
    print("渲染失败，未获得图片")

if args.save_blend and blend_path:
    shutil.copy(blend_path, args.save_blend)
    print(f".blend 已保存: {args.save_blend}")
