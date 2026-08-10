"""
render_raw_data_images.py — 批量渲染场景的 5 视角初始图片

对--raw_data_dir 下每个子目录读取 pcg_scene.json，调用 PCG 渲染服务渲染
front/back/left/right/topdown 五个视角，输出到该子目录的 image/ 下。

用法：
    python render_raw_data_images.py \
        --raw_data_dir ./render_in_blender/assets/cmds/ \
        --server http://localhost:8080 \
        --quality "低质量 (快速预览)"
"""

import argparse
import json
import math
import os
import shutil
import tempfile
import time

# gradio_client 默认把返回图片缓存到 /tmp，大批量渲染容易打满。
# 用 GRADIO_TEMP_DIR 指到空间充足的盘上；默认放仓库内的 .gradio_tmp/。
_GRADIO_TMP = os.environ.get(
    "GRADIO_TEMP_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gradio_tmp"),
)
os.makedirs(_GRADIO_TMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _GRADIO_TMP
tempfile.tempdir = _GRADIO_TMP


def compute_camera_params(actors):
    """计算 5 视角相机参数（front/back/left/right/topdown）"""
    SCALE = 0.01  # cm → m

    xs = [a["pos"][0] * SCALE for a in actors if "pos" in a]
    ys = [-a["pos"][1] * SCALE for a in actors if "pos" in a]
    zs = [a["pos"][2] * SCALE for a in actors if "pos" in a]

    if not xs:
        cx, cy, cz, span = 0, 0, 0, 40
    else:
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        cz = (max(zs) + min(zs)) / 2
        span_xy = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
        span_z  = max(max(zs) - min(zs), 0.1)
        span    = max(span_z * 4.0, span_xy * 0.8, 10.0)

    dist      = span * 0.9
    h_offset  = dist * math.sin(math.radians(30))
    d_horiz   = dist * math.cos(math.radians(30))
    cam_z     = cz + h_offset
    target_z  = cz + (max(zs) - min(zs) if zs else 0) * 0.15

    cameras = [
        (f"{cx:.2f},{cy - d_horiz:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),   # front
        (f"{cx:.2f},{cy + d_horiz:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),   # back
        (f"{cx - d_horiz:.2f},{cy:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),   # left
        (f"{cx + d_horiz:.2f},{cy:.2f},{cam_z:.2f}", f"{cx:.2f},{cy:.2f},{target_z:.2f}"),   # right
        (f"{cx:.2f},{cy:.2f},{cz + dist * 1.5:.2f}", f"{cx:.2f},{cy:.2f},{cz:.2f}"),          # topdown
    ]

    return "\n".join(c[0] for c in cameras), "\n".join(c[1] for c in cameras)


def extract_path(item):
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "image" in item:
            img = item["image"]
            if isinstance(img, dict):
                return img.get("path") or img.get("url")
            return img
        return item.get("path") or item.get("name") or item.get("url")
    if isinstance(item, (list, tuple)) and item:
        return extract_path(item[0])
    return None


def render_scene(client, pcg_json_path, image_dir, quality, lens=31):
    with open(pcg_json_path, encoding="utf-8") as f:
        pcg_data = json.load(f)

    actors = []
    for chunk in pcg_data:
        actors.extend(chunk.get("actors", []))

    json_text = json.dumps(pcg_data, ensure_ascii=False)

    cam_params_path = os.path.join(os.path.dirname(pcg_json_path), "camera_params.json")
    if os.path.exists(cam_params_path):
        with open(cam_params_path, encoding="utf-8") as f:
            cp = json.load(f)
        cam_pos_str    = cp["cam_pos"]
        cam_target_str = cp["cam_target"]
        lens           = cp.get("lens", lens)
    else:
        cam_pos_str, cam_target_str = compute_camera_params(actors)

    images, _, status = client.predict(
        "（不使用预设）",
        None,
        json_text,
        "自定义",
        cam_pos_str,
        cam_target_str,
        lens,
        quality,
        api_name="/render",
    )

    view_names = [
        "debug_cmd_data_result_0.jpg",
        "debug_cmd_data_result_1.jpg",
        "debug_cmd_data_result_2.jpg",
        "debug_cmd_data_result_3.jpg",
        "debug_cmd_data_result_4.jpg",
    ]

    os.makedirs(image_dir, exist_ok=True)
    saved = 0
    if images and isinstance(images, list):
        for i, item in enumerate(images):
            path = extract_path(item)
            if path and os.path.exists(str(path)):
                dst = os.path.join(image_dir, view_names[i] if i < len(view_names) else f"debug_cmd_data_result_{i}.jpg")
                shutil.copy(str(path), dst)
                saved += 1

    return saved, status


def main():
    parser = argparse.ArgumentParser(description="批量渲染 raw_data 场景图片")
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--quality", default="低质量 (快速预览)",
                        choices=["低质量 (快速预览)", "中质量 (默认)", "高质量"])
    parser.add_argument("--lens", type=float, default=31)
    args = parser.parse_args()

    from gradio_client import Client
    print(f"连接渲染服务: {args.server}")
    client = Client(args.server, verbose=False)
    print("✅ 连接成功\n")

    sample_dirs = sorted(
        [d for d in os.listdir(args.raw_data_dir)
         if os.path.isdir(os.path.join(args.raw_data_dir, d))],
        key=lambda x: int(x) if x.isdigit() else x
    )
    print(f"共 {len(sample_dirs)} 个场景待渲染\n")

    success = skip = fail = 0
    for sample_id in sample_dirs:
        sample_path = os.path.join(args.raw_data_dir, sample_id)
        pcg_path    = os.path.join(sample_path, "pcg_scene.json")
        image_dir   = os.path.join(sample_path, "image")

        if os.path.exists(image_dir) and len(os.listdir(image_dir)) >= 5:
            print(f"  [{sample_id:>4}] ⏭️  已有图片，跳过")
            skip += 1
            success += 1
            continue

        if not os.path.exists(pcg_path):
            print(f"  [{sample_id:>4}] ⚠️  无 pcg_scene.json，跳过")
            fail += 1
            continue

        print(f"  [{sample_id:>4}] 渲染中...", end="", flush=True)
        t0 = time.time()
        try:
            saved, status = render_scene(client, pcg_path, image_dir, args.quality, args.lens)
            elapsed = time.time() - t0
            if saved >= 5:
                print(f" ✅ {saved}张 ({elapsed:.1f}s)")
                success += 1
            else:
                print(f" ⚠️  只得到{saved}张 ({elapsed:.1f}s)")
                for line in str(status).split("\n")[:3]:
                    print(f"       {line}")
                fail += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f" ❌ 失败 ({elapsed:.1f}s): {e}")
            fail += 1

    total = len(sample_dirs)
    print(f"\n完成: {success}/{total} 成功  {skip} 跳过  {fail} 失败")


if __name__ == "__main__":
    main()
