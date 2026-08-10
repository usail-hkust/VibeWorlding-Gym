"""
Gradio service for Blender scene rendering.
Input:  scene JSON + optional camera parameters
Output: rendered PNG image
"""

import gradio as gr
import subprocess
import tempfile
import os
import json
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
# Path to the Blender 4.2.x binary. Override with the BLENDER_EXE env var.
BLENDER_EXE   = os.environ.get("BLENDER_EXE", "/opt/blender-4.2.0-linux-x64/blender")
RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), "render_scene.py")
# Each worker writes into its own output sub-directory so that running several
# workers behind session_proxy.py cannot clobber each other's renders.
WORKER_ID     = os.environ.get("WORKER_ID") or os.environ.get("PORT", "default")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output", WORKER_ID)
CMDS_DIR      = os.path.join(os.path.dirname(__file__), "assets", "cmds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _get_preset_scenes():
    files = sorted(f for f in os.listdir(CMDS_DIR) if f.endswith(".json"))
    return ["（不使用预设）"] + files

QUALITY_SAMPLES = {"低质量 (快速预览)":  8,
                   "中质量 (默认)":      32,
                   "高质量":            128}

PRESETS = {
    "自动 (根据场景计算)":      ("", ""),
    "等轴测 (斜45°俯视)":       ("auto_iso",   "auto_iso"),
    "正面 (从前方平视)":        ("auto_front",  "auto_front"),
    "俯视 (正上方朝下)":        ("auto_top",    "auto_top"),
    "自定义":                   ("custom",      "custom"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _scene_center_span(actors):
    """Return (cx, cy, cz, span) in Blender meters from actor pos list."""
    SCALE = 0.01
    # Flip Y to convert UE -> Blender
    xs = [a["pos"][0] * SCALE for a in actors if "pos" in a]
    ys = [-a["pos"][1] * SCALE for a in actors if "pos" in a]
    zs = [a["pos"][2] * SCALE for a in actors if "pos" in a]
    if not xs:
        return 0, 0, 0, 10
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
    return cx, cy, cz, span


def _preset_to_args(preset, scene_json_str):
    """Compute cam_pos / cam_target strings from preset name."""
    try:
        data = json.loads(scene_json_str)
        actors = []
        for item in data:
            actors.extend(item.get("actors", []))
        cx, cy, cz, span = _scene_center_span(actors)
    except Exception:
        cx, cy, cz, span = 0, 0, 0, 10

    if preset == "自动 (根据场景计算)":
        return "", ""
    elif preset == "等轴测 (斜45°俯视)":
        dist, h = span * 1.5, span * 0.8
        cam    = f"{cx},{cy - dist},{cz + h}"
        target = f"{cx},{cy + span * 0.3},{cz + span * 0.25}"
    elif preset == "正面 (从前方平视)":
        dist = span * 1.5
        cam    = f"{cx},{cy - dist},{cz + span * 0.15}"
        target = f"{cx},{cy},{cz + span * 0.15}"
    elif preset == "俯视 (正上方朝下)":
        cam    = f"{cx},{cy},{cz + span * 2}"
        target = f"{cx},{cy},{cz}"
    else:  # 自定义：返回空，由用户自填
        return "", ""
    return cam, target


# ── Core render function ──────────────────────────────────────────────────────
def render(scene_preset, json_file, json_text, preset, cam_pos, cam_target, lens, quality):
    # 1. Get JSON content
    if scene_preset and scene_preset != "（不使用预设）":
        preset_path = os.path.join(CMDS_DIR, scene_preset)
        with open(preset_path, encoding="utf-8") as f:
            scene_json_str = f.read()
    elif json_file is not None:
        with open(json_file.name, encoding="utf-8") as f:
            scene_json_str = f.read()
    elif json_text.strip():
        scene_json_str = json_text.strip()
    else:
        return None, None, "❌ 请上传或粘贴场景 JSON"

    # Validate JSON
    try:
        json.loads(scene_json_str)
    except json.JSONDecodeError as e:
        return None, None, f"❌ JSON 格式错误: {e}"

    # 2. Resolve camera args
    samples = QUALITY_SAMPLES.get(quality, 64)

    if preset != "自定义":
        final_cam_pos, final_cam_target = _preset_to_args(preset, scene_json_str)
    else:
        final_cam_pos    = cam_pos.strip()
        final_cam_target = cam_target.strip()

    lens_str    = str(lens) if lens else "28"
    samples_str = str(samples)

    # 3. Write temp files
    tmpdir = tempfile.mkdtemp(dir=OUTPUT_DIR)
    scene_path  = os.path.join(tmpdir, "scene.json")
    output_path = os.path.join(tmpdir, "render.png")
    blend_path  = os.path.join(tmpdir, "scene.blend")

    with open(scene_path, "w", encoding="utf-8") as f:
        f.write(scene_json_str)

    # 4. Call Blender
    cmd = [
        BLENDER_EXE, "--background", "--python", RENDER_SCRIPT,
        "--",
        scene_path, output_path,
        final_cam_pos, final_cam_target,
        lens_str, samples_str,
        blend_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=900,          # 15 min hard limit
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, "❌ 渲染超时（超过15分钟）"
    except FileNotFoundError:
        return None, f"❌ 找不到 Blender: {BLENDER_EXE}"

    # 5. Parse output
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        log = f"returncode={result.returncode}\n--- stdout ---\n{stdout[-2000:]}\n--- stderr ---\n{stderr[-1000:]}"
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None, f"❌ 渲染失败:\n{log}"

    if not os.path.exists(output_path):
        # Multi-view: look for output_0.png etc.
        base, ext = os.path.splitext(output_path)
        multi = sorted(p for p in os.listdir(tmpdir)
                       if p.startswith(os.path.basename(base) + "_") and p.endswith(ext))
        if not multi:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, None, f"❌ 渲染完成但未找到输出文件\n{stdout[-1000:]}"
        image_paths = [os.path.join(tmpdir, p) for p in multi]
    else:
        image_paths = [output_path]

    # Extract useful stats from stdout
    lines = stdout.splitlines()
    stats = "\n".join(l for l in lines if any(
        kw in l for kw in ["Scene:", "Loaded:", "Missing:", "Saved:", "center:", "span:", "Camera", "] "]
    ))
    status = f"✅ 渲染完成（{len(image_paths)} 张图）\n{stats}"

    blend_out = blend_path if os.path.exists(blend_path) else None
    return image_paths, blend_out, status


# ── UI ────────────────────────────────────────────────────────────────────────
def update_cam_visibility(preset):
    visible = preset == "自定义"
    return (
        gr.update(visible=visible),  # cam_pos
        gr.update(visible=visible),  # cam_target
        gr.update(visible=visible),  # lens
    )


with gr.Blocks(title="Blender 场景渲染", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏯 Blender 场景渲染服务")
    gr.Markdown("上传场景 JSON，选择相机和质量，点击渲染。")

    with gr.Row():
        # ── 左栏：输入 ──
        with gr.Column(scale=1):
            gr.Markdown("### 场景输入")
            scene_preset = gr.Dropdown(
                choices=_get_preset_scenes(),
                value="（不使用预设）",
                label="预设场景",
            )
            json_file = gr.File(
                label="上传 JSON 文件",
                file_types=[".json"],
            )
            json_text = gr.Textbox(
                label="或直接粘贴 JSON",
                placeholder='[{"actors": [...]}]',
                lines=6,
                max_lines=20,
            )

            gr.Markdown("### 相机设置")
            preset = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="自动 (根据场景计算)",
                label="视角预设",
            )
            with gr.Group(visible=False) as custom_group:
                cam_pos = gr.Textbox(
                    label="相机位置 (x,y,z)  多视角每行一组",
                    placeholder="例: 0,-20,15\n20,-20,15",
                    lines=3,
                )
                cam_target = gr.Textbox(
                    label="Look-at 目标 (x,y,z)  多视角每行一组",
                    placeholder="例: 0,0,2\n0,0,2",
                    lines=3,
                )
                lens = gr.Slider(
                    minimum=10, maximum=200, value=28, step=1,
                    label="焦距 (mm)",
                )

            gr.Markdown("### 渲染质量")
            quality = gr.Radio(
                choices=list(QUALITY_SAMPLES.keys()),
                value="中质量 (默认, ~3min)",
                label="",
            )

            render_btn = gr.Button("🎬 开始渲染", variant="primary")

        # ── 右栏：输出 ──
        with gr.Column(scale=1):
            gr.Markdown("### 渲染结果")
            output_img = gr.Gallery(
                label="渲染图片（支持多视角）",
                height=480,
                columns=2,
                object_fit="contain",
                preview=True,
            )
            status_box = gr.Textbox(
                label="状态",
                lines=6,
                interactive=False,
            )
            blend_file = gr.File(
                label="下载 .blend 文件",
                visible=True,
            )

    # ── Events ──
    preset.change(
        fn=update_cam_visibility,
        inputs=[preset],
        outputs=[cam_pos, cam_target, lens],
    )
    # Show/hide custom group
    preset.change(
        fn=lambda p: gr.update(visible=(p == "自定义")),
        inputs=[preset],
        outputs=[custom_group],
    )

    render_btn.click(
        fn=render,
        inputs=[scene_preset, json_file, json_text, preset, cam_pos, cam_target, lens, quality],
        outputs=[output_img, blend_file, status_box],
    )

if __name__ == "__main__":
    # 注意：theme 是 gr.Blocks() 的参数，不是 launch() 的（见上面的 gr.Blocks(...)）。
    demo.launch(
        server_name=os.environ.get("RENDER_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "8080")),
        show_error=True,
    )
