"""
render_scene.py  —  Blender headless script called by gradio_app.py
Reads a cmds JSON, imports all assets, sets up lighting + camera, renders PNG.

CLI (called by gradio_app.py):
    blender --background --python render_scene.py -- \
        <scene_json> <output_png> [cam_pos] [cam_target] [lens] [samples]

    cam_pos / cam_target : "x,y,z" in Blender metres, or "" for auto
    lens                 : focal length mm (default 28)
    samples              : Cycles samples (default 64)
"""

import bpy
import sys
import os
import json
import math
import mathutils
import time
import subprocess
import threading

_t0 = time.time()
def _log(msg):
    print(f"[{time.time()-_t0:6.2f}s] {msg}", flush=True)

# ── GPU monitor (background thread during render) ──────────────────────────────
_gpu_samples = []

def _gpu_monitor(stop_event, interval=1.0):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=3
            ).decode().strip()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                _gpu_samples.append({
                    "t": round(time.time() - _t0, 2),
                    "idx": parts[0], "name": parts[1],
                    "gpu%": parts[2], "mem%": parts[3],
                    "mem_used": parts[4], "mem_total": parts[5],
                })
        except Exception:
            pass
        stop_event.wait(interval)

# ── CLI args ───────────────────────────────────────────────────────────────────

argv = sys.argv
try:
    args = argv[argv.index("--") + 1:]
except ValueError:
    args = []

def _arg(i, default=""):
    return args[i].strip() if i < len(args) else default

SCENE_JSON  = _arg(0)
OUTPUT_PNG  = _arg(1)
CAM_POS_STR = _arg(2)   # "x,y,z" or ""
CAM_TGT_STR = _arg(3)   # "x,y,z" or ""
LENS        = float(_arg(4, "28"))
SAMPLES     = int(_arg(5, "64"))
OUTPUT_BLEND = _arg(6)   # optional, save .blend alongside PNG

CORRECTIONS_JSON = os.path.join(os.path.dirname(__file__),
                                "assets", "glb_corrections.json")
ITEM_INFOS_JSON  = os.path.join(os.path.dirname(__file__),
                                "assets", "item_infos.json")
# GLB asset root (one sub-directory per 5-digit type_id, containing <type_id>.glb).
# Download from https://huggingface.co/datasets/usail-hkust/VWE-Bench and place
# under assets/models/clone/, or point VIBEWORLD_MODELS_DIR at your own copy.
MODELS_DIR       = os.environ.get("VIBEWORLD_MODELS_DIR") or os.path.join(
                                os.path.dirname(__file__),
                                "assets", "models", "clone")

if not SCENE_JSON or not OUTPUT_PNG:
    print("ERROR: scene_json and output_png are required.")
    sys.exit(1)

print(f"Scene JSON : {SCENE_JSON}")
print(f"Output PNG : {OUTPUT_PNG}")
print(f"Models dir : {MODELS_DIR}")
print(f"cam_pos='{CAM_POS_STR}'  cam_target='{CAM_TGT_STR}'  lens={LENS}  samples={SAMPLES}")
_log("START")

# ── Load display names from item_infos.json ────────────────────────────────────

id_to_name = {}
if os.path.isfile(ITEM_INFOS_JSON):
    with open(ITEM_INFOS_JSON, encoding="utf-8") as f:
        _infos = json.load(f)
    for tid, info in _infos.items():
        id_to_name[tid] = info.get("NameChinese", tid)
    print(f"Loaded {len(id_to_name)} asset names.")
_log("names loaded")

def _glb_path(type_id):
    return os.path.join(MODELS_DIR, type_id, f"{type_id}.glb")

print(f"Paths resolved on demand from {MODELS_DIR}.")
_log("mapping loaded")

# ── Load GLB correction table (per-asset size + pivot) ─────────────────────────
# HY3D AI 重建的 GLB 各轴比例与原始资产不一致，需要各轴独立校正。
# corrections[type_id] = {
#   "correction_x": float,  # mat_S X轴 = sca.x * correction_x
#   "correction_y": float,  # mat_S Y轴 = sca.y * correction_y
#   "correction_z": float,  # mat_S Z轴 = sca.z * correction_z
#   "pivot_z_local": float, # GLB 底部 local-Z 位置（负值，m），贴地补偿用
# }

_glb_corrections = {}
if os.path.isfile(CORRECTIONS_JSON):
    with open(CORRECTIONS_JSON, encoding="utf-8") as f:
        _glb_corrections = json.load(f)
    print(f"Loaded {len(_glb_corrections)} GLB correction entries.")
    _log("glb_corrections loaded")
else:
    print(f"WARNING: {CORRECTIONS_JSON} not found, GLB assets will render at raw scale (~1m)")
    _log("glb_corrections: missing, no per-asset correction")

# ── Load scene JSON ────────────────────────────────────────────────────────────

with open(SCENE_JSON, encoding="utf-8") as f:
    scene_data = json.load(f)

actors = []
for chunk in scene_data:
    actors.extend(chunk.get("actors", []))

print(f"Scene: {len(actors)} actors.")
_log("scene JSON loaded")

# ── Clear default scene ────────────────────────────────────────────────────────

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# ── Constants ─────────────────────────────────────────────────────────────────

SCALE      = 0.01
yup_to_zup = mathutils.Matrix.Rotation(math.radians(90), 4, "X")

# ── Asset import cache（支持 OBJ 和 GLB）─────────────────────────────────────
# 返回 (meshes, is_glb)：
#   is_glb=True  → HY3D GLB，需要 yup_to_zup + per-asset 尺寸校正
#   is_glb=False → OBJ（Y-up），actor 放置时需要乘 yup_to_zup

_mesh_cache = {}   # obj_path -> (meshes_or_None, is_glb)

def _import_obj_cached(obj_path):
    """导入 OBJ 或 GLB 文件，返回 (meshes, is_glb)（结果缓存）"""
    if obj_path in _mesh_cache:
        return _mesh_cache[obj_path]
    if not os.path.isfile(obj_path):
        print(f"  WARNING: asset not found: {obj_path}")
        _mesh_cache[obj_path] = (None, False)
        return _mesh_cache[obj_path]
    before = set(bpy.data.objects.keys())
    ext = os.path.splitext(obj_path)[1].lower()
    is_glb = ext in (".glb", ".gltf")
    try:
        if is_glb:
            bpy.ops.import_scene.gltf(filepath=obj_path)
        else:
            try:
                bpy.ops.wm.obj_import(filepath=obj_path)
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=obj_path)
    except Exception as e:
        print(f"  WARNING: import failed for {obj_path}: {e}")
        _mesh_cache[obj_path] = (None, is_glb)
        return _mesh_cache[obj_path]
    after    = set(bpy.data.objects.keys())
    new_objs = [bpy.data.objects[n] for n in (after - before)]
    meshes   = []
    for obj in new_objs:
        if obj.type != 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        mesh = obj.data
        # Clear custom split normals (can cause Cycles to render pure black)
        if mesh.has_custom_normals:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        meshes.append(mesh)
        bpy.data.objects.remove(obj, do_unlink=True)
    _mesh_cache[obj_path] = (meshes if meshes else None, is_glb)
    return _mesh_cache[obj_path]

# ── Import actors ──────────────────────────────────────────────────────────────

missing_ids = set()
loaded = skipped = 0

for actor in actors:
    type_id = str(actor.get("typeId", "")).strip()
    gname   = actor.get("gname", f"actor_{actor.get('id', 0)}")
    name    = actor.get("name", id_to_name.get(type_id, type_id))
    pos     = actor.get("pos", [0.0, 0.0, 0.0])
    rot     = actor.get("rot", [0.0, 0.0, 0.0, 1.0])
    sca     = actor.get("sca", [1.0, 1.0, 1.0])

    if type_id not in id_to_name:
        if type_id not in missing_ids:
            print(f"  MISSING typeId={type_id}  name={name} (not in item_infos)")
            missing_ids.add(type_id)

    meshes, is_glb = _import_obj_cached(_glb_path(type_id))
    if meshes is None:
        skipped += 1
        continue

    # UE -> Blender: flip Y (Y-axis mirror).
    # Quaternion under Y-mirror: (w,x,y,z) -> (w, -x, y, -z)
    loc   = mathutils.Vector([pos[0]*SCALE, -pos[1]*SCALE, pos[2]*SCALE])
    rx, ry, rz, rw = rot[0], rot[1], rot[2], rot[3]
    rot_q    = mathutils.Quaternion((rw, -rx, ry, -rz))
    sv       = mathutils.Vector(sca)
    # sca 是无单位缩放比，不需要乘 SCALE；位置已由 loc 处理了单位转换
    mat_S    = mathutils.Matrix.Diagonal((sv.x, sv.y, sv.z, 1.0))

    if is_glb:
        corr_entry   = _glb_corrections.get(type_id, {})
        cx           = corr_entry.get("correction_x",  1.0)
        cy           = corr_entry.get("correction_y",  1.0)
        cz           = corr_entry.get("correction_z",  1.0)
        pivot_z_loc  = corr_entry.get("pivot_z_local", 0.0)
        is_normalized = corr_entry.get("normalized",   False)

        if is_normalized:
            # ── 预处理 normalized GLB (step4 normalize_glb_blender.py 输出) ──
            # LOCAL 顶点已是 Blender Z-up，底部 Zmin=0，XY 居中，尺寸已对齐 BBox
            # 不需要 yup_to_zup，不需要 z_offset，correction=1，只有 sca 做场景缩放
            mat_S_norm = mathutils.Matrix.Diagonal((sv.x * cx, sv.y * cy, sv.z * cz, 1.0))
            full_mat = (mathutils.Matrix.Translation(loc)
                        @ rot_q.to_matrix().to_4x4()
                        @ mat_S_norm)
        else:
            # ── 原始 HY3D GLB（Y-up，node 变换未烘焙）─────────────────────────
            mat_S_glb = mathutils.Matrix.Diagonal((sv.x * cx, sv.y * cy, sv.z * cz, 1.0))
            z_offset  = abs(pivot_z_loc) * sv.z * cz
            loc_corrected = mathutils.Vector([loc.x, loc.y, loc.z + z_offset])
            full_mat = (mathutils.Matrix.Translation(loc_corrected)
                        @ rot_q.to_matrix().to_4x4()
                        @ mat_S_glb
                        @ yup_to_zup)
            if max(cx, cy, cz) > 1.1 or min(cx, cy, cz) < 0.9:
                print(f"  GLB correction: typeId={type_id} "
                      f"cx={cx:.2f} cy={cy:.2f} cz={cz:.2f} "
                      f"z_offset={z_offset:.3f}m")
    else:
        full_mat = mathutils.Matrix.Translation(loc) @ rot_q.to_matrix().to_4x4() @ mat_S @ yup_to_zup

    empty = bpy.data.objects.new(f"{name}_{gname}", None)
    bpy.context.collection.objects.link(empty)
    empty.matrix_world = full_mat

    mesh_objs = []
    for i, mesh_data in enumerate(meshes):
        label    = f"{name}_mesh" if len(meshes) == 1 else f"{name}_mesh.{i:03d}"
        mesh_obj = bpy.data.objects.new(label, mesh_data.copy() if len(meshes) > 1 else mesh_data)
        bpy.context.collection.objects.link(mesh_obj)
        mesh_obj.parent       = empty
        mesh_obj["typeId"]    = type_id
        mesh_obj["assetName"] = name
        mesh_objs.append(mesh_obj)

    # ── 逐 actor 装配诊断日志（世界坐标 BBox + 贴地状态）──────────────────
    # 用于 curl 渲染时在 status 日志里直接看到每个资产的真实坐标，定位悬空
    try:
        bpy.context.view_layer.update()
        _mn = mathutils.Vector(( 1e18,  1e18,  1e18))
        _mx = mathutils.Vector((-1e18, -1e18, -1e18))
        for _o in mesh_objs:
            for _c in _o.bound_box:
                _w = _o.matrix_world @ mathutils.Vector(_c)
                for _i in range(3):
                    _mn[_i] = min(_mn[_i], _w[_i]); _mx[_i] = max(_mx[_i], _w[_i])
        _kind = "norm-GLB" if (is_glb and _glb_corrections.get(type_id, {}).get("normalized")) \
                else ("raw-GLB" if is_glb else "OBJ")
        # normalized GLB / OBJ：pivot 在 bbox 几何中心（与轻游一致）
        #   pos.z=0 时中心在地面 → 底部 Zbot≈-半高 是【正确】的，不是悬空
        #   判定"异常离地"：中心 Z 明显偏离 pos.z（说明 pivot 不在中心或被错误偏移）
        _zc = (_mn.z + _mx.z) / 2.0
        _expected_zc = pos[2] * 0.01  # pos.z 转米（pivot 中心应落在此高度）
        _ok = abs(_zc - _expected_zc) < 0.1
        _ground = "中心贴地OK" if _ok else f"异常!中心z={_zc:.2f}(应={_expected_zc:.2f})"
        print(f"  ACTOR [{_kind:8s}] tid={type_id} {name} | pos={pos} sca={[round(s,2) for s in sca]} | "
              f"worldZ[{_mn.z:.2f},{_mx.z:.2f}] centerZ={_zc:.2f} WDH=({_mx.x-_mn.x:.2f},{_mx.y-_mn.y:.2f},{_mx.z-_mn.z:.2f}) | {_ground}")
    except Exception as _e:
        print(f"  ACTOR diag fail tid={type_id}: {_e}")

    loaded += 1

print(f"Loaded: {loaded}  Skipped: {skipped}  Missing IDs: {len(missing_ids)}")
_log(f"assets imported ({loaded} actors, {len(_mesh_cache)} unique OBJs)")

# ── Lighting ───────────────────────────────────────────────────────────────────

sun_d = bpy.data.lights.new("Sun", type="SUN")
sun_d.energy = 3.0
sun_d.color  = (1.0, 0.95, 0.85)
sun_d.angle  = math.radians(5)
sun_o = bpy.data.objects.new("Sun", sun_d)
bpy.context.collection.objects.link(sun_o)
sun_o.rotation_euler = (math.radians(55), 0, math.radians(45))
sun_o.hide_render    = False

fill_d = bpy.data.lights.new("Fill", type="SUN")
fill_d.energy = 0.8
fill_d.color  = (0.7, 0.85, 1.0)
fill_d.angle  = math.radians(30)
fill_o = bpy.data.objects.new("Fill", fill_d)
bpy.context.collection.objects.link(fill_o)
fill_o.rotation_euler = (math.radians(30), 0, math.radians(225))
fill_o.hide_render    = False

world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value    = (0.6, 0.7, 0.9, 1.0)
    bg.inputs["Strength"].default_value = 1.2

# ── Scene bounding box — from actor positions (game cm -> m) ──────────────────
# Using actor positions is more reliable than Blender bound_box after yup_to_zup
# rotation, which can inflate Z due to Y-depth in original OBJ vertices.

SCALE = 0.01
# Flip Y to convert UE -> Blender
act_xs = [a["pos"][0]*SCALE for a in actors if "pos" in a]
act_ys = [-a["pos"][1]*SCALE for a in actors if "pos" in a]
act_zs = [a["pos"][2]*SCALE for a in actors if "pos" in a]

if act_xs:
    cx = (max(act_xs) + min(act_xs)) / 2
    cy = (max(act_ys) + min(act_ys)) / 2
    cz = (max(act_zs) + min(act_zs)) / 2
    span_xy = max(max(act_xs)-min(act_xs), max(act_ys)-min(act_ys), 0.1)
    # Estimate building height from scale — use 15m as default if unknown
    span_z  = 15.0
    span    = max(span_z * 4.0, span_xy * 0.8, 10.0)
else:
    cx = cy = cz = 0
    span = 40

print(f"Scene center: ({cx:.2f}, {cy:.2f}, {cz:.2f})  span: {span:.2f}")

# ── Camera(s) ──────────────────────────────────────────────────────────────────

def _parse_vec(s):
    parts = [float(v) for v in s.split(",")]
    return mathutils.Vector(parts)

def _parse_vec_list(s):
    """Parse newline-separated '(x,y,z)' entries into a list of Vectors.
    Empty string returns []. Single line returns single-element list."""
    if not s.strip():
        return []
    return [_parse_vec(line) for line in s.splitlines() if line.strip()]

cam_pos_list    = _parse_vec_list(CAM_POS_STR)
cam_target_list = _parse_vec_list(CAM_TGT_STR)

if cam_pos_list and cam_target_list:
    # Broadcast if one has 1 entry and the other has many
    if len(cam_pos_list) == 1 and len(cam_target_list) > 1:
        cam_pos_list = cam_pos_list * len(cam_target_list)
    elif len(cam_target_list) == 1 and len(cam_pos_list) > 1:
        cam_target_list = cam_target_list * len(cam_pos_list)
    if len(cam_pos_list) != len(cam_target_list):
        print(f"ERROR: cam_pos has {len(cam_pos_list)} entries but cam_target has {len(cam_target_list)}")
        sys.exit(1)
    cameras = list(zip(cam_pos_list, cam_target_list))
else:
    # Auto isometric (single view)
    dist = span * 1.2
    h    = span * 1.0
    cameras = [(mathutils.Vector((cx, cy - dist, cz + h)),
                mathutils.Vector((cx, cy, cz)))]

print(f"Cameras: {len(cameras)} view(s)")
for i, (cl, ct) in enumerate(cameras):
    print(f"  [{i}] {tuple(round(v,2) for v in cl)}  ->  {tuple(round(v,2) for v in ct)}")

cam_data = bpy.data.cameras.new("Camera")
cam_obj  = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.data.lens = LENS

# ── Render (Cycles) ────────────────────────────────────────────────────────────

scene = bpy.context.scene
scene.render.engine  = "CYCLES"
scene.cycles.samples = SAMPLES

# GPU acceleration: enable CUDA/OptiX/Metal/HIP, fall back to CPU if unavailable
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.refresh_devices()
gpu_types = ("CUDA", "OPTIX", "METAL", "HIP", "ONEAPI")
activated = False
for gpu_type in gpu_types:
    try:
        prefs.compute_device_type = gpu_type
        prefs.refresh_devices()
        devices = prefs.get_devices_for_type(gpu_type)
        if devices:
            for d in devices:
                d.use = "CPU" not in d.name.upper() and "INTEL" not in d.name.upper()
            if not any(d.use for d in devices):
                for d in devices:
                    d.use = True  # fallback: enable all if filter too aggressive
            scene.cycles.device = "GPU"
            activated = True
            _log(f"Cycles {gpu_type}: {[d.name for d in devices if d.use]}")
            break
    except Exception:
        continue
if not activated:
    scene.cycles.device = "CPU"
    _log("Cycles GPU: no compatible device found, using CPU")
_log(f"render setup done (engine=CYCLES device={scene.cycles.device})")
scene.render.resolution_x           = 1280
scene.render.resolution_y           = 720
scene.render.filepath               = OUTPUT_PNG
scene.render.image_settings.file_format = "PNG"

# Denoise: reduces samples needed for clean result
scene.cycles.use_denoising = True
try:
    scene.cycles.denoiser = "OPENIMAGEDENOISE"  # CPU-based, always available
    if activated and scene.cycles.device == "GPU":
        scene.cycles.denoiser = "OPTIX"         # GPU denoiser, faster on NVIDIA
except Exception:
    pass

os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

# Compute per-view output paths: single view keeps original name, multi-view uses _N suffix
output_paths = []
if len(cameras) == 1:
    output_paths = [OUTPUT_PNG]
else:
    base, ext = os.path.splitext(OUTPUT_PNG)
    output_paths = [f"{base}_{i}{ext}" for i in range(len(cameras))]

_log("render start")
_stop = threading.Event()
_monitor = threading.Thread(target=_gpu_monitor, args=(_stop,), daemon=True)
_monitor.start()

for i, ((cam_loc, cam_target), out_path) in enumerate(zip(cameras, output_paths)):
    cam_obj.location       = cam_loc
    cam_obj.rotation_euler = (cam_target - cam_loc).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath  = out_path
    _log(f"render view {i+1}/{len(cameras)} -> {out_path}")
    bpy.ops.render.render(write_still=True)

_stop.set()
_monitor.join(timeout=3)
_log("render done")

# Print GPU utilization summary
if _gpu_samples:
    # Group by GPU index
    by_gpu = {}
    for s in _gpu_samples:
        by_gpu.setdefault(s["idx"], []).append(s)
    for idx, samples in by_gpu.items():
        name     = samples[0]["name"]
        gpu_vals = [int(s["gpu%"]) for s in samples]
        mem_vals = [int(s["mem_used"]) for s in samples]
        mem_tot  = samples[0]["mem_total"]
        _log(f"GPU[{idx}] {name} | util: avg={sum(gpu_vals)//len(gpu_vals)}% max={max(gpu_vals)}% | "
             f"mem: avg={sum(mem_vals)//len(mem_vals)}MB max={max(mem_vals)}MB / {mem_tot}MB "
             f"({len(samples)} samples)")
else:
    _log("GPU monitor: no data (nvidia-smi not available or no GPU)")
for p in output_paths:
    print(f"Saved: {p}")

# ── Save .blend (optional) ─────────────────────────────────────────────────────
if OUTPUT_BLEND:
    _log("saving .blend")
    # Pack all external textures into the .blend so it is self-contained
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
    _log("blend saved")
    print(f"Blend saved: {OUTPUT_BLEND}")
_log("DONE")
