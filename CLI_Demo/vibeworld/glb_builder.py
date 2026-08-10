"""
glb_builder.py — 把当前场景 actors 拼装成一个 GLB，供 <model-viewer> 交互式展示。

资产为预导出 GLB：
    VibeWorlding/render_in_blender/assets/models/clone/<typeId>/<typeId>.glb
typeId 为 5 位字符串（如 "00443"），与资产检索服务返回的 type_id 对齐。

坐标变换完全复刻 render_in_blender/import_scene.py：
    full_mat = T(pos*0.01) @ R(quat) @ S(sca) @ yup_to_zup
    pos: 游戏 cm -> 米 (* 0.01)，UE->Blender 翻转 Y
    rot: 游戏 xyzw -> Quaternion(w, -x, y, -z)
    yup_to_zup: 绕 X 轴 +90°

合并所有 actor 网格到一个 trimesh.Scene 后导出 GLB。缺资产/全部失败时返回 None，
HTML 端据此降级为只显示 5 视角拍照图。
"""

import logging
import math
import os
from typing import List, Optional

logger = logging.getLogger("glb_builder")

# ── 资产目录 ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "../../render_in_blender/assets/models/clone")
)

SCALE = 0.01

# 延迟导入 trimesh / numpy（缺失时整体降级，不影响 5 图渲染）
try:
    import numpy as np
    import trimesh
    _HAS_TRIMESH = True
except Exception as _e:  # pragma: no cover
    _HAS_TRIMESH = False
    logger.warning(f"trimesh/numpy 不可用，交互式 3D 降级关闭: {_e}")


# ── GLB 几何缓存：同一 typeId 只加载一次 ─────────────────────────────────────────
_geom_cache: dict = {}


def _glb_path(type_id: str) -> str:
    type_id = str(type_id).strip()
    return os.path.join(MODELS_DIR, type_id, f"{type_id}.glb")


def _load_base_geometry(type_id: str):
    """加载单个资产 GLB，返回合并后的单个 trimesh.Trimesh（已转 Z-up 之前的原始坐标）。

    缓存命中直接返回；缺文件 / 加载失败返回 None。
    """
    if type_id in _geom_cache:
        return _geom_cache[type_id]

    path = _glb_path(type_id)
    if not os.path.isfile(path):
        logger.warning(f"GLB 缺失 typeId={type_id}: {path}")
        _geom_cache[type_id] = None
        return None

    try:
        loaded = trimesh.load(path, process=False, force="scene")
    except Exception as e:
        logger.warning(f"GLB 加载失败 typeId={type_id}: {e}")
        _geom_cache[type_id] = None
        return None

    # 把 scene 内部各 geometry 按其自身变换 dump 成网格列表再合并，
    # 保留 GLB 内部节点的相对摆放与材质。
    try:
        if isinstance(loaded, trimesh.Scene):
            dumped = loaded.dump()  # 应用各节点 transform，返回 Trimesh 列表
            meshes = [g for g in dumped if isinstance(g, trimesh.Trimesh) and len(g.faces)]
            if not meshes:
                _geom_cache[type_id] = None
                return None
            geom = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        elif isinstance(loaded, trimesh.Trimesh):
            geom = loaded
        else:
            _geom_cache[type_id] = None
            return None
    except Exception as e:
        logger.warning(f"GLB 合并失败 typeId={type_id}: {e}")
        _geom_cache[type_id] = None
        return None

    _geom_cache[type_id] = geom
    return geom


def _yup_to_zup_matrix():
    """绕 X 轴 +90° 的 4x4 矩阵（与 mathutils.Matrix.Rotation(radians(90),'X') 等价）。"""
    c, s = math.cos(math.radians(90)), math.sin(math.radians(90))
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def _zup_to_yup_matrix():
    """绕 X 轴 -90°：把 Z-up 场景转成 glTF 的 Y-up 约定。

    我们的场景（复刻 Blender import_scene）是 Z-up；而 glTF/model-viewer 默认 Y-up，
    且 auto-rotate 绕模型的竖直轴（glTF +Y）旋转。若不转换，场景会「躺倒」且绕错轴转。
    施加此变换后：Z→Y（竖直向上），模型直立、auto-rotate 绕真正的竖直轴旋转。
    """
    c, s = math.cos(math.radians(-90)), math.sin(math.radians(-90))
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def _quat_to_matrix(w, x, y, z):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵。"""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _actor_matrix(pos, rot, sca):
    """复刻 import_scene.py：T(loc) @ R(quat) @ S(sca) @ yup_to_zup。

    loc  = (px*0.01, -py*0.01, pz*0.01)
    quat = (rw, -rx, ry, -rz)
    """
    px, py, pz = (pos + [0, 0, 0])[:3]
    rx, ry, rz, rw = (rot + [0, 0, 0, 1])[:4]
    sx, sy, sz = (sca + [1, 1, 1])[:3]

    loc = np.array([px * SCALE, -py * SCALE, pz * SCALE], dtype=float)

    mat_T = np.eye(4)
    mat_T[:3, 3] = loc

    mat_R = np.eye(4)
    mat_R[:3, :3] = _quat_to_matrix(rw, -rx, ry, -rz)

    mat_S = np.diag([float(sx), float(sy), float(sz), 1.0])

    return mat_T @ mat_R @ mat_S @ _yup_to_zup_matrix()


def build_scene_glb(actors: List[dict], output_path: str) -> Optional[str]:
    """把 actors 拼成一个 GLB，写到 output_path。

    Args:
        actors: [{typeId, name, pos(cm), rot(xyzw), sca}, ...]
        output_path: 目标 .glb 路径

    Returns:
        成功返回 output_path；trimesh 不可用 / 无可用资产 / 导出失败返回 None。
    """
    if not _HAS_TRIMESH:
        return None
    if not actors:
        return None

    scene = trimesh.Scene()
    loaded = 0
    missing = set()

    # 整场景 Z-up → Y-up，供 model-viewer 正确直立显示并绕竖直轴自转
    zup_to_yup = _zup_to_yup_matrix()

    for i, actor in enumerate(actors):
        type_id = str(actor.get("typeId", "")).strip()
        if not type_id:
            continue
        base = _load_base_geometry(type_id)
        if base is None:
            missing.add(type_id)
            continue

        mat = zup_to_yup @ _actor_matrix(
            actor.get("pos", [0, 0, 0]),
            actor.get("rot", [0, 0, 0, 1]),
            actor.get("sca", [1, 1, 1]),
        )
        # copy 几何，避免共享网格被多次变换污染缓存
        mesh = base.copy()
        node_name = f"{actor.get('name', type_id)}_{i}"
        scene.add_geometry(mesh, node_name=node_name, transform=mat)
        loaded += 1

    if loaded == 0:
        logger.warning(f"build_scene_glb: 无可用资产（missing={len(missing)}），跳过 GLB")
        return None

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        scene.export(output_path)
    except Exception as e:
        logger.warning(f"GLB 导出失败: {e}")
        return None

    logger.info(f"GLB 已生成: {output_path} (actors={loaded}, missing={len(missing)})")
    return output_path
