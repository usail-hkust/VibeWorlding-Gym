"""Compose a whole seed 3D world into one GLB.

Only 251 of the 323 seed worlds ship a pre-exported seed_N.glb, and the visually
richest ones (ids >= 300) are exactly the ones missing. But every world has a
pcg_scene.json listing its actors (typeId + pos/rot/sca), and every referenced
asset has its own GLB, so the world can be rebuilt by instancing.

glTF lets many nodes share one mesh, so each distinct typeId is merged in once
and then referenced per placement -- a 2,000-actor world costs only as much
geometry as its unique assets.

Conventions (matched against the official exports):
  * pcg pos is in centimetres, glTF is metres  -> scale 0.01
  * pcg/Blender is Z-up, glTF is Y-up          -> root node rotated -90 deg about X
  * quaternions are [x, y, z, w] in both       -> passed through unchanged
"""
import struct, json, io, os, math

def _pad(b, n=4, fill=b"\x00"):
    r = len(b) % n
    return b + fill * (n - r) if r else b


def read_glb(path):
    d = open(path, "rb").read()
    magic, ver, length = struct.unpack_from("<III", d, 0)
    assert magic == 0x46546C67, "not a glb: %s" % path
    off, J, B = 12, None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", d, off)
        body = d[off + 8:off + 8 + clen]
        t = ctype.to_bytes(4, "little")
        if t == b"JSON":
            J = json.loads(body.decode("utf-8"))
        elif t[:3] == b"BIN":
            B = body
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    return J, B


def write_glb(path, J, B):
    j = _pad(json.dumps(J, separators=(",", ":")).encode("utf-8"), 4, b" ")
    b = _pad(B, 4)
    total = 12 + 8 + len(j) + 8 + len(b)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A)); f.write(j)
        f.write(struct.pack("<II", len(b), 0x004E4942)); f.write(b)
    return total


class Doc:
    """Accumulates merged glTF sub-documents into one buffer."""

    def __init__(self):
        self.J = {"asset": {"version": "2.0", "generator": "VibeWorlding web composer"},
                  "scene": 0, "scenes": [{"nodes": []}], "nodes": [],
                  "meshes": [], "materials": [], "textures": [], "images": [],
                  "samplers": [], "accessors": [], "bufferViews": []}
        self.bin = bytearray()
        self.ext = set()

    def _view(self, data, src_view):
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        for k in ("byteStride", "target"):
            if k in src_view:
                v[k] = src_view[k]
        self.bin += data
        self.J["bufferViews"].append(v)
        return len(self.J["bufferViews"]) - 1

    def merge(self, J, B, tex_bytes=None):
        """Merge one asset document; returns its mesh-index remap.

        Only data reachable from the meshes/images is copied. Decimation leaves
        the superseded high-poly buffers behind, and copying those would make the
        output *bigger* than the input."""
        vmap, amap, immap, smap, tmap, mmap, meshmap = {}, {}, {}, {}, {}, {}, {}

        used_acc = set()
        for m in J.get("meshes", []):
            for pr in m.get("primitives", []):
                used_acc.update(pr.get("attributes", {}).values())
                if "indices" in pr:
                    used_acc.add(pr["indices"])
        used_view = set()
        for i in used_acc:
            a = J["accessors"][i]
            if "bufferView" in a:
                used_view.add(a["bufferView"])
        for im in J.get("images", []):
            if "bufferView" in im:
                used_view.add(im["bufferView"])

        for i, v in enumerate(J.get("bufferViews", [])):
            if i not in used_view:
                continue
            off, ln = v.get("byteOffset", 0), v["byteLength"]
            data = B[off:off + ln]
            if tex_bytes and i in tex_bytes:
                data = tex_bytes[i]
            vmap[i] = self._view(data, v)

        for i in sorted(used_acc):
            a = dict(J["accessors"][i])
            if "bufferView" in a:
                a["bufferView"] = vmap[a["bufferView"]]
            if "sparse" in a:            # rare; drop rather than mis-remap
                a.pop("sparse")
            self.J["accessors"].append(a)
            amap[i] = len(self.J["accessors"]) - 1

        for i, im in enumerate(J.get("images", [])):
            im = dict(im)
            if "bufferView" in im:
                im["bufferView"] = vmap[im["bufferView"]]
            self.J["images"].append(im)
            immap[i] = len(self.J["images"]) - 1

        for i, sm in enumerate(J.get("samplers", [])):
            self.J["samplers"].append(dict(sm))
            smap[i] = len(self.J["samplers"]) - 1

        for i, t in enumerate(J.get("textures", [])):
            t = dict(t)
            if "source" in t:
                t["source"] = immap[t["source"]]
            if "sampler" in t:
                t["sampler"] = smap[t["sampler"]]
            self.J["textures"].append(t)
            tmap[i] = len(self.J["textures"]) - 1

        def fix_tex(d):
            if isinstance(d, dict):
                out = {}
                for k, v in d.items():
                    if k == "index" and isinstance(v, int):
                        out[k] = tmap.get(v, v)
                    else:
                        out[k] = fix_tex(v)
                return out
            if isinstance(d, list):
                return [fix_tex(x) for x in d]
            return d

        for i, m in enumerate(J.get("materials", [])):
            self.J["materials"].append(fix_tex(dict(m)))
            mmap[i] = len(self.J["materials"]) - 1

        for i, mesh in enumerate(J.get("meshes", [])):
            mesh = dict(mesh)
            prims = []
            for pr in mesh.get("primitives", []):
                pr = dict(pr)
                pr["attributes"] = {k: amap[v] for k, v in pr.get("attributes", {}).items()}
                if "indices" in pr:
                    pr["indices"] = amap[pr["indices"]]
                if "material" in pr:
                    pr["material"] = mmap[pr["material"]]
                pr.pop("targets", None)
                prims.append(pr)
            mesh["primitives"] = prims
            self.J["meshes"].append(mesh)
            meshmap[i] = len(self.J["meshes"]) - 1

        for e in J.get("extensionsUsed", []):
            self.ext.add(e)
        return meshmap

    def finish(self, path):
        self.J["buffers"] = [{"byteLength": len(self.bin)}]
        for k in ("materials", "textures", "images", "samplers"):
            if not self.J[k]:
                del self.J[k]
        if self.ext:
            self.J["extensionsUsed"] = sorted(self.ext)
        return write_glb(path, self.J, bytes(self.bin))


def _asset_root_meshes(J):
    """Mesh indices reachable from the asset's scene, with their local transforms."""
    out = []
    nodes = J.get("nodes", [])
    scene = J.get("scenes", [{}])[J.get("scene", 0)]

    def walk(ni, parent):
        n = nodes[ni]
        t = n.get("translation", [0, 0, 0])
        r = n.get("rotation", [0, 0, 0, 1])
        s = n.get("scale", [1, 1, 1])
        local = (t, r, s)
        if "mesh" in n:
            out.append((n["mesh"], local, parent))
        for c in n.get("children", []):
            walk(c, local)

    for ni in scene.get("nodes", []):
        walk(ni, None)
    return out


_COMP = {5126: "f", 5123: "H", 5125: "I", 5121: "B", 5122: "h", 5120: "b"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_accessor(J, B, ai):
    a = J["accessors"][ai]
    v = J["bufferViews"][a["bufferView"]]
    import numpy as np
    fmt = _COMP[a["componentType"]]
    n = _NCOMP[a["type"]]
    itemsize = np.dtype(fmt).itemsize * n
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride") or itemsize
    raw = B[off:off + stride * a["count"]]
    arr = np.frombuffer(raw, dtype=np.uint8)
    if stride == itemsize:
        return np.frombuffer(raw[:itemsize * a["count"]], dtype=np.dtype(fmt)).reshape(a["count"], n)
    out = np.empty((a["count"], n), dtype=np.dtype(fmt))
    for i in range(a["count"]):
        out[i] = np.frombuffer(raw[i * stride:i * stride + itemsize], dtype=np.dtype(fmt))
    return out


def _decimate_doc(J, B, max_tris):
    """Quadric-decimate any primitive above max_tris, in place, returning new (J, B).

    A handful of foliage assets carry ~100k-900k triangles, which dominates a
    composed world far more than textures do. Everything else is left alone."""
    import numpy as np
    import fast_simplification

    heavy = []
    for mi, m in enumerate(J.get("meshes", [])):
        for pi, pr in enumerate(m.get("primitives", [])):
            if "indices" not in pr or "POSITION" not in pr.get("attributes", {}):
                continue
            if J["accessors"][pr["indices"]]["count"] // 3 > max_tris:
                heavy.append((mi, pi))
    if not heavy:
        return J, B

    B = bytearray(B)
    for mi, pi in heavy:
        pr = J["meshes"][mi]["primitives"][pi]
        try:
            pos = _read_accessor(J, B, pr["attributes"]["POSITION"]).astype(np.float32)
            idx = _read_accessor(J, B, pr["indices"]).reshape(-1).astype(np.uint32)
            tris = idx.reshape(-1, 3)
            ratio = min(0.9, 1.0 - (max_tris / len(tris)))
            nv, nf = fast_simplification.simplify(pos, tris, ratio)
            nv = np.asarray(nv, dtype=np.float32)
            nf = np.asarray(nf, dtype=np.uint32)
        except Exception:
            continue

        # append new POSITION + indices; drop the other attributes (UV/normal no
        # longer index-aligned after decimation) and fall back to flat material
        def add(data, comp_type, typ, count, extra=None):
            while len(B) % 4:
                B.append(0)
            v = {"buffer": 0, "byteOffset": len(B), "byteLength": len(data)}
            B.extend(data)
            J["bufferViews"].append(v)
            acc = {"bufferView": len(J["bufferViews"]) - 1, "componentType": comp_type,
                   "count": count, "type": typ}
            if extra:
                acc.update(extra)
            J["accessors"].append(acc)
            return len(J["accessors"]) - 1

        pmin = nv.min(axis=0).tolist()
        pmax = nv.max(axis=0).tolist()
        ai_pos = add(nv.tobytes(), 5126, "VEC3", len(nv), {"min": pmin, "max": pmax})
        ai_idx = add(nf.reshape(-1).astype(np.uint32).tobytes(), 5125, "SCALAR", nf.size)
        pr["attributes"] = {"POSITION": ai_pos}
        pr["indices"] = ai_idx
    return J, bytes(B)


def compose(scene_json, clone_dir, out_path, tex_max=512, tex_q=80, corrections=None,
            max_tris=None):
    from PIL import Image

    j = json.load(open(scene_json, encoding="utf-8"))
    actors = []
    if isinstance(j, list):
        for b in j:
            if isinstance(b, dict):
                actors += b.get("actors", [])
    elif isinstance(j, dict):
        actors = j.get("actors", [])

    tids = []
    for a in actors:
        t = a.get("typeId")
        if t and t not in tids and os.path.exists(os.path.join(clone_dir, t, t + ".glb")):
            tids.append(t)

    doc = Doc()
    mesh_of = {}
    for t in tids:
        J, B = read_glb(os.path.join(clone_dir, t, t + ".glb"))
        if max_tris:
            J, B = _decimate_doc(J, B, max_tris)
        # shrink textures during the merge -- source atlases are 4096^2 PNGs
        tex = {}
        for im in J.get("images", []):
            if "bufferView" not in im:
                continue
            bv = J["bufferViews"][im["bufferView"]]
            raw = B[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
            try:
                p = Image.open(io.BytesIO(raw))
                alpha = p.mode in ("RGBA", "LA") or (p.mode == "P" and "transparency" in p.info)
                w, h = p.size
                if max(w, h) > tex_max:
                    sc = tex_max / max(w, h)
                    p = p.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS)
                buf = io.BytesIO()
                if alpha:
                    p.convert("RGBA").save(buf, "PNG", optimize=True)
                    im["mimeType"] = "image/png"
                else:
                    p.convert("RGB").save(buf, "JPEG", quality=tex_q, optimize=True)
                    im["mimeType"] = "image/jpeg"
                tex[im["bufferView"]] = buf.getvalue()
            except Exception:
                pass
        mm = doc.merge(J, B, tex)
        mesh_of[t] = [(mm[mi], loc) for mi, loc, _p in _asset_root_meshes(J)]

    # Z-up (pcg/Blender) -> Y-up (glTF), and centimetres -> metres.
    S = 0.01
    root_children = []
    placed = 0
    for a in actors:
        t = a.get("typeId")
        if t not in mesh_of:
            continue
        pos = a.get("pos") or [0, 0, 0]
        rot = a.get("rot") or [0, 0, 0, 1]
        sca = a.get("sca") or [1, 1, 1]
        corr = (corrections or {}).get(t) or {}
        cx = corr.get("correction_x", 1.0) or 1.0
        cy = corr.get("correction_y", 1.0) or 1.0
        cz = corr.get("correction_z", 1.0) or 1.0
        # Mirror render_scene.py exactly:
        #   loc   = (x, -y, z) * 0.01          (UE -> Blender: flip Y)
        #   rot_q = (w, -x, y, -z)             (quaternion under Y-mirror)
        #   scale = sca * correction
        # `normalized` assets need no extra basis change; raw ones are Y-up and
        # additionally get yup_to_zup plus a pivot_z lift.
        normalized = bool(corr.get("normalized"))
        pivot_z = corr.get("pivot_z_local", 0.0) or 0.0

        kids = []
        for mi, (lt, lr, ls) in mesh_of[t]:
            child = {"mesh": mi}
            if list(lt) != [0, 0, 0]:
                child["translation"] = list(lt)
            if list(lr) != [0, 0, 0, 1]:
                child["rotation"] = list(lr)
            if list(ls) != [1, 1, 1]:
                child["scale"] = list(ls)
            doc.J["nodes"].append(child)
            kids.append(len(doc.J["nodes"]) - 1)

        # Blender's glTF importer always rotates +90 deg about X (Y-up -> Z-up),
        # so the mesh render_scene.py transforms is already Z-up. Reproduce that
        # here. Non-normalized assets then get render_scene's extra yup_to_zup.
        h = math.sin(math.pi / 4)
        RX90 = [h, 0.0, 0.0, math.cos(math.pi / 4)]
        for _ in range(1 if normalized else 2):
            doc.J["nodes"].append({"rotation": RX90, "children": kids})
            kids = [len(doc.J["nodes"]) - 1]

        z = pos[2] * S
        if not normalized:
            z += abs(pivot_z) * sca[2] * cz

        doc.J["nodes"].append({
            "translation": [pos[0] * S, -pos[1] * S, z],
            "rotation": [-rot[0], rot[1], -rot[2], rot[3]],
            "scale": [sca[0] * cx, sca[1] * cy, sca[2] * cz],
            "children": kids,
        })
        root_children.append(len(doc.J["nodes"]) - 1)
        placed += 1

    r = math.sin(-math.pi / 4)
    doc.J["nodes"].append({"name": "ZupToYup", "rotation": [r, 0, 0, math.cos(-math.pi / 4)],
                           "children": root_children})
    doc.J["scenes"][0]["nodes"] = [len(doc.J["nodes"]) - 1]
    n = doc.finish(out_path)
    return n, placed, len(tids)


if __name__ == "__main__":
    import sys
    SEED = "/mnt/ceph-AT/private/yansongning/AgenticRL4UGC/ai4ugc_map_gen/VibeWorlding/raw_data/seed_3dworld"
    CLONE = "/mnt/ceph-AT/private/yansongning/AgenticRL4UGC/ai4ugc_map_gen/VibeWorlding/render_in_blender/assets/models/clone"
    CORR = json.load(open("/mnt/ceph-AT/private/yansongning/AgenticRL4UGC/ai4ugc_map_gen/"
                          "VibeWorlding-Gym/render_in_blender/assets/glb_corrections.json",
                          encoding="utf-8"))
    sid, out = sys.argv[1], sys.argv[2]
    tm = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    mt = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    n, placed, uniq = compose(os.path.join(SEED, sid, "pcg_scene.json"), CLONE, out,
                              tex_max=tm, corrections=CORR, max_tris=(mt or None))
    print("seed %s: %d actors, %d unique assets -> %.2f MB" % (sid, placed, uniq, n / 1e6))
