"""本地 smoke test：直接构造 RetrieverService 跑几条 query，不起 HTTP。

涵盖与参考部署 ``recommendation_qy/asset_recommendation_v1`` 一致的 4 个推荐接口
的内部调用形态（同 entity / 同 query 走相同翻译路径）。

用法：
    cd <repo-root>/assets_retrieval
    python -m D1_deploy.scripts.smoke_test
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _bootstrap_sys_path() -> None:
    this_file = Path(__file__).resolve()
    pkg_parent = this_file.parent.parent.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))


_bootstrap_sys_path()

from D1_deploy.asset_recommend_d1.recommend_service import RetrieverService  # noqa: E402


def _print(label: str, res, top_k: int = 5) -> None:
    print(f"\n=== {label} ===")
    print(f"  query_text = {res.query_text!r}")
    print(f"  view = {res.view}")
    print(f"  total_candidates = {res.total_candidates}")
    print(f"  message = {res.message}")
    for i, item in enumerate(res.items[:top_k]):
        print(f"  [{i+1}] {item.asset_id}  score={item.score:.4f}  name={item.name!r}")


def main() -> int:
    print("[smoke] 初始化 RetrieverService（首次会触发预编码） ...", flush=True)
    t0 = time.time()
    svc = RetrieverService.get()
    print(f"[smoke] 初始化耗时 {time.time() - t0:.1f}s（命中缓存应 < 30s）", flush=True)
    print(f"[smoke] N_assets={svc.index.n} dim={svc.index.dim} device={svc.encoder.device}",
          flush=True)

    # ── 4 个推荐接口的典型调用 ────────────────────────────────────────
    cases = [
        # /recommend/single_slot —— V1.2 路径（仅 entity_name）
        {
            "label": "single_slot [V1.2 entity-only]: '书架'",
            "kwargs": {
                "entity_name": "书架", "top_k": 5,
                "return_fields": ["type", "category", "image_uri"],
            },
        },
        # /recommend/single_slot —— V1.1 路径（带 query）
        {
            "label": "single_slot [V1.1 query+entity]: '木栅栏' + 中世纪村庄",
            "kwargs": {
                "entity_name": "木栅栏",
                "query": "中世纪村庄清晨，雨后湿润",
                "top_k": 5,
                "return_fields": ["type", "scene_limit", "style"],
            },
        },
        # /recommend/single_slot —— 类型过滤
        {
            "label": "single_slot [filter type=花草]: '高大挺拔的仙人掌'",
            "kwargs": {
                "entity_name": "高大挺拔的仙人掌",
                "filters": {"type": "花草"},
                "top_k": 5,
                "return_fields": ["type", "image_uri"],
            },
        },
    ]

    for c in cases:
        t = time.time()
        res = svc.recommend_single_slot(**c["kwargs"])
        latency = (time.time() - t) * 1000
        _print(c["label"], res)
        print(f"  latency = {latency:.1f} ms", flush=True)

    # ── /recommend/combination & /recommend/entity 内部都走 recommend_batch ──
    print("\n=== combination/entity [shared query]: 日式庭院 + 多 entity ===")
    t = time.time()
    out = svc.recommend_batch(
        entities=[
            {"entity_name": "樱花树"},
            {"entity_name": "石灯笼", "filters": {"type": "建筑"}},
        ],
        query="日式庭院的樱花林，傍晚柔和光线",
        top_k=3,
    )
    print(f"  batch latency = {(time.time() - t) * 1000:.1f} ms")
    for ent_name, res in out.items():
        print(f"  - entity: {ent_name}  view={res.view}")
        for it in res.items:
            print(f"      {it.asset_id}  score={it.score:.4f}  name={it.name!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
