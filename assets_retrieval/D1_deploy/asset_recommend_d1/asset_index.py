"""资产嵌入缓存：把 6560 条资产卡一次性 encode 到 fp32 numpy，常驻内存。

设计要点：
  - 缓存文件 schema：``np.savez_compressed(asset_ids=..., embeddings=...)``
    asset_ids: object array（asset_id 字符串）；embeddings: (N, D) float32
  - 缓存文件名带 ckpt 路径 hash 作 tag，多 ckpt 共存
  - 启动时优先加载缓存；缺失或 force_reencode=True 时重编码并写盘
  - 推理期所有 query → top-K 都走 numpy / torch 矩阵乘法（cosine = 内积，
    因为 emb 已 L2-normalize）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..common.logger import get_logger
from .config import D1RecommendConfig
from .text_utils import load_asset_cards, serialize_asset_card

log = get_logger(__name__)


@dataclass
class AssetEmbeddingIndex:
    """常驻内存的资产嵌入索引（query 侧 top-K 检索用）。

    字段：
      - asset_ids:   List[str]，长度 N，与 embeddings 行对齐
      - cards:       Dict[asset_id -> raw card dict]，过滤 / fields 透出用
      - embeddings:  np.ndarray[N, D] float32，**已 L2-normalize**
      - device_emb:  torch.Tensor[N, D] (与编码器同 device，可选；启动时 lazy 上设备)
    """
    asset_ids: List[str]
    cards: Dict[str, dict]
    embeddings: "Any"                                # np.ndarray[N, D]
    device_emb: Optional["Any"] = None               # torch.Tensor[N, D] on device

    def to_device(self, device: str) -> None:
        """把 embeddings 搬到 GPU/NPU 备好（可选；不搬时检索走 CPU torch）。"""
        import torch
        if self.device_emb is not None:
            return
        t = torch.from_numpy(self.embeddings)         # CPU fp32
        # 在 NPU/CUDA 上跑矩阵乘法显著快；CPU 6560×2560 fp32 也才 ~10ms 量级，可接受。
        try:
            self.device_emb = t.to(device)
            log.info("asset embeddings → %s, shape=%s", device, tuple(self.device_emb.shape))
        except Exception as e:
            log.warning("无法把 embeddings 移到 %s: %r；保留在 CPU", device, e)
            self.device_emb = t

    @property
    def n(self) -> int:
        return len(self.asset_ids)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.size else 0


# ─── 加载 / 编码 / 保存 ─────────────────────────────────────────────────
def _try_load_cache(cache_path: Path) -> Optional[Tuple[List[str], "Any"]]:
    if not cache_path.exists():
        return None
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("需要 numpy: pip install numpy") from e
    try:
        z = np.load(cache_path, allow_pickle=True)
        ids = list(z["asset_ids"].tolist())
        embs = z["embeddings"].astype("float32", copy=False)
        if embs.ndim != 2 or embs.shape[0] != len(ids):
            log.warning("缓存形态异常：ids=%d, emb=%s；忽略缓存", len(ids), embs.shape)
            return None
        log.info("命中嵌入缓存 %s: N=%d, D=%d", cache_path, len(ids), embs.shape[1])
        return ids, embs
    except Exception as e:
        log.warning("加载缓存 %s 失败: %r；将重新编码", cache_path, e)
        return None


def _save_cache(cache_path: Path, asset_ids: Sequence[str], embeddings: "Any") -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(
        cache_path,
        asset_ids=np.array(list(asset_ids), dtype=object),
        embeddings=embeddings.astype("float32", copy=False),
    )
    size_mb = cache_path.stat().st_size / 1e6
    log.info("已保存嵌入缓存 %s (%.1f MB)", cache_path, size_mb)


def build_or_load_asset_index(
    cfg: D1RecommendConfig,
    encoder,                                          # InferenceEncoder
) -> AssetEmbeddingIndex:
    """启动时调用一次：加载资产卡 → 命中缓存 / 否则用 encoder 编码 → 写盘。

    返回的 ``AssetEmbeddingIndex`` 直接被 ``RetrieverService`` 持有。
    """
    import numpy as np
    cards_path = Path(cfg.asset_cards_jsonl)
    cards_dict = load_asset_cards(cards_path)
    asset_ids_full = list(cards_dict.keys())          # 保留 jsonl 顺序，便于 debug
    if not asset_ids_full:
        raise RuntimeError(f"asset_cards 为空，路径={cards_path}")

    cache_path = cfg.emb_cache_path()
    cached: Optional[Tuple[List[str], "Any"]] = None
    if not cfg.force_reencode:
        cached = _try_load_cache(cache_path)

    if cached is not None:
        cached_ids, cached_emb = cached
        # 校验：缓存里的 ids 必须是 cards 的子集（顺序也必须一致）。
        # 资产库扩量时缓存 ids 只能 ≤ cards；这种情况下也强制重编码（小库下重编码很快）。
        if cached_ids == asset_ids_full:
            return AssetEmbeddingIndex(
                asset_ids=cached_ids, cards=cards_dict, embeddings=cached_emb,
            )
        log.warning(
            "缓存的 asset_ids 与 asset_cards 不一致（cache N=%d, cards N=%d）；"
            "强制重编码以保证 embeddings ↔ cards 对齐",
            len(cached_ids), len(asset_ids_full),
        )

    # ── 重编码 ────────────────────────────────────────────────────────
    log.info("开始全量 asset 编码：N=%d (这一步在 4B 模型 + GPU 上 ~1-3 分钟)", len(asset_ids_full))
    doc_texts = [serialize_asset_card(cards_dict[aid]) for aid in asset_ids_full]
    emb_t = encoder.encode_corpus(
        doc_texts, is_query=False, batch_size=getattr(cfg, "encode_batch_size", 32),
    )
    # to numpy fp32（内存中常驻 fp32，~6560×2560×4B ≈ 64 MB；可接受）
    embeddings = emb_t.detach().to("cpu").numpy().astype("float32", copy=False)
    _save_cache(cache_path, asset_ids_full, embeddings)
    return AssetEmbeddingIndex(
        asset_ids=asset_ids_full, cards=cards_dict, embeddings=embeddings,
    )
