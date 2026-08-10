"""单实体推荐服务（D1 部署主路径）。

调用图：
    HTTP request
       ↓
    main.py FastAPI 路由
       ↓
    RetrieverService.recommend_single_slot(entity_name, query?,
                                          filters?, top_k, asset_ids?, excluded_ids?)
       ↓
    1) build query text + instruction (与训练侧严格一致)
        - query 非空 → 类型 1 (V1.1 baseline.query_entity)
        - query 为空 → 类型 2 (V1.2 baseline.entity_only)
    2) encoder.encode([wrapped_query], is_query=True)  → q_emb [1, D]
    3) sims = q_emb @ asset_emb.T                       → [1, N]
    4) 应用 filters / asset_ids / excluded_ids 在 N 维 mask
       —— filters 直接命中 CSV 列（见 csv_attrs.py），不再做 phase1 翻译
    5) torch.topk(top_k)
    6) 由 csv_attrs 透出 item.name / item.attributes（fields 白名单按 CSV 列名）

设计选择：
  - 进程级单例：encoder + asset_index + csv_attrs 都不重复加载
  - 单条 query 编码 ~30-100ms（GPU），矩阵乘法 ~1ms
  - filters / fields 透出走纯 Python（小库 6560，遍历 < 1ms）
  - **属性 / filter 的真相源 = module1b CSV**；phase1 jsonl 仅用于 doc 编码

返回结构：
    RecommendItem:    asset_id / name / score / attributes(可选)
    RecommendResult:  items / query_text / view / total_candidates
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..common.logger import get_logger
from .asset_index import AssetEmbeddingIndex, build_or_load_asset_index
from .config import D1RecommendConfig, get_config
from .csv_attrs import CsvAttrTable, load_csv_attrs
from .encoder import InferenceEncoder, InferenceEncoderConfig
from .text_utils import (
    build_query_text,
    infer_view,
    load_view_instructions,
    wrap_with_instruction,
)

log = get_logger(__name__)


# ─── 数据结构 ───────────────────────────────────────────────────────────
@dataclass
class RecommendItem:
    asset_id: str
    name: str
    score: float
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class RecommendResult:
    items: List[RecommendItem]
    query_text: str
    view: str
    total_candidates: int                 # 应用 filter / excluded 后剩下的池子大小
    success: bool = True
    message: str = ""


# ─── 检索服务（进程级单例） ────────────────────────────────────────────
class RetrieverService:
    """单例容器：encoder + asset_index + view_instructions。"""

    _instance: Optional["RetrieverService"] = None
    _lock = threading.Lock()

    def __init__(self, cfg: Optional[D1RecommendConfig] = None):
        self.cfg = cfg or get_config()

        log.info("初始化 InferenceEncoder ckpt=%s", self.cfg.ckpt_dir)
        self.encoder = InferenceEncoder(InferenceEncoderConfig(
            ckpt_dir=self.cfg.ckpt_dir,
            backbone_name=self.cfg.backbone_name,
            max_q_len=self.cfg.max_q_len,
            max_d_len=self.cfg.max_d_len,
            bf16=self.cfg.bf16,
            flash_attn=self.cfg.flash_attn,
        ))

        log.info("加载 / 编码资产卡库 (phase1 jsonl, doc 编码用) ...")
        self.index: AssetEmbeddingIndex = build_or_load_asset_index(self.cfg, self.encoder)
        # 把 asset embeddings 搬到 encoder 同 device，加速 sims 矩阵乘法
        self.index.to_device(self.encoder.device)

        log.info("加载资产属性 CSV (filters & fields 透出唯一数据源) from %s",
                 self.cfg.asset_csv)
        self.attrs: CsvAttrTable = load_csv_attrs(
            self.cfg.asset_csv, color_csv=getattr(self.cfg, "color_csv", None),
        )
        # 一致性提醒：CSV 的 asset_id 集合应覆盖 phase1 jsonl
        missing = [aid for aid in self.index.asset_ids if not self.attrs.has(aid)]
        if missing:
            log.warning(
                "[csv_attrs] %d/%d 个资产在 CSV 里找不到（asset_id 不一致）；"
                "示例: %s。这些资产仍可被检索到，但 attributes 会为 None / name 退化为 asset_id。",
                len(missing), self.index.n, missing[:5],
            )

        log.info("加载 view 指令模板 from %s", self.cfg.view_prefixes_yaml)
        self.instr_table = load_view_instructions(self.cfg.view_prefixes_yaml)

        log.info(
            "RetrieverService 就绪：N_assets=%d, dim=%d, csv_attrs=%d, device=%s",
            self.index.n, self.index.dim, self.attrs.n, self.encoder.device,
        )

    @classmethod
    def get(cls) -> "RetrieverService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 公共接口 ──────────────────────────────────────────────────────
    def recommend_single_slot(
        self,
        *,
        entity_name: str,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        asset_ids: Optional[List[str]] = None,
        excluded_ids: Optional[List[str]] = None,
        top_k: int = 10,
        return_fields: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
        asset_ids_as_whitelist: bool = False,
    ) -> RecommendResult:
        """单实体检索主入口 —— 严格对齐 embedding 模型训练时的 2 类输入。

        模型见过的 2 类 query 形态（在 view_prefixes.yaml 里各对应一条 instruction）：

          类型 1（V1.1 baseline.query_entity）—— query + entity name
              "Instruct: 为下面的资产检索 query 和实体名找到匹配的资产\\n
               [实体] {entity_name}\\n[原始 query] {query}"

          类型 2（V1.2 baseline.entity_only）—— entity name only
              "Instruct: 为下面的实体名找到匹配的资产\\n
               [实体] {entity_name}"

        Args:
            entity_name:     必填，实体名（如 "残旧的木栅栏"）
            query:           可选；非空字符串走类型 1，None / 空字符串走类型 2
            filters:         硬过滤 dict（schema 见 csv_attrs.match_filters）
            asset_ids:       与参考部署字段名兼容；**默认是 anchor 语义**，仅日志记录
                             不参与过滤（D1 无 rerank，anchor 无实际打分作用）。
                             如确需当作硬白名单，传 ``asset_ids_as_whitelist=True``。
            excluded_ids:    黑名单（一定生效）
            top_k:           返回条数
            return_fields:   按白名单透出的资产属性 keys
            score_threshold: 低于该 cosine 分数丢弃
            asset_ids_as_whitelist: True 时 asset_ids 当硬白名单（旧行为）

        Returns:
            RecommendResult
        """
        import torch

        ent = (entity_name or "").strip()
        if not ent:
            return RecommendResult(
                items=[], query_text="", view="", total_candidates=0,
                success=False, message="entity_name 必填且非空",
            )

        # ── 1) query 文本拼装（与训练侧 2 类形态严格一致） ─────────────
        # query 非空 → 类型 1（V1.1）；为空 → 类型 2（V1.2）
        view = infer_view(query)
        q_text = build_query_text(ent, query)
        wrapped = wrap_with_instruction(view, q_text, self.instr_table)

        # ── 2) encode query ────────────────────────────────────────────
        # 推理路径必须显式 no_grad：encoder.encode() 内部不包 no_grad
        # （与训练侧一致，仅 encode_corpus 才包），不加会构建梯度图导致显存暴涨。
        with torch.no_grad():
            q_emb = self.encoder.encode([wrapped], is_query=True)  # [1, D]

        # ── 3) sims：cosine = inner product（已 L2-normalize） ─────────
        d_emb = self.index.device_emb
        if d_emb is None:
            # CPU 兜底：把 numpy 转一个临时 tensor（应在 to_device 时已搬好，这里冗余保护）
            d_emb = torch.from_numpy(self.index.embeddings)
        # 把 q_emb 也搬到同设备
        if q_emb.device != d_emb.device:
            q_emb = q_emb.to(d_emb.device)
        sims = (q_emb @ d_emb.T).squeeze(0)  # [N]
        sims = sims.float()

        # ── 4) filter / asset_ids / excluded_ids → 池子掩码 ────────────
        # filters / fields 透出全部走 csv_attrs（CSV 是真相源），
        # phase1 jsonl 仅用于 doc 编码。
        N = self.index.n
        keep_mask: List[bool] = [True] * N
        # filter
        if filters:
            for i, aid in enumerate(self.index.asset_ids):
                if not self.attrs.match_filters(aid, filters):
                    keep_mask[i] = False
        # asset_ids：默认是 anchor 语义（与参考部署字段命名兼容）—— D1 无 rerank
        # 模块，anchor 不参与过滤；仅当 asset_ids_as_whitelist=True 时才当硬白名单。
        if asset_ids:
            if asset_ids_as_whitelist:
                wl = set(asset_ids)
                for i, aid in enumerate(self.index.asset_ids):
                    if aid not in wl:
                        keep_mask[i] = False
            else:
                log.debug("asset_ids 收到 %d 个 anchor（D1 无 rerank，仅记录不参与过滤）",
                          len(asset_ids))
        # excluded_ids 黑名单
        if excluded_ids:
            bl = set(excluded_ids)
            for i, aid in enumerate(self.index.asset_ids):
                if aid in bl:
                    keep_mask[i] = False

        n_pool = sum(1 for x in keep_mask if x)
        if n_pool == 0:
            return RecommendResult(
                items=[], query_text=q_text, view=view, total_candidates=0,
                success=True,
                message=(
                    f"过滤后候选池为空（filters={bool(filters)}, "
                    f"whitelist={bool(asset_ids)}, blacklist={bool(excluded_ids)}）"
                ),
            )

        # 屏蔽不在 pool 的位置：sims[i] = finfo.min（fp32 最小值，topk 后稳定排到末尾）
        neg_inf = torch.finfo(sims.dtype).min
        if n_pool < N:
            mask_t = torch.tensor(keep_mask, dtype=torch.bool, device=sims.device)
            sims = sims.masked_fill(~mask_t, neg_inf)

        # ── 5) top-K ─────────────────────────────────────────────────────
        k_eff = min(top_k, n_pool, self.cfg.max_top_k)
        topk = torch.topk(sims, k=k_eff, dim=0)
        scores = topk.values.detach().to("cpu").tolist()
        idxs = topk.indices.detach().to("cpu").tolist()

        thr = score_threshold if score_threshold is not None else self.cfg.similarity_threshold
        # 阈值：被 mask 掉的位置分数等于 finfo.min（fp32 ≈ -3.4e38，不是 -inf）。
        # 用「半数阈值」鲁棒判定：任何小于 finfo.min/2 的分数都视作被 mask。
        mask_threshold = neg_inf / 2.0

        items: List[RecommendItem] = []
        for sc, idx in zip(scores, idxs):
            if sc <= mask_threshold:               # 被 mask 掉的位置
                continue
            if thr and sc < thr:
                continue
            aid = self.index.asset_ids[idx]
            items.append(RecommendItem(
                asset_id=aid,
                name=self.attrs.get_name(aid),
                score=float(sc),
                attributes=self.attrs.select_fields(aid, return_fields),
            ))

        return RecommendResult(
            items=items, query_text=q_text, view=view,
            total_candidates=n_pool,
            success=True,
            message=f"返回 {len(items)} 条 / 候选池 {n_pool}",
        )

    def recommend_batch(
        self,
        entities: List[Dict[str, Any]],
        *,
        query: Optional[str] = None,
        global_filters: Optional[Dict[str, Any]] = None,
        asset_ids: Optional[List[str]] = None,
        excluded_ids: Optional[List[str]] = None,
        top_k: int = 10,
        return_fields: Optional[List[str]] = None,
    ) -> Dict[str, RecommendResult]:
        """批量：每个 entity 独立做 single_slot 检索（每实体可单独覆写 filters）。

        与 ``recommend_single_slot`` 一样严格对齐模型训练时的 2 类输入：
        ``query`` 非空 → 全部实体走类型 1（V1.1 query+entity）；
        ``query`` 为空 → 全部实体走类型 2（V1.2 entity-only）。

        ``entities[i]`` schema:
            {
              "entity_name": "...",   # 必填
              "filters": {...},        # 可选，与 global_filters 合并（实体级覆盖）
            }
        """
        out: Dict[str, RecommendResult] = {}
        for ent in entities:
            name = str(ent.get("entity_name", "")).strip()
            if not name:
                continue
            ent_filters = dict(global_filters or {})
            ent_filters.update(ent.get("filters") or {})  # entity-level 覆盖 global

            res = self.recommend_single_slot(
                entity_name=name,
                query=query,
                filters=ent_filters,
                asset_ids=asset_ids,
                excluded_ids=excluded_ids,
                top_k=top_k,
                return_fields=return_fields,
            )
            out[name] = res
        return out
