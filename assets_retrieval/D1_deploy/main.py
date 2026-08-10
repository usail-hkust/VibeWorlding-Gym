"""D1_deploy.main — D1 single-entity 推荐 HTTP 服务（FastAPI 入口）。

API 与参考部署 ``recommendation_qy/asset_recommendation_v1`` **完全兼容**：
4 个推荐路由（combination / single_slot / scene / entity）+ recommend_with_plan
+ search_strategy3 全部保留，请求 / 响应字段 1:1 对齐；调用方迁移过来零改动。

内部把所有路由翻译成 embedding 模型见过的 **2 类输入**（每个 entity 独立路由）：

  类型 1（V1.1 baseline.query_entity）—— query + entity_name
      Instruct: 为下面的资产检索 query 和实体名找到匹配的资产
      [实体] {entity_name}
      [原始 query] {query}

  类型 2（V1.2 baseline.entity_only）—— entity_name only
      Instruct: 为下面的实体名找到匹配的资产
      [实体] {entity_name}

翻译规则（每个 entity 独立路由）：
    q_concat = " ".join(filter(None, [scene_description, theme, entity_description])).strip()
    if q_concat:    走类型 1（query 文本 = q_concat）
    else:           走类型 2

启动:
    cd <repo-root>/assets_retrieval
    python -m D1_deploy.main
    # 或
    python -m uvicorn D1_deploy.main:app --host 0.0.0.0 --port 8084 --workers 1

环境变量:
    PORT, D1_CKPT_DIR, D1_ASSET_CARDS_JSONL, D1_FORCE_REENCODE, D1_LOG_LEVEL

接口（与参考部署一致，调用方零改动）:
    POST /recommend/combination   组合推荐（实体列表 → 每实体 top-K，再拼成组合）
    POST /recommend/single_slot   单实体检索
    POST /recommend/scene         场景描述（D1 无 LLM Planner，降级为单实体路径）
    POST /recommend/entity        批量实体（每实体独立 top-K）
    POST /recommend_with_plan     消费外部 plan_result（slots → 每 slot top-K）
    POST /search_strategy3        向后兼容（等同 /recommend/scene）
    GET  /health  /info  /

⚠️ 必须 ``--workers 1``：本服务用进程级单例（4B encoder + 6560 资产嵌入全在内存里）。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .asset_recommend_d1.config import get_config
from .asset_recommend_d1.recommend_service import (
    RecommendItem,
    RecommendResult,
    RetrieverService,
)
from .common.logger import get_logger

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#   return_mode 枚举（D1 无组合优化模块，combined / both 静默降级为 per_slot）
# ═════════════════════════════════════════════════════════════════════════════

RETURN_MODE_PER_SLOT = "per_slot"
RETURN_MODE_COMBINED = "combined"
RETURN_MODE_BOTH = "both"
_VALID_RETURN_MODES = {RETURN_MODE_PER_SLOT, RETURN_MODE_COMBINED, RETURN_MODE_BOTH}
_DEPRECATED_WARNED = False

# 是否允许把内部异常 message 透传给 HTTP 客户端
# 生产建议关闭（避免泄漏内部路径 / 栈信息）；调试设 1 打开
_EXPOSE_INTERNAL_ERROR = os.environ.get("D1_EXPOSE_INTERNAL_ERROR", "0").strip() not in ("", "0", "false", "False")


def _safe_error_detail(e: Exception, fallback: str = "internal server error") -> str:
    """把 exception 转成对外 detail；默认仅返回类名 + fallback，详情走 logger。"""
    if _EXPOSE_INTERNAL_ERROR:
        return f"{type(e).__name__}: {e}"
    return f"{fallback} ({type(e).__name__})"


def _normalize_return_mode(req_mode: str) -> str:
    global _DEPRECATED_WARNED
    if req_mode in (RETURN_MODE_COMBINED, RETURN_MODE_BOTH):
        if not _DEPRECATED_WARNED:
            logger.warning(
                "DEPRECATED return_mode=%r 已废弃（D1 无组合优化），降级为 per_slot",
                req_mode,
            )
            _DEPRECATED_WARNED = True
        return RETURN_MODE_PER_SLOT
    return req_mode


# ═════════════════════════════════════════════════════════════════════════════
#   Pydantic 请求模型 —— 字段 1:1 对齐参考部署
# ═════════════════════════════════════════════════════════════════════════════

class EntitySpec(BaseModel):
    """单实体描述（与参考部署 EntitySpec 字段完全一致）。

    向后兼容：``entity_description`` 在 D1 模型训练侧不是独立字段，路由层会
    把它和 scene_description / theme 拼成「原始 query」喂给类型 1。
    """
    entity_name: str = Field(..., description='实体名称')
    entity_description: Optional[str] = Field(
        None,
        description="实体文本描述（可选）。D1 内部会与 scene_description / theme 拼接 "
                    "进入「原始 query」段，作为类型 1（V1.1 query+entity）的 query 文本。",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description='实体级硬过滤；与请求顶层 filters 合并（实体级覆盖全局）。',
    )
    model_config = {"extra": "forbid"}


class CombinationRequest(BaseModel):
    """``POST /recommend/combination`` —— 字段与参考部署完全一致。"""
    entities: List[EntitySpec] = Field(..., description="实体列表（至少 1 个）")
    scene_description: Optional[str] = Field(None, description="全局场景/用户 query 文本（可选）")
    theme: Optional[str] = Field(None, description="全局主题（可选）")
    asset_ids: Optional[List[str]] = Field(default_factory=list, description="anchor 资产 IDs")
    excluded_ids: Optional[List[str]] = Field(default_factory=list, description="需排除的资产 IDs")
    return_mode: str = Field(
        RETURN_MODE_BOTH,
        description=f"返回策略: {RETURN_MODE_PER_SLOT} | {RETURN_MODE_COMBINED} | {RETURN_MODE_BOTH}。"
                    "D1 简化版只跑 per_slot，combined / both 静默降级。",
    )
    top_k: int = Field(10, ge=1, le=50, description="每实体保留的资产数（与参考部署一致：上限 50）")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="全局过滤；与 entity.filters 合并")
    fields: Optional[List[str]] = Field(None, description="按白名单透出资产属性到 item.attributes")


class SceneRequest(BaseModel):
    """``POST /recommend/scene`` —— 字段与参考部署完全一致。

    D1 部署无 LLM Planner：内部退化为「把 scene_description 当成单实体 query」
    跑一次类型 1 检索（取代参考部署里的 Gemini 规划阶段）。
    """
    scene_description: str = Field(..., description="全局场景/用户 query 文本（必填）")
    theme: Optional[str] = Field(None)
    asset_ids: Optional[List[str]] = Field(default_factory=list)
    excluded_ids: Optional[List[str]] = Field(default_factory=list)
    return_mode: str = Field(RETURN_MODE_BOTH)
    # 与参考部署保持一致：默认 1，上限 50
    top_k: int = Field(1, ge=1, le=50, description="每实体返回的资产数（与参考部署一致）")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    fields: Optional[List[str]] = Field(None)


class SingleSlotRequest(BaseModel):
    """``POST /recommend/single_slot`` —— 字段与参考部署完全一致。

    D1 扩展：``score_threshold``（可选；低于该 cosine 分数的丢弃；参考部署无该字段，
    但属于"传不传都不破坏向后兼容"的可选扩展）。
    """
    entity_name: str = Field(..., description='实体名称（必填非空）')
    entity_description: Optional[str] = Field(None)
    theme: Optional[str] = Field(None)
    scene_description: Optional[str] = Field(None)
    asset_ids: Optional[List[str]] = Field(default_factory=list)
    excluded_ids: Optional[List[str]] = Field(default_factory=list)
    # 与参考部署保持一致：上限 100
    top_k: int = Field(10, ge=1, le=100, description="返回资产上限（按融合分降序，与参考部署一致）")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    fields: Optional[List[str]] = Field(None)
    score_threshold: Optional[float] = Field(
        None, description="可选；低于该 cosine 分数的丢弃（D1 扩展字段，参考部署无）",
    )
    # 注意：保留 extra=forbid 与参考部署一致；score_threshold 是 D1 显式声明的扩展字段
    # 已声明在 schema 里，不会被 forbid 拒绝。
    model_config = {"extra": "forbid"}


class EntityRequest(BaseModel):
    """``POST /recommend/entity`` —— 字段与参考部署完全一致。"""
    entities: List[EntitySpec] = Field(..., description="实体列表（至少 1 个）")
    theme: Optional[str] = Field(None)
    scene_description: Optional[str] = Field(None)
    asset_ids: Optional[List[str]] = Field(default_factory=list)
    excluded_ids: Optional[List[str]] = Field(default_factory=list)
    # 与参考部署保持一致：上限 50
    top_k: int = Field(10, ge=1, le=50, description="每个实体召回的 top-k 上限（与参考部署一致）")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    fields: Optional[List[str]] = Field(None)


class RecommendWithPlanRequest(BaseModel):
    """``POST /recommend_with_plan`` —— 字段与参考部署完全一致。

    D1 直接读 ``plan_result.slots``，每个 slot 取 ``name`` / ``description`` /
    ``filters`` 当成 entity 转交给批量逻辑。

    D1 扩展：相对参考部署多一个顶层 ``filters`` 字段（与 ``entity.filters`` 合并；
    向后兼容：参考部署调用方不传该字段时行为完全一致）。
    """
    plan_result: Dict[str, Any] = Field(
        ...,
        description="规划结果 dict（query/slots/slot_relations/reasoning/anchor_asset_ids/excluded_ids）",
    )
    scene_description: Optional[str] = Field(
        None, description="可选；不传则使用 plan_result.query 作为全局场景文本",
    )
    theme: Optional[str] = Field(None)
    asset_ids: Optional[List[str]] = Field(default_factory=list)
    excluded_ids: Optional[List[str]] = Field(default_factory=list)
    return_mode: str = Field(RETURN_MODE_BOTH)
    # 与参考部署保持一致：默认 1，上限 100
    top_k: int = Field(1, ge=1, le=100, description="每实体返回的资产数（与参考部署一致）")
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="可选全局过滤；与 plan_result.slots[*].filters 合并（slot 级覆盖全局）。"
                    "D1 扩展字段：参考部署不接受顶层 filters，但传 None / {} 时行为一致。",
    )
    fields: Optional[List[str]] = Field(None)


class Strategy3CompatRequest(BaseModel):
    """``POST /search_strategy3`` —— 字段与参考部署完全一致。"""
    query: str
    theme: Optional[str] = None
    terrain_type: Optional[str] = None
    fields: Optional[List[str]] = Field(None)


# ═════════════════════════════════════════════════════════════════════════════
#   响应模型 —— 1:1 对齐参考部署 ResponseResult
# ═════════════════════════════════════════════════════════════════════════════

class CombinationItem(BaseModel):
    """组合返回项（字段名与参考部署完全一致）。

    ``type_id`` = asset_id（保留参考部署历史字段名）。
    """
    type_id: str
    name: str
    score: Optional[float] = None
    attributes: Optional[Dict[str, Any]] = None


class CombinationDetail(BaseModel):
    combination_list: List[CombinationItem]
    source: str = ""
    total_relation_score: float = 0.0


class ResponseResult(BaseModel):
    combinations: List[CombinationDetail] = Field(default_factory=list)
    per_entity_results: Dict[str, List[CombinationItem]] = Field(default_factory=dict)
    success: bool = True
    message: str = ""


# ═════════════════════════════════════════════════════════════════════════════
#   翻译层：API 字段 → embedding 模型 2 类输入
# ═════════════════════════════════════════════════════════════════════════════

def _compose_query_text(
    scene_description: Optional[str] = None,
    theme: Optional[str] = None,
    entity_description: Optional[str] = None,
) -> Optional[str]:
    """把 3 个全局/实体级文本拼成一段「原始 query」喂给类型 1 编码。

    规则：按 scene_description → theme → entity_description 顺序非空拼接；
    全空返回 None（由调用方决定走类型 2）。

    分隔符策略（B15 修复）：
      训练时 query 是单段连续中文（模型没见过中英文空格分词后的 query）。
      为减少 OOD：
        - 若所有片段全是中文 / 全是非 ASCII，用全角分号「；」分隔（中文里的合理停顿）
        - 否则用单空格（兼容含英文 / 数字的混合 query）
      这样比一律 " " join 更接近训练分布。

    设计取舍：
      - 训练时模型见到的原始 query 多为「场景句」（如"中世纪村庄清晨"），
        把场景 + 主题 + 实体补充描述 concat 是和训练分布最接近的兼容做法。
      - 不再像 v1 那样把字段拆开喂多套 view —— 训练侧只有 2 个 view，多套
        view 拼装会引入分布漂移。
    """
    parts = [
        (scene_description or "").strip(),
        (theme or "").strip(),
        (entity_description or "").strip(),
    ]
    parts = [p for p in parts if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    # 检测是否全是中文/非 ASCII（粗略：每段都不含 ASCII letter/digit）
    def _is_pure_cjk(s: str) -> bool:
        return not any((c.isascii() and (c.isalnum() or c == "_")) for c in s)
    if all(_is_pure_cjk(p) for p in parts):
        return "；".join(parts)
    return " ".join(parts)


def _item_from_recommend(
    item: RecommendItem,
    keep_score: bool = True,
) -> CombinationItem:
    """RecommendItem → CombinationItem（字段名映射：asset_id → type_id）。"""
    return CombinationItem(
        type_id=item.asset_id,
        name=item.name,
        score=float(item.score) if keep_score else None,
        attributes=item.attributes,
    )


def _items_from_result(res: RecommendResult) -> List[CombinationItem]:
    return [_item_from_recommend(it) for it in res.items]


def _build_response(
    per_entity: Dict[str, List[CombinationItem]],
    return_mode: str,
    extra_combinations: Optional[List[CombinationDetail]] = None,
    message: str = "",
) -> ResponseResult:
    """按 return_mode 组装响应（与参考部署 _build_response 行为一致）。

    - per_slot：只透出 per_entity_results；combinations 留空
    - combined / both：D1 已退化为 per_slot；额外把 entity_concat 作为一个组合
      塞进 combinations[0] 方便老调用方还能拿到「拼接后的总列表」。
    """
    combinations: List[CombinationDetail] = list(extra_combinations or [])
    if return_mode != RETURN_MODE_PER_SLOT:
        # entity_concat：把所有 entity 的 top-K 顺序拼起来作为一个组合
        flat: List[CombinationItem] = []
        for ent_name, items in per_entity.items():
            flat.extend(items)
        if flat:
            combinations.insert(0, CombinationDetail(
                combination_list=flat,
                source="entity_concat",
                total_relation_score=0.0,
            ))
    out_per_entity = per_entity if return_mode != RETURN_MODE_COMBINED else {}

    return ResponseResult(
        combinations=combinations,
        per_entity_results=out_per_entity,
        success=True,
        message=message or (
            f"推荐成功：combinations={len(combinations)}，per_entity={len(out_per_entity)}"
        ),
    )


def _run_for_entity(
    svc: RetrieverService,
    *,
    entity_name: str,
    entity_description: Optional[str],
    scene_description: Optional[str],
    theme: Optional[str],
    filters: Optional[Dict[str, Any]],
    asset_ids: Optional[List[str]],
    excluded_ids: Optional[List[str]],
    top_k: int,
    return_fields: Optional[List[str]],
    score_threshold: Optional[float] = None,
) -> RecommendResult:
    """所有路由的统一调用点：拼 query 文本 → service.recommend_single_slot。"""
    q = _compose_query_text(scene_description, theme, entity_description)
    return svc.recommend_single_slot(
        entity_name=entity_name,
        query=q,
        filters=filters,
        asset_ids=asset_ids,
        excluded_ids=excluded_ids,
        top_k=top_k,
        return_fields=return_fields,
        score_threshold=score_threshold,
    )


def _merge_filters(
    entity_filters: Optional[Dict[str, Any]],
    global_filters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """合并优先级：entity.filters > global_filters（与参考部署一致）。"""
    merged: Dict[str, Any] = {}
    if global_filters:
        merged.update(global_filters)
    if entity_filters:
        merged.update(entity_filters)
    return merged


# ═════════════════════════════════════════════════════════════════════════════
#   FastAPI 应用
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="D1 Recommendation API (qy-v1 compatible)",
    description=(
        "基于 phase4_training_v2 训练的 Qwen3-Embedding-4B dense retriever。"
        "API 与 recommendation_qy/asset_recommendation_v1 完全兼容；"
        "内部翻译为 embedding 模型见过的 2 类输入（query+entity / entity-only）。"
    ),
    version="2.0.0",
)


@app.on_event("startup")
async def _startup() -> None:
    """异步启动钩子。

    RetrieverService.get() 是同步阻塞（首次会跑 4B 加载 + 6560 doc 编码 1-3min），
    直接在 async 上下文里调会卡住整个 event loop —— 期间 /health 也无响应，
    k8s readiness / systemd 健康检测会误判服务挂掉。
    用 run_in_executor 把它扔到默认线程池，event loop 期间仍可处理（极少量的）
    其他请求（/health 之类的轻量探测）。
    """
    import asyncio
    cfg = get_config()
    logger.info(
        "[D1 startup] 启动预热中：ckpt=%s, asset_cards=%s",
        cfg.ckpt_dir, cfg.asset_cards_jsonl,
    )
    loop = asyncio.get_event_loop()
    svc = await loop.run_in_executor(None, RetrieverService.get)
    logger.info(
        "[D1 startup] ready：N_assets=%d dim=%d device=%s",
        svc.index.n, svc.index.dim, svc.encoder.device,
    )


# ═════════════════════════════════════════════════════════════════════════════
#   路由 1：/recommend/combination —— 组合推荐（实体列表 → 每实体 top-K）
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/recommend/combination", response_model=ResponseResult)
async def recommend_combination(req: CombinationRequest) -> ResponseResult:
    """组合推荐。D1 无组合优化，return_mode=combined/both 静默降级为 per_slot。"""
    try:
        if not req.entities:
            raise HTTPException(status_code=400, detail="entities 至少 1 个")
        if req.return_mode not in _VALID_RETURN_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid return_mode={req.return_mode!r}; 合法值: {sorted(_VALID_RETURN_MODES)}",
            )
        rm = _normalize_return_mode(req.return_mode)
        svc = RetrieverService.get()

        per_entity: Dict[str, List[CombinationItem]] = {}
        for ent in req.entities:
            merged = _merge_filters(ent.filters, req.filters)
            res = _run_for_entity(
                svc,
                entity_name=ent.entity_name,
                entity_description=ent.entity_description,
                scene_description=req.scene_description,
                theme=req.theme,
                filters=merged,
                asset_ids=req.asset_ids or None,
                excluded_ids=req.excluded_ids or None,
                top_k=req.top_k,
                return_fields=req.fields,
            )
            per_entity[ent.entity_name] = _items_from_result(res)

        return _build_response(per_entity, rm)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[/recommend/combination] 执行出错")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e))


# ═════════════════════════════════════════════════════════════════════════════
#   路由 2：/recommend/single_slot —— 单实体检索
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/recommend/single_slot", response_model=ResponseResult)
async def recommend_single_slot(req: SingleSlotRequest) -> ResponseResult:
    try:
        svc = RetrieverService.get()
        res = _run_for_entity(
            svc,
            entity_name=req.entity_name,
            entity_description=req.entity_description,
            scene_description=req.scene_description,
            theme=req.theme,
            filters=req.filters,
            asset_ids=req.asset_ids or None,
            excluded_ids=req.excluded_ids or None,
            top_k=req.top_k,
            return_fields=req.fields,
            score_threshold=req.score_threshold,
        )
        items = _items_from_result(res)
        return ResponseResult(
            combinations=[CombinationDetail(combination_list=items, source="single_slot")],
            per_entity_results={req.entity_name: items},
            success=res.success,
            message=res.message or f"单实体检索成功：返回 {len(items)} 条",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[/recommend/single_slot] 执行出错")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e))


# ═════════════════════════════════════════════════════════════════════════════
#   路由 3：/recommend/scene —— 场景描述（D1 无 LLM Planner，降级单实体路径）
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/recommend/scene", response_model=ResponseResult)
async def recommend_scene(req: SceneRequest) -> ResponseResult:
    """场景描述推荐。

    D1 没有 LLM Planner（参考部署的 Gemini 规划阶段不在 D1 范畴），
    退化为「把 scene_description 整段当成 query 文本，对自身做单实体类型 1 检索」。
    调用方拿到的依旧是 ResponseResult 形态（与参考部署一致）。
    """
    try:
        if req.return_mode not in _VALID_RETURN_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid return_mode={req.return_mode!r}",
            )
        rm = _normalize_return_mode(req.return_mode)
        svc = RetrieverService.get()

        # 退化策略：D1 没有 LLM Planner，把 scene_description 当成 [原始 query]
        # 走类型 1 检索。`[实体]` 段塞半截场景句子（如截前 12 字符）会让 query 进入
        # 训练分布外（训练里 [实体] 段都是名词性短语）。改用一个**通用占位 entity**：
        # 训练时也见过的 entity_only 路径里出现过的"通用场景"风格短语；这里固定用
        # "场景资产" —— 名词、短、不引入语义偏置，让模型主要用 [原始 query] 段做匹配。
        scene = (req.scene_description or "").strip()
        ent_anchor = "场景资产"
        res = _run_for_entity(
            svc,
            entity_name=ent_anchor,
            entity_description=None,
            scene_description=scene,
            theme=req.theme,
            filters=req.filters,
            asset_ids=req.asset_ids or None,
            excluded_ids=req.excluded_ids or None,
            top_k=req.top_k,
            return_fields=req.fields,
        )
        items = _items_from_result(res)
        return _build_response(
            per_entity={ent_anchor: items},
            return_mode=rm,
            message=(
                f"场景检索成功（D1 无 LLM Planner，降级为单实体路径）：返回 {len(items)} 条"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[/recommend/scene] 执行出错")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e))


# ═════════════════════════════════════════════════════════════════════════════
#   路由 4：/recommend/entity —— 批量实体（每实体独立 top-K）
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/recommend/entity", response_model=ResponseResult)
async def recommend_entity(req: EntityRequest) -> ResponseResult:
    try:
        if not req.entities:
            raise HTTPException(status_code=400, detail="entities 至少 1 个")
        svc = RetrieverService.get()

        per_entity: Dict[str, List[CombinationItem]] = {}
        per_entity_details: List[CombinationDetail] = []
        for ent in req.entities:
            merged = _merge_filters(ent.filters, req.filters)
            res = _run_for_entity(
                svc,
                entity_name=ent.entity_name,
                entity_description=ent.entity_description,
                scene_description=req.scene_description,
                theme=req.theme,
                filters=merged,
                asset_ids=req.asset_ids or None,
                excluded_ids=req.excluded_ids or None,
                top_k=req.top_k,
                return_fields=req.fields,
            )
            items = _items_from_result(res)
            per_entity[ent.entity_name] = items
            per_entity_details.append(CombinationDetail(
                combination_list=items,
                source=f"entity:{ent.entity_name}",
                total_relation_score=0.0,
            ))

        # entity 接口默认两块都透出（参考部署同款）：per_entity + per_entity_details + entity_concat
        return _build_response(
            per_entity=per_entity,
            return_mode=RETURN_MODE_BOTH,
            extra_combinations=per_entity_details,
            message=f"批量实体推荐成功：{len(per_entity)} 个实体",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[/recommend/entity] 执行出错")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e))


# ═════════════════════════════════════════════════════════════════════════════
#   路由 5：/recommend_with_plan —— 消费外部 plan_result
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/recommend_with_plan", response_model=ResponseResult)
async def recommend_with_plan(req: RecommendWithPlanRequest) -> ResponseResult:
    """消费外部规划结果（如 llm_planning_external.plan() 输出）。

    D1 直接读 ``plan_result.slots``，每个 slot 抽取：
      - ``name``        → entity_name
      - ``description`` → entity_description（可选）
      - ``filters``     → 实体级硬过滤（可选）
    剩余字段（slot_relations / reasoning）D1 不消费。
    """
    try:
        if req.return_mode not in _VALID_RETURN_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid return_mode={req.return_mode!r}",
            )
        rm = _normalize_return_mode(req.return_mode)

        plan = req.plan_result or {}
        if not isinstance(plan, dict):
            raise HTTPException(
                status_code=400,
                detail=f"plan_result 必须为 dict，收到 {type(plan).__name__}",
            )
        slots = plan.get("slots") or []
        if not isinstance(slots, list) or not slots:
            raise HTTPException(
                status_code=400,
                detail="plan_result.slots 必须是非空 list",
            )

        # scene_description 缺省回退到 plan_result.query（与参考部署一致）
        scene = req.scene_description or plan.get("query") or None
        # asset_ids / excluded_ids 缺省回退到 plan_result.* （与参考部署一致）
        asset_ids = req.asset_ids or plan.get("anchor_asset_ids") or None
        excluded_ids = req.excluded_ids or plan.get("excluded_ids") or None

        svc = RetrieverService.get()
        per_entity: Dict[str, List[CombinationItem]] = {}
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            name = str(slot.get("name", "") or "").strip()
            if not name:
                continue
            desc = slot.get("description")
            slot_filters = slot.get("filters") or {}
            merged = _merge_filters(slot_filters, req.filters)
            res = _run_for_entity(
                svc,
                entity_name=name,
                entity_description=desc if isinstance(desc, str) else None,
                scene_description=scene,
                theme=req.theme,
                filters=merged,
                asset_ids=asset_ids,
                excluded_ids=excluded_ids,
                top_k=req.top_k,
                return_fields=req.fields,
            )
            per_entity[name] = _items_from_result(res)

        return _build_response(
            per_entity=per_entity,
            return_mode=rm,
            message=f"plan 推荐成功：slots={len(per_entity)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[/recommend_with_plan] 执行出错")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e))


# ═════════════════════════════════════════════════════════════════════════════
#   路由 6：/search_strategy3 —— 向后兼容（等同 /recommend/scene）
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/search_strategy3", response_model=ResponseResult)
async def search_strategy3(req: Strategy3CompatRequest) -> ResponseResult:
    """向后兼容入口（与参考部署一致：等同 /recommend/scene 默认参数）。

    ``terrain_type`` 透传为 ``available_terrain`` 过滤器（参考部署同款语义）。
    """
    filters: Dict[str, Any] = {}
    if req.terrain_type:
        # available_terrain 是多值字段；用单值传入会被 _match_one 当 scalar 处理
        # （asset 端是 list 时检查 condition 是否在 list 中，命中即放行）。
        filters["available_terrain"] = req.terrain_type
    scene_req = SceneRequest(
        scene_description=req.query,
        theme=req.theme,
        return_mode=RETURN_MODE_BOTH,
        top_k=10,
        filters=filters,
        fields=req.fields,
    )
    return await recommend_scene(scene_req)


# ═════════════════════════════════════════════════════════════════════════════
#   健康检查 + 路由列表
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> Dict[str, Any]:
    # 服务名与 DEPLOY.md §6 文档保持一致（监控脚本按此 key 报警）
    return {"status": "healthy", "service": "d1-single-entity-recommendation"}


@app.get("/info")
async def info() -> Dict[str, Any]:
    cfg = get_config()
    try:
        svc = RetrieverService.get()
        return {
            "n_assets": svc.index.n,
            "embedding_dim": svc.index.dim,
            "ckpt_dir": cfg.ckpt_dir,
            "asset_cards_jsonl": cfg.asset_cards_jsonl,
            "device": svc.encoder.device,
            "max_q_len": cfg.max_q_len,
            "max_d_len": cfg.max_d_len,
            "ready": True,
        }
    except Exception as e:
        # 服务未就绪 / 启动出错 → 503 Service Unavailable，方便 k8s readiness probe
        # 直接判定。返回 200 + ready=False 会让监控误以为是健康响应。
        logger.exception("[/info] 服务未就绪")
        raise HTTPException(
            status_code=503,
            detail=_safe_error_detail(e, fallback="service not ready"),
        )


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "D1 Recommendation API (qy-v1 compatible)",
        "version": "2.0.0",
        "model_input_modes": {
            "type1_query_plus_entity":
                "(scene_description / theme / entity_description) 任一非空 → V1.1 baseline.query_entity",
            "type2_entity_only":
                "三者全空 → V1.2 baseline.entity_only",
        },
        "endpoints": {
            "POST /recommend/combination":  "组合推荐（实体列表 → 每实体 top-K）",
            "POST /recommend/single_slot":  "单实体检索",
            "POST /recommend/scene":        "场景描述（D1 无 LLM Planner，降级为单实体）",
            "POST /recommend/entity":       "批量实体（每实体独立 top-K）",
            "POST /recommend_with_plan":    "消费外部 plan_result",
            "POST /search_strategy3":       "向后兼容（等同 /recommend/scene）",
            "GET  /health":                 "健康检查",
            "GET  /info":                   "服务运行信息",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#   入口
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """python -m D1_deploy.main 启动。"""
    logging.basicConfig(
        level=os.environ.get("D1_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = get_config()
    logger.info("[D1] 服务即将启动：host=%s port=%d", cfg.host, cfg.port)
    uvicorn.run(
        "D1_deploy.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
