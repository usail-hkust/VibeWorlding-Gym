"""Asset Retrieval Client — 资产检索服务客户端.

封装 assets_retrieval 服务的 /recommend/single_slot 端点,
基于 VibeWorlder-Embedding-4B + VWE-Bench 资产库.

服务地址默认读环境变量 VIBEWORLD_RETRIEVE_SERVER（默认 http://localhost:8081）,
服务部署见 assets_retrieval/README.md。

设计要点:
- **无 cache**:服务时延 < 1s,在线直连
- **trust_env=False**:避免读到环境里的 http_proxy 干扰
- 指数退避重试 3 次 (HTTP 5xx / Network / Timeout)
- 返回 simplified 列表,不暴露原始 combinations 结构

核心 API:
    client = AssetRetrievalClient()
    results = client.retrieve("高大挺拔的松树", top_k=5, theme="中式园林")
    # results = [
    #   {"type_id": "20007733", "name": "主题02松树02", "score": 0.34, ...},
    #   ...
    # ]

服务返回响应实测结构(/recommend/single_slot):
    {
      "per_entity_results": {
        "<entity_name>": [
          {
            "type_id": "20007733",       # 8 位 dream_creator,与 PCG ID 体系同源
            "name": "主题02松树02",
            "score": 0.3400,             # cosine ∈ [0, 1]
            "attributes": {              # 仅当请求 fields 时存在
              "category_minor": "植被",
              "type": "树木",
              "subtype": "乔木",
              "size_class": "大尺寸物体",
              "placement": "落地 - 独立落地",
              "scene_limit": "无限制",
              "image_uri": "..."
            }
          }
        ]
      },
      "combinations": [...],   # 历史兼容,本 client 不暴露
      "success": true
    }
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# 默认 fields:平衡 verbosity 和评估需要(供 verifier H5 用 + 看板展示 + agent 视觉选型)
# 2026-06-02 起服务支持 caption_visual(整段中文视觉描述)+ colors(颜色数组),
# 用于让 agent 基于视觉/颜色挑选最契合场景主题的资产。
DEFAULT_FIELDS: List[str] = [
    "category_minor",
    "type",
    "subtype",
    "size_class",
    "placement",
    "scene_limit",
    "image_uri",
    "caption_visual",
    "colors",
]

# 默认服务地址；用 VIBEWORLD_RETRIEVE_SERVER 环境变量覆盖
DEFAULT_BASE_URL = os.environ.get("VIBEWORLD_RETRIEVE_SERVER", "http://localhost:8081")

# 客户端默认 timeout(s):服务声称稳态 30~100ms,实测 < 1s,5s 足够
DEFAULT_TIMEOUT = 5.0

# 指数退避重试上限
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 1.0   # 1s, 2s, 4s


class RetrieveError(Exception):
    """检索失败 (重试耗尽 / 服务返回 success=false 等)."""


class AssetRetrievalClient:
    """qy 资产检索服务的同步客户端.

    线程安全:**不**线程安全 — 每个线程请单独构造一个 client 实例.
    httpx.Client 内部连接池会自然复用 keepalive.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        client: Optional[httpx.Client] = None,
    ):
        """
        Args:
            base_url: 服务根 URL,无尾斜杠
            timeout: 单次请求 timeout(秒)
            retries: 重试次数(不含首次).0 表示不重试
            retry_backoff: 退避基数(秒).第 N 次等 retry_backoff * 2^(N-1)
            client: 可选;允许注入自定义 httpx.Client(用于单测 mock)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff

        # **关键**:trust_env=False 避免读 http_proxy 干扰
        # (Session 28 教训)
        self._client = client or httpx.Client(
            timeout=timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._owns_client = client is None

    def __enter__(self) -> "AssetRetrievalClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        entity_name: str,
        top_k: int = 5,
        theme: Optional[str] = None,
        scene_description: Optional[str] = None,
        size_class: Optional[str] = None,
        scene_limit: Optional[str] = None,
        labels: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """单实体检索.

        Args:
            entity_name: 中文实体名 (e.g. "高大挺拔的松树")
            top_k: 返回候选数 (上限 100)
            theme: 可选,丰富检索语义
            scene_description: 可选
            size_class: 可选硬过滤 — "大尺寸物体" / "中尺寸物体" / "小尺寸物体"
            scene_limit: 可选硬过滤 — "无限制" / "室内" / "沙地" / ...
            labels: 可选 OR 过滤 — e.g. ["卡通", "中国风"]
            fields: 资产属性透出白名单;None 取 DEFAULT_FIELDS

        Returns:
            simplified 列表:
            [
              {
                "type_id": "20007733",
                "name": "主题02松树02",
                "score": 0.3400,
                "category_minor": "植被",
                "type": "树木",
                "subtype": "乔木",
                "size_class": "大尺寸物体",
                "placement": "落地 - 独立落地",
                "scene_limit": "无限制",
                "image_uri": "..."
              },
              ...  # 共 top_k 条
            ]
            返回空列表 [] 表示服务正常但召回为空(filter 太严或库里没有).

        Raises:
            RetrieveError: 网络重试耗尽 / 服务 5xx 持续失败.
        """
        if not entity_name or not isinstance(entity_name, str):
            raise ValueError(f"entity_name must be a non-empty str, got {entity_name!r}")
        if top_k <= 0 or top_k > 100:
            raise ValueError(f"top_k must be in [1, 100], got {top_k}")

        fields = fields if fields is not None else list(DEFAULT_FIELDS)
        filters = _build_filters(size_class=size_class, scene_limit=scene_limit, labels=labels)

        payload: Dict[str, Any] = {
            "entity_name": entity_name,
            "top_k": top_k,
            "fields": fields,
        }
        if theme:
            payload["theme"] = theme
        if scene_description:
            payload["scene_description"] = scene_description
        if filters:
            payload["filters"] = filters

        resp = self._post_with_retry("/recommend/single_slot", payload)
        return _flatten_results(resp, entity_name)

    def retrieve_batch(
        self,
        entity_names: List[str],
        top_k: int = 5,
        theme: Optional[str] = None,
        scene_description: Optional[str] = None,
        global_filters: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """批量多实体检索 (走 /recommend/entity).

        Returns:
            {entity_name: [simplified items], ...}
            如果某个 entity 召回为空,对应 value = []
        """
        if not entity_names:
            return {}
        if top_k <= 0 or top_k > 50:
            # 注意:/recommend/entity 上限 50 (single_slot 才是 100)
            raise ValueError(f"top_k must be in [1, 50] for batch, got {top_k}")

        fields = fields if fields is not None else list(DEFAULT_FIELDS)

        payload: Dict[str, Any] = {
            "entities": [{"entity_name": e} for e in entity_names],
            "top_k": top_k,
            "fields": fields,
        }
        if theme:
            payload["theme"] = theme
        if scene_description:
            payload["scene_description"] = scene_description
        if global_filters:
            payload["filters"] = global_filters

        resp = self._post_with_retry("/recommend/entity", payload)

        per_entity = resp.get("per_entity_results") or {}
        return {
            name: _simplify_items(per_entity.get(name) or [])
            for name in entity_names
        }

    def health(self) -> Dict[str, Any]:
        """健康检查 (不重试,直接返回 dict)."""
        try:
            r = self._client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def info(self) -> Dict[str, Any]:
        """服务信息 (不重试)."""
        try:
            r = self._client.get(f"{self.base_url}/info")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post_with_retry(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(url, json=payload)
                # 4xx 不重试 (是请求方错误)
                if 400 <= r.status_code < 500:
                    raise RetrieveError(
                        f"HTTP {r.status_code} on {path}: {r.text[:300]}"
                    )
                r.raise_for_status()
                data = r.json()
                if not data.get("success", True):
                    # 服务显式返回 success=false
                    raise RetrieveError(
                        f"service returned success=false: {data.get('message', 'no message')}"
                    )
                return data
            except RetrieveError:
                # 4xx 直接抛,不重试
                raise
            except (httpx.HTTPError, httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                if attempt >= self.retries:
                    break
                backoff = self.retry_backoff * (2 ** attempt)
                logger.warning(
                    "[asset_retrieval] %s attempt %d/%d failed (%s), retrying in %.1fs",
                    path, attempt + 1, self.retries + 1, type(e).__name__, backoff,
                )
                time.sleep(backoff)

        raise RetrieveError(
            f"all {self.retries + 1} attempts failed for {path}: {last_err}"
        ) from last_err


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_filters(
    size_class: Optional[str] = None,
    scene_limit: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if size_class:
        f["size_class"] = size_class
    if scene_limit:
        f["scene_limit"] = scene_limit
    if labels:
        f["labels"] = labels
    return f


def _flatten_results(resp: Dict[str, Any], entity_name: str) -> List[Dict[str, Any]]:
    """从服务响应抽取 simplified 列表."""
    per_entity = resp.get("per_entity_results") or {}
    items = per_entity.get(entity_name) or []
    return _simplify_items(items)


def _simplify_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 [{type_id, name, score, attributes:{...}}] 摊平成单层 dict."""
    out: List[Dict[str, Any]] = []
    for it in items:
        flat: Dict[str, Any] = {
            "type_id": it.get("type_id"),
            "name": it.get("name"),
            "score": it.get("score"),
        }
        attrs = it.get("attributes") or {}
        flat.update(attrs)
        out.append(flat)
    return out


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------


def _cli_smoke() -> None:
    """命令行冒烟测试.

    用法:  python utils/asset_retrieval_client.py
    """
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 60)
    print("AssetRetrievalClient — CLI smoke test")
    print("=" * 60)

    with AssetRetrievalClient() as client:
        print("\n[1] /health")
        print(json.dumps(client.health(), ensure_ascii=False, indent=2))

        print("\n[2] /info")
        info = client.info()
        # 截断超长字段,只看关键的
        for key in ("n_assets", "embedding_dim", "device", "ready"):
            print(f"    {key}: {info.get(key)}")

        print("\n[3] retrieve('高大挺拔的松树', theme='中式园林', top_k=3)")
        results = client.retrieve(
            "高大挺拔的松树",
            top_k=3,
            theme="中式园林",
            scene_description="古朴雅致的庭院",
        )
        for r in results:
            cap = (r.get("caption_visual") or "")[:60]
            print(f"    type_id={r['type_id']} name={r['name']} score={r['score']:.3f}"
                  f" size={r.get('size_class')} cat={r.get('category_minor')}")
            print(f"        colors={r.get('colors')} caption={cap}...")

        print("\n[4] retrieve_batch(['樱花树', '石灯笼', '青苔石头'], theme='日式庭院', top_k=2)")
        batch = client.retrieve_batch(
            ["樱花树", "石灯笼", "青苔石头"],
            top_k=2,
            theme="日式庭院",
        )
        for name, items in batch.items():
            print(f"    [{name}]")
            for r in items:
                print(f"      type_id={r['type_id']} name={r['name']} score={r['score']:.3f}")

        print("\n[5] OOD test: retrieve('想象中的发光蘑菇')")
        ood = client.retrieve("想象中的发光蘑菇", top_k=2)
        for r in ood:
            print(f"    type_id={r['type_id']} name={r['name']} score={r['score']:.3f}")

    print("\n✅ All smoke tests passed.")


if __name__ == "__main__":
    _cli_smoke()
