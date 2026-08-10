"""D1 部署服务全局配置。

设计与 recommendation_qy/asset_recommendation_v1/asset_recommend_qy/config.py 对齐：
统一一份 dataclass 暴露所有外部依赖（模型 ckpt 路径、资产卡 jsonl、可选 ES 等）。
迁机时**只改这一份文件**。

> 与参考部署的关键差异：
>   - 参考部署走 *外部 embedding HTTP*（Qwen3-Embedding-8B vllm 服务）
>     + 离线把全量 asset 编码进 ES 做 kNN。
>   - 本部署直接加载 phase4 训练（v2 / v3）产物的 *本地 ckpt*，进程内 forward；
>     对全量 asset 一次性预编码后驻留 GPU/CPU 内存做矩阵乘法 top-K，
>     **零外部依赖**（不依赖 ES / embedding API）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── 路径基准：D1_deploy 目录所在的父目录 ─────────────────────────────────
# 历史上指向 `recommendation_qy_v2/`，当前迁到 `recommendation_yansong/` 下。
# 训练产物 / 资产卡 / view_prefixes 等已改用绝对路径，这里仅作 fallback 用。
_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parent.parent           # .../D1_deploy/
_REPO_ROOT = _PKG_ROOT.parent                  # .../recommendation_yansong/


@dataclass
class D1RecommendConfig:
    """D1 single-entity 推荐服务配置。

    `default_factory` 全部用绝对路径，避免 cwd 漂移导致路径解析失败。
    """

    # ── 检索模型 ckpt（VibeWorlder-Embedding-4B） ─────────────────────────
    # transformers 全参格式：config.json + model-*.safetensors + tokenizer.json。
    # 从 https://huggingface.co/collections/usail-hkust/vibeworlder 下载后，
    # 用 D1_CKPT_DIR 环境变量指向本地目录（默认相对本仓库的 models/ 子目录）。
    ckpt_dir: str = field(default_factory=lambda: os.environ.get(
        "D1_CKPT_DIR", "./models/VibeWorlder-Embedding-4B"
    ))

    # backbone HF 名（仅在 ckpt 是 LoRA adapter 时用作 base，普通全参 ckpt 无所谓）
    backbone_name: str = field(default_factory=lambda: os.environ.get(
        "D1_BACKBONE_NAME", "./models/VibeWorlder-Embedding-4B"
    ))

    # ── 资产卡数据（phase1 输出，doc 编码用） ─────────────────────────────
    # 默认指向与本服务同目录的 data/asset_cards.jsonl；可用
    # D1_ASSET_CARDS_JSONL 环境变量覆写。
    asset_cards_jsonl: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "data" / "asset_cards.jsonl"
    ))

    # ── 资产属性 / 过滤的真相源 CSV（filters & fields 透出从这里查） ──────
    # 数据来源：knowledge_graph/output/module1b 输出的标准化资产库 CSV，
    # 包含 26 列原始字段（asset_id / name_clean / category_major / category_minor /
    # type / subtype / artifact_nature / labels_json / env_flags / ...）。
    # 推理服务启动时一次性加载到内存（~6560 行，几 MB）。
    # 可用 D1_ASSET_CSV 环境变量覆写。
    asset_csv: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "data" / "standardized_asset_library_with_caption.csv"
    ))

    # ── 颜色 / 形状补充 CSV（module1b_color_shape，按 asset_id join 主库） ──
    # 主库 CSV 没有颜色字段；这份补充表提供 `colors`（顿号分隔的中文色名，
    # 服务侧切成 list[str]）与 `shape_description`（形状文字描述）。
    # 列：asset_id / status / has_image / colors / shape_description / confidence。
    # 启动时按 asset_id left-join 进主库属性表；缺失资产对应字段为 [] / None。
    # 置空字符串可关闭该 join（仅用主库字段）。
    # 可用 D1_COLOR_CSV 环境变量覆写；置空关闭颜色 join。
    color_csv: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "data" / "color_shape_detail.csv"
    ))

    # ── view → instruction 模板（与训练评测使用同一份 yaml） ──────────────
    # 默认指向与本服务同目录的 data/view_prefixes.yaml；可用
    # D1_VIEW_PREFIXES_YAML 环境变量覆写。
    view_prefixes_yaml: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "data" / "view_prefixes.yaml"
    ))

    # ── 推理参数（编码） ───────────────────────────────────────────────────
    max_q_len: int = 1024              # 训练时也是 1024，保持一致
    max_d_len: int = 1024
    bf16: bool = True
    flash_attn: bool = True            # CUDA 上自动启用，CPU/NPU 自动 fallback
    encode_batch_size: int = 32        # asset 全量预编码的 batch；GPU OOM 时调小

    # ── 资产嵌入缓存：避免每次启动重跑 6560 次 forward ─────────────────────
    # 文件名带 ckpt_dir 的最后一段（一般是 best/checkpoint-N），便于多 ckpt 共存。
    # 缓存 schema:
    #   {"asset_ids": [...], "embeddings": np.ndarray[N, D, float32]}
    asset_emb_cache_dir: str = field(default_factory=lambda: str(_PKG_ROOT / "cache"))
    # 是否在启动时强制重编码（忽略缓存）。开发期切换 ckpt 但缓存名相同会用得到。
    force_reencode: bool = False

    # ── 检索 / 过滤参数 ────────────────────────────────────────────────────
    default_top_k: int = 10
    max_top_k: int = 100
    similarity_threshold: float = 0.0  # 默认不过滤；> 0 时丢弃低分

    # ── 可选 ES 后端（默认关闭；本部署主路径不用 ES） ──────────────────────
    # 如需 ES kNN（例如资产库 > 100k 时），把 enable_es=True 并填好其余字段。
    enable_es: bool = False
    es_host: str = field(default_factory=lambda: os.environ.get("ES_HOST", "https://localhost:9200"))
    es_user: str = field(default_factory=lambda: os.environ.get("ES_USER", "elastic"))
    es_password: str = field(default_factory=lambda: os.environ.get("ES_PASSWORD", "your_es_password"))
    es_index_name: str = "d1_asset_v1"
    es_verify_certs: bool = False

    # ── 服务参数 ───────────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: os.environ.get("D1_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8080")))
    log_level: str = "info"

    def __post_init__(self) -> None:
        # 允许通过环境变量覆盖关键路径（不破坏默认值，便于运维切机）
        env_ckpt = os.environ.get("D1_CKPT_DIR")
        if env_ckpt:
            self.ckpt_dir = env_ckpt
        env_cards = os.environ.get("D1_ASSET_CARDS_JSONL")
        if env_cards:
            self.asset_cards_jsonl = env_cards
        env_csv = os.environ.get("D1_ASSET_CSV")
        if env_csv:
            self.asset_csv = env_csv
        env_color_csv = os.environ.get("D1_COLOR_CSV")
        if env_color_csv is not None:
            # 允许显式置空（""）关闭颜色 join
            self.color_csv = env_color_csv
        env_view_yaml = os.environ.get("D1_VIEW_PREFIXES_YAML")
        if env_view_yaml:
            self.view_prefixes_yaml = env_view_yaml
        env_force = os.environ.get("D1_FORCE_REENCODE")
        if env_force is not None:
            self.force_reencode = env_force.strip() not in ("", "0", "false", "False")
        # uvicorn 期望 lower-case log level（"info" / "debug" / ...）
        env_log = os.environ.get("D1_LOG_LEVEL")
        if env_log:
            self.log_level = env_log.strip().lower()

    # ── 工具方法 ──────────────────────────────────────────────────────────
    def emb_cache_path(self) -> Path:
        """根据 ckpt_dir 生成稳定缓存文件名。"""
        ck = Path(self.ckpt_dir)
        # ckpt 路径最后两段（如 v2sweep_A4_.../best）作为 tag，避免冲突
        tag = f"{ck.parent.name}__{ck.name}".replace("/", "_")
        # tag 可能很长，做 sha1 截断保留可读性
        import hashlib
        digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:10]
        # 保留人类可读前缀：方便 ls 时分辨
        readable = ck.parent.name[:60].replace("/", "_") + "__" + ck.name
        readable = readable.replace(":", "_")
        return Path(self.asset_emb_cache_dir) / f"asset_emb__{readable}__{digest}.npz"


# 默认单例（main.py / 推理服务直接用）；测试可自己 new 一个改字段。
_DEFAULT_CFG: Optional[D1RecommendConfig] = None


def get_config() -> D1RecommendConfig:
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = D1RecommendConfig()
    return _DEFAULT_CFG


def set_config(cfg: D1RecommendConfig) -> None:
    """测试 / CLI 覆写默认配置。"""
    global _DEFAULT_CFG
    _DEFAULT_CFG = cfg
