"""离线预编码脚本：在服务启动前一次性把 6560 条资产卡编码成 .npz 缓存。

为什么需要：
    服务启动时 ``RetrieverService.get()`` 会自动检测缓存；首次部署 / 切换 ckpt 时
    缓存不存在，会**在 startup hook 里同步编码**，导致 uvicorn 卡在
    "Waiting for application startup" 1-3 分钟（GPU）/ 30+ 分钟（CPU）。
    用本脚本提前生成缓存，部署时启动 < 10s。

用法（建议在 recommendation_qy_v2 目录下跑）：
    cd <repo-root>/assets_retrieval
    python -m D1_deploy.scripts.precompute_asset_emb

    # 切 ckpt 后强制重编码
    D1_FORCE_REENCODE=1 python -m D1_deploy.scripts.precompute_asset_emb

    # 用绝对路径跑（cwd 随意；脚本自带 sys.path 自修复）
    python D1_deploy/scripts/precompute_asset_emb.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_sys_path() -> None:
    """让脚本既能 ``python -m D1_deploy.scripts.precompute_asset_emb`` 也能
    ``python /abs/path/precompute_asset_emb.py`` 跑通。"""
    this_file = Path(__file__).resolve()
    # 找到 D1_deploy 的父目录（即 recommendation_qy_v2/）
    pkg_parent = this_file.parent.parent.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))


_bootstrap_sys_path()

from D1_deploy.asset_recommend_d1.config import D1RecommendConfig, set_config  # noqa: E402
from D1_deploy.asset_recommend_d1.encoder import (                              # noqa: E402
    InferenceEncoder,
    InferenceEncoderConfig,
)
from D1_deploy.asset_recommend_d1.asset_index import build_or_load_asset_index  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="D1: 预编码资产嵌入到 .npz 缓存")
    p.add_argument("--ckpt", type=str, default=None, help="覆写 ckpt 路径")
    p.add_argument("--asset-cards", type=str, default=None, help="覆写 asset_cards.jsonl 路径")
    p.add_argument("--cache-dir", type=str, default=None, help="覆写缓存目录")
    p.add_argument("--batch-size", type=int, default=None, help="编码 batch_size（GPU OOM 调小）")
    p.add_argument("--force", action="store_true", help="忽略已有缓存，强制重编码")
    args = p.parse_args()

    # 构造配置
    cfg = D1RecommendConfig()
    if args.ckpt:
        cfg.ckpt_dir = args.ckpt
    if args.asset_cards:
        cfg.asset_cards_jsonl = args.asset_cards
    if args.cache_dir:
        cfg.asset_emb_cache_dir = args.cache_dir
    if args.batch_size:
        cfg.encode_batch_size = args.batch_size
    # 环境变量语义与 D1RecommendConfig.__post_init__ 保持一致：
    # 仅当 D1_FORCE_REENCODE 为非空且不在 ("0","false","False") 时才视为开启。
    env_force_raw = os.environ.get("D1_FORCE_REENCODE")
    env_force = env_force_raw is not None and env_force_raw.strip() not in ("", "0", "false", "False")
    if args.force or env_force:
        cfg.force_reencode = True
    set_config(cfg)

    print(f"[precompute] ckpt = {cfg.ckpt_dir}", flush=True)
    print(f"[precompute] asset_cards = {cfg.asset_cards_jsonl}", flush=True)
    print(f"[precompute] cache_path = {cfg.emb_cache_path()}", flush=True)
    print(f"[precompute] force_reencode = {cfg.force_reencode}", flush=True)

    enc = InferenceEncoder(InferenceEncoderConfig(
        ckpt_dir=cfg.ckpt_dir,
        backbone_name=cfg.backbone_name,
        max_q_len=cfg.max_q_len,
        max_d_len=cfg.max_d_len,
        bf16=cfg.bf16,
        flash_attn=cfg.flash_attn,
    ))

    idx = build_or_load_asset_index(cfg, enc)
    print(
        f"[precompute] DONE: N_assets={idx.n}, dim={idx.dim}, "
        f"cache={cfg.emb_cache_path()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
