"""轻量 logger 工厂；不引入第三方依赖。

每个模块用 `from D1_deploy.common.logger import get_logger`
拿一个绑定名字的 logger，避免根 logger 被 uvicorn / fastapi 重置。
"""
from __future__ import annotations

import logging
import os


_INITIALIZED = False


def _ensure_root_config() -> None:
    """全局只设置一次 basicConfig。

    uvicorn CLI 启动时不会调 basicConfig，我们自己在第一次取 logger 时配。
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    level_name = os.environ.get("D1_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    _ensure_root_config()
    return logging.getLogger(name)
