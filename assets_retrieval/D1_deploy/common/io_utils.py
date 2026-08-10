"""JSONL / YAML 读取小工具。

避免依赖 D1_single_entity.src.common.io_utils（那条线还和训练 trainer 强耦合）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def iter_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_yaml(path: Optional[Path | str]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("需要 PyYAML：pip install PyYAML") from e
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
