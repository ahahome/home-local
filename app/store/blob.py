"""内容寻址 blob 存储：SHA256 命名、原子写（tmp + fsync + os.replace）、天然去重。"""
from __future__ import annotations

import os
from pathlib import Path

from app.config import get_settings
from app.util import sha256_bytes


def _path_for(h: str) -> Path:
    root = get_settings().abs_blob_dir
    return root / h[:2] / h


def put(data: bytes) -> str:
    """写入并返回内容哈希。已存在则直接返回（幂等）。"""
    h = sha256_bytes(data)
    dest = _path_for(h)
    if dest.exists():
        return h
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)  # 原子
    return h


def get(h: str) -> bytes:
    return _path_for(h).read_bytes()


def exists(h: str) -> bool:
    return _path_for(h).exists()


def uri(h: str) -> str:
    return f"blob://{h}"
