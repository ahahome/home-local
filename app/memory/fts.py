"""中文分词 → FTS5。jieba 优先；不可用时退化为「CJK 连续段 + 字母数字段」分词。"""
from __future__ import annotations

import re

try:
    import jieba  # type: ignore
    _HAS_JIEBA = True
except Exception:  # noqa: BLE001
    _HAS_JIEBA = False

_RUN = re.compile(r"[一-鿿]+|[A-Za-z0-9_]+")
_QUOTE = re.compile(r'"')


def tokenize(text: str) -> str:
    """把文本切成空格连接的 token 串，供 FTS5 索引/查询。"""
    if not text:
        return ""
    if _HAS_JIEBA:
        return " ".join(w.strip() for w in jieba.cut(text) if w.strip())
    return " ".join(_RUN.findall(text))


def fts_match_query(text: str) -> str | None:
    """把查询文本转成 FTS5 MATCH 表达式（词组 AND）。"""
    toks = [t for t in tokenize(text).split() if t and t not in '\"\'']
    if not toks:
        return None
    return " AND ".join(f'"{_QUOTE.sub("", t)}"' for t in toks)
