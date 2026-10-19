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
_QUERY_FILLERS = frozenset({
    '的', '了', '呢', '吗', '呀', '啊', '吧', '是', '什么', '怎么', '怎样', '如何',
    '哪个', '哪些', '哪种', '请', '请问', '告诉', '我', '想', '想要', '知道', '一下',
    '关于', '有关', '有什么', '有', '和', '与', '对', '他', '她', '它', '他们', '她们',
    '你', '您', '我们',
    'what', 'which', 'how', 'is', 'are', 'does', 'do', 'please', 'tell', 'me',
    'about', 'the', 'a', 'an', 'of', 'to', 'for', 'with',
})


def tokenize(text: str) -> str:
    """把文本切成空格连接的 token 串，供 FTS5 索引/查询。"""
    if not text:
        return ""
    if _HAS_JIEBA:
        return " ".join(w.strip() for w in jieba.cut(text) if w.strip())
    return " ".join(_RUN.findall(text))


def query_terms(text: str) -> list[str]:
    """Query-only normalization leaves the persisted index format unchanged."""
    terms = []
    for term in tokenize(text).split():
        term = _QUOTE.sub('', term).casefold()
        if not any(char.isalnum() for char in term) or term in _QUERY_FILLERS or term in terms:
            continue
        terms.append(term)
        if len(terms) == 32:
            break
    return terms


def fts_match_query(text: str, *, match_all: bool = True) -> str | None:
    toks = query_terms(text)
    if not toks:
        return None
    return (' AND ' if match_all else ' OR ').join(f'"{term}"' for term in toks)
