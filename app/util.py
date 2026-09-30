"""通用工具：id、时间、哈希、JSON 修复解析。"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime

from app.config import tz


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def now_ms() -> int:
    return int(time.time() * 1000)


def today_local() -> str:
    """本地时区的 YYYY-MM-DD。"""
    return datetime.now(tz()).strftime("%Y-%m-%d")


def ms_to_local_str(ms: int | None) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz()).strftime("%Y-%m-%d %H:%M")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalize_value(s: str) -> str:
    """归一化字段值用于幂等指纹：去空白、统一全半角、小写。"""
    if s is None:
        return ""
    s = str(s).strip()
    # 全角转半角（常见标点和字母数字）
    table = str.maketrans(
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９，。！？：；（）",
        "abcdefghijklmnopqrstuvwxyz0123456789,.!?:;()",
    )
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


_MATCH_PUNCT = re.compile(r"[，。！？；：、（）()\[\]【】《》\"'‘’“”,.!?;:~`·—\-]+")
_MATCH_WS = re.compile(r"[\s\u3000]+")


def flatten_for_match(s) -> str:
    """去空白与标点后归一化，用于判断「某片段是否出现在原文中」。

    抽取器与评测器共用同一实现，避免两边规则漂移导致
    「评测说 evidence 不在原文，但代码认为在」这种自相矛盾。
    """
    if s is None:
        return ""
    return _MATCH_PUNCT.sub("", _MATCH_WS.sub("", normalize_value(str(s))))


def patch_fingerprint(person_id: str, field: str, value: str, source_msg_id: str) -> str:
    return sha256_text(f"{person_id}|{field}|{normalize_value(value)}|{source_msg_id}")


# ---------- 宽松的 JSON 解析（FR-CHT-10 同源：修复缺括号/全角/代码围栏） ----------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_loose(raw: str):
    """尽力把模型输出解析成 JSON。处理：代码围栏、前后废话、全角引号、尾逗号。"""
    if raw is None:
        raise ValueError("empty")
    s = raw.strip()
    s = _FENCE.sub("", s).strip()

    # 截取最外层的 {} 或 []
    start = None
    open_ch = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            open_ch = ch
            break
    if start is None:
        raise ValueError("no json object found")
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    end = None
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        # 尝试补全
        frag = s[start:]
        frag += close_ch * max(1, frag.count(open_ch) - frag.count(close_ch))
        s2 = frag
    else:
        s2 = s[start:end + 1]

    # 全角引号转半角（字符串外的全角引号会破坏解析）
    s2 = s2.replace("“", '"').replace("”", '"').replace("：", ":")
    # 尾逗号
    s2 = re.sub(r",(\s*[}\]])", r"\1", s2)
    return json.loads(s2)


def clamp(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
