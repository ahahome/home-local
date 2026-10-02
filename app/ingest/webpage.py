"""网页抓取（FR-ING-04）：httpx 抓取 + 清洗。抓取内容必须标注来源与时间。"""
from __future__ import annotations

import re

import httpx

from app.util import clamp

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120"}


def fetch_and_clean(url: str, max_chars: int = 15000) -> dict:
    resp = httpx.get(url, headers=_UA, timeout=20.0, follow_redirects=True)
    resp.raise_for_status()
    html = resp.text
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    body = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text).strip()
    return {"url": url, "title": title, "text": clamp(text, max_chars)}
