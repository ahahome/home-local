"""Deep Research 编排（FR-RES-01~09）：
预检索 → 提案 → 子 agent 并行（边界互斥）→ 主 agent 复核（无来源丢弃）→ 合成报告。
草稿持续落库，中断可恢复（FR-RES-06）；过薄报告标记 thin（FR-RES-07）。
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.llm import get_llm
from app.store import repo
from app.util import clamp, now_ms

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def websearch(query: str, n: int = 5) -> list[dict]:
    """Bing HTML 检索（零密钥，沙箱内 DuckDuckGo 被代理拦截）。失败返回空列表。"""
    try:
        resp = httpx.get("https://www.bing.com/search",
                         params={"q": query, "setlang": "zh-CN", "count": str(n * 2)},
                         headers=_UA, timeout=15.0, follow_redirects=True)
        html = resp.text
    except Exception:  # noqa: BLE001
        return []
    results = []
    # Bing 真实结构：<h2 class=""><a target="_blank" href="URL">标题</a></h2>
    for m in re.finditer(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        url = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if url.startswith("http") and title:
            results.append({"title": title, "url": url})
        if len(results) >= n:
            break
    return results


def fetch_text(url: str, max_chars: int = 3000) -> str:
    try:
        resp = httpx.get(url, headers=_UA, timeout=15.0, follow_redirects=True)
        html = resp.text
    except Exception:  # noqa: BLE001
        return ""
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return clamp(text.strip(), max_chars)


def _proposal(topic: str, seed: list[dict]) -> dict:
    llm = get_llm()
    seed_txt = "\n".join(f"- {s['title']} {s['url']}" for s in seed) or "（预检索无结果）"
    obj = llm.chat_json([
        {"role": "system", "content": "你是调研负责人。基于预检索结果，为一个深度调研制定方案。"
         "把调研拆成 3 个【边界互斥】的子任务（每个子任务只负责一个角度，不要重叠）。"
         "只输出 JSON。"},
        {"role": "user", "content": f"调研主题：{topic}\n\n预检索结果：\n{seed_txt}\n\n"
         "输出：{\"questions\":[\"要回答的核心问题\"],"
         "\"subtasks\":[{\"scope\":\"子任务边界\",\"queries\":[\"检索词1\",\"检索词2\"]}]}"},
    ], tier="mid")
    if not isinstance(obj.get("subtasks"), list) or not obj["subtasks"]:
        obj["subtasks"] = [
            {"scope": f"{topic} 的基本事实与背景", "queries": [topic]},
            {"scope": f"{topic} 的近期动态与数据", "queries": [f"{topic} 最新"]},
            {"scope": f"{topic} 的评价与争议", "queries": [f"{topic} 评价"]},
        ]
    return obj


def _run_subtask(topic: str, sub: dict) -> dict:
    """子 agent：只在自己边界内检索与阅读，产出带来源的结论。"""
    llm = get_llm()
    scope = sub.get("scope", "")
    snippets = []
    sources: list[dict] = []
    for q in (sub.get("queries") or [topic])[:2]:
        for hit in websearch(f"{topic} {q}", n=3):
            text = fetch_text(hit["url"])
            if text:
                sources.append(hit)
                snippets.append(f"【{hit['title']}】{hit['url']}\n{text[:1200]}")
    if not snippets:
        return {"scope": scope, "claims": [], "sources": [], "note": "无外部来源"}
    obj = llm.chat_json([
        {"role": "system", "content": f"你只负责这个边界：{scope}。不要超出。"
         "基于资料产出结论，每条结论必须带来源 URL；资料里没有的不要编。只输出 JSON。"},
        {"role": "user", "content": f"主题：{topic}\n\n资料：\n" + "\n\n".join(snippets)[:6000] +
         "\n\n输出：{\"claims\":[{\"claim\":\"结论\",\"source\":\"URL\",\"confidence\":0.0}]}"},
    ], tier="mid")
    claims = obj.get("claims") if isinstance(obj, dict) else []
    return {"scope": scope, "claims": claims or [], "sources": sources}


def _review(topic: str, results: list[dict]) -> dict:
    """主 agent 复核：无来源丢弃，冲突标注。"""
    llm = get_llm()
    all_claims = []
    for r in results:
        for c in r.get("claims", []):
            if c.get("source"):
                all_claims.append(c)
    if not all_claims:
        return {"claims": [], "conflicts": []}
    obj = llm.chat_json([
        {"role": "system", "content": "复核调研结论：1) 丢掉没有来源或与来源明显不符的；"
         "2) 相互矛盾的保留双方都标注冲突；3) 去重。只输出 JSON。"},
        {"role": "user", "content": f"主题：{topic}\n\n结论：\n{json.dumps(all_claims, ensure_ascii=False)}\n\n"
         "输出：{\"claims\":[{\"claim\":\"...\",\"source\":\"URL\",\"confidence\":0.0}],"
         "\"conflicts\":[\"冲突描述\"]}"},
    ], tier="think")
    return obj if isinstance(obj, dict) else {"claims": all_claims, "conflicts": []}


def _synthesize(topic: str, proposal: dict, reviewed: dict) -> str:
    llm = get_llm()
    claims = reviewed.get("claims", [])
    claims_txt = "\n".join(
        f"- {c['claim']}（来源：{c['source']}）" for c in claims) or "（无可靠结论）"
    questions = "、".join(proposal.get("questions", [])) or topic
    return llm.chat([
        {"role": "system", "content": "把复核后的结论写成一份结构化中文报告。"
         "要求：每条事实性结论后用 [来源URL] 标注；结论不足时如实说明；不要引入资料外的新事实。"},
        {"role": "user", "content": f"主题：{topic}\n要回答的问题：{questions}\n\n"
         f"已复核结论：\n{claims_txt}\n\n冲突：{json.dumps(reviewed.get('conflicts', []), ensure_ascii=False)}\n\n"
         "输出 Markdown 报告：# 标题 / ## 核心结论 / ## 分维度分析 / ## 冲突与不确定性 / ## 来源列表"},
    ], tier="pro", max_tokens=4000)["content"]


def run_research(conn, topic: str) -> dict:
    rid = repo.create_research(conn, topic)
    conn.commit()
    try:
        seed = websearch(topic, n=5)
        proposal = _proposal(topic, seed)
        repo.update_research(conn, rid, status="running",
                             proposal=json.dumps(proposal, ensure_ascii=False))
        conn.commit()

        with ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(lambda s: _run_subtask(topic, s), proposal["subtasks"]))
        # 草稿落库（FR-RES-06）
        repo.update_research(conn, rid, draft=json.dumps(results, ensure_ascii=False))
        conn.commit()

        reviewed = _review(topic, results)
        report = _synthesize(topic, proposal, reviewed)
        sources = sorted({c["source"] for c in reviewed.get("claims", []) if c.get("source")})
        thin = len(reviewed.get("claims", [])) < 3 or len(report) < 400

        repo.update_research(conn, rid, status="done", report_md=report,
                             sources=json.dumps(sources, ensure_ascii=False))
        conn.commit()
        first_para = report.split("\n\n")[1] if "\n\n" in report else report[:200]
        return {"id": rid, "summary": clamp(first_para.replace("#", "").strip(), 200),
                "report_md": report, "sources": sources, "thin": thin}
    except Exception as e:  # noqa: BLE001
        repo.update_research(conn, rid, status="failed", report_md=str(e))
        conn.commit()
        raise
