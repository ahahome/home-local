"""Deep Research 编排（FR-RES-01~09）：
预检索 → 提案 → 子 agent 并行（边界互斥）→ 主 agent 复核（无来源丢弃）→ 合成报告。
草稿持续落库，中断可恢复（FR-RES-06）；过薄报告标记 thin（FR-RES-07）。
"""
from __future__ import annotations

import json
import re
import html as html_lib
import base64
from urllib.parse import urlparse, parse_qs, unquote
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
        resp.raise_for_status()
    except Exception:  # noqa: BLE001
        html = ""
    results = []
    # Bing 真实结构：<h2 class=""><a target="_blank" href="URL">标题</a></h2>
    for m in re.finditer(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        url = html_lib.unescape(m.group(1))
        if "bing.com/ck/" in url:
            encoded = parse_qs(urlparse(url).query).get("u", [""])[0]
            if encoded.startswith("a1"):
                try:
                    url = base64.urlsafe_b64decode(encoded[2:] + "=" * (-len(encoded[2:]) % 4)).decode()
                except (ValueError, UnicodeDecodeError):
                    continue
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if url.startswith("http") and title:
            results.append({"title": title, "url": url})
        if len(results) >= n:
            break
    if not results:
        try:
            response = httpx.get("https://html.duckduckgo.com/html/", params={"q": query},
                                 headers=_UA, timeout=15, follow_redirects=True)
            response.raise_for_status()
            for match in re.finditer(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.S):
                target = html_lib.unescape(match[1])
                url = parse_qs(urlparse(target).query).get("uddg", [target])[0]
                title = re.sub(r"<[^>]+>", "", match[2]).strip()
                if url.startswith(("http://", "https://")):
                    results.append({"url": url, "title": html_lib.unescape(title)})
                if len(results) >= n:
                    break
        except httpx.HTTPError:
            pass
    return results


def fetch_text(url: str, max_chars: int = 3000) -> str:
    try:
        resp = httpx.get(url, headers=_UA, timeout=15.0, follow_redirects=True)
        html = resp.text
        resp.raise_for_status()
    except Exception:  # noqa: BLE001
        return ""
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return clamp(html_lib.unescape(text.strip()), max_chars)


def _proposal(topic: str, seed: list[dict]) -> dict:
    llm = get_llm()
    seed_txt = "\n".join(f"- {s['title']} {s['url']}\n{s.get('excerpt', '')[:1600]}" for s in seed) or "（预检索无结果）"
    obj = llm.chat_json([
        {"role": "system", "content": "你是调研负责人。基于预检索结果，为一个深度调研制定方案。"
         "先理解资料中的实际对象，不要凭名称假设功能或技术。范围必须贴近主题和公开原文；未公开的细节标为待验证。"
         "把调研拆成 3 个【边界互斥】的子任务（每个子任务只负责一个角度，不要重叠）。"
         "只输出 JSON。"},
        {"role": "user", "content": f"调研主题：{topic}\n\n预检索结果：\n{seed_txt}\n\n"
         "输出：{\"questions\":[\"要回答的核心问题\"],"
         "\"subtasks\":[{\"scope\":\"子任务边界\",\"queries\":[\"检索词1\",\"检索词2\"]}]}"},
    ], tier="mid")
    if not isinstance(obj, dict):
        obj = {}
    if not isinstance(obj.get("subtasks"), list) or not obj["subtasks"]:
        obj["subtasks"] = [
            {"scope": f"{topic} 的基本事实与背景", "queries": [topic]},
            {"scope": f"{topic} 的近期动态与数据", "queries": [f"{topic} 最新"]},
            {"scope": f"{topic} 的评价与争议", "queries": [f"{topic} 评价"]},
        ]
    obj["subtasks"] = [s for s in obj["subtasks"][:3] if isinstance(s, dict) and s.get("scope")]
    return obj


def direct_sources(topic: str) -> list[dict]:
    """User-supplied public URLs also work when search providers are unavailable."""
    urls = dict.fromkeys(re.findall(r"https?://[^\s<>\[\]（），。]+", topic))
    return [{"title": urlparse(u).netloc, "url": u.rstrip('),.;')} for u in urls]


def _run_subtask(topic: str, sub: dict, seed=None) -> dict:
    """子 agent：只在自己边界内检索与阅读，产出带来源的结论。"""
    llm = get_llm()
    scope = sub.get("scope", "")
    snippets = []
    sources: list[dict] = []
    hits = list(seed or direct_sources(topic))
    domains = {urlparse(s['url']).netloc.removeprefix('www.') for s in direct_sources(topic)}
    for q in (sub.get("queries") or [topic])[:2]:
        hits.extend(websearch(f"{topic} {q}", n=3))
    seen = set()
    for hit in hits[:10]:
        if domains and urlparse(hit['url']).netloc.removeprefix('www.') not in domains:
            continue
        if hit["url"] not in seen:
            seen.add(hit["url"])
            text = fetch_text(hit["url"])
            if text:
                sources.append({**hit, "excerpt": text})
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
    urls = {s["url"] for s in sources}
    claims = [c for c in claims or [] if isinstance(c, dict) and c.get("source") in urls]
    return {"scope": scope, "claims": claims, "sources": sources}


def _review(topic: str, results: list[dict]) -> dict:
    """主 agent 复核：无来源丢弃，冲突标注。"""
    llm = get_llm()
    all_claims = []
    sources = {s["url"]: s.get("excerpt", "") for r in results for s in r.get("sources", [])}
    for r in results:
        for c in r.get("claims", []):
            if c.get("source") in sources:
                all_claims.append(c)
    if not all_claims:
        return {"claims": [], "conflicts": []}
    passages = {url: [text[i:i + 300] for i in range(0, len(text), 300)] for url, text in sources.items()}
    numbered = {url: [{'index': i, 'text': text} for i, text in enumerate(parts)] for url, parts in passages.items()}
    obj = llm.chat_json([
        {"role": "system", "content": "复核调研结论：1) 丢掉没有来源或与来源明显不符的；"
         "2) 相互矛盾的保留双方都标注冲突；3) 去重。每条结论必须选择确实支持它的来源片段的 evidence_index。"
         "结论可以用中文，但证据必须引用提供的原文编号，不得翻译或捏造原文。只输出 JSON。"},
        {"role": "user", "content": f"主题：{topic}\n\n结论：\n{json.dumps(all_claims, ensure_ascii=False)}\n\n"
         f"来源原文（每个 URL 独立编号）：{json.dumps(numbered, ensure_ascii=False)[:30000]}\n\n"
         "输出：{\"claims\":[{\"claim\":\"...\",\"source\":\"URL\",\"confidence\":0.0,\"evidence_index\":0}],"
         "\"conflicts\":[\"冲突描述\"]}"},
    ], tier="think")
    if not isinstance(obj, dict):
        return {"claims": [], "conflicts": []}
    verified = []
    for c in obj.get('claims', []):
        if not isinstance(c, dict) or c.get('source') not in sources or not c.get('claim'):
            continue
        index = c.get('evidence_index')
        if type(index) is int and 0 <= index < len(passages[c['source']]):
            c['evidence'] = passages[c['source']][index]
        if c.get('evidence') and c['evidence'] in sources[c['source']]:
            verified.append(c)
    obj['claims'] = verified
    return obj


def _synthesize(topic: str, proposal: dict, reviewed: dict) -> str:
    llm = get_llm()
    claims = reviewed.get("claims", [])
    claims_txt = "\n".join(
        f"- {'待核实推断' if c.get('confidence', 0) < .85 else '来源披露'}：{c['claim']}（[来源]({c['source']})）"
        for c in claims) or "（无可靠结论）"
    if not claims:
        return f"# {topic}\n\n## 资料不足\n\n本次没有得到可核验的结论。请补充公开资料网址、调整研究范围，或稍后重试。\n\n资料抓取和审查过程保留在任务记录中；这里不补写未经验证的事实。"
    questions = "、".join(proposal.get("questions", [])) or topic
    return llm.chat([
        {"role": "system", "content": "把复核后的结论写成一份结构化中文报告。"
         "要求：每条事实性结论后用 [来源](URL) 的 Markdown 链接标注；低置信推断单独标明，不能写成已证实事实。"
         "区分来源的自述与独立验证；结论不足时如实说明；不要引入资料外的新事实。"},
        {"role": "user", "content": f"主题：{topic}\n要回答的问题：{questions}\n\n"
         f"已复核结论：\n{claims_txt}\n\n冲突：{json.dumps(reviewed.get('conflicts', []), ensure_ascii=False)}\n\n"
         "输出 Markdown 报告：# 标题 / ## 核心结论 / ## 分维度分析 / ## 冲突与不确定性 / ## 来源列表"},
    ], tier="pro", max_tokens=4000)["content"]


def prepare_research(conn, topic, emit=None, rid=None):
    rid = rid or repo.create_research(conn, topic)
    conn.commit()
    emit = emit or (lambda kind, data: None)
    emit("progress", {"message": "预检索并生成研究提案"})
    seed = direct_sources(topic) or websearch(topic, n=5)
    seed = list({s['url']: s for s in seed}.values())
    for hit in seed:
        hit['excerpt'] = fetch_text(hit['url'], max_chars=2400)
    proposal = _proposal(topic, seed)
    proposal["seed_sources"] = seed
    repo.update_research(conn, rid, status="proposed", proposal=json.dumps(proposal, ensure_ascii=False))
    conn.commit()
    return {"id": rid, "topic": topic, "status": "proposed", "proposal": proposal,
            "sources": seed, "note": "预检索无结果，执行时仍将尝试检索" if not seed else "请确认研究范围"}


def run_research(conn, topic: str, rid=None, emit=None) -> dict:
    emit = emit or (lambda kind, data: None)
    if not rid:
        prepared = prepare_research(conn, topic)
        rid = prepared["id"]
    conn.commit()
    try:
        row = repo.get_research(conn, rid)
        proposal = json.loads(row["proposal"])
        repo.update_research(conn, rid, status="running",
                             proposal=json.dumps(proposal, ensure_ascii=False))
        conn.commit()

        results = json.loads(row["draft"]) if row["draft"] else []
        completed = {r["scope"] for r in results}
        todo = [s for s in proposal["subtasks"] if s["scope"] not in completed]
        emit("progress", {"message": f"并行调研 {len(todo)} 个方向，已恢复 {len(results)} 份草稿"})
        with ThreadPoolExecutor(max_workers=3) as ex:
            for result in ex.map(lambda s: _run_subtask(topic, s, seed=proposal.get('seed_sources')), todo):
                results.append(result)
                repo.update_research(conn, rid, draft=json.dumps(results, ensure_ascii=False))
                conn.commit()
                emit("progress", {"message": f"已完成：{result['scope']}"})

        emit("progress", {"message": "对照来源原文审查结论"})
        reviewed = _review(topic, results)
        emit("progress", {"message": "整理报告与证据"})
        report = _synthesize(topic, proposal, reviewed)
        sources = sorted({c["source"] for c in reviewed.get("claims", []) if c.get("source")})
        thin = len(reviewed.get("claims", [])) < 3 or len(report) < 400

        repo.update_research(conn, rid, status="done", report_md=report,
                             sources=json.dumps(sources, ensure_ascii=False))
        conn.commit()
        first_para = report.split("\n\n")[1] if "\n\n" in report else report[:200]
        return {"id": rid, "summary": clamp(first_para.replace("#", "").strip(), 200),
                "report_md": report, "sources": sources, "thin": thin,
                "evidence": reviewed.get("claims", []),
                "chart": [{"scope": r["scope"], "sources": len(r["sources"]), "claims": len(r["claims"])} for r in results]}
    except Exception as e:  # noqa: BLE001
        repo.update_research(conn, rid, status="failed", report_md=str(e))
        conn.commit()
        raise
