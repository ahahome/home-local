"""信息抽取：把任意文本（消息/截图转写/文档/网页）抽成结构化 patch（FR-MEM-02）。

输出统一为带 confidence 与 evidence 的 patch 结构，交由 apply.py 分档落库。
"""
from __future__ import annotations

from typing import Any

from app.llm import get_llm

SYSTEM = """你是一个信息抽取器。从文本中抽取关于「人」的事实，只输出 JSON。

硬规则：
1. 只有能确定指向一个真实人物的名字才抽取。"我对象""那个客户""前台"这类角色词不算人名；无法确定是谁的称呼不算。
2. 每条事实必须带 evidence：从原文【逐字】截取的依据片段，不要改写。
3. confidence 校准：明确陈述=0.9以上；合理推断=0.6~0.8；猜测=0.6以下。涉及金钱从严，不确定给 0.75 以下。
4. 相对日期（"下周三""前两天"）基于给出的今天日期换算成 YYYY-MM-DD；无法确定填 null。
5. 客套话（"改天吃饭""回头联系"）没有具体内容或时间时，不是承诺。
6. 关于"用户自己"的目标、偏好、状态，不要放进 people，放进 my_notes。
7. 只输出一个 JSON 对象。不要解释、不要 markdown 代码围栏。

输出结构（没有的项给空数组/空对象）：
{
  "people": [
    {
      "name": "原文中的称呼",
      "canonical_name": "判断出的规范名（真名或最常用称呼），不确定就同 name",
      "fields": [{"field":"company|title|city|birthday|phone|email|trait.特点|need.需求","value":"...","confidence":0.9,"evidence":"原文片段"}],
      "aliases": [{"alias":"此人的其他称呼","confidence":0.8}],
      "relations": [{"to":"另一个人名","kind":"colleague|family|referred_by|friend|know","confidence":0.7,"evidence":"原文片段"}],
      "facts": [{"text":"客观发生的事，一句话，含时间与人物","confidence":0.9,"evidence":"原文片段"}],
      "promises": [{"who":"me|them","content":"承诺内容","due":"YYYY-MM-DD或null","confidence":0.9}],
      "ledger": [{"direction":"in|out","amount":123.45,"note":"说明","occurred":"YYYY-MM-DD","confidence":0.85}],
      "value_read": {"text":"对此人特点/价值的判断","confidence":0.7,"evidence":"原文片段"}
    }
  ],
  "my_notes": [{"text":"关于用户自己的事实","kind":"semantic|episodic","confidence":0.9}]
}"""


def _f(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def _s(x: Any) -> str:
    return "" if x is None else str(x).strip()


def sanitize(obj: Any) -> dict:
    """把模型输出规整成稳定的 patch 结构，金额统一为分。"""
    out: dict[str, Any] = {"people": [], "my_notes": []}
    if not isinstance(obj, dict):
        return out
    for p in obj.get("people") or []:
        if not isinstance(p, dict):
            continue
        name = _s(p.get("name"))
        if not name:
            continue
        person = {
            "name": name,
            "canonical_name": _s(p.get("canonical_name")) or name,
            "fields": [], "aliases": [], "relations": [], "facts": [],
            "promises": [], "ledger": [], "value_read": None,
        }
        for f in p.get("fields") or []:
            if isinstance(f, dict) and _s(f.get("field")) and _s(f.get("value")):
                person["fields"].append({
                    "field": _s(f["field"]), "value": _s(f["value"]),
                    "confidence": _f(f.get("confidence")), "evidence": _s(f.get("evidence")),
                })
        for a in p.get("aliases") or []:
            if isinstance(a, dict) and _s(a.get("alias")):
                person["aliases"].append({"alias": _s(a["alias"]), "confidence": _f(a.get("confidence"))})
        for r in p.get("relations") or []:
            if isinstance(r, dict) and _s(r.get("to")):
                person["relations"].append({
                    "to": _s(r["to"]), "kind": _s(r.get("kind")) or "know",
                    "confidence": _f(r.get("confidence")), "evidence": _s(r.get("evidence")),
                })
        for fa in p.get("facts") or []:
            if isinstance(fa, dict) and _s(fa.get("text")):
                person["facts"].append({
                    "text": _s(fa["text"]), "confidence": _f(fa.get("confidence"), 0.85),
                    "evidence": _s(fa.get("evidence")),
                })
        for pr in p.get("promises") or []:
            if isinstance(pr, dict) and _s(pr.get("content")):
                person["promises"].append({
                    "who": pr.get("who") if pr.get("who") in ("me", "them") else "me",
                    "content": _s(pr["content"]), "due": _s(pr.get("due")) or None,
                    "confidence": _f(pr.get("confidence"), 0.85),
                })
        for le in p.get("ledger") or []:
            if isinstance(le, dict) and le.get("amount") is not None:
                try:
                    cents = int(round(float(le["amount"]) * 100))
                except (TypeError, ValueError):
                    continue
                if cents <= 0:
                    continue
                person["ledger"].append({
                    "direction": le.get("direction") if le.get("direction") in ("in", "out") else "out",
                    "amount_cents": cents, "note": _s(le.get("note")),
                    "occurred": _s(le.get("occurred")) or None,
                    "confidence": _f(le.get("confidence"), 0.7),
                })
        vr = p.get("value_read")
        if isinstance(vr, dict) and _s(vr.get("text")):
            person["value_read"] = {
                "text": _s(vr["text"]), "confidence": _f(vr.get("confidence"), 0.7),
                "evidence": _s(vr.get("evidence")),
            }
        out["people"].append(person)

    for n in obj.get("my_notes") or []:
        if isinstance(n, dict) and _s(n.get("text")):
            out["my_notes"].append({
                "text": _s(n["text"]),
                "kind": n.get("kind") if n.get("kind") in ("semantic", "episodic") else "semantic",
                "confidence": _f(n.get("confidence"), 0.9),
            })
    return out


def extract_from_text(text: str, known_names: list[str], today: str) -> dict:
    llm = get_llm()
    names = "、".join(known_names) if known_names else "（暂无）"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"今天是 {today}。已知人物：{names}\n\n待抽取文本：\n{text}"},
    ]
    return sanitize(llm.chat_json(messages, tier="mid", max_tokens=4000))
