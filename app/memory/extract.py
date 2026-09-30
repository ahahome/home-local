"""信息抽取：把任意文本（消息/截图转写/文档/网页）抽成结构化 patch（FR-MEM-02）。

输出统一为带 confidence 与 evidence 的 patch 结构，交由 apply.py 分档落库。

SL1 评测暴露的三类缺陷，处理方式不一样——记在这里免得后人以为都靠 prompt：
- A 承诺误判：SYSTEM 第 5 条给正/反例降概率，_is_pleasantry() 在代码里兜底拦截。
  （只改 prompt 不够：实测仍会低置信地猜出「以后常见面」。）
- B evidence 拼接：_evidence() 强制 evidence 必须是原文中的连续片段。纯代码保证。
- C 字段归位（薪资误入账本、城市误入 company）：只靠 SYSTEM 的归位规则，没有代码兜底。
  归位是语义问题，硬规则容易误伤，所以留在 prompt 层，靠评测监控。
"""
from __future__ import annotations

import re
from typing import Any

from app.llm import get_llm
from app.util import flatten_for_match

SYSTEM = """你是一个信息抽取器。从文本中抽取关于「人」的事实，只输出 JSON。

硬规则：
1. 只有能确定指向一个真实人物的名字才抽取。"我对象""那个客户""前台"这类角色词不算人名；无法确定是谁的称呼不算。
2. 每条事实必须带 evidence：从原文【逐字、连续】截取的依据片段。不要改写；
   不要用「；」把不相邻的两处原文拼成一条；如果找不到一整段连续的原文，就把 evidence 留空。
3. confidence 校准：明确陈述=0.9以上；合理推断=0.6~0.8；猜测=0.6以下。涉及金钱从严，不确定给 0.75 以下。
4. 相对日期（"下周三""前两天"）基于给出的今天日期换算成 YYYY-MM-DD；无法确定填 null。
5. 承诺 = 具体事项 + 可执行动作（通常还带时间）。下列一律不算承诺，无论它出现在单句、
   多轮对话还是聊天记录片段里：「改天吃饭」「回头联系」「以后常见面」「以后常联系」
   「有机会聚聚」「保持联系」「有空一起…」。这些是社交客套，只是寒暄。
   - 不算：「改天一起吃个饭」「以后咱们能常见面」「回头我联系你」
   - 算：  「我答应下周三之前把报价单发给他」「9/8 之前给你确认」
   不要因为对话轮次变多、语气变正式就放宽标准——语境不改变一句话是不是客套。
6. 关于"用户自己"的目标、偏好、状态，不要放进 people，放进 my_notes。
7. 只输出一个 JSON 对象。不要解释、不要 markdown 代码围栏。

字段归位规则（这三类是最常见的放错位置）：
- ledger 只记「你和某人之间真实发生的收付款」：转账、借款、还款、份子钱、代付、报销。
  判断依据是「钱有没有在你和这个人之间移动」。某人的薪资/年薪/奖金/收入、公司规模/营收/
  融资额、房价、报价、商品单价——钱没在这个人和你之间移动，都不是 ledger，应记为 facts 或对应字段。
- 地点：「搬到/住在/去了/从…来 + 城市名」→ city 字段，不要放进 company。
  特别注意「公司搬到杭州了」这种情况：说的是公司在哪，但「杭州」是 city，不是 company。
  只有公司/机构名（如「恒远进出口」「宁德时代」）才放 company。
- 「他是宁波人」是 city；「他女儿在加拿大读书」是家属信息，不是他的 city。

别把「事实」也一起丢掉（上面否掉的是客套和角色词，不是事实本身）：
- 某人单位裁员/搬迁/出事、某人换工作/升职/结婚/生病/住院/生孩子/搬家——这些是明确发生过的事，
  即使不涉及金钱、也不是承诺，也必须抽成一条 facts。
- 每抽到一个人，至少给他留下「为什么值得记」的一条：一个字段、一条 facts 或一条 value_read 都可以；
  但不要为了凑条数硬造内容。

同一字段只给一条值：
- 一次抽取里，同一个 field 只出现一次。如果同时出现城市和更细的区/街道
  （如「杭州」与「滨江」「钱江新城」「浦东」），city 只取城市层级（杭州），区/街道不要再单独给一条 city。

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
      "value_read": {"text":"对此人特点/价值的判断","confidence":0.7,"evidence":"原文中的一段连续片段"}
    }
  ],
  "my_notes": [{"text":"关于用户自己的事实","kind":"semantic|episodic","confidence":0.9}]
}"""

# 切分 evidence 用：模型最常见的拼接分隔符
_EV_SPLIT = re.compile(r"[，。！？；：、\n\r]+")

# 社交客套（缺陷 A 的代码兜底）。
# 提示词第 5 条已经写了「客套不是承诺」，但实测模型偶尔仍会低置信地猜出一条
# （例如 0.4 分的「以后在杭州能常见面」）。统计上它从「高置信断言」降成了
# 「待确认」，但客套就是客套，跟置信度无关——所以在代码里直接拦掉，不交给模型自觉。
_PLEASANTRY = re.compile(
    r"(改天|哪天|有空|回头|下次|以后|日后|有时间)"
    r".{0,8}"
    r"(吃饭|吃个饭|吃|聚|聊|见面|见|喝|约|坐|玩|联系)"
    r"|(保持|经常|常常|常)联系"
)
# 但「下次见面把合同签了」「以后每周五发报表」是真承诺——带具体动作/时间的不算客套。
# 注意别把数字写得太宽：「一起」的一不能算具体量，所以只认「数词+单位」。
_CONCRETE = re.compile(
    r"把|将|[0-9０-９]"
    r"|[一二三四五六七八九十百千万]+(?:块|元|钱|万|亿|个|天|周|月|日|号|点|次|份|张|件|条)"
    r"|之前|以前|周[一二三四五六日天]|月|号|点前|以后.{0,3}(?:每|第)"
)


def _f(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def _s(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _is_pleasantry(content: str) -> bool:
    """这句话是不是社交客套（「改天吃饭」「以后常见面」这类寒暄）。

    命中客套模式、且没有任何具体动作/时间标记时才算。带「把…给…」「周X」「号」「点前」
    这种可执行细节的，即使含「以后」「下次」也照常当承诺保留。
    """
    text = _s(content)
    if not text or not _PLEASANTRY.search(text):
        return False
    return not _CONCRETE.search(text)


def _evidence(ev: Any, src_flat: str) -> str:
    """把模型给的 evidence 收敛成「原文中的一段连续片段」。

    模型偶尔会把不相邻的两处原文用「；」拼成一条 evidence，这违反「逐字截取」，
    评测里会表现为 evidence 不在原文（缺陷 B）。这里做两级兜底：
    1. 整条能对上原文 → 原样保留；
    2. 否则按标点切开，取第一段确实逐字出现的 → 保留该段（仍是连续片段）；
    3. 都对不上 → 返回空串：宁可这条没有凭证，也不留一条伪造的凭证。

    注意这里【只】处理凭证，不动 confidence。凭证不可用 ≠ 事实不可信，
    悄悄改 confidence 会掩盖模型自身的校准误差，那是另一条独立规则该管的事。
    """
    ev = _s(ev)
    if not ev or not src_flat:
        return ev
    if flatten_for_match(ev) in src_flat:
        return ev
    for part in _EV_SPLIT.split(ev):
        part = part.strip()
        if part and flatten_for_match(part) in src_flat:
            return part
    return ""


def sanitize(obj: Any, source: str = "") -> dict:
    """把模型输出规整成稳定的 patch 结构，金额统一为分。

    source：原文。给定时会强制 evidence 必须是原文里的连续片段（缺陷 B 的兜底）。
    """
    out: dict[str, Any] = {"people": [], "my_notes": []}
    if not isinstance(obj, dict):
        return out
    src_flat = flatten_for_match(source) if source else ""
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
                    "confidence": _f(f.get("confidence")),
                    "evidence": _evidence(f.get("evidence"), src_flat),
                })
        for a in p.get("aliases") or []:
            if isinstance(a, dict) and _s(a.get("alias")):
                person["aliases"].append({"alias": _s(a["alias"]), "confidence": _f(a.get("confidence"))})
        for r in p.get("relations") or []:
            if isinstance(r, dict) and _s(r.get("to")):
                person["relations"].append({
                    "to": _s(r["to"]), "kind": _s(r.get("kind")) or "know",
                    "confidence": _f(r.get("confidence")),
                    "evidence": _evidence(r.get("evidence"), src_flat),
                })
        for fa in p.get("facts") or []:
            if isinstance(fa, dict) and _s(fa.get("text")):
                person["facts"].append({
                    "text": _s(fa["text"]), "confidence": _f(fa.get("confidence"), 0.85),
                    "evidence": _evidence(fa.get("evidence"), src_flat),
                })
        for pr in p.get("promises") or []:
            if not isinstance(pr, dict) or not _s(pr.get("content")):
                continue
            content = _s(pr["content"])
            if _is_pleasantry(content):
                continue  # 客套不算承诺，直接丢掉（缺陷 A 的代码兜底）
            person["promises"].append({
                "who": pr.get("who") if pr.get("who") in ("me", "them") else "me",
                "content": content, "due": _s(pr.get("due")) or None,
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
                "evidence": _evidence(vr.get("evidence"), src_flat),
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
    return sanitize(llm.chat_json(messages, tier="mid", max_tokens=4000), source=text)
