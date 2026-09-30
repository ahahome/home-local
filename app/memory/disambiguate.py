"""身份消歧（FR-ID-04）：四路召回 → 加权打分 → 三档决策。

核心不变量：永不静默合并。分数高 ≠ 可以自动合并，还要看领先差值。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.store import repo


@dataclass
class Candidate:
    person_id: str
    display_name: str
    score: float
    via: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    kind: str                      # bound | confirm | new
    person_id: str | None = None
    display_name: str | None = None
    score: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def _alias_match(name: str, aliases: list[str]) -> float:
    """别名匹配强度：精确=1.0，包含=0.8，编辑距离≤1=0.7，否则 0。"""
    best = 0.0
    for al in aliases:
        if al == name:
            return 1.0
        if name and al and (name in al or al in name):
            best = max(best, 0.8)
        elif 0 < _levenshtein(name, al) <= 1:
            best = max(best, 0.7)
    return best


def _context_overlap(name_ctx_tokens: set[str], field_values: list[str]) -> float:
    """上下文里是否出现了此人已知字段的值（如公司名）→ 字段自洽度。"""
    if not name_ctx_tokens or not field_values:
        return 0.0
    hit = 0
    for v in field_values:
        if not v:
            continue
        vt = set(v)
        if vt and vt & name_ctx_tokens:
            hit += 1
    return min(1.0, hit / max(1, len(field_values)))


def recall(conn, name: str) -> dict[str, Candidate]:
    """四路召回，任一命中即入池。"""
    cands: dict[str, Candidate] = {}

    def put(pid: str, dname: str, via: str):
        c = cands.get(pid)
        if c is None:
            cands[pid] = Candidate(pid, dname, 0.0, [via])
        elif via not in c.via:
            c.via.append(via)

    for row in repo.alias_rows(conn):
        pid, dname, al = row["person_id"], row["display_name"], row["alias"]
        if al == name:
            put(pid, dname, "alias_exact")
        elif name and al and (name in al or al in name):
            put(pid, dname, "alias_partial")
        elif 0 < _levenshtein(name, al) <= 1:
            put(pid, dname, "alias_fuzzy")
    return cands


def score_candidates(conn, name: str, cands: dict[str, Candidate],
                     context_tokens: set[str] | None = None,
                     cooccur_ids: set[str] | None = None,
                     name_sim: dict[str, float] | None = None) -> list[Candidate]:
    context_tokens = context_tokens or set()
    cooccur_ids = cooccur_ids or set()
    name_sim = name_sim or {}
    for c in cands.values():
        aliases = repo.aliases_of(conn, c.person_id)
        am = _alias_match(name, aliases)
        cs = name_sim.get(c.person_id, 0.0)
        co = 1.0 if c.person_id in cooccur_ids else 0.0
        cur = repo.current_fields(conn, c.person_id)
        fc = _context_overlap(context_tokens, [v["value"] for v in cur.values() if v["value"]])
        # 别名精确命中是强信号：直接给 0.9 基线。
        # 单个精确命中 → bound；多个并列精确命中 → gap 规则拦下 → confirm（AC-15）
        base = 0.9 if am == 1.0 else 0.45 * am
        c.score = min(1.0, base + 0.25 * cs + 0.20 * co + 0.10 * fc)
    return sorted(cands.values(), key=lambda c: c.score, reverse=True)


def resolve(conn, name: str, context_tokens: set[str] | None = None,
            cooccur_ids: set[str] | None = None,
            name_sim: dict[str, float] | None = None) -> Resolution:
    s = get_settings()
    cands = recall(conn, name)
    if not cands:
        return Resolution(kind="new")
    scored = score_candidates(conn, name, cands, context_tokens, cooccur_ids, name_sim)
    best = scored[0]

    # 领先差值不足 = 有并列候选，即使绝对分够也不自动绑定（3 个"小张"场景）
    ambiguous = len(scored) > 1 and (scored[0].score - scored[1].score) < s.disambiguate_gap
    if best.score >= s.disambiguate_auto and not ambiguous:
        return Resolution(kind="bound", person_id=best.person_id,
                          display_name=best.display_name, score=best.score, candidates=scored)
    if best.score >= s.disambiguate_confirm:
        return Resolution(kind="confirm", person_id=best.person_id,
                          display_name=best.display_name, score=best.score, candidates=scored)
    return Resolution(kind="new", candidates=scored)
