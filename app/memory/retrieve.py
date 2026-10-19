"""混合检索（FR-MEM-04）：结构化 + FTS5(jieba) + 向量，融合排序 + 时间衰减 + 人物重要度。"""
from __future__ import annotations

import math

from app.agent.memory_sources import interaction_source, memory_source
from app.memory.fts import fts_match_query
from app.store import repo
from app.util import now_ms

W_VEC, W_FTS = 0.6, 0.4


_VISIBLE_INTERACTION = (
    "NOT EXISTS (SELECT 1 FROM interaction_person ip LEFT JOIN person p ON p.id=ip.person_id "
    "WHERE ip.interaction_id=i.id AND (p.id IS NULL OR p.deleted_at IS NOT NULL OR p.status='merged'))"
)


def _scope_filters(person_id, scope):
    if scope not in {'all', 'personal'}:
        raise ValueError('记忆范围须为 all 或 personal')
    if person_id is not None and scope == 'personal':
        raise ValueError('个人记忆范围不能同时指定人物')
    memory, interaction, params = repo.ACTIVE_MEMORY_WHERE, _VISIBLE_INTERACTION, ()
    if person_id is not None:
        memory += ' AND m.person_id=?'
        interaction += (' AND EXISTS (SELECT 1 FROM interaction_person scoped '
                        'WHERE scoped.interaction_id=i.id AND scoped.person_id=?)')
        params = (person_id,)
    elif scope == 'personal':
        memory += ' AND m.person_id IS NULL'
        interaction += ' AND NOT EXISTS (SELECT 1 FROM interaction_person ip WHERE ip.interaction_id=i.id)'
    return memory, interaction, params


def _fetch_items(conn, keys: list[tuple[str, str]], *, person_id=None, scope='all') -> dict[tuple[str, str], dict]:
    out = {}
    for owner_type, oid in keys:
        if owner_type == "memory":
            source = memory_source(conn, oid)
            if source:
                pid = (source['person'] or {}).get('id')
                if person_id is not None and pid != person_id or scope == 'personal' and pid is not None:
                    continue
                out[(owner_type, oid)] = {
                    "type": "memory", "id": oid, "text": source["text"], "ts": source["ts"],
                    "kind": source["kind"], "person": (source['person'] or {}).get('display_name', ''),
                    "source": source,
                }
        elif owner_type == "interaction":
            source = interaction_source(conn, oid)
            if source:
                people = source['people']
                if person_id is not None and not any(p['id'] == person_id for p in people) \
                        or scope == 'personal' and people:
                    continue
                out[(owner_type, oid)] = {
                    "type": "interaction", "id": oid, "text": source["text"], "ts": source["ts"],
                    "kind": source["kind"], "channel": source["channel"],
                    "person": "、".join(p['display_name'] for p in people), "source": source,
                }
    return out


def hybrid_search(conn, query: str, top_k: int = 8, *, person_id: str | None = None,
                  scope: str = 'all') -> list[dict]:
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('检索条数须为 1 至 100 的整数')
    memory_filter, interaction_filter, params = _scope_filters(person_id, scope)
    if person_id is not None:
        person = repo.get_person(conn, person_id)
        if not person or person['deleted_at'] is not None or person['status'] == 'merged':
            raise ValueError('人物不存在或已归档')
    vec_scores: dict[tuple[str, str], float] = {}
    fts_scores: dict[tuple[str, str], float] = {}

    # 向量通道
    try:
        from app.llm import get_llm
        from app.memory.vector import get_index
        qv = get_llm().embed([query])[0]
        idx = get_index()
        for owner_type, table, alias, clause in (
                ('memory', 'memory_item', 'm', memory_filter),
                ('interaction', 'interaction', 'i', interaction_filter)):
            allowed = {row['id'] for row in conn.execute(
                f'SELECT {alias}.id FROM {table} {alias} WHERE {clause}', params)}
            for oid, _text, sim in idx.search(conn, owner_type, qv, top_k=top_k * 2, allowed_ids=allowed):
                vec_scores[(owner_type, oid)] = max(vec_scores.get((owner_type, oid), 0.0), sim)
    except Exception:  # noqa: BLE001
        pass

    # FTS 通道
    match = fts_match_query(query)
    if match:
        try:
            rows = conn.execute(
                "SELECT owner_type, owner_id, bm25(memory_fts) s FROM memory_fts"
                " WHERE memory_fts MATCH ? AND ((owner_type='memory' AND EXISTS "
                f'(SELECT 1 FROM memory_item m WHERE m.id=memory_fts.owner_id AND {memory_filter})) '
                "OR (owner_type='interaction' AND EXISTS "
                f'(SELECT 1 FROM interaction i WHERE i.id=memory_fts.owner_id AND {interaction_filter}))) '
                'ORDER BY bm25(memory_fts) LIMIT ?',
                (match, *params, *params, top_k * 2)).fetchall()
            if rows:
                # bm25 为负值（越小越好），归一化到 (0,1]
                best = abs(rows[0]["s"]) or 1.0
                for r in rows:
                    fts_scores[(r["owner_type"], r["owner_id"])] = min(1.0, abs(r["s"]) / best)
        except Exception:  # noqa: BLE001
            pass

    keys = set(vec_scores) | set(fts_scores)
    if not query.strip() and (person_id is not None or scope == 'personal'):
        for owner_type, table, alias, clause in (
                ('memory', 'memory_item', 'm', memory_filter),
                ('interaction', 'interaction', 'i', interaction_filter)):
            for row in conn.execute(f'SELECT {alias}.id FROM {table} {alias} WHERE {clause} '
                                    f'ORDER BY {alias}.ts DESC,{alias}.id LIMIT ?', (*params, top_k * 2)):
                key = (owner_type, row['id'])
                keys.add(key)
                fts_scores[key] = 1.0
    if not keys:
        return []
    items = _fetch_items(conn, list(keys), person_id=person_id, scope=scope)
    now = now_ms()
    results = []
    for key, item in items.items():
        base = W_VEC * vec_scores.get(key, 0.0) + W_FTS * fts_scores.get(key, 0.0)
        age_days = max(0.0, (now - item["ts"]) / 86400000.0)
        recency = 0.7 + 0.3 * math.exp(-age_days / 30.0)
        score = base * recency
        item = dict(item)
        item["score"] = round(score, 4)
        results.append(item)
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ---------------- 结构化检索（工具用） ----------------

def search_people(conn, query: str, top_k: int = 8) -> list[dict]:
    """按别名模糊 + 人名向量找人。"""
    from app.memory.disambiguate import _levenshtein
    found: dict[str, dict] = {}
    for p in repo.all_active_persons(conn):
        aliases = repo.aliases_of(conn, p["id"])
        hit = any(a == query or (query and (query in a or a in query)) or _levenshtein(query, a) <= 1
                  for a in aliases)
        if hit:
            found[p["id"]] = {"person_id": p["id"], "display_name": p["display_name"],
                              "aliases": aliases, "importance": p["importance"]}
    try:
        from app.llm import get_llm
        from app.memory.vector import get_index
        qv = get_llm().embed([query])[0]
        for pid, _t, sim in get_index().search(conn, "person_name", qv, top_k=top_k, min_sim=0.55):
            if pid not in found:
                p = repo.get_person(conn, pid)
                if p and not p["deleted_at"] and p["status"] != "merged":
                    found[pid] = {"person_id": pid, "display_name": p["display_name"],
                                  "aliases": repo.aliases_of(conn, pid),
                                  "importance": p["importance"], "via_vector": round(sim, 3)}
    except Exception:  # noqa: BLE001
        pass
    out = sorted(found.values(), key=lambda x: -x["importance"])
    return out[:top_k]


def quiet_people(conn, days: int = 14, limit: int = 10) -> list[dict]:
    """最近 N 天无实际联系的人；资料记录和第三方提及不刷新联系时间。"""
    cutoff = now_ms() - days * 86400000
    out = []
    for p in repo.all_active_persons(conn):
        last = repo.last_contact(conn, p["id"])
        if last is None or last < cutoff:
            out.append({"person_id": p["id"], "display_name": p["display_name"],
                        "last_interaction": last, "last_contact": last, "importance": p["importance"]})
    out.sort(key=lambda x: -x["importance"])
    return out[:limit]
