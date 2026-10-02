"""Atomic confirmation and discard of every pending item kind."""
import json

from app.memory import apply as apply_mod
from app.memory.writes import resolve_person, discard_key
from app.store import repo
from app.store.db import savepoint
from app.util import normalize_value, now_ms


def resolve(conn, item_id, action, person_id=None):
    if action not in {"confirm", "discard"}:
        raise ValueError("操作必须为 confirm 或 discard")
    item = repo.get_pending(conn, item_id)
    if not item or item["resolved_at"]:
        raise ValueError("待确认项不存在或已处理")
    p = json.loads(item["payload"])
    kind = item["kind"]
    with savepoint(conn, "resolve_pending"):
        if action == "discard":
            if kind == "ledger" and p.get("ledger_id"):
                conn.execute("UPDATE ledger_entry SET status='void' WHERE id=? AND status='pending'", (p["ledger_id"],))
            repo.resolve_pending(conn, item_id, "discarded")
            fingerprint = p.get('fingerprint') or p.get('dedupe_hash')
            if fingerprint and p.get('source_msg_id'):
                repo.mark_patch(conn, discard_key(kind, fingerprint, p['source_msg_id']),
                                p.get('person_id'), 'discard', '', p['source_msg_id'])
        else:
            if kind == "person_identity":
                target = resolve_person(conn, person_id=person_id) if person_id and person_id != "new" else \
                    repo.create_person(conn, p["canonical_name"], status="active")
                repo.add_alias(conn, target, p["mention"], p.get("channel", "chat"), 1.0)
                if p.get("patch"):
                    apply_mod.apply_extraction(conn, {"people": [p["patch"]]}, p["source_msg_id"],
                        p.get("source_text", p["mention"]), channel=p.get("channel", "chat"),
                        forced={p["patch"]["canonical_name"]: target}, evidence_id=p.get("evidence_id"), commit=False)
            else:
                if p.get("person_id"):
                    resolve_person(conn, person_id=p["person_id"])
                if kind == "field":
                    repo.upsert_field(conn, p["person_id"], p["field"], p["value"], 1.0,
                                      p.get("evidence_id"), apply_mod._date_ms(p.get("valid_from")))
                elif kind == "ledger":
                    if p.get("ledger_id"):
                        conn.execute("UPDATE ledger_entry SET status='confirmed' WHERE id=? AND status='pending'",
                                     (p["ledger_id"],))
                    else:
                        repo.add_ledger(conn, p["direction"], p["amount_cents"], p.get("person_id"),
                            p.get("note"), apply_mod._date_ms(p.get("occurred")) or now_ms(),
                            "confirmed", p["dedupe_hash"], p.get("evidence_id"))
                elif kind == "promise":
                    pid = repo.add_promise(conn, p["who"], p["content"], p.get("person_id"),
                                           apply_mod._date_ms(p.get("due")), p.get("evidence_id"))
                    if p.get("due"):
                        repo.add_reminder(conn, apply_mod._date_ms(p["due"]), p["content"], p.get("person_id"), pid)
                elif kind == "memory":
                    repo.add_memory(conn, p["kind"], p["text"], p.get("person_id"), 1.0, p.get("evidence_id"))
                elif kind == "relation":
                    target = resolve_person(conn, p["to"], person_id if person_id != "new" else None, create=True)
                    repo.upsert_edge(conn, p["person_id"], target, p["kind"], 1.0, p.get("evidence_id"))
                else:
                    raise ValueError("未知的待确认类型")
                if p.get("fingerprint"):
                    repo.mark_patch(conn, p["fingerprint"], p.get("person_id"), kind,
                        normalize_value(p.get("value") or p.get("text") or p.get("content") or ""), p.get("source_msg_id", item_id))
            repo.resolve_pending(conn, item_id, "confirmed")
    conn.commit()
    return {"ok": True, "resolution": "confirmed" if action == "confirm" else "discarded"}
