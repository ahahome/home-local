"""数据访问层：所有表的 CRUD。只在这里写 SQL。

不变量（对应 PRD）：
- 人物删除是软删 + 墓碑，永不物理删除（FR-ID-07 的"不复活"由写入侧 status/deleted_at 检查保证）
- person_field 双时态：upsert = 关闭旧版本 + 追加新版本（FR-ID-03）
- interaction / ledger / applied_patch 都有幂等键，重放不产生重复（FR-ING-06）
- 待确认的条目不进入任何汇总（FR-MEM-03 / FR-LED-03）
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.util import new_id, now_ms, sha256_text, today_local


# ---------------- person ----------------

def create_person(conn, name: str, status: str = "unconfirmed", alias_source: str = "chat",
                  alias_conf: float = 1.0) -> str:
    pid = new_id()
    t = now_ms()
    conn.execute(
        "INSERT INTO person(id,display_name,status,created_at,updated_at) VALUES(?,?,?,?,?)",
        (pid, name, status, t, t),
    )
    add_alias(conn, pid, name, alias_source, alias_conf)
    return pid


def get_person(conn, pid: str):
    return conn.execute("SELECT * FROM person WHERE id=?", (pid,)).fetchone()


def get_person_by_name(conn, name: str):
    return conn.execute(
        "SELECT p.* FROM person p JOIN person_alias a ON a.person_id=p.id "
        "WHERE a.alias=? AND p.deleted_at IS NULL AND p.status!='merged' LIMIT 1", (name,),
    ).fetchone()


def all_active_persons(conn) -> list:
    return conn.execute(
        "SELECT * FROM person WHERE deleted_at IS NULL AND status!='merged' ORDER BY importance DESC"
    ).fetchall()


def add_alias(conn, person_id: str, alias: str, source: str, confidence: float = 1.0) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO person_alias(person_id,alias,source,confidence,created_at) VALUES(?,?,?,?,?)",
        (person_id, alias, source, confidence, now_ms()),
    )


def aliases_of(conn, person_id: str) -> list[str]:
    """按 alias 去重（同一别名可能来自多个来源）。"""
    rows = conn.execute("SELECT alias FROM person_alias WHERE person_id=?", (person_id,)).fetchall()
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        if r["alias"] not in seen:
            seen.add(r["alias"])
            out.append(r["alias"])
    return out


def alias_rows(conn) -> list:
    return conn.execute(
        "SELECT a.*, p.display_name, p.deleted_at, p.status FROM person_alias a "
        "JOIN person p ON p.id=a.person_id WHERE p.deleted_at IS NULL AND p.status!='merged'"
    ).fetchall()


def set_person_status(conn, pid: str, status: str) -> None:
    conn.execute("UPDATE person SET status=?, updated_at=? WHERE id=?", (status, now_ms(), pid))


def touch_person(conn, pid: str) -> None:
    conn.execute("UPDATE person SET updated_at=? WHERE id=?", (now_ms(), pid))


def soft_delete_person(conn, pid: str) -> None:
    t = now_ms()
    conn.execute("UPDATE person SET deleted_at=? WHERE id=?", (t, pid))
    conn.execute("UPDATE memory_item SET deleted_at=? WHERE person_id=?", (t, pid))


def update_value_read(conn, pid: str, value_read: str, evidence: str | None) -> None:
    conn.execute("UPDATE person SET value_read=?, value_read_evidence=?, updated_at=? WHERE id=?",
                 (value_read, evidence, now_ms(), pid))


def recompute_importance(conn, pid: str) -> None:
    """重要度 = 互动数（log）与最近度的合成，供检索排序用。"""
    import math
    cnt = conn.execute(
        "SELECT COUNT(*) c FROM interaction_person WHERE person_id=?", (pid,)).fetchone()["c"]
    last = conn.execute(
        "SELECT MAX(i.ts) m FROM interaction i JOIN interaction_person ip ON ip.interaction_id=i.id "
        "WHERE ip.person_id=?", (pid,)).fetchone()["m"]
    rec = 0.0
    if last:
        days = (now_ms() - last) / 86400000.0
        rec = math.exp(-days / 30.0)
    imp = min(1.0, 0.5 * math.log1p(cnt) / 3.0 + 0.5 * rec)
    conn.execute("UPDATE person SET importance=? WHERE id=?", (imp, pid))


# ---------------- person_field（双时态） ----------------

def current_fields(conn, person_id: str) -> dict[str, dict]:
    rows = conn.execute(
        "SELECT * FROM person_field WHERE person_id=? AND valid_to IS NULL ORDER BY field",
        (person_id,)).fetchall()
    return {r["field"]: dict(r) for r in rows}


def field_at(conn, person_id: str, field: str, ts: int):
    return conn.execute(
        "SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from<=? "
        "AND (valid_to IS NULL OR valid_to>?) ORDER BY valid_from DESC LIMIT 1",
        (person_id, field, ts, ts)).fetchone()


def upsert_field(conn, person_id: str, field: str, value: str, confidence: float,
                 evidence_id: str | None, valid_from: int | None = None) -> None:
    """关闭旧版本 + 追加新版本。值相同则不产生新版本（幂等）。"""
    now = now_ms()
    vf = valid_from or now
    cur = conn.execute(
        "SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_to IS NULL",
        (person_id, field)).fetchone()
    if cur and (cur["value"] or "") == (value or ""):
        # 相同值：只提升置信度/证据，不产生新版本
        if confidence > cur["confidence"]:
            conn.execute(
                "UPDATE person_field SET confidence=?, evidence_id=COALESCE(?,evidence_id) "
                "WHERE person_id=? AND field=? AND valid_to IS NULL",
                (confidence, evidence_id, person_id, field))
        return
    if cur:
        # 防 PK 冲突：新版本的 valid_from 必须严格大于旧版本（同毫秒写入时递增）
        if vf <= cur["valid_from"]:
            vf = cur["valid_from"] + 1
        conn.execute(
            "UPDATE person_field SET valid_to=? WHERE person_id=? AND field=? AND valid_to IS NULL",
            (vf, person_id, field))
    conn.execute(
        "INSERT INTO person_field(person_id,field,value,confidence,evidence_id,valid_from,valid_to,recorded_at)"
        " VALUES(?,?,?,?,?,?,NULL,?)",
        (person_id, field, value, confidence, evidence_id, vf, now))


def field_history(conn, person_id: str, field: str) -> list:
    return conn.execute(
        "SELECT * FROM person_field WHERE person_id=? AND field=? ORDER BY valid_from",
        (person_id, field)).fetchall()


# ---------------- interaction ----------------

def insert_interaction(conn, ts: int, channel: str, content: str, source_ref: str | None,
                       person_ids: list[tuple[str, str]] | None = None) -> tuple[str, bool]:
    """返回 (id, created)。content_hash 去重：已存在则返回 (existing_id, False)。"""
    ch = sha256_text(f"{channel}|{content.strip()}")
    exist = conn.execute("SELECT id FROM interaction WHERE content_hash=?", (ch,)).fetchone()
    if exist:
        return exist["id"], False
    iid = new_id()
    conn.execute(
        "INSERT INTO interaction(id,ts,channel,source_ref,content,content_hash,created_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (iid, ts, channel, source_ref, content, ch, now_ms()))
    for pid, role in (person_ids or []):
        conn.execute(
            "INSERT OR IGNORE INTO interaction_person(interaction_id,person_id,role) VALUES(?,?,?)",
            (iid, pid, role))
    from app.memory.fts import tokenize  # 延迟导入避免循环
    conn.execute(
        "INSERT INTO memory_fts(tokens,owner_type,owner_id) VALUES(?,?,?)",
        (tokenize(content), "interaction", iid))
    return iid, True


def recent_interactions(conn, person_id: str, limit: int = 5) -> list:
    return conn.execute(
        "SELECT i.* FROM interaction i JOIN interaction_person ip ON ip.interaction_id=i.id "
        "WHERE ip.person_id=? ORDER BY i.ts DESC LIMIT ?", (person_id, limit)).fetchall()


def interactions_since(conn, since_ms: int) -> list:
    return conn.execute(
        "SELECT * FROM interaction WHERE ts>=? ORDER BY ts", (since_ms,)).fetchall()


def interaction_person_names(conn, interaction_id: str) -> list[str]:
    return [r["display_name"] for r in conn.execute(
        "SELECT p.display_name FROM interaction_person ip JOIN person p ON p.id=ip.person_id "
        "WHERE ip.interaction_id=?", (interaction_id,)).fetchall()]


# ---------------- memory_item ----------------

def add_memory(conn, kind: str, text: str, person_id: str | None = None, confidence: float = 1.0,
               evidence_id: str | None = None, ts: int | None = None) -> str:
    mid = new_id()
    t = ts or now_ms()
    conn.execute(
        "INSERT INTO memory_item(id,person_id,kind,text,confidence,evidence_id,ts,created_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (mid, person_id, kind, text, confidence, evidence_id, t, now_ms()))
    from app.memory.fts import tokenize
    conn.execute(
        "INSERT INTO memory_fts(tokens,owner_type,owner_id) VALUES(?,?,?)",
        (tokenize(text), "memory", mid))
    return mid


def memories_of(conn, person_id: str, kinds: list[str] | None = None, limit: int = 50) -> list:
    if kinds:
        ph = ",".join("?" * len(kinds))
        return conn.execute(
            f"SELECT * FROM memory_item WHERE person_id=? AND deleted_at IS NULL AND kind IN ({ph})"
            f" ORDER BY ts DESC LIMIT ?", (person_id, *kinds, limit)).fetchall()
    return conn.execute(
        "SELECT * FROM memory_item WHERE person_id=? AND deleted_at IS NULL ORDER BY ts DESC LIMIT ?",
        (person_id, limit)).fetchall()


def memories_since(conn, since_ms: int, kinds: list[str] | None = None) -> list:
    base = "SELECT * FROM memory_item WHERE deleted_at IS NULL AND ts>=?"
    params: list[Any] = [since_ms]
    if kinds:
        ph = ",".join("?" * len(kinds))
        base += f" AND kind IN ({ph})"
        params += kinds
    return conn.execute(base + " ORDER BY ts", params).fetchall()


def my_profile_memories(conn) -> list:
    """用户自己的目标/偏好/个人档案 —— FR-MEM-05：每轮全量读取。"""
    return conn.execute(
        "SELECT * FROM memory_item WHERE person_id IS NULL AND deleted_at IS NULL "
        "AND kind IN ('semantic','episodic') ORDER BY ts DESC LIMIT 200").fetchall()


# ---------------- relation_edge ----------------

def upsert_edge(conn, from_p: str, to_p: str, kind: str, strength: float = 0.5,
                evidence_id: str | None = None) -> None:
    conn.execute(
        "INSERT INTO relation_edge(id,from_person,to_person,kind,strength,evidence_id,created_at)"
        " VALUES(?,?,?,?,?,?,?) ON CONFLICT(from_person,to_person,kind) DO UPDATE SET"
        " strength=MAX(strength,excluded.strength)",
        (new_id(), from_p, to_p, kind, strength, evidence_id, now_ms()))


def edges_all(conn) -> list:
    return conn.execute("SELECT * FROM relation_edge").fetchall()


# ---------------- promise ----------------

def add_promise(conn, who: str, content: str, person_id: str | None = None, due_at: int | None = None,
                evidence_id: str | None = None) -> str:
    pid = new_id()
    t = now_ms()
    conn.execute(
        "INSERT INTO promise(id,person_id,who,content,due_at,status,evidence_id,created_at,updated_at)"
        " VALUES(?,?,?,?,?,'open',?,?,?)",
        (pid, person_id, who, content, due_at, evidence_id, t, t))
    return pid


def open_promises(conn, person_id: str | None = None) -> list:
    if person_id:
        return conn.execute(
            "SELECT * FROM promise WHERE status='open' AND person_id=? ORDER BY due_at IS NULL, due_at",
            (person_id,)).fetchall()
    return conn.execute(
        "SELECT p.*, pe.display_name FROM promise p LEFT JOIN person pe ON pe.id=p.person_id "
        "WHERE p.status='open' ORDER BY p.due_at IS NULL, p.due_at").fetchall()


def set_promise_status(conn, pid: str, status: str) -> None:
    conn.execute("UPDATE promise SET status=?, updated_at=? WHERE id=?", (status, now_ms(), pid))


# ---------------- ledger ----------------

def add_ledger(conn, direction: str, amount_cents: int, person_id: str | None, note: str | None,
               occurred_at: int, status: str, dedupe_hash: str, evidence_id: str | None = None) -> tuple[str, bool]:
    exist = conn.execute("SELECT id FROM ledger_entry WHERE dedupe_hash=?", (dedupe_hash,)).fetchone()
    if exist:
        return exist["id"], False
    lid = new_id()
    conn.execute(
        "INSERT INTO ledger_entry(id,person_id,direction,amount_cents,note,occurred_at,status,dedupe_hash,evidence_id,created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (lid, person_id, direction, amount_cents, note, occurred_at, status, dedupe_hash, evidence_id, now_ms()))
    return lid, True


def find_similar_transfer(conn, person_id: str | None, amount_cents: int, since_ms: int) -> list:
    """去重窗口外的相似历史转账，用于走确认流程（FR-LED-02）。"""
    return conn.execute(
        "SELECT * FROM ledger_entry WHERE amount_cents=? AND occurred_at<? AND status!='void'"
        " AND (person_id IS ? OR person_id=?) ORDER BY occurred_at DESC LIMIT 3",
        (amount_cents, since_ms, person_id, person_id)).fetchall()


def ledger_totals(conn, person_id: str | None = None) -> dict:
    """待确认与作废不计入总额（FR-LED-03）。"""
    where = "status='confirmed'"
    params: list[Any] = []
    if person_id:
        where += " AND person_id=?"
        params.append(person_id)
    row = conn.execute(
        f"SELECT COALESCE(SUM(CASE WHEN direction='in' THEN amount_cents ELSE 0 END),0) tin,"
        f" COALESCE(SUM(CASE WHEN direction='out' THEN amount_cents ELSE 0 END),0) tout,"
        f" COUNT(*) n FROM ledger_entry WHERE {where}", params).fetchone()
    return {"in_cents": row["tin"], "out_cents": row["tout"], "count": row["n"]}


def ledger_entries(conn, person_id: str | None = None, include_pending: bool = True) -> list:
    sql = "SELECT l.*, p.display_name FROM ledger_entry l LEFT JOIN person p ON p.id=l.person_id"
    conds, params = ["l.status!='void'"], []
    if person_id:
        conds.append("l.person_id=?")
        params.append(person_id)
    if not include_pending:
        conds.append("l.status='confirmed'")
    return conn.execute(sql + " WHERE " + " AND ".join(conds) + " ORDER BY l.occurred_at DESC LIMIT 100",
                        params).fetchall()


# ---------------- skill ----------------

def save_skill(conn, name: str, body: str, person_id: str | None = None, trigger_hint: str | None = None,
               style_notes: str | None = None) -> tuple[str, bool]:
    cur = conn.execute("SELECT id FROM skill WHERE name=?", (name,)).fetchone()
    t = now_ms()
    if cur:
        conn.execute(
            "UPDATE skill SET body=?, person_id=?, trigger_hint=?, style_notes=?, updated_at=? WHERE id=?",
            (body, person_id, trigger_hint, style_notes, t, cur["id"]))
        return cur["id"], False
    sid = new_id()
    conn.execute(
        "INSERT INTO skill(id,name,person_id,trigger_hint,style_notes,body,created_at,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (sid, name, person_id, trigger_hint, style_notes, body, t, t))
    return sid, True


def get_skill(conn, name: str):
    return conn.execute("SELECT * FROM skill WHERE name=?", (name,)).fetchone()


def list_skills(conn) -> list:
    return conn.execute(
        "SELECT s.*, p.display_name FROM skill s LEFT JOIN person p ON p.id=s.person_id ORDER BY s.updated_at DESC"
    ).fetchall()


# ---------------- evidence ----------------

def add_evidence(conn, kind: str, blob_hash: str | None = None, source_ref: str | None = None,
                 excerpt: str | None = None) -> str:
    eid = new_id()
    conn.execute(
        "INSERT INTO evidence(id,kind,blob_hash,source_ref,excerpt,created_at) VALUES(?,?,?,?,?,?)",
        (eid, kind, blob_hash, source_ref, excerpt, now_ms()))
    return eid


# ---------------- ingest_job ----------------

def create_job(conn, kind: str, payload: str, idempotency_key: str | None = None) -> tuple[str, bool]:
    if idempotency_key:
        exist = conn.execute("SELECT id FROM ingest_job WHERE idempotency_key=?", (idempotency_key,)).fetchone()
        if exist:
            return exist["id"], False
    jid = new_id()
    t = now_ms()
    conn.execute(
        "INSERT INTO ingest_job(id,kind,status,payload,idempotency_key,created_at,updated_at)"
        " VALUES(?,?,'queued',?,?,?,?)", (jid, kind, payload, idempotency_key, t, t))
    return jid, True


def update_job(conn, jid: str, status: str, result: str | None = None, error: str | None = None) -> None:
    conn.execute(
        "UPDATE ingest_job SET status=?, result=COALESCE(?,result), error=?, attempts=attempts+1, updated_at=?"
        " WHERE id=?", (status, result, error, now_ms(), jid))


def get_job(conn, jid: str):
    return conn.execute("SELECT * FROM ingest_job WHERE id=?", (jid,)).fetchone()


# ---------------- merge / split ----------------

def add_merge_log(conn, survivor_id: str, merged_id: str, snapshot: dict) -> str:
    lid = new_id()
    conn.execute(
        "INSERT INTO merge_log(id,survivor_id,merged_id,snapshot,created_at) VALUES(?,?,?,?,?)",
        (lid, survivor_id, merged_id, json.dumps(snapshot, ensure_ascii=False), now_ms()))
    return lid


def get_merge_log(conn, lid: str):
    return conn.execute("SELECT * FROM merge_log WHERE id=?", (lid,)).fetchone()


def find_merge_log_by_merged(conn, merged_id: str):
    return conn.execute(
        "SELECT * FROM merge_log WHERE merged_id=? AND undone=0 ORDER BY created_at DESC LIMIT 1",
        (merged_id,)).fetchone()


def mark_merge_undone(conn, lid: str) -> None:
    conn.execute("UPDATE merge_log SET undone=1 WHERE id=?", (lid,))


# ---------------- applied_patch / pending ----------------

def has_patch(conn, fingerprint: str) -> bool:
    return conn.execute("SELECT 1 FROM applied_patch WHERE fingerprint=?", (fingerprint,)).fetchone() is not None


def mark_patch(conn, fingerprint: str, person_id: str | None, field: str, value_norm: str,
               source_msg_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO applied_patch(fingerprint,person_id,field,value_norm,source_msg_id,applied_at)"
        " VALUES(?,?,?,?,?,?)",
        (fingerprint, person_id, field, value_norm, source_msg_id, now_ms()))


def add_pending(conn, kind: str, payload: dict, reason: str) -> str:
    pid = new_id()
    conn.execute(
        "INSERT INTO pending_item(id,kind,payload,reason,created_at) VALUES(?,?,?,?,?)",
        (pid, kind, json.dumps(payload, ensure_ascii=False), reason, now_ms()))
    return pid


def list_pending(conn, kind: str | None = None) -> list:
    if kind:
        return conn.execute(
            "SELECT * FROM pending_item WHERE resolved_at IS NULL AND kind=? ORDER BY created_at", (kind,)).fetchall()
    return conn.execute("SELECT * FROM pending_item WHERE resolved_at IS NULL ORDER BY created_at").fetchall()


def resolve_pending(conn, pid: str, resolution: str) -> None:
    conn.execute("UPDATE pending_item SET resolved_at=?, resolution=? WHERE id=?",
                 (now_ms(), resolution, pid))


def get_pending(conn, pid: str):
    return conn.execute("SELECT * FROM pending_item WHERE id=?", (pid,)).fetchone()


# ---------------- journal ----------------

def upsert_journal(conn, day: str, text: str) -> None:
    conn.execute(
        "INSERT INTO journal(id,day,text,created_at) VALUES(?,?,?,?)"
        " ON CONFLICT(day) DO UPDATE SET text=excluded.text",
        (new_id(), day, text, now_ms()))


def get_journal(conn, day: str):
    return conn.execute("SELECT * FROM journal WHERE day=?", (day,)).fetchone()


def list_journals(conn, limit: int = 30) -> list:
    return conn.execute("SELECT * FROM journal ORDER BY day DESC LIMIT ?", (limit,)).fetchall()


# ---------------- conversation / message ----------------

def create_conversation(conn, title: str = "新对话") -> str:
    cid = new_id()
    t = now_ms()
    conn.execute("INSERT INTO conversation(id,title,created_at,updated_at) VALUES(?,?,?,?)", (cid, title, t, t))
    return cid


def list_conversations(conn) -> list:
    return conn.execute("SELECT * FROM conversation ORDER BY updated_at DESC LIMIT 100").fetchall()


def get_conversation(conn, cid: str):
    return conn.execute("SELECT * FROM conversation WHERE id=?", (cid,)).fetchone()


def set_conversation_title(conn, cid: str, title: str) -> None:
    conn.execute("UPDATE conversation SET title=?, updated_at=? WHERE id=?", (title, now_ms(), cid))


def add_message(conn, conversation_id: str, role: str, content: str | None,
                tool_calls: str | None = None) -> str:
    mid = new_id()
    t = now_ms()
    conn.execute(
        "INSERT INTO message(id,conversation_id,role,content,tool_calls,created_at) VALUES(?,?,?,?,?,?)",
        (mid, conversation_id, role, content, tool_calls, t))
    conn.execute("UPDATE conversation SET updated_at=? WHERE id=?", (t, conversation_id))
    return mid


def list_messages(conn, conversation_id: str, limit: int = 200) -> list:
    return conn.execute(
        "SELECT * FROM message WHERE conversation_id=? ORDER BY created_at LIMIT ?",
        (conversation_id, limit)).fetchall()


def last_message_ts(conn, conversation_id: str) -> int | None:
    r = conn.execute(
        "SELECT MAX(created_at) m FROM message WHERE conversation_id=?", (conversation_id,)).fetchone()
    return r["m"]


# ---------------- reminder / notification ----------------

def add_reminder(conn, fire_at: int, text: str, person_id: str | None = None,
                 promise_id: str | None = None) -> str:
    rid = new_id()
    conn.execute(
        "INSERT INTO reminder(id,person_id,promise_id,fire_at,text,status,created_at) VALUES(?,?,?,?,?,'pending',?)",
        (rid, person_id, promise_id, fire_at, text, now_ms()))
    return rid


def due_reminders(conn, now: int) -> list:
    return conn.execute(
        "SELECT * FROM reminder WHERE status='pending' AND fire_at<=? ORDER BY fire_at LIMIT 50", (now,)).fetchall()


def mark_reminder_fired(conn, rid: str) -> None:
    conn.execute("UPDATE reminder SET status='fired', fired_at=? WHERE id=?", (now_ms(), rid))


def upcoming_reminders(conn, limit: int = 50) -> list:
    return conn.execute(
        "SELECT r.*, p.display_name FROM reminder r LEFT JOIN person p ON p.id=r.person_id "
        "WHERE r.status='pending' ORDER BY r.fire_at LIMIT ?", (limit,)).fetchall()


def add_notification(conn, kind: str, text: str) -> str:
    nid = new_id()
    conn.execute("INSERT INTO notification(id,kind,text,created_at) VALUES(?,?,?,?)",
                 (nid, kind, text, now_ms()))
    return nid


def list_notifications(conn, unread_only: bool = False) -> list:
    if unread_only:
        return conn.execute("SELECT * FROM notification WHERE read=0 ORDER BY created_at DESC LIMIT 100").fetchall()
    return conn.execute("SELECT * FROM notification ORDER BY created_at DESC LIMIT 100").fetchall()


def mark_notifications_read(conn) -> None:
    conn.execute("UPDATE notification SET read=1 WHERE read=0")


# ---------------- research ----------------

def create_research(conn, topic: str) -> str:
    rid = new_id()
    t = now_ms()
    conn.execute(
        "INSERT INTO research(id,topic,status,created_at,updated_at) VALUES(?,?,'proposed',?,?)",
        (rid, topic, t, t))
    return rid


def update_research(conn, rid: str, **fields) -> None:
    fields["updated_at"] = now_ms()
    cols = ",".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE research SET {cols} WHERE id=?", (*fields.values(), rid))


def get_research(conn, rid: str):
    return conn.execute("SELECT * FROM research WHERE id=?", (rid,)).fetchone()


def list_research(conn) -> list:
    return conn.execute("SELECT * FROM research ORDER BY updated_at DESC LIMIT 50").fetchall()


# ---------------- vector ----------------

def upsert_vector(conn, owner_type: str, owner_id: str, text: str, vector: list[float]) -> None:
    import numpy as np
    arr = np.asarray(vector, dtype=np.float32)
    conn.execute(
        "INSERT INTO vector_item(id,owner_type,owner_id,text,dim,vector) VALUES(?,?,?,?,?,?)"
        " ON CONFLICT(owner_type,owner_id) DO UPDATE SET text=excluded.text, vector=excluded.vector, dim=excluded.dim",
        (new_id(), owner_type, owner_id, text, arr.size, arr.tobytes()))


def all_vectors(conn, owner_type: str) -> list:
    return conn.execute("SELECT * FROM vector_item WHERE owner_type=?", (owner_type,)).fetchall()


def delete_vector(conn, owner_type: str, owner_id: str) -> None:
    conn.execute("DELETE FROM vector_item WHERE owner_type=? AND owner_id=?", (owner_type, owner_id))


# ---------------- kv ----------------

def get_kv(conn, key: str, default: str | None = None) -> str | None:
    r = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_kv(conn, key: str, value: str) -> None:
    conn.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))


# ---------------- atlas ----------------

def atlas_graph(conn) -> dict:
    persons = [dict(p) for p in all_active_persons(conn)]
    edges = [dict(e) for e in edges_all(conn)]
    return {"nodes": persons, "edges": edges}
