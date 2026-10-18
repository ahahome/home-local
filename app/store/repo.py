"""数据访问层：所有表的 CRUD。只在这里写 SQL。

不变量（对应 PRD）：
- 人物删除分两级：**归档**（archive_person：打 deleted_at，可恢复，默认动作）与
  **彻底清除**（purge_person：物理删除，含字段/别名/记忆/账本/关系边/凭证/blob）。
  「删除后不得复活」由写入侧 status/deleted_at 检查保证（FR-ID-07）；
  「删人就删别名」由 purge_person + verify_purged 保证（FR-ID-02）
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
        "INSERT INTO person(id,display_name,status,created_at,updated_at,importance) VALUES(?,?,?,?,?,0)",
        (pid, name, status, t, t),
    )
    add_alias(conn, pid, name, alias_source, alias_conf)
    return pid


def get_person(conn, pid: str):
    return conn.execute("SELECT * FROM person WHERE id=?", (pid,)).fetchone()


def get_person_by_name(conn, name: str):
    rows = conn.execute(
        "SELECT DISTINCT p.* FROM person p JOIN person_alias a ON a.person_id=p.id "
        "WHERE a.alias=? AND p.deleted_at IS NULL AND p.status!='merged' LIMIT 2", (name,),
    ).fetchall()
    return rows[0] if len(rows) == 1 else None


def all_active_persons(conn) -> list:
    return conn.execute(
        "SELECT * FROM person WHERE deleted_at IS NULL AND status!='merged' ORDER BY importance DESC"
    ).fetchall()


def archived_persons(conn) -> list:
    """已归档的人物（deleted_at 非空、且不是合并墓碑）——「已归档」面板用它提供恢复入口。

    有归档而没有恢复入口，等于「可恢复」只是一句口号。
    """
    return conn.execute(
        "SELECT * FROM person WHERE deleted_at IS NOT NULL AND status!='merged'"
        " ORDER BY deleted_at DESC").fetchall()


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


def archive_person(conn, pid: str) -> None:
    """**归档**：打 deleted_at，人物/字段/别名/记忆都留在库里，可恢复。

    P5 的语义是两级（见 eval/results/BASELINE.md 第八节的拍板）：
    - 日常点「删除」是**归档**——从所有列表/检索里消失、不复活，但可 restore；
    - 真要抹掉痕迹走 `purge_person()`（物理删除，含别名/blob）。
    这样既满足「删除后不得复活」，又给了误删后悔的余地。
    """
    t = now_ms()
    conn.execute("UPDATE person SET deleted_at=?, updated_at=? WHERE id=? AND deleted_at IS NULL", (t, t, pid))
    conn.execute("UPDATE memory_item SET deleted_at=?, revision=revision+1, updated_at=? "
                 "WHERE person_id=? AND deleted_at IS NULL", (t, t, pid))


def restore_person(conn, pid: str) -> bool:
    """把归档的人恢复回来。返回是否真的恢复了（墓碑/不存在都返回 False）。"""
    row = conn.execute("SELECT deleted_at, status FROM person WHERE id=?", (pid,)).fetchone()
    if not row or row["deleted_at"] is None or row["status"] == "merged":
        return False
    t = now_ms()
    conn.execute("UPDATE person SET deleted_at=NULL, updated_at=? WHERE id=?", (t, pid))
    conn.execute("UPDATE memory_item SET deleted_at=NULL, revision=revision+1, updated_at=? "
                 "WHERE person_id=? AND deleted_at IS NOT NULL", (t, pid))
    return True


# 旧名兼容：当年的 soft_delete 就是现在的「归档」
soft_delete_person = archive_person


# 彻底清除要清的表：(表名, 外键列)
_PURGE_LINKS = (
    ("person_field", "person_id"),
    ("person_alias", "person_id"),
    ("interaction_person", "person_id"),
    ("memory_item", "person_id"),
    ("promise", "person_id"),
    ("ledger_entry", "person_id"),
    ("reminder", "person_id"),
    ("applied_patch", "person_id"),
)


def purge_person(conn, pid: str) -> dict:
    """**彻底清除**：物理删除该人的全部痕迹（FR-ID-02）。

    与归档的区别是「真的没了」——删完之后：
    - person / person_field / person_alias / memory_item / promise / ledger_entry /
      reminder / interaction_person / applied_patch / relation_edge 里不许再有该人的行；
    - 别名行一并删除（FR-ID-02 的硬要求：删人必须删别名）；
    - 该人独占的 evidence 连同 blob 文件一起删（隐私缺口就在这里）；
    - 全库搜任何曾用名都返回 0 条。

    为什么 evidence 要判「独占」：一次 apply_extraction 只建**一条** evidence，
    同一批次里所有人都指向它。若无脑删，清一个人会把别人的凭证一起删掉。

    为什么 skill 只解绑不删：话术/流程是用户自己攒下的资产，不该跟着人一起消失。
    （单测里可用 `verify=True` 做删除后自检；不通过就返回 ok=False，让上层报
    「部分删除」而不是「已删除」。）

    返回 {ok, deleted:{表:行数}, unlinked:{表:行数}, blobs:[hash], leftovers:[...]}。
    """
    row = conn.execute("SELECT * FROM person WHERE id=?", (pid,)).fetchone()
    if not row:
        return {"ok": False, "deleted": {}, "unlinked": {}, "blobs": [],
                "leftovers": ["person 不存在"]}
    if row["status"] == "merged":
        return {"ok": False, "deleted": {}, "unlinked": {}, "blobs": [],
                "leftovers": ["该人物是合并后的墓碑，行数据已迁到存活方，请清除存活方"]}

    aliases = [r["alias"] for r in conn.execute(
        "SELECT DISTINCT alias FROM person_alias WHERE person_id=?", (pid,)).fetchall()]

    # 先取要清理的 id 集合（删掉行之后就查不到了）
    memory_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM memory_item WHERE person_id=?", (pid,)).fetchall()]
    ev_ids = _evidence_ids_of(conn, pid)
    exclusive_interactions = [r[0] for r in conn.execute(
        "SELECT interaction_id FROM interaction_person WHERE person_id=? AND NOT EXISTS "
        "(SELECT 1 FROM interaction_person other WHERE other.interaction_id=interaction_person.interaction_id "
        "AND other.person_id!=?)", (pid, pid))]

    deleted: dict[str, int] = {}

    def _del(table: str, where: str, params: tuple) -> None:
        cur = conn.execute(f"DELETE FROM {table} WHERE {where}", params)
        if cur.rowcount:
            deleted[table] = deleted.get(table, 0) + cur.rowcount

    # FTS 索引（memory 的名字/内容副本）必须一起删，否则搜索仍能命中
    for mid in memory_ids:
        _del("memory_fts", "owner_type='memory' AND owner_id=?", (mid,))
    for table, col in _PURGE_LINKS:
        _del(table, f"{col}=?", (pid,))
    # Preparation tasks keep complete snapshots in both result and event data.
    # Resolve them while followups still exist and the purge holds the write lock.
    followup_tasks = (
        "SELECT t.id FROM task t WHERE t.kind='followup_prepare' AND ("
        "t.id IN (SELECT task_id FROM followup WHERE person_id=?) OR "
        "json_extract(t.payload,'$.followup_id') IN (SELECT id FROM followup WHERE person_id=?))")
    _del("task_event", f"task_id IN ({followup_tasks})", (pid, pid))
    _del("task", f"id IN ({followup_tasks})", (pid, pid))
    memory_outcomes = (
        "SELECT o.id FROM followup_outcome o JOIN followup f ON f.id=o.followup_id "
        "WHERE f.person_id=? UNION SELECT outcome_id FROM outcome_memory_analysis WHERE person_id=?")
    memory_tasks = (
        "SELECT id FROM task WHERE kind='outcome_memory' AND "
        f"json_extract(payload,'$.outcome_id') IN ({memory_outcomes})")
    _del("task_event", f"task_id IN ({memory_tasks})", (pid, pid))
    _del("task", f"id IN ({memory_tasks})", (pid, pid))
    _del("pending_item", "kind='outcome_memory' AND (json_extract(payload,'$.person_id')=? OR "
         f"json_extract(payload,'$.outcome_id') IN ({memory_outcomes}))", (pid, pid, pid))
    correction_persons = [r[0] for r in conn.execute(
        "WITH RECURSIVE ids(id) AS (SELECT ? UNION SELECT p.id FROM person p JOIN ids ON p.merged_into=ids.id "
        "WHERE p.status='merged') SELECT id FROM ids", (pid,))]
    for correction_pid in correction_persons:
        _del("correction_log", "person_id=? OR related_person_id=?", (correction_pid, correction_pid))
    for iid in exclusive_interactions:
        _del("memory_fts", "owner_type='interaction' AND owner_id=?", (iid,))
        _del("vector_item", "owner_type='interaction' AND owner_id=?", (iid,))
        _del("interaction", "id=?", (iid,))
    _del("relation_edge", "from_person=? OR to_person=?", (pid, pid))
    for mid in memory_ids:
        _del("vector_item", "owner_type='memory' AND owner_id=?", (mid,))
    _del("vector_item", "owner_type='person_name' AND owner_id=?", (pid,))
    # 人物最后删（person_alias 有外键指向它）
    _del("person", "id=?", (pid,))

    # 人物专属的凭证与 blob
    blobs = _purge_evidence(conn, ev_ids, deleted)

    unlinked: dict[str, int] = {}
    cur = conn.execute("UPDATE skill SET person_id=NULL, updated_at=? WHERE person_id=?",
                       (now_ms(), pid))
    if cur.rowcount:
        unlinked["skill"] = cur.rowcount

    return {"ok": True, "deleted": deleted, "unlinked": unlinked, "blobs": blobs,
            "leftovers": [], "aliases": aliases}


def verify_purged(conn, pid: str, aliases: list[str]) -> list[str]:
    """删除后自检：还有残留就返回清单（上层据此报「部分删除」而不是「已删除」）。

    别名检查**按 person_id 限定**：schema 允许两个人物共享同一个别名（三个「小张」），
    所以「全库搜不到这个别名」是不成立的验收口径——成立的口径是
    「这个别名再也指不到被清除的那个人」。
    """
    leftovers: list[str] = []
    if conn.execute("SELECT COUNT(*) n FROM person WHERE id=?", (pid,)).fetchone()["n"]:
        leftovers.append("person")
    for table, col in _PURGE_LINKS:
        n = conn.execute(f"SELECT COUNT(*) n FROM {table} WHERE {col}=?", (pid,)).fetchone()["n"]
        if n:
            leftovers.append(f"{table}({n})")
    n = conn.execute(
        "SELECT COUNT(*) n FROM relation_edge WHERE from_person=? OR to_person=?",
        (pid, pid)).fetchone()["n"]
    if n:
        leftovers.append(f"relation_edge({n})")
    n = conn.execute(
        "SELECT COUNT(*) n FROM vector_item WHERE owner_type='person_name' AND owner_id=?",
        (pid,)).fetchone()["n"]
    if n:
        leftovers.append(f"vector_item({n})")
    for alias in aliases:
        if conn.execute("SELECT 1 FROM person_alias WHERE alias=? AND person_id=? LIMIT 1",
                        (alias, pid)).fetchone():
            leftovers.append(f"alias:{alias}")
    return leftovers


def _evidence_ids_of(conn, pid: str) -> list[str]:
    """该人名下所有行引用到的 evidence id。"""
    ids: set[str] = set()
    for sql in (
        "SELECT DISTINCT evidence_id e FROM person_field WHERE person_id=?",
        "SELECT DISTINCT evidence_id e FROM memory_item WHERE person_id=?",
        "SELECT DISTINCT origin_evidence_id e FROM memory_item WHERE person_id=?",
        "SELECT DISTINCT c.evidence_id e FROM memory_correction c "
        "JOIN memory_item m ON m.id=c.memory_id WHERE m.person_id=?",
        "SELECT DISTINCT evidence_id e FROM promise WHERE person_id=?",
        "SELECT DISTINCT evidence_id e FROM ledger_entry WHERE person_id=?",
        "SELECT DISTINCT evidence_id e FROM relation_edge WHERE from_person=? OR to_person=?",
        "SELECT DISTINCT a.evidence_id e FROM outcome_memory_analysis a "
        "JOIN followup f ON f.id=a.followup_id WHERE a.person_id=? OR f.person_id=?",
    ):
        for r in conn.execute(sql, (pid, pid) if sql.count('?') == 2 else (pid,)).fetchall():
            if r["e"]:
                ids.add(r["e"])
    return sorted(_evidence_chain(conn, ids))


def _evidence_chain(conn, evidence_ids) -> set[str]:
    """Review evidence can retain an explicit link to original communication evidence."""
    found: set[str] = set()
    pending = list(evidence_ids)
    while pending:
        eid = pending.pop()
        if not eid or eid in found:
            continue
        found.add(eid)
        row = conn.execute("SELECT excerpt FROM evidence WHERE id=?", (eid,)).fetchone()
        if not row:
            continue
        try:
            source = json.loads(row["excerpt"])
        except (TypeError, ValueError):
            continue
        linked = source.get("source_evidence_id") if isinstance(source, dict) else None
        if isinstance(linked, str) and linked:
            pending.append(linked)
    return found


def _purge_evidence(conn, ev_ids: list[str], deleted: dict[str, int]) -> list[str]:
    """删掉「只被这个人引用」的 evidence 及其 blob（被别人引用的留着）。"""
    from app.store import blob
    blobs: list[str] = []
    roots: set[str] = set()
    for table, column in (("person_field", "evidence_id"), ("memory_item", "evidence_id"),
                          ("memory_item", "origin_evidence_id"), ("memory_correction", "evidence_id"),
                          ("promise", "evidence_id"), ("ledger_entry", "evidence_id"),
                          ("relation_edge", "evidence_id"), ("outcome_memory_analysis", "evidence_id")):
        roots.update(r[0] for r in conn.execute(
            f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"))
    protected = _evidence_chain(conn, roots)
    for eid in ev_ids:
        if eid in protected:
            continue
        ev = conn.execute("SELECT blob_hash FROM evidence WHERE id=?", (eid,)).fetchone()
        if not ev:
            continue
        conn.execute("DELETE FROM evidence WHERE id=?", (eid,))
        deleted["evidence"] = deleted.get("evidence", 0) + 1
        h = ev["blob_hash"] if ev else None
        if h and not conn.execute(
                "SELECT 1 FROM evidence WHERE blob_hash=? LIMIT 1", (h,)).fetchone():
            blob.delete(h)
            blobs.append(h)
    return blobs


def update_value_read(conn, pid: str, value_read: str, evidence: str | None) -> None:
    conn.execute("UPDATE person SET value_read=?, value_read_evidence=?, updated_at=? WHERE id=?",
                 (value_read, evidence, now_ms(), pid))


def recompute_importance(conn, pid: str) -> None:
    """重要度只由实际参与的联系次数与最近度合成，资料提及不计入。"""
    import math
    cnt = conn.execute(
        "SELECT COUNT(*) c FROM interaction i JOIN interaction_person ip ON ip.interaction_id=i.id "
        "WHERE ip.person_id=? AND i.kind='contact' AND ip.role='participant'", (pid,)).fetchone()["c"]
    last = last_contact(conn, pid)
    rec = 0.0
    if last is not None:
        days = max(0.0, (now_ms() - last) / 86400000.0)
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
    if not valid_from and cur and (cur["value"] or "") == (value or ""):
        # 相同值：只提升置信度/证据，不产生新版本
        if confidence > cur["confidence"]:
            conn.execute(
                "UPDATE person_field SET confidence=?, evidence_id=COALESCE(?,evidence_id) "
                "WHERE person_id=? AND field=? AND valid_to IS NULL",
                (confidence, evidence_id, person_id, field))
        return
    if valid_from is not None:
        existing = conn.execute("SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from=?",
                                (person_id, field, vf)).fetchone()
        if existing:
            if existing["value"] == value:
                return
            vf += 1
        while conn.execute("SELECT 1 FROM person_field WHERE person_id=? AND field=? AND valid_from=?",
                           (person_id, field, vf)).fetchone():
            vf += 1
        nxt = conn.execute("SELECT MIN(valid_from) t FROM person_field WHERE person_id=? AND field=? AND valid_from>?",
                           (person_id, field, vf)).fetchone()["t"]
        prev = conn.execute("SELECT MAX(valid_from) t FROM person_field WHERE person_id=? AND field=? AND valid_from<?",
                            (person_id, field, vf)).fetchone()["t"]
        if prev is not None:
            conn.execute("UPDATE person_field SET valid_to=? WHERE person_id=? AND field=? AND valid_from=? "
                         "AND (valid_to IS NULL OR valid_to>?)",
                         (vf, person_id, field, prev, vf))
        conn.execute("INSERT INTO person_field VALUES(?,?,?,?,?,?,?,?)",
                     (person_id, field, value, confidence, evidence_id, vf, nxt, now))
        return
    if cur:
        # 防 PK 冲突：新版本的 valid_from 必须严格大于旧版本（同毫秒写入时递增）
        if vf <= cur["valid_from"]:
            vf = cur["valid_from"] + 1
        conn.execute(
            "UPDATE person_field SET valid_to=? WHERE person_id=? AND field=? AND valid_to IS NULL",
            (vf, person_id, field))
    # A manually removed field has no current row, but still has historical keys.
    latest = conn.execute("SELECT MAX(valid_from) FROM person_field WHERE person_id=? AND field=?",
                          (person_id, field)).fetchone()[0]
    if latest is not None and vf <= latest:
        vf = latest + 1
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
                       person_ids: list[tuple[str, str]] | None = None,
                       kind: str = "note", dedupe_key: str | None = None) -> tuple[str, bool]:
    """写入资料或实际联系；重复内容不改变原类型、时间或参与人。

    只有 contact 中显式标为 participant 的人物计入联系统计。
    dedupe_key 标识独立来源事件；缺省沿用 channel + 内容去重。
    """
    if kind not in {"note", "contact"}:
        raise ValueError("互动类型必须为 note 或 contact")
    if dedupe_key is not None and (not isinstance(dedupe_key, str) or not dedupe_key.strip()):
        raise ValueError("去重标识不能为空")
    ch = sha256_text(f"{channel}|{content.strip()}") if dedupe_key is None else \
        sha256_text(json.dumps(["interaction-source", channel, dedupe_key], ensure_ascii=False))
    exist = conn.execute("SELECT id FROM interaction WHERE content_hash=?", (ch,)).fetchone()
    if exist:
        return exist["id"], False
    iid = new_id()
    conn.execute(
        "INSERT INTO interaction(id,ts,channel,source_ref,content,content_hash,created_at,kind)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (iid, ts, channel, source_ref, content, ch, now_ms(), kind))
    roles = {}
    for pid, role in (person_ids or []):
        if roles.get(pid) != "participant":
            roles[pid] = role
    for pid, role in roles.items():
        conn.execute(
            "INSERT OR IGNORE INTO interaction_person(interaction_id,person_id,role) VALUES(?,?,?)",
            (iid, pid, role))
        recompute_importance(conn, pid)
    from app.memory.fts import tokenize  # 延迟导入避免循环
    conn.execute(
        "INSERT INTO memory_fts(tokens,owner_type,owner_id) VALUES(?,?,?)",
        (tokenize(content), "interaction", iid))
    return iid, True


def recent_interactions(conn, person_id: str, limit: int = 5, kind: str | None = None) -> list:
    """返回人物相关资料与联系；kind 可筛选 note/contact，保留该人的角色。"""
    if kind is not None and kind not in {"note", "contact"}:
        raise ValueError("互动类型必须为 note 或 contact")
    sql = (
        "SELECT i.*,ip.role AS person_role FROM interaction i "
        "JOIN interaction_person ip ON ip.interaction_id=i.id WHERE ip.person_id=?")
    params = [person_id]
    if kind is not None:
        sql += " AND i.kind=?"
        params.append(kind)
    return conn.execute(sql + " ORDER BY i.ts DESC,i.created_at DESC,i.id DESC LIMIT ?",
                        (*params, limit)).fetchall()


def last_contact(conn, person_id: str) -> int | None:
    """最近一次有明确 participant 角色的实际联系时间，未知返回 None。"""
    return conn.execute(
        "SELECT MAX(i.ts) FROM interaction i JOIN interaction_person ip ON ip.interaction_id=i.id "
        "WHERE ip.person_id=? AND i.kind='contact' AND ip.role='participant'", (person_id,)).fetchone()[0]


def recent_contacts(conn, person_id: str, limit: int = 5) -> list:
    """仅返回该人物实际参与的联系，其他人的联系中被提及不计入。"""
    return conn.execute(
        "SELECT i.*,ip.role AS person_role FROM interaction i "
        "JOIN interaction_person ip ON ip.interaction_id=i.id "
        "WHERE ip.person_id=? AND i.kind='contact' AND ip.role='participant' "
        "ORDER BY i.ts DESC,i.created_at DESC,i.id DESC LIMIT ?", (person_id, limit)).fetchall()


def interactions_since(conn, since_ms: int) -> list:
    return conn.execute(
        "SELECT i.* FROM interaction i WHERE ts>=? AND NOT EXISTS (SELECT 1 FROM interaction_person ip "
        "JOIN person p ON p.id=ip.person_id WHERE ip.interaction_id=i.id AND "
        "(p.deleted_at IS NOT NULL OR p.status='merged')) ORDER BY ts", (since_ms,)).fetchall()


def interaction_person_names(conn, interaction_id: str) -> list[str]:
    return [r["display_name"] for r in conn.execute(
        "SELECT p.display_name FROM interaction_person ip JOIN person p ON p.id=ip.person_id "
        "WHERE ip.interaction_id=?", (interaction_id,)).fetchall()]


# ---------------- memory_item ----------------

# All shared consumers use the same final visibility check, including vector hits.
ACTIVE_MEMORY_WHERE = (
    "m.status='active' AND m.deleted_at IS NULL AND (m.person_id IS NULL OR EXISTS "
    "(SELECT 1 FROM person p WHERE p.id=m.person_id AND p.deleted_at IS NULL AND p.status!='merged'))"
)


def add_memory(conn, kind: str, text: str, person_id: str | None = None, confidence: float = 1.0,
               evidence_id: str | None = None, ts: int | None = None) -> str:
    mid = new_id()
    t = ts or now_ms()
    created = now_ms()
    conn.execute(
        "INSERT INTO memory_item(id,person_id,kind,text,confidence,evidence_id,ts,created_at,updated_at,origin_evidence_id)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (mid, person_id, kind, text, confidence, evidence_id, t, created, created, evidence_id))
    from app.memory.fts import tokenize
    conn.execute(
        "INSERT INTO memory_fts(tokens,owner_type,owner_id) VALUES(?,?,?)",
        (tokenize(text), "memory", mid))
    return mid


def memories_of(conn, person_id: str, kinds: list[str] | None = None, limit: int = 50) -> list:
    if kinds:
        ph = ",".join("?" * len(kinds))
        return conn.execute(
            f"SELECT m.* FROM memory_item m WHERE m.person_id=? AND {ACTIVE_MEMORY_WHERE} AND m.kind IN ({ph})"
            f" ORDER BY m.ts DESC LIMIT ?", (person_id, *kinds, limit)).fetchall()
    return conn.execute(
        f"SELECT m.* FROM memory_item m WHERE m.person_id=? AND {ACTIVE_MEMORY_WHERE} ORDER BY m.ts DESC LIMIT ?",
        (person_id, limit)).fetchall()


def memories_since(conn, since_ms: int, kinds: list[str] | None = None) -> list:
    base = f"SELECT m.* FROM memory_item m WHERE {ACTIVE_MEMORY_WHERE} AND m.ts>=?"
    params: list[Any] = [since_ms]
    if kinds:
        ph = ",".join("?" * len(kinds))
        base += f" AND m.kind IN ({ph})"
        params += kinds
    return conn.execute(base + " ORDER BY m.ts", params).fetchall()


def my_profile_memories(conn) -> list:
    """近期个人记忆；更早的内容保留在库中，可通过全文检索读取。"""
    return conn.execute(
        f"SELECT m.* FROM memory_item m WHERE m.person_id IS NULL AND {ACTIVE_MEMORY_WHERE} "
        "AND m.kind IN ('semantic','episodic') ORDER BY m.ts DESC LIMIT 200").fetchall()


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
    if status != "open":
        conn.execute("UPDATE reminder SET status='cancelled' WHERE promise_id=? AND status='pending'", (pid,))


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
    else:
        where += " AND (person_id IS NULL OR person_id IN (SELECT id FROM person WHERE deleted_at IS NULL AND status!='merged'))"
    row = conn.execute(
        f"SELECT COALESCE(SUM(CASE WHEN direction='in' THEN amount_cents ELSE 0 END),0) tin,"
        f" COALESCE(SUM(CASE WHEN direction='out' THEN amount_cents ELSE 0 END),0) tout,"
        f" COUNT(*) n FROM ledger_entry WHERE {where}", params).fetchone()
    return {"in_cents": row["tin"], "out_cents": row["tout"], "count": row["n"]}


def ledger_entries(conn, person_id: str | None = None, include_pending: bool = True, include_void: bool = False,
                   limit: int = 100, offset: int = 0) -> list:
    sql = "SELECT l.*, p.display_name FROM ledger_entry l LEFT JOIN person p ON p.id=l.person_id"
    conds, params = (["1=1"] if include_void else ["l.status!='void'"]), []
    if person_id:
        conds.append("l.person_id=?")
        params.append(person_id)
    else:
        conds.append("(l.person_id IS NULL OR (p.deleted_at IS NULL AND p.status!='merged'))")
    if not include_pending:
        conds.append("l.status='confirmed'")
    return conn.execute(sql + " WHERE " + " AND ".join(conds) + " ORDER BY l.occurred_at DESC,l.id DESC LIMIT ? OFFSET ?",
                        [*params, limit, offset]).fetchall()


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
    if source_ref:
        existing = conn.execute("SELECT id FROM evidence WHERE kind=? AND source_ref=? AND blob_hash IS ?",
                                (kind, source_ref, blob_hash)).fetchone()
        if existing:
            return existing["id"]
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
            job = get_job(conn, exist["id"])
            if job["status"] in {"failed", "partial"}:
                conn.execute("UPDATE ingest_job SET status='queued',error=NULL WHERE id=?", (job["id"],))
                return job["id"], True
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
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    existing = conn.execute("SELECT id FROM pending_item WHERE kind=? AND payload=? "
                            "AND resolved_at IS NULL", (kind, encoded)).fetchone()
    if existing:
        return existing["id"]
    for key in ("fingerprint", "dedupe_hash"):
        if payload.get(key):
            existing = conn.execute("SELECT id FROM pending_item WHERE kind=? AND resolved_at IS NULL "
                "AND json_extract(payload,?)=?", (kind, "$." + key, payload[key])).fetchone()
            if existing:
                return existing["id"]
    pid = new_id()
    conn.execute(
        "INSERT INTO pending_item(id,kind,payload,reason,created_at) VALUES(?,?,?,?,?)",
        (pid, kind, encoded, reason, now_ms()))
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
        "SELECT * FROM (SELECT rowid ordering, * FROM message WHERE conversation_id=? "
        "ORDER BY created_at DESC, rowid DESC LIMIT ?) ORDER BY created_at, ordering",
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
        "SELECT r.* FROM reminder r LEFT JOIN person p ON p.id=r.person_id WHERE r.status='pending' "
        "AND r.fire_at<=? AND (r.person_id IS NULL OR (p.id IS NOT NULL AND p.deleted_at IS NULL "
        "AND p.status!='merged')) ORDER BY r.fire_at LIMIT 50", (now,)).fetchall()


def mark_reminder_fired(conn, rid: str) -> None:
    conn.execute("UPDATE reminder SET status='fired', fired_at=? WHERE id=?", (now_ms(), rid))


def upcoming_reminders(conn, limit: int = 50) -> list:
    return conn.execute(
        "SELECT r.*, p.display_name FROM reminder r LEFT JOIN person p ON p.id=r.person_id "
        "WHERE r.status='pending' AND (r.person_id IS NULL OR (p.deleted_at IS NULL AND p.status!='merged')) "
        "ORDER BY r.fire_at LIMIT ?", (limit,)).fetchall()


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
