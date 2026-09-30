"""FastAPI 层：对话 / 摄入 / 人物 / 待确认 / 星图 / 提醒 / 日志 / 技能 / 研究。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent import harness
from app.ingest import chatlog, importer, normalize
from app.memory import apply as apply_mod
from app.memory import compile_card, retrieve
from app.store import repo
from app.store.db import get_conn
from app.util import ms_to_local_str, normalize_value, now_ms

ROOT = Path(__file__).resolve().parent.parent.parent
app = FastAPI(title="home", version="0.1.0")


@app.on_event("startup")
def _startup():
    get_conn()
    from app.jobs.scheduler import start_scheduler
    start_scheduler()


# ---------------- 页面 ----------------

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


# ---------------- 对话 ----------------

class NewConv(BaseModel):
    title: str = "新对话"


@app.post("/api/conversations")
def create_conv(body: NewConv):
    conn = get_conn()
    cid = repo.create_conversation(conn, body.title)
    conn.commit()
    return {"id": cid, "title": body.title}


@app.get("/api/conversations")
def list_convs():
    conn = get_conn()
    return {"conversations": [
        {"id": c["id"], "title": c["title"], "updated_at": ms_to_local_str(c["updated_at"])}
        for c in repo.list_conversations(conn)]}


@app.get("/api/conversations/{cid}/messages")
def list_msgs(cid: str):
    conn = get_conn()
    return {"messages": [
        {"id": m["id"], "role": m["role"], "content": m["content"],
         "created_at": ms_to_local_str(m["created_at"])}
        for m in repo.list_messages(conn, cid) if m["role"] in ("user", "assistant")]}


class ChatIn(BaseModel):
    text: str


@app.post("/api/conversations/{cid}/messages")
def send_msg(cid: str, body: ChatIn):
    conn = get_conn()
    if not repo.get_conversation(conn, cid):
        raise HTTPException(404, "对话不存在")
    return harness.run_turn(conn, cid, body.text)


# ---------------- 摄入 ----------------

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "空文件")
    conn = get_conn()
    try:
        return normalize.ingest_file(conn, file.filename or "unnamed", data)
    except ValueError as e:
        raise HTTPException(400, str(e))


class UrlIn(BaseModel):
    url: str


@app.post("/api/ingest/chatlog")
async def ingest_chatlog(file: UploadFile = File(...), me: str = ""):
    """导入微信/QQ 导出聊天记录（.txt）。me 可显式指定「我」在记录里的名字。"""
    data = await file.read()
    if not data:
        raise HTTPException(400, "空文件")
    conn = get_conn()
    try:
        return chatlog.import_chatlog(conn, file.filename or "chatlog.txt", data,
                                      me=me.strip() or None)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/ingest/url")
def ingest_url(body: UrlIn):
    conn = get_conn()
    return normalize.ingest_url(conn, body.url)


class ImportIn(BaseModel):
    text: str


@app.post("/api/import/names")
def import_names(body: ImportIn):
    conn = get_conn()
    return importer.import_csv_names(conn, body.text)


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    conn = get_conn()
    j = repo.get_job(conn, jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    return dict(j)


# ---------------- 人物 ----------------

@app.get("/api/persons")
def list_persons():
    conn = get_conn()
    out = []
    for p in repo.all_active_persons(conn):
        fields = repo.current_fields(conn, p["id"])
        out.append({"id": p["id"], "display_name": p["display_name"],
                    "status": p["status"], "importance": round(p["importance"], 3),
                    "aliases": repo.aliases_of(conn, p["id"]),
                    "field_count": len(fields),
                    "value_read": p["value_read"]})
    return {"persons": out}


@app.get("/api/persons/archived")
def list_archived_persons():
    """已归档的人物——归档要能恢复，就必须有个入口找到它们。

    注意：这条必须声明在 /api/persons/{pid} 之前，否则会被当成 pid="archived"。
    """
    conn = get_conn()
    return {"persons": [
        {"id": p["id"], "display_name": p["display_name"],
         "aliases": repo.aliases_of(conn, p["id"]),
         "archived_at": ms_to_local_str(p["deleted_at"])}
        for p in repo.archived_persons(conn)]}


@app.get("/api/persons/{pid}")
def person_detail(pid: str):
    conn = get_conn()
    p = repo.get_person(conn, pid)
    if not p:
        raise HTTPException(404, "人物不存在")
    fields = repo.current_fields(conn, pid)
    return {
        "person": dict(p),
        "card": compile_card.build_card(conn, pid),
        "aliases": repo.aliases_of(conn, pid),
        "fields": {k: {"value": v["value"], "confidence": v["confidence"],
                       "since": ms_to_local_str(v["valid_from"])[:10]} for k, v in fields.items()},
        "interactions": [{"ts": ms_to_local_str(i["ts"]), "channel": i["channel"],
                          "content": i["content"][:200]} for i in repo.recent_interactions(conn, pid, 10)],
        "open_promises": [{"id": r["id"], "who": r["who"], "content": r["content"],
                           "due": ms_to_local_str(r["due_at"])[:10] if r["due_at"] else None}
                          for r in repo.open_promises(conn, pid)],
        "ledger": {"totals": repo.ledger_totals(conn, pid),
                   "entries": [{"direction": e["direction"], "amount_yuan": e["amount_cents"] / 100,
                                "note": e["note"], "status": e["status"],
                                "when": ms_to_local_str(e["occurred_at"])[:10]}
                               for e in repo.ledger_entries(conn, pid)]},
    }


class FieldIn(BaseModel):
    field: str
    value: str


@app.post("/api/persons/{pid}/fields")
def edit_field(pid: str, body: FieldIn):
    conn = get_conn()
    if not repo.get_person(conn, pid):
        raise HTTPException(404, "人物不存在")
    repo.upsert_field(conn, pid, body.field, body.value, 1.0, None)
    conn.commit()
    return {"ok": True}


@app.delete("/api/persons/{pid}")
def archive_person(pid: str):
    """归档（旧接口语义改了）：从所有列表/检索里消失、不复活，但可恢复。

    P5 把「删除」拆成两级，这里给的是**可恢复**的那一级——真正抹掉痕迹走 /purge。
    """
    conn = get_conn()
    if not repo.get_person(conn, pid):
        raise HTTPException(404, "人物不存在")
    repo.archive_person(conn, pid)
    conn.commit()
    return {"ok": True, "action": "archived",
            "message": "已归档（含别名与记忆，不会复活；可恢复，彻底清除请用 purge）"}


@app.post("/api/persons/{pid}/restore")
def restore_person(pid: str):
    conn = get_conn()
    if not repo.restore_person(conn, pid):
        raise HTTPException(400, "该人物不处于可恢复的归档状态")
    conn.commit()
    return {"ok": True, "action": "restored"}


@app.post("/api/persons/{pid}/purge")
def purge_person(pid: str):
    """彻底清除：物理删除（字段/别名/记忆/账本/关系边/凭证/blob）。

    FR-ID-02：删完必须能通过「全库搜该人的任何曾用名返回 0 条」的校验；
    校验不过要报「部分删除」，而不是含糊地说「已删除」。
    """
    conn = get_conn()
    rep = repo.purge_person(conn, pid)
    if not rep.get("ok") and "person 不存在" in rep.get("leftovers", []):
        raise HTTPException(404, "人物不存在")
    aliases = rep.pop("aliases", [])
    conn.commit()
    leftovers = repo.verify_purged(conn, pid, aliases)
    if leftovers:
        raise HTTPException(500, f"部分删除：仍有残留 {leftovers}（未声称已删除）")
    return {"ok": True, "action": "purged", "message": "已彻底清除（含别名与凭证）",
            "deleted": rep["deleted"], "unlinked": rep["unlinked"], "blobs": rep["blobs"]}


class MergeIn(BaseModel):
    survivor_id: str
    merged_id: str


@app.post("/api/persons/merge")
def merge(body: MergeIn):
    conn = get_conn()
    try:
        log_id = apply_mod.merge_persons(conn, body.survivor_id, body.merged_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "merge_log_id": log_id}


class SplitIn(BaseModel):
    merge_log_id: str


@app.post("/api/persons/split")
def split(body: SplitIn):
    conn = get_conn()
    try:
        apply_mod.split_merge(conn, body.merge_log_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


# ---------------- 待确认 ----------------

@app.get("/api/pending")
def list_pending():
    conn = get_conn()
    return {"pending": [
        {"id": p["id"], "kind": p["kind"], "reason": p["reason"],
         "payload": json.loads(p["payload"]), "created_at": ms_to_local_str(p["created_at"])}
        for p in repo.list_pending(conn)]}


class ResolveIn(BaseModel):
    action: str                 # confirm | discard
    person_id: str | None = None  # person_identity 确认时选定的对象，"new" 表示新建


@app.post("/api/pending/{pid}/resolve")
def resolve_pending(pid: str, body: ResolveIn):
    conn = get_conn()
    item = repo.get_pending(conn, pid)
    if not item or item["resolved_at"]:
        raise HTTPException(404, "待确认项不存在或已处理")
    payload = json.loads(item["payload"])

    if body.action == "discard":
        repo.resolve_pending(conn, pid, "discarded")
        conn.commit()
        return {"ok": True, "resolution": "discarded"}

    kind = item["kind"]
    if kind == "person_identity":
        if body.person_id and body.person_id != "new":
            target = body.person_id
            if not repo.get_person(conn, target):
                raise HTTPException(404, "选定的人物不存在")
        else:
            target = repo.create_person(conn, payload["canonical_name"], status="active")
        repo.add_alias(conn, target, payload["mention"], payload.get("channel", "chat"), 0.95)
        patch = payload.get("patch")
        if patch:
            # 已确认身份 → 强制绑定，重放该人的 patch（绕过二次消歧）
            apply_mod.apply_extraction(conn, {"people": [patch], "my_notes": []},
                                       payload["source_msg_id"], payload["mention"],
                                       channel=payload.get("channel", "chat"),
                                       forced={patch["canonical_name"]: target})
        repo.resolve_pending(conn, pid, f"bound:{target}")
    elif kind == "field":
        repo.upsert_field(conn, payload["person_id"], payload["field"], payload["value"],
                          payload["confidence"], payload.get("evidence_id"))
        repo.mark_patch(conn, payload["fingerprint"], payload["person_id"], payload["field"],
                        normalize_value(payload["value"]), payload["source_msg_id"])
        repo.resolve_pending(conn, pid, "confirmed")
    elif kind == "ledger":
        from datetime import datetime
        from app.config import tz
        occurred = now_ms()
        if payload.get("occurred"):
            try:
                occurred = int(datetime.strptime(payload["occurred"], "%Y-%m-%d")
                               .replace(hour=12, tzinfo=tz()).timestamp() * 1000)
            except ValueError:
                pass
        repo.add_ledger(conn, payload["direction"], payload["amount_cents"], payload["person_id"],
                        payload["note"], occurred, "confirmed", payload["dedupe_hash"],
                        payload.get("evidence_id"))
        repo.resolve_pending(conn, pid, "confirmed")
    elif kind == "promise":
        from app.memory.apply import _date_ms
        pmid = repo.add_promise(conn, payload["who"], payload["content"], payload["person_id"],
                                _date_ms(payload.get("due")), payload.get("evidence_id"))
        repo.resolve_pending(conn, pid, f"confirmed:{pmid}")
    else:
        raise HTTPException(400, f"未知的待确认类型 {kind}")

    conn.commit()
    return {"ok": True, "resolution": "confirmed"}


# ---------------- 星图 / 提醒 / 通知 / 日志 / 技能 / 研究 ----------------

@app.get("/api/atlas")
def atlas():
    conn = get_conn()
    g = repo.atlas_graph(conn)
    return {**g, "quiet": retrieve.quiet_people(conn, 14)}


@app.get("/api/reminders")
def reminders():
    conn = get_conn()
    return {"reminders": [
        {"id": r["id"], "text": r["text"], "person": r["display_name"],
         "fire_at": ms_to_local_str(r["fire_at"])} for r in repo.upcoming_reminders(conn)]}


@app.get("/api/notifications")
def notifications(unread: bool = False):
    conn = get_conn()
    return {"notifications": [
        {"id": n["id"], "kind": n["kind"], "text": n["text"], "read": bool(n["read"]),
         "created_at": ms_to_local_str(n["created_at"])}
        for n in repo.list_notifications(conn, unread)]}


@app.post("/api/notifications/read")
def read_notifications():
    conn = get_conn()
    repo.mark_notifications_read(conn)
    conn.commit()
    return {"ok": True}


@app.get("/api/journals")
def journals():
    conn = get_conn()
    return {"journals": [{"day": j["day"], "text": j["text"]} for j in repo.list_journals(conn)]}


@app.get("/api/journal/{day}")
def journal(day: str):
    conn = get_conn()
    j = repo.get_journal(conn, day)
    return {"day": day, "text": j["text"] if j else None}


class JournalGen(BaseModel):
    day: str | None = None


@app.post("/api/jobs/journal")
def gen_journal(body: JournalGen):
    from app.jobs.scheduler import generate_journal
    text = generate_journal(body.day)
    return {"ok": True, "text": text}


@app.get("/api/skills")
def skills():
    conn = get_conn()
    return {"skills": [dict(s) for s in repo.list_skills(conn)]}


@app.get("/api/research")
def researches():
    conn = get_conn()
    return {"research": [
        {"id": r["id"], "topic": r["topic"], "status": r["status"],
         "updated_at": ms_to_local_str(r["updated_at"])} for r in repo.list_research(conn)]}


@app.get("/api/research/{rid}")
def research_detail(rid: str):
    conn = get_conn()
    r = repo.get_research(conn, rid)
    if not r:
        raise HTTPException(404, "研究不存在")
    return {"id": r["id"], "topic": r["topic"], "status": r["status"],
            "report_md": r["report_md"],
            "sources": json.loads(r["sources"]) if r["sources"] else []}
