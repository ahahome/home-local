"""FastAPI 层：对话 / 摄入 / 人物 / 待确认 / 星图 / 提醒 / 日志 / 技能 / 研究。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Literal
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
@asynccontextmanager
async def lifespan(app):
    get_conn()
    from app.jobs.scheduler import start_scheduler
    from app.jobs.tasks import recover
    scheduler = start_scheduler()
    recover()
    try:
        yield
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)


app = FastAPI(title="home", version="0.3.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "web")), name="static")
from app.api.workspace import router
app.include_router(router)
from app.api.followups import router as followups_router
app.include_router(followups_router)


# ---------------- 页面 ----------------

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


# ---------------- 对话 ----------------

class NewConv(BaseModel):
    title: str = "新对话"
    project_id: str | None = None


@app.post("/api/conversations")
def create_conv(body: NewConv):
    conn = get_conn()
    if body.project_id and not conn.execute("SELECT 1 FROM project WHERE id=?", (body.project_id,)).fetchone():
        raise HTTPException(400, "项目不存在")
    cid = repo.create_conversation(conn, body.title)
    if body.project_id:
        conn.execute("UPDATE conversation SET project_id=? WHERE id=?", (body.project_id, cid))
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
         "created_at": ms_to_local_str(m["created_at"]),
         "metadata": json.loads(m["tool_calls"]) if m["tool_calls"] and m["tool_calls"].startswith("{") else None}
        for m in repo.list_messages(conn, cid) if m["role"] in ("user", "assistant") and not (m["role"] == "assistant" and m["tool_calls"] and m["tool_calls"].startswith("["))]}


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
def ingest_file(file: UploadFile = File(...)):
    data = file.file.read()
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
def ingest_chatlog(file: UploadFile = File(...), me: str = Form("")):
    """导入微信/QQ 导出聊天记录（.txt）。me 可显式指定「我」在记录里的名字。"""
    data = file.file.read()
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
                    "summary": ' · '.join(str(fields[k]['value']) for k in ('company','title','city') if k in fields and fields[k]['value']),
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
                       "valid_from": v["valid_from"], "since": ms_to_local_str(v["valid_from"])[:10]} for k, v in fields.items()},
        "last_contact": repo.last_contact(conn, pid),
        "interactions": [{"id": i["id"], "kind": i["kind"], "person_role": i["person_role"],
                          "ts": ms_to_local_str(i["ts"]), "channel": i["channel"],
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
    items = []
    for p in repo.list_pending(conn):
        payload = json.loads(p['payload'])
        for candidate in payload.get('candidates', []):
            fields = repo.current_fields(conn, candidate['person_id'])
            candidate['hint'] = ' · '.join(str(fields[k]['value']) for k in ('company','title','city') if k in fields and fields[k]['value'])
        if p['kind'] == 'ledger' and payload.get('evidence_id') and not payload.get('evidence'):
            evidence = conn.execute('SELECT excerpt FROM evidence WHERE id=?', (payload['evidence_id'],)).fetchone()
            if evidence:
                payload['evidence'] = evidence['excerpt']
        person = repo.get_person(conn, payload.get('person_id')) if payload.get('person_id') else None
        items.append({'id': p['id'], 'kind': p['kind'], 'payload': payload,
                      'person_name': person['display_name'] if person else payload.get('mention'),
                      'reason': p['reason'], 'created_at': ms_to_local_str(p['created_at'])})
    return {'pending': items}


class ResolveIn(BaseModel):
    action: Literal["confirm", "discard"]
    person_id: str | None = None  # person_identity 确认时选定的对象，"new" 表示新建
    direction: Literal['in', 'out'] | None = None  # 方向不明的账目必须明确选择


@app.post("/api/pending/{pid}/resolve")
def resolve_pending(pid: str, body: ResolveIn):
    from app.memory.pending import resolve
    try:
        return resolve(get_conn(), pid, body.action, body.person_id, body.direction)
    except ValueError as e:
        raise HTTPException(400, str(e))


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
            "proposal": json.loads(r["proposal"]) if r["proposal"] else None,
            "draft": json.loads(r["draft"]) if r["draft"] else [],
            "sources": json.loads(r["sources"]) if r["sources"] else []}
