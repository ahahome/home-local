from __future__ import annotations

import asyncio
import io
import json
import re
import sqlite3
import zipfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import Response, StreamingResponse, FileResponse
from pydantic import BaseModel, Field, StrictInt

from app.config import get_settings
from app.jobs import tasks
from app.memory import retrieve
from app.memory import corrections
from app.store import repo, blob
from app.store.db import get_conn
from app.util import now_ms, new_id, ms_to_local_str

router = APIRouter(prefix="/api")


def task_or_404(tid):
    try:
        return tasks.read(get_conn(), tid)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/status")
def status():
    s = get_settings()
    return {"configured": bool(s.llm_api_key), "base_url": s.llm_base_url,
            "models": {"chat": s.model_mid, "vision": s.model_vision, "research": s.model_pro},
            "embedding_enabled": bool(s.model_embed), "timezone": s.timezone,
            "storage": "本地 SQLite 与文件", "inference": "内容通过配置的 API 网关进行推理"}


@router.get("/home")
def home():
    c = get_conn()
    persons = repo.all_active_persons(c)
    promises = [dict(p) for p in repo.open_promises(c)
                if not p["person_id"] or any(x["id"] == p["person_id"] for x in persons)]
    for p in promises:
        p["due"] = ms_to_local_str(p["due_at"])[:10] if p["due_at"] else None
        p["overdue"] = bool(p["due_at"] and p["due_at"] < now_ms())
    changes = [dict(r) for r in c.execute(
        "SELECT f.*,p.display_name FROM person_field f JOIN person p ON p.id=f.person_id "
        "WHERE p.deleted_at IS NULL AND p.status!='merged' ORDER BY f.recorded_at DESC LIMIT 8")]
    quiet = retrieve.quiet_people(c, 14, 6)
    for p in quiet:
        p["last"] = ms_to_local_str(p["last_interaction"])[:10] if p["last_interaction"] else "尚无互动"
    return {"stats": {"persons": len(persons), "pending": len(repo.list_pending(c)),
                       "promises": len(promises), "overdue": sum(p["overdue"] for p in promises)},
            "promises": promises[:8], "changes": changes, "quiet": quiet,
            "ledger": repo.ledger_totals(c), "reminders": [dict(r) for r in repo.upcoming_reminders(c, 5)]}


class TurnIn(BaseModel):
    text: str = Field(min_length=1, max_length=30000)
    request_id: str | None = None


@router.post("/conversations/{cid}/turns", status_code=202)
def turn(cid: str, body: TurnIn):
    c = get_conn()
    if not repo.get_conversation(c, cid):
        raise HTTPException(404, "对话不存在")
    active = c.execute("SELECT id FROM task WHERE kind='chat' AND status IN ('queued','running') "
                       "AND json_extract(payload,'$.conversation_id')=?", (cid,)).fetchone()
    if active:
        if body.request_id:
            old = c.execute("SELECT id FROM task WHERE idempotency_key=?", (f"chat:{cid}:{body.request_id}",)).fetchone()
            if old:
                return tasks.read(c, old[0])
        raise HTTPException(409, "该对话已有回复任务，请等待或取消后再发")
    try:
        return tasks.submit(c, "chat", body.text[:30], {"conversation_id": cid, "text": body.text},
                            f"chat:{cid}:{body.request_id}" if body.request_id else None)
    except sqlite3.IntegrityError:
        c.rollback()
        if body.request_id:
            old = c.execute("SELECT id FROM task WHERE idempotency_key=?", (f"chat:{cid}:{body.request_id}",)).fetchone()
            if old:
                return tasks.read(c, old[0])
        raise HTTPException(409, "该对话已有回复任务，请等待或取消后再发")


@router.get("/tasks")
def list_tasks():
    c = get_conn()
    return {"tasks": [tasks.read(c, r[0]) for r in c.execute("SELECT id FROM task ORDER BY created_at DESC LIMIT 100")]}


@router.post('/conversations/{cid}/messages/{mid}/reanalyze', status_code=202)
def reanalyze_message(cid: str, mid: str):
    from app.jobs import chat_reanalysis
    c = get_conn()
    if not repo.get_conversation(c, cid):
        raise HTTPException(404, '对话不存在')
    try:
        return chat_reanalysis.submit(c, cid, mid)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    except sqlite3.IntegrityError:
        c.rollback()
        existing = c.execute('SELECT id FROM task WHERE idempotency_key=?',
                             (f'chat-reanalysis:{cid}:{mid}',)).fetchone()
        if existing:
            return tasks.read(c, existing[0])
        raise HTTPException(409, '该对话已有回复任务，请等待或取消后再发') from None


@router.get("/tasks/{tid}")
def task_detail(tid: str):
    return task_or_404(tid)


@router.get("/tasks/{tid}/events")
async def task_events(tid: str, after: int = 0):
    task_or_404(tid)
    async def stream():
        cursor = after
        heartbeat = 0
        while True:
            c = get_conn()
            rows = c.execute("SELECT * FROM task_event WHERE task_id=? AND seq>? ORDER BY seq", (tid, cursor)).fetchall()
            for row in rows:
                cursor = row["seq"]
                yield f"id: {cursor}\nevent: {row['kind']}\ndata: {row['data']}\n\n"
            state = tasks.read(c, tid)
            if state["status"] in tasks.TERMINAL:
                yield "event: end\ndata: " + json.dumps({"status": state["status"], "error": state["error"]}, ensure_ascii=False) + "\n\n"
                break
            heartbeat += 1
            if heartbeat % 30 == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(.3)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/tasks/{tid}/cancel")
def cancel_task(tid: str):
    t = task_or_404(tid)
    if t["status"] in tasks.TERMINAL:
        return t
    c = get_conn()
    c.execute("UPDATE task SET cancel_requested=1,updated_at=? WHERE id=?", (now_ms(), tid))
    if t["status"] == "queued":
        c.execute("UPDATE task SET status='cancelled' WHERE id=?", (tid,))
    c.commit()
    return tasks.read(c, tid)


@router.post("/tasks/{tid}/retry", status_code=202)
def retry_task(tid: str):
    task_or_404(tid)
    try:
        return tasks.retry(get_conn(), tid)
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.post("/tasks/file", status_code=202)
def file_task(file: UploadFile = File(...), mode: str = Form("file"), me: str = Form("")):
    data = file.file.read(50 * 1024 * 1024 + 1)
    if not data or len(data) > 50 * 1024 * 1024:
        raise HTTPException(400, "文件为空或超过 50 MB")
    if mode not in {"file", "chatlog"}:
        raise HTTPException(400, "导入模式无效")
    return tasks.submit(get_conn(), mode, "导入 " + (file.filename or "文件"),
        {"filename": file.filename or "file.txt", "blob_hash": blob.put(data), "me": me.strip() or None})


class UrlIn(BaseModel):
    url: str = Field(min_length=1, max_length=2000)


@router.post("/tasks/url", status_code=202)
def url_task(body: UrlIn):
    if not body.url.startswith(("https://", "http://")):
        raise HTTPException(400, "请输入 http 或 https 网页地址")
    return tasks.submit(get_conn(), "url", "导入网页", {"url": body.url})


class ResearchIn(BaseModel):
    topic: str = Field(min_length=2, max_length=1000)


@router.post("/research", status_code=202)
def research_prepare(body: ResearchIn):
    c = get_conn()
    rid = repo.create_research(c, body.topic)
    c.commit()
    return tasks.submit(c, "research_prepare", "研究提案：" + body.topic[:25], {"topic": body.topic, "research_id": rid})


@router.post("/research/{rid}/start", status_code=202)
def research_start(rid: str):
    c = get_conn()
    r = repo.get_research(c, rid)
    if not r:
        raise HTTPException(404, "研究不存在")
    if not r["proposal"] or r["status"] not in {"proposed", "failed", "cancelled"}:
        raise HTTPException(409, "研究未准备好或已开始，请通过任务中心重试中断任务")
    repo.update_research(c, rid, status="queued")
    c.commit()
    return tasks.submit(c, "research", "研究：" + r["topic"][:25], {"topic": r["topic"], "research_id": rid})


class SkillIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    body: str = Field(min_length=1, max_length=30000)
    person_id: str | None = None
    trigger_hint: str = ""
    style_notes: str = ""


@router.post("/skills")
def save_skill(body: SkillIn):
    c = get_conn()
    if body.person_id:
        from app.memory.writes import resolve_person
        try:
            resolve_person(c, person_id=body.person_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
    sid, created = repo.save_skill(c, body.name.strip().lstrip("/"), body.body, body.person_id, body.trigger_hint, body.style_notes)
    c.commit()
    return {"ok": True, "id": sid, "created": created}


@router.delete("/skills/{sid}")
def delete_skill(sid: str):
    c = get_conn()
    c.execute("DELETE FROM skill WHERE id=?", (sid,))
    c.commit()
    return {"ok": True}


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=5000)
    person_ids: list[str] = Field(default_factory=list)


@router.get("/projects")
def projects():
    c = get_conn()
    out = []
    for r in c.execute("SELECT * FROM project ORDER BY updated_at DESC"):
        p = dict(r)
        p["persons"] = [dict(x) for x in c.execute("SELECT p.id,p.display_name FROM project_person pp "
            "JOIN person p ON p.id=pp.person_id WHERE pp.project_id=? AND p.deleted_at IS NULL", (p["id"],))]
        p["conversations"] = [dict(x) for x in c.execute("SELECT id,title FROM conversation WHERE project_id=?", (p["id"],))]
        out.append(p)
    return {"projects": out}


@router.post("/projects")
def create_project(body: ProjectIn):
    c = get_conn()
    ids = list(dict.fromkeys(body.person_ids))
    if any(not repo.get_person(c, pid) for pid in ids):
        raise HTTPException(400, "项目成员不存在")
    pid, ts = new_id(), now_ms()
    c.execute("INSERT INTO project VALUES(?,?,?,?,?)", (pid, body.name, body.description, ts, ts))
    for person in ids:
        c.execute("INSERT INTO project_person VALUES(?,?)", (pid, person))
    c.commit()
    return {"ok": True, "id": pid}


@router.delete("/projects/{pid}")
def delete_project(pid: str):
    c = get_conn()
    c.execute("UPDATE conversation SET project_id=NULL WHERE project_id=?", (pid,))
    c.execute("DELETE FROM project WHERE id=?", (pid,))
    c.commit()
    return {"ok": True}


@router.get("/persons/{pid}/history")
def field_history(pid: str, at: str | None = None):
    c = get_conn()
    if not repo.get_person(c, pid):
        raise HTTPException(404, "人物不存在")
    rows = c.execute("SELECT f.*,e.excerpt,e.source_ref,e.blob_hash,e.kind evidence_kind FROM person_field f "
        "LEFT JOIN evidence e ON e.id=f.evidence_id WHERE f.person_id=? ORDER BY f.field,f.valid_from DESC", (pid,)).fetchall()
    result = [dict(r) for r in rows]
    if at:
        from app.memory.apply import _date_ms
        ts = _date_ms(at)
        if not ts:
            raise HTTPException(400, "日期格式应为 YYYY-MM-DD")
        result = [r for r in result if r["valid_from"] <= ts and (r["valid_to"] is None or r["valid_to"] > ts)]
    return {"history": result, "corrections": corrections.logs(c, 'person', pid)}


def correct(operation):
    """Serialize version checks and writes, including both sides of a transfer."""
    c = get_conn()
    try:
        c.execute('BEGIN IMMEDIATE')
        result = operation(c)
        c.commit()
        return result
    except corrections.Conflict as e:
        c.rollback()
        raise HTTPException(409, str(e))
    except ValueError as e:
        c.rollback()
        raise HTTPException(400, str(e))
    except Exception:
        c.rollback()
        raise


class ManualFieldIn(BaseModel):
    field: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z][A-Za-z0-9_.-]*$')
    value: str = Field(min_length=1, max_length=5000)
    expected_valid_from: StrictInt | None = None
    reason: str = Field(default='', max_length=1000)


@router.post('/persons/{pid}/fields')
def manual_field(pid: str, body: ManualFieldIn):
    return correct(lambda c: corrections.set_field(c, pid, body.field, body.value,
        body.expected_valid_from, body.reason, check='expected_valid_from' in body.model_fields_set))


class FieldCorrectionIn(BaseModel):
    field: str = Field(min_length=1, max_length=100)
    action: Literal['delete', 'restore', 'transfer']
    expected_valid_from: StrictInt | None
    version: StrictInt | None = None
    target_person_id: str | None = None
    target_expected_valid_from: StrictInt | None = None
    replace_target: bool = False
    reason: str = Field(default='', max_length=1000)


@router.post('/persons/{pid}/fields/correct')
def field_correction(pid: str, body: FieldCorrectionIn):
    if body.action == 'delete':
        return correct(lambda c: corrections.remove_field(c, pid, body.field, body.expected_valid_from, body.reason))
    if body.action == 'restore':
        if body.version is None:
            raise HTTPException(400, '请选择历史版本')
        return correct(lambda c: corrections.restore_field(c, pid, body.field, body.version,
                                                           body.expected_valid_from, body.reason))
    if not body.target_person_id:
        raise HTTPException(400, '请选择接收人物')
    return correct(lambda c: corrections.transfer_field(c, pid, body.field, body.target_person_id,
        body.expected_valid_from, body.target_expected_valid_from, body.replace_target, body.reason))


@router.get('/evidence/{eid}/file')
def evidence_file(eid: str):
    e = get_conn().execute('SELECT * FROM evidence WHERE id=?', (eid,)).fetchone()
    if not e or not e['blob_hash'] or not re.fullmatch(r'[a-f0-9]{64}', e['blob_hash']):
        raise HTTPException(404, '原始附件不存在')
    path = blob._path_for(e['blob_hash'])
    if not path.is_file():
        raise HTTPException(404, '原始附件已被清除')
    return FileResponse(path, filename=Path(e['source_ref'] or 'attachment').name)


@router.post("/promises/{pid}/complete")
def complete_promise(pid: str):
    c = get_conn()
    if not c.execute("SELECT 1 FROM promise WHERE id=?", (pid,)).fetchone():
        raise HTTPException(404, "承诺不存在")
    repo.set_promise_status(c, pid, "done")
    c.commit()
    return {"ok": True}


@router.get("/ledger")
def ledger(include_void: bool = False, limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)):
    c = get_conn()
    rows = repo.ledger_entries(c, include_void=include_void, limit=limit + 1, offset=offset)
    return {"totals": repo.ledger_totals(c), "entries": [dict(r) for r in rows[:limit]],
            "offset": offset, "limit": limit, "has_more": len(rows) > limit}


@router.get('/ledger/{lid}')
def ledger_detail(lid: str):
    c = get_conn()
    try:
        row = corrections.ledger_row(c, lid)
    except ValueError as e:
        raise HTTPException(404, str(e))
    e = c.execute('SELECT * FROM evidence WHERE id=?', (row['evidence_id'],)).fetchone()
    return {'entry': row, 'evidence': dict(e) if e else None,
            'corrections': corrections.logs(c, 'ledger', lid)}


class LedgerCorrectionIn(BaseModel):
    expected_revision: StrictInt = Field(ge=0)
    person_id: str | None = None
    direction: Literal['in', 'out']
    amount_cents: StrictInt = Field(gt=0, le=100000000000000)
    occurred_on: date
    note: str = Field(default='', max_length=2000)
    reason: str = Field(default='', max_length=1000)


@router.patch('/ledger/{lid}')
def ledger_correction(lid: str, body: LedgerCorrectionIn):
    from app.config import tz
    occurred_at = int(datetime.combine(body.occurred_on, time(12), tzinfo=tz()).timestamp() * 1000)
    values = {'person_id': body.person_id, 'direction': body.direction, 'amount_cents': body.amount_cents,
              'occurred_at': occurred_at, 'note': body.note.strip()}
    return correct(lambda c: corrections.change_ledger(c, lid, body.expected_revision, values, reason=body.reason))


class LedgerVoidIn(BaseModel):
    expected_revision: StrictInt = Field(ge=0)
    reason: str = Field(default='', max_length=1000)


@router.post('/ledger/{lid}/void')
def ledger_void(lid: str, body: LedgerVoidIn):
    return correct(lambda c: corrections.change_ledger(c, lid, body.expected_revision, void=True, reason=body.reason))


@router.get("/promises")
def promises():
    rows = []
    c = get_conn()
    for row in repo.open_promises(c):
        p = dict(row)
        if p["person_id"]:
            person = repo.get_person(c, p["person_id"])
            if not person or person["deleted_at"] or person["status"] == "merged":
                continue
        p["due"] = ms_to_local_str(p["due_at"])[:10] if p["due_at"] else None
        rows.append(p)
    return {"promises": rows}


@router.post("/tasks/journal", status_code=202)
def journal_task():
    return tasks.submit(get_conn(), "journal", "生成今天的关系日志", {})


@router.get("/export")
def export():
    snapshot = sqlite3.connect(":memory:")
    get_conn().backup(snapshot)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("home.db", snapshot.serialize())
        root = get_settings().abs_blob_dir
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and ".tmp." not in p.name:
                    archive.write(p, "blobs/" + p.relative_to(root).as_posix())
    snapshot.close()
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="home-backup.zip"'})
