"""Durable local tasks. Work is resumed explicitly after interruption."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

from app.store import repo, blob
from app.store.db import get_conn
from app.util import new_id, now_ms

_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="home-task")
_running = set()
_lock = threading.Lock()
TERMINAL = {"done", "partial", "failed", "cancelled", "interrupted"}


class Cancelled(RuntimeError):
    pass


def read(conn, tid):
    row = conn.execute("SELECT * FROM task WHERE id=?", (tid,)).fetchone()
    if not row:
        raise ValueError("任务不存在")
    result = dict(row)
    result["payload"] = json.loads(result["payload"])
    result["result"] = json.loads(result["result"]) if result["result"] else None
    last = conn.execute("SELECT data FROM task_event WHERE task_id=? AND kind IN ('progress','queued','started') ORDER BY seq DESC LIMIT 1", (tid,)).fetchone()
    result['progress'] = json.loads(last[0]).get('message', '') if last else ''
    return result


def event(conn, tid, kind, data):
    row = conn.execute("SELECT cancel_requested FROM task WHERE id=?", (tid,)).fetchone()
    if row and row[0] and kind not in {"cancelled", "failed", "interrupted"}:
        raise Cancelled("任务已取消；此前已确认的写入仍保留")
    conn.execute("INSERT INTO task_event(task_id,kind,data,created_at) VALUES(?,?,?,?)",
                 (tid, kind, json.dumps(data, ensure_ascii=False), now_ms()))
    conn.execute("UPDATE task SET updated_at=? WHERE id=?", (now_ms(), tid))
    conn.commit()


def submit(conn, kind, title, payload, key=None, dispatch=True):
    if key:
        existing = conn.execute("SELECT id FROM task WHERE idempotency_key=?", (key,)).fetchone()
        if existing:
            return read(conn, existing[0])
    tid, ts = new_id(), now_ms()
    conn.execute("INSERT INTO task(id,kind,title,payload,idempotency_key,created_at,updated_at) "
                 "VALUES(?,?,?,?,?,?,?)", (tid, kind, title, json.dumps(payload, ensure_ascii=False), key, ts, ts))
    conn.commit()
    event(conn, tid, "queued", {"message": "任务已排队"})
    if dispatch:
        start(tid)
    return read(conn, tid)


def start(tid):
    with _lock:
        if tid in _running:
            return
        _running.add(tid)
    _pool.submit(_run, tid)


def _run(tid):
    conn = get_conn()
    try:
        t = read(conn, tid)
        if t["status"] != "queued":
            return
        conn.execute("UPDATE task SET status='running',error=NULL WHERE id=?", (tid,))
        conn.commit()
        emit = lambda kind, data: event(conn, tid, kind, data)
        emit("started", {"message": "开始处理"})
        p = t["payload"]
        if t["kind"] == "chat":
            from app.agent.harness import run_turn
            if not p.get("user_message_id"):
                p["user_message_id"] = repo.add_message(conn, p["conversation_id"], "user", p["text"])
                conn.execute("UPDATE task SET payload=? WHERE id=?", (json.dumps(p, ensure_ascii=False), tid))
                conn.commit()
            result = run_turn(conn, p["conversation_id"], p["text"], emit=emit,
                              user_message_id=p["user_message_id"])
        elif t["kind"] in {"file", "chatlog", "url"}:
            from app.ingest import normalize, chatlog
            emit("progress", {"message": "解析内容并提取人物信息"})
            if t["kind"] == "url":
                result = normalize.ingest_url(conn, p["url"])
            else:
                data = blob.get(p["blob_hash"])
                if t["kind"] == "chatlog":
                    result = chatlog.import_chatlog(conn, p["filename"], data, me=p.get("me"))
                else:
                    result = normalize.ingest_file(conn, p["filename"], data)
            emit("progress", {"message": "解析完成，人物与回执已保存"})
        elif t["kind"] == "research_prepare":
            from app.agent.research import prepare_research
            result = prepare_research(conn, p["topic"], emit=emit, rid=p.get("research_id"))
        elif t["kind"] == "research":
            from app.agent.research import run_research
            result = run_research(conn, p["topic"], rid=p["research_id"], emit=emit)
        elif t["kind"] == "journal":
            from app.jobs.scheduler import generate_journal
            result = {"text": generate_journal(p.get("day"))}
        else:
            raise ValueError("不支持的任务类型")
        failed = result.get("failed") or [o for o in result.get("outcomes", []) if o["status"] == "failed"]
        status = "partial" if failed or result.get("status") == "partial" else "done"
        emit("result", result)
        conn.execute("UPDATE task SET status=?,result=?,error=NULL,updated_at=? WHERE id=?",
                     (status, json.dumps(result, ensure_ascii=False), now_ms(), tid))
        repo.add_notification(conn, "task", f"{t['title']}：{'部分完成，请查看回执' if status == 'partial' else '已完成'}")
        conn.commit()
    except Cancelled as e:
        conn.rollback()
        if t["kind"] in {"research", "research_prepare"} and t["payload"].get("research_id"):
            repo.update_research(conn, t["payload"]["research_id"], status="cancelled")
        conn.execute("UPDATE task SET status='cancelled',error=?,updated_at=? WHERE id=?", (str(e), now_ms(), tid))
        conn.commit()
        event(conn, tid, "cancelled", {"message": str(e)})
    except Exception as e:
        conn.rollback()
        if t["kind"] in {"research", "research_prepare"} and t["payload"].get("research_id"):
            repo.update_research(conn, t["payload"]["research_id"], status="failed")
        from app.config import get_settings
        message = str(e).replace(get_settings().llm_api_key, "[REDACTED]") if get_settings().llm_api_key else str(e)
        conn.execute("UPDATE task SET status='failed',error=?,updated_at=? WHERE id=?", (message, now_ms(), tid))
        conn.commit()
        event(conn, tid, "failed", {"message": message})
    finally:
        with _lock:
            _running.discard(tid)


def retry(conn, tid):
    with _lock:
        if tid in _running:
            raise ValueError("任务仍在结束中，请稍后重试")
    t = read(conn, tid)
    if t["status"] not in {"failed", "cancelled", "interrupted", "partial"}:
        raise ValueError("仅失败、取消、中断或部分完成的任务可以重试")
    conn.execute("UPDATE task SET status='queued',cancel_requested=0,error=NULL,result=NULL,updated_at=? WHERE id=?",
                 (now_ms(), tid))
    conn.commit()
    event(conn, tid, "queued", {"message": "正在重试，已保存内容不会重复写入"})
    start(tid)
    return read(conn, tid)


def recover():
    conn = get_conn()
    conn.execute("UPDATE task SET status='interrupted',error='应用关闭导致中断，可重试',updated_at=? "
                 "WHERE status='running'", (now_ms(),))
    conn.commit()
    for row in conn.execute("SELECT id FROM task WHERE status='queued'").fetchall():
        start(row[0])
