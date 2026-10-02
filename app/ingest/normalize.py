"""摄入路由：文件/URL → 转写/解析 → 抽取 → 落库。全程状态可见（FR-ING-08）。

顺序：先抽取+落库（人物会被创建/绑定），再插入 interaction 并链接到已解析的人。
"""
from __future__ import annotations

import json
from pathlib import Path

from app.ingest import chatlog, documents, screenshot, webpage
from app.memory import apply as apply_mod
from app.memory import extract
from app.store import blob, repo
from app.util import clamp, now_ms, sha256_bytes, today_local

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DOC_EXT = {".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".txt", ".md"}


def _known_names(conn) -> list[str]:
    return [p["display_name"] for p in repo.all_active_persons(conn)][:100]


def _link_person_ids(conn, applied: dict) -> list[tuple[str, str]]:
    return [(pid, "mentioned") for pid in applied.get("person_ids", [])]


def ingest_file(conn, filename: str, data: bytes, source_msg_id: str | None = None) -> dict:
    ext = Path(filename).suffix.lower()
    # 聊天记录导出（微信/QQ）走专用解析：保留发言人+时间，按人切块抽取
    if ext in (".txt", ".md") and chatlog.looks_like_chatlog(chatlog.decode_text(data)):
        return chatlog.import_chatlog(conn, filename, data, source_msg_id=source_msg_id)
    kind = "screenshot" if ext in IMAGE_EXT else "document"
    jid, created = repo.create_job(conn, kind, payload=filename,
                                   idempotency_key=sha256_bytes(data))
    if not created:
        return {"job_id": jid, "status": repo.get_job(conn, jid)["status"], "duplicate": True,
                "message": "相同文件已摄入过，未重复处理"}
    conn.commit()
    repo.update_job(conn, jid, "parsing")
    conn.commit()

    blob_hash = blob.put(data)
    try:
        if ext in IMAGE_EXT:
            text = screenshot.transcribe_screenshot(data)
            channel = "screenshot"
        elif ext in DOC_EXT:
            text = documents.extract_text(filename, data)
            channel = "document"
        else:
            raise ValueError(f"不支持的文件类型 {ext}")

        evidence_id = repo.add_evidence(conn, "blob", blob_hash=blob_hash,
                                        source_ref=filename, excerpt=clamp(text, 300))
        extraction = extract.extract_from_text(text, _known_names(conn), today_local())
        applied = apply_mod.apply_extraction(conn, extraction, source_msg_id or jid,
                                             text, channel=channel, evidence_id=evidence_id)
        iid, _ = repo.insert_interaction(conn, now_ms(), channel, clamp(text, 4000),
                                         filename, _link_person_ids(conn, applied))
        n_ok = sum(1 for o in applied["outcomes"] if o["status"] == "ok")
        n_pending = sum(1 for o in applied["outcomes"] if o["status"] == "pending")
        n_failed = sum(o["status"] == "failed" for o in applied["outcomes"])
        status = "partial" if n_pending or n_failed else "done"
        repo.update_job(conn, jid, status,
                        result=json.dumps({"ok": n_ok, "pending": n_pending,
                                           "interaction_id": iid, "evidence_id": evidence_id},
                                          ensure_ascii=False))
        conn.commit()
        return {"job_id": jid, "status": status, "channel": channel,
                "transcript": clamp(text, 2000), "interaction_id": iid, **applied}
    except Exception as e:  # noqa: BLE001
        repo.update_job(conn, jid, "failed", error=str(e))
        conn.commit()
        raise


def ingest_url(conn, url: str, source_msg_id: str | None = None) -> dict:
    jid, created = repo.create_job(conn, "webpage", payload=url,
                                   idempotency_key=sha256_bytes(url.encode()))
    if not created:
        return {"job_id": jid, "status": repo.get_job(conn, jid)["status"], "duplicate": True,
                "message": "相同 URL 已摄入过，未重复处理"}
    conn.commit()
    repo.update_job(conn, jid, "parsing")
    conn.commit()
    try:
        page = webpage.fetch_and_clean(url)
        text = f"标题：{page['title']}\n来源：{url}\n\n{page['text']}"
        evidence_id = repo.add_evidence(conn, "url", source_ref=url, excerpt=clamp(text, 300))
        extraction = extract.extract_from_text(text, _known_names(conn), today_local())
        applied = apply_mod.apply_extraction(conn, extraction, source_msg_id or jid,
                                             text, channel="web", evidence_id=evidence_id)
        iid, _ = repo.insert_interaction(conn, now_ms(), "web", clamp(text, 4000), url,
                                         _link_person_ids(conn, applied))
        status = "partial" if any(o["status"] in {"failed", "pending"} for o in applied["outcomes"]) else "done"
        repo.update_job(conn, jid, status,
                        result=json.dumps({"title": page["title"], "interaction_id": iid,
                                           "evidence_id": evidence_id}, ensure_ascii=False))
        conn.commit()
        return {"job_id": jid, "status": status, "title": page["title"],
                "interaction_id": iid, **applied}
    except Exception as e:  # noqa: BLE001
        repo.update_job(conn, jid, "failed", error=str(e))
        conn.commit()
        raise
