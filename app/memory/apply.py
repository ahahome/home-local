"""幂等写入管线（FR-ING-06 / FR-ING-07）+ 合并/拆分（FR-ID-05/06）。

语义保证：
- 每条 patch 一个 SAVEPOINT，局部失败只回滚该条，其余保持已保存
- 应用层指纹幂等：重放同一条消息不产生重复记忆/账本（applied_patch + 各表唯一键）
- 逐条回报 outcome，绝不声称全部成功（P2）
- 待确认条目不进入任何汇总（FR-MEM-03 / FR-LED-03）
- 合并可反向拆分（merge_log 快照）
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config import get_settings, tz
from app.memory import disambiguate
from app.memory.extract import is_role_person
from app.store import repo
from app.store.db import savepoint
from app.util import (clamp, new_id, normalize_value, now_ms, patch_fingerprint,
                      sha256_text, today_local)


def _date_ms(datestr: str | None) -> int | None:
    """YYYY-MM-DD → 本地时区当日 12:00 的毫秒时间戳（避免时区边界）。"""
    if not datestr:
        return None
    try:
        d = datetime.strptime(datestr.strip(), "%Y-%m-%d").replace(hour=12, tzinfo=tz())
        return int(d.timestamp() * 1000)
    except ValueError:
        return None


class Outcome(dict):
    def __init__(self, kind: str, status: str, detail: str = ""):
        super().__init__(kind=kind, status=status, detail=detail)


def apply_extraction(conn, extraction: dict, source_msg_id: str, evidence_excerpt: str,
                     channel: str = "chat", forced: dict[str, str] | None = None) -> dict:
    """把 extract 的结构化结果落库。返回 {outcomes, person_ids, pending_ids}。

    forced: {canonical_name: person_id}，用于「用户已确认身份」的重放——绕过二次消歧。
    """
    s = get_settings()
    forced = forced or {}
    outcomes: list[dict] = []
    person_ids: list[str] = []
    pending_ids: list[str] = []

    # 本批次共用一个证据（截图时为 blob 证据，文本为 text 证据）
    evidence_id = repo.add_evidence(conn, "text", source_ref=source_msg_id,
                                    excerpt=clamp(evidence_excerpt, 500))

    for p in extraction.get("people", []):
        name = p["canonical_name"]
        # ---- FR-ID-01：角色词不建档（落库层是最后一道闸门） ----
        # sanitize() 已经拦过一道，这里再拦一道：apply 可能被直接调用
        # （例如待确认面板重放 patch），不能因为「上游应该过滤了」就放行。
        if is_role_person(name) or is_role_person(p.get("name")):
            outcomes.append(Outcome("person", "skipped",
                                    f"{name} 是角色词/亲属称谓，不建档（FR-ID-01）"))
            continue
        # ---- 身份消歧（forced 命中的跳过：用户已确认身份） ----
        if name in forced:
            person_id = forced[name]
        else:
            res = disambiguate.resolve(conn, name)
            if res.kind == "confirm":
                # 存疑必回问（P1）：整条 patch 挂起，等用户确认身份后再应用
                pid_pending = repo.add_pending(conn, "person_identity", {
                    "mention": p["name"], "canonical_name": name,
                    "candidates": [{"person_id": c.person_id, "display_name": c.display_name,
                                    "score": round(c.score, 3)} for c in res.candidates[:3]],
                    "patch": p, "source_msg_id": source_msg_id, "channel": channel,
                    "evidence_id": evidence_id,
                }, reason=f"身份不确定（最高分 {res.score:.2f} 且存在并列候选）")
                pending_ids.append(pid_pending)
                outcomes.append(Outcome("person", "pending", f"{name} 待确认身份"))
                continue

            if res.kind == "bound":
                person_id = res.person_id
            else:  # new：明确的新人物直接创建，不卡在确认循环（FR-ID-01）
                person_id = repo.create_person(conn, name, status="active")
                outcomes.append(Outcome("person", "ok", f"新建人物 {name}"))
        person_ids.append(person_id)
        repo.add_alias(conn, person_id, p["name"], channel, 0.9)
        _index_person_name(conn, person_id, name)

        # ---- 字段（双时态 upsert + 置信度分档 + 指纹幂等） ----
        for f in p["fields"]:
            fp = patch_fingerprint(person_id, f["field"], f["value"], source_msg_id)
            if repo.has_patch(conn, fp):
                outcomes.append(Outcome("field", "duplicate", f"{f['field']}={f['value']}"))
                continue
            try:
                with savepoint(conn, f"sp_{new_id()[:8]}"):
                    if f["confidence"] >= s.confidence_auto:
                        repo.upsert_field(conn, person_id, f["field"], f["value"],
                                          f["confidence"], evidence_id)
                        repo.mark_patch(conn, fp, person_id, f["field"],
                                        normalize_value(f["value"]), source_msg_id)
                        outcomes.append(Outcome("field", "ok", f"{f['field']}={f['value']}"))
                    elif f["confidence"] >= s.confidence_confirm:
                        pid2 = repo.add_pending(conn, "field", {
                            "person_id": person_id, "field": f["field"], "value": f["value"],
                            "confidence": f["confidence"], "evidence": f["evidence"],
                            "evidence_id": evidence_id, "source_msg_id": source_msg_id,
                            "fingerprint": fp,
                        }, reason="置信度不足，待确认")
                        pending_ids.append(pid2)
                        outcomes.append(Outcome("field", "pending", f"{f['field']}={f['value']}"))
                    else:
                        outcomes.append(Outcome("field", "skipped",
                                                f"{f['field']}={f['value']}（置信度过低 {f['confidence']:.2f}）"))
            except Exception as e:  # noqa: BLE001
                outcomes.append(Outcome("field", "failed", f"{f['field']}: {e}"))

        # ---- 别名 ----
        for a in p["aliases"]:
            if a["confidence"] >= 0.7:
                repo.add_alias(conn, person_id, a["alias"], channel, a["confidence"])

        # ---- 事实 → 情节记忆 ----
        for fa in p["facts"]:
            fp = sha256_text(f"fact|{person_id}|{normalize_value(fa['text'])}")
            if repo.has_patch(conn, fp):
                outcomes.append(Outcome("fact", "duplicate", clamp(fa["text"], 30)))
                continue
            try:
                with savepoint(conn, f"sp_{new_id()[:8]}"):
                    mid = repo.add_memory(conn, "episodic", fa["text"], person_id,
                                          fa["confidence"], evidence_id)
                    repo.mark_patch(conn, fp, person_id, "fact",
                                    normalize_value(fa["text"]), source_msg_id)
                    _embed_memory(conn, mid, fa["text"])
                    outcomes.append(Outcome("fact", "ok", clamp(fa["text"], 30)))
            except Exception as e:  # noqa: BLE001
                outcomes.append(Outcome("fact", "failed", str(e)))

        # ---- 承诺 ----
        for pr in p["promises"]:
            fp = sha256_text(f"promise|{person_id}|{normalize_value(pr['content'])}|{pr.get('due')}")
            if repo.has_patch(conn, fp):
                outcomes.append(Outcome("promise", "duplicate", clamp(pr["content"], 30)))
                continue
            try:
                with savepoint(conn, f"sp_{new_id()[:8]}"):
                    if pr["confidence"] >= s.confidence_confirm:
                        due_ms = _date_ms(pr.get("due"))
                        pmid = repo.add_promise(conn, pr["who"], pr["content"], person_id,
                                                due_ms, evidence_id)
                        repo.mark_patch(conn, fp, person_id, "promise",
                                        normalize_value(pr["content"]), source_msg_id)
                        if due_ms:
                            who_txt = "我答应" if pr["who"] == "me" else f"{name} 答应"
                            repo.add_reminder(conn, due_ms, f"{who_txt}：{pr['content']}",
                                              person_id, pmid)
                        outcomes.append(Outcome("promise", "ok", clamp(pr["content"], 30)))
                    else:
                        pid2 = repo.add_pending(conn, "promise", {
                            "person_id": person_id, **pr, "evidence_id": evidence_id,
                            "source_msg_id": source_msg_id, "fingerprint": fp,
                        }, reason="承诺置信度不足，待确认")
                        pending_ids.append(pid2)
                        outcomes.append(Outcome("promise", "pending", clamp(pr["content"], 30)))
            except Exception as e:  # noqa: BLE001
                outcomes.append(Outcome("promise", "failed", str(e)))

        # ---- 账本（去重窗口 + 待确认不计入总额） ----
        for le in p["ledger"]:
            occurred_ms = _date_ms(le.get("occurred")) or now_ms()
            dh = sha256_text(f"ledger|{person_id}|{le['direction']}|{le['amount_cents']}|{le.get('occurred')}")
            try:
                with savepoint(conn, f"sp_{new_id()[:8]}"):
                    # 窗口外的相似历史 → 必须走确认（FR-LED-02）
                    window_ms = s.ledger_dedupe_days * 86400000
                    similar = repo.find_similar_transfer(conn, person_id, le["amount_cents"],
                                                         occurred_ms - window_ms)
                    if similar and le["confidence"] < s.confidence_auto:
                        pid2 = repo.add_pending(conn, "ledger", {
                            "person_id": person_id, **le, "evidence_id": evidence_id,
                            "source_msg_id": source_msg_id, "dedupe_hash": dh,
                            "similar_ids": [r["id"] for r in similar],
                        }, reason="与历史转账相似，需确认是否重复")
                        pending_ids.append(pid2)
                        outcomes.append(Outcome("ledger", "pending",
                                                f"{le['direction']} {le['amount_cents']/100:.2f}（疑重复）"))
                        continue
                    status = "confirmed" if le["confidence"] >= s.confidence_auto else "pending"
                    lid, created = repo.add_ledger(conn, le["direction"], le["amount_cents"],
                                                   person_id, le["note"], occurred_ms, status,
                                                   dh, evidence_id)
                    if not created:
                        outcomes.append(Outcome("ledger", "duplicate",
                                                f"{le['direction']} {le['amount_cents']/100:.2f}"))
                    else:
                        repo.mark_patch(conn, sha256_text("ledgerpatch|" + dh), person_id,
                                        "ledger", str(le["amount_cents"]), source_msg_id)
                        outcomes.append(Outcome("ledger", "ok" if status == "confirmed" else "pending",
                                                f"{le['direction']} {le['amount_cents']/100:.2f}"))
            except Exception as e:  # noqa: BLE001
                outcomes.append(Outcome("ledger", "failed", str(e)))

        # ---- 价值评估（不覆盖用户手动修改：仅在更高置信度时更新） ----
        vr = p.get("value_read")
        if vr and vr["confidence"] >= 0.7:
            cur = repo.get_person(conn, person_id)
            if cur and (cur["value_read"] or "") != vr["text"]:
                repo.update_value_read(conn, person_id, vr["text"], vr.get("evidence"))
                outcomes.append(Outcome("value_read", "ok", clamp(vr["text"], 30)))

        # ---- 关系边 ----
        for r in p["relations"]:
            # FR-ID-01 旁路：不解这一条，「前台」「王芳的老公」还是会因为出现在
            # relation.to 里被 create_person 建出来，闸门等于白设。
            if is_role_person(r.get("to")):
                outcomes.append(Outcome("relation", "skipped",
                                        f"{r.get('to')} 是角色词，不建档也不建边（FR-ID-01）"))
                continue
            try:
                with savepoint(conn, f"sp_{new_id()[:8]}"):
                    target = repo.get_person_by_name(conn, r["to"])
                    if target is None:
                        tid = repo.create_person(conn, r["to"], status="active")
                        _index_person_name(conn, tid, r["to"])
                    else:
                        tid = target["id"]
                    if tid != person_id:
                        repo.upsert_edge(conn, person_id, tid, r["kind"],
                                         strength=max(0.3, r["confidence"]), evidence_id=evidence_id)
                        outcomes.append(Outcome("relation", "ok", f"{name} -{r['kind']}-> {r['to']}"))
            except Exception as e:  # noqa: BLE001
                outcomes.append(Outcome("relation", "failed", str(e)))

        repo.recompute_importance(conn, person_id)

    # ---- 用户自己的目标/偏好（FR-MEM-05 的写入侧） ----
    for n in extraction.get("my_notes", []):
        fp = sha256_text(f"mynote|{normalize_value(n['text'])}")
        if repo.has_patch(conn, fp):
            continue
        try:
            with savepoint(conn, f"sp_{new_id()[:8]}"):
                mid = repo.add_memory(conn, n["kind"], n["text"], None, n["confidence"], evidence_id)
                repo.mark_patch(conn, fp, None, "mynote", normalize_value(n["text"]), source_msg_id)
                _embed_memory(conn, mid, n["text"])
                outcomes.append(Outcome("note", "ok", clamp(n["text"], 30)))
        except Exception as e:  # noqa: BLE001
            outcomes.append(Outcome("note", "failed", str(e)))

    conn.commit()
    return {"outcomes": outcomes, "person_ids": person_ids, "pending_ids": pending_ids}


def _index_person_name(conn, person_id: str, name: str) -> None:
    try:
        from app.llm import get_llm
        from app.memory.vector import get_index
        vec = get_llm().embed([name])[0]
        get_index().add(conn, "person_name", person_id, name, vec)
    except Exception:  # noqa: BLE001
        pass  # 向量索引失败不阻塞主流程


def _embed_memory(conn, memory_id: str, text: str) -> None:
    try:
        from app.llm import get_llm
        from app.memory.vector import get_index
        vec = get_llm().embed([text])[0]
        get_index().add(conn, "memory", memory_id, text, vec)
    except Exception:  # noqa: BLE001
        pass


# ---------------- 合并 / 拆分（FR-ID-05 / FR-ID-06） ----------------

def merge_persons(conn, survivor_id: str, merged_id: str) -> str:
    """a 存活，b 变墓碑。单事务 + 快照，支持拆分。永不物理删除（P1/P5）。"""
    survivor = repo.get_person(conn, survivor_id)
    merged = repo.get_person(conn, merged_id)
    if not survivor or not merged:
        raise ValueError("人物不存在")
    if survivor["deleted_at"] or merged["deleted_at"]:
        raise ValueError("不能合并已删除的人物")

    b = merged_id
    snapshot: dict[str, Any] = {
        "person": dict(merged),
        "aliases": [[r["alias"], r["source"], r["confidence"]]
                    for r in conn.execute("SELECT * FROM person_alias WHERE person_id=?", (b,)).fetchall()],
        "fields": [[r["field"], r["valid_from"]]
                   for r in conn.execute("SELECT field,valid_from FROM person_field WHERE person_id=?", (b,)).fetchall()],
        "memories": [r["id"] for r in conn.execute(
            "SELECT id FROM memory_item WHERE person_id=?", (b,)).fetchall()],
        "promises": [r["id"] for r in conn.execute(
            "SELECT id FROM promise WHERE person_id=?", (b,)).fetchall()],
        "ledger": [r["id"] for r in conn.execute(
            "SELECT id FROM ledger_entry WHERE person_id=?", (b,)).fetchall()],
        "interactions": [r["interaction_id"] for r in conn.execute(
            "SELECT interaction_id FROM interaction_person WHERE person_id=?", (b,)).fetchall()],
        "edges": [dict(r) for r in conn.execute(
            "SELECT * FROM relation_edge WHERE from_person=? OR to_person=?", (b, b)).fetchall()],
    }

    now = now_ms()
    # 别名：迁到 a（OR IGNORE 语义）
    for alias, source, conf in snapshot["aliases"]:
        conn.execute("UPDATE OR IGNORE person_alias SET person_id=? WHERE alias=? AND source=?",
                     (survivor_id, alias, source))
    conn.execute("DELETE FROM person_alias WHERE person_id=?", (b,))
    for alias, source, conf in snapshot["aliases"]:
        conn.execute("INSERT OR IGNORE INTO person_alias(person_id,alias,source,confidence,created_at)"
                     " VALUES(?,?,?,?,?)", (survivor_id, alias, source, conf, now))
    # 其余按 id/键迁移
    conn.execute("UPDATE person_field SET person_id=? WHERE person_id=?", (survivor_id, b))
    conn.execute("UPDATE memory_item SET person_id=? WHERE person_id=?", (survivor_id, b))
    conn.execute("UPDATE promise SET person_id=? WHERE person_id=?", (survivor_id, b))
    conn.execute("UPDATE ledger_entry SET person_id=? WHERE person_id=?", (survivor_id, b))
    for iid in snapshot["interactions"]:
        conn.execute("INSERT OR IGNORE INTO interaction_person(interaction_id,person_id,role)"
                     " VALUES(?,?,'mentioned')", (iid, survivor_id))
    conn.execute("DELETE FROM interaction_person WHERE person_id=?", (b,))
    for e in snapshot["edges"]:
        other = e["to_person"] if e["from_person"] == b else e["from_person"]
        if other == b:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO relation_edge(id,from_person,to_person,kind,strength,evidence_id,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (new_id(), survivor_id, other, e["kind"], e["strength"], e["evidence_id"], now))
    conn.execute("DELETE FROM relation_edge WHERE from_person=? OR to_person=?", (b, b))
    if not survivor["value_read"] and merged["value_read"]:
        conn.execute("UPDATE person SET value_read=?, value_read_evidence=? WHERE id=?",
                     (merged["value_read"], merged["value_read_evidence"], survivor_id))
    # b → 墓碑
    conn.execute("UPDATE person SET status='merged', merged_into=?, deleted_at=?, updated_at=? WHERE id=?",
                 (survivor_id, now, now, b))
    log_id = repo.add_merge_log(conn, survivor_id, b, snapshot)
    repo.recompute_importance(conn, survivor_id)
    conn.commit()
    return log_id


def split_merge(conn, merge_log_id: str) -> None:
    """按快照把合并拆回去（FR-ID-06）。"""
    import json
    log = repo.get_merge_log(conn, merge_log_id)
    if not log or log["undone"]:
        raise ValueError("合并记录不存在或已撤销")
    snap = json.loads(log["snapshot"])
    b = snap["person"]["id"]
    survivor_id = log["survivor_id"]

    # b 恢复为未确认状态
    conn.execute("UPDATE person SET status='unconfirmed', merged_into=NULL, deleted_at=NULL, updated_at=? WHERE id=?",
                 (now_ms(), b))
    # 按快照把行指回 b
    for field, valid_from in snap["fields"]:
        conn.execute("UPDATE person_field SET person_id=? WHERE person_id=? AND field=? AND valid_from=?",
                     (b, survivor_id, field, valid_from))
    for table, key in (("memory_item", "memories"), ("promise", "promises"), ("ledger_entry", "ledger")):
        for rid in snap[key]:
            conn.execute(f"UPDATE {table} SET person_id=? WHERE id=?", (b, rid))
    for alias, source, conf in snap["aliases"]:
        conn.execute("UPDATE OR IGNORE person_alias SET person_id=? WHERE alias=? AND source=?",
                     (b, alias, source))
    for iid in snap["interactions"]:
        conn.execute("INSERT OR IGNORE INTO interaction_person(interaction_id,person_id,role)"
                     " VALUES(?,?,'mentioned')", (iid, b))
    for e in snap["edges"]:
        conn.execute(
            "INSERT OR IGNORE INTO relation_edge(id,from_person,to_person,kind,strength,evidence_id,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (new_id(), e["from_person"], e["to_person"], e["kind"], e["strength"],
             e["evidence_id"], now_ms()))
    repo.mark_merge_undone(conn, merge_log_id)
    repo.recompute_importance(conn, b)
    repo.recompute_importance(conn, survivor_id)
    conn.commit()
