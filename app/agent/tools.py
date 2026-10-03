"""工具集：OpenAI function-calling schema + 执行分发。

所有写操作工具必须返回 receipt（P2：写入必有回执）。
"""
from __future__ import annotations

import json

from app.memory import compile_card, retrieve
from app.money import amount_to_cents
from app.store import repo
from app.util import ms_to_local_str, now_ms

TOOLS = [
    {"type": "function", "function": {
        "name": "search_people",
        "description": "按名字/昵称模糊搜索人物，返回候选人列表。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "名字、昵称或拼音"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_person",
        "description": "获取某人的完整档案卡：字段、最近互动、未闭合承诺、账本汇总、价值评估。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "save_person_fact",
        "description": "用户明确告知某人信息时，保存一个字段（如 company/title/city/birthday/trait.*）。",
        "parameters": {"type": "object", "properties": {
            "person": {"type": "string"}, "field": {"type": "string"},
            "value": {"type": "string"}, "evidence": {"type": "string", "description": "用户原话"}},
            "required": ["person", "field", "value"]}}},
    {"type": "function", "function": {
        "name": "query_memory",
        "description": "跨全部记忆与互动做语义+全文检索（谁说过什么、发生过什么）。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer", "default": 8}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_promise",
        "description": "记录一条承诺（我答应的或对方答应的），可带截止日期，会自动建提醒。",
        "parameters": {"type": "object", "properties": {
            "person": {"type": "string"}, "who": {"type": "string", "enum": ["me", "them"]},
            "content": {"type": "string"}, "due_date": {"type": "string", "description": "YYYY-MM-DD，可空"}},
            "required": ["who", "content"]}}},
    {"type": "function", "function": {
        "name": "list_open_promises",
        "description": "列出未闭合的承诺。不传 person 则列出全部。",
        "parameters": {"type": "object", "properties": {"person": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "complete_promise",
        "description": "把一条承诺标记为已完成。",
        "parameters": {"type": "object", "properties": {
            "promise_id": {"type": "string"}}, "required": ["promise_id"]}}},
    {"type": "function", "function": {
        "name": "record_transfer",
        "description": "记录一笔与某人的资金往来（转账/收款）。",
        "parameters": {"type": "object", "properties": {
            "person": {"type": "string"},
            "direction": {"type": "string", "enum": ["in", "out"], "description": "in=我收到 out=我付出"},
            "amount_yuan": {"type": "number"}, "note": {"type": "string"},
            "occurred_date": {"type": "string", "description": "YYYY-MM-DD，默认今天"}},
            "required": ["direction", "amount_yuan"]}}},
    {"type": "function", "function": {
        "name": "ledger_summary",
        "description": "查看账本汇总与明细（待确认不计入总额）。",
        "parameters": {"type": "object", "properties": {"person": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "save_skill",
        "description": "把一套打法/某人的沟通风格保存为可复用技能，之后用 /技能名 唤起。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "person": {"type": "string"},
            "style_notes": {"type": "string", "description": "沟通风格要点"},
            "body": {"type": "string", "description": "技能正文：步骤/话术/注意事项"},
            "trigger_hint": {"type": "string"}},
            "required": ["name", "body"]}}},
    {"type": "function", "function": {
        "name": "run_skill",
        "description": "取出一个已保存技能的完整内容，用于指导当前回复。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list_skills",
        "description": "列出所有已保存的技能。", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "deep_research",
        "description": "对一个人/公司/行业做深度调研，出一份带来源的报告。耗时较长。",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string"}}, "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "create_reminder",
        "description": "创建一个到点提醒。",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "fire_date": {"type": "string", "description": "YYYY-MM-DD"},
            "person": {"type": "string"}},
            "required": ["text", "fire_date"]}}},
    {"type": "function", "function": {
        "name": "who_quiet",
        "description": "找出最近 N 天没有互动、按重要度排序的人（谁最近安静了）。",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "default": 14}}}}},
    {"type": "function", "function": {
        "name": "note",
        "description": "记录一条关于用户自己的目标/偏好/状态。",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}}, "required": ["text"]}}},
]


for _tool in TOOLS:
    if _tool['function']['name'] in {'get_person', 'save_person_fact', 'create_promise', 'list_open_promises',
                                    'record_transfer', 'ledger_summary', 'save_skill', 'create_reminder'}:
        _tool['function']['parameters']['properties']['person_id'] = {
            'type': 'string', 'description': '已知人物 ID；有同名人物时必须使用检索返回的 ID'}


def _resolve_person_id(conn, name: str | None, person_id: str | None = None) -> str | None:
    if not name and not person_id:
        return None
    from app.memory.writes import resolve_person
    return resolve_person(conn, name, person_id=person_id)


def execute_tool(conn, name: str, args: dict, source_text: str | None = None,
                 source_msg_id: str | None = None) -> dict:
    from app.memory.writes import AmbiguousPerson
    try:
        if name in {"save_person_fact", "record_transfer", "create_promise", "note"}:
            return _shared_write(conn, name, args, source_text, source_msg_id)
        return _execute_tool(conn, name, args)
    except AmbiguousPerson as e:
        return {"ok": False, "needs_confirmation": True, "error": str(e), "candidates": e.candidates}
    except (ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": str(e)}


def _shared_write(conn, name, args, source_text, source_msg_id):
    from app.memory.apply import apply_extraction, _date_ms
    from app.memory.writes import resolve_person, promise_key, matching_promise
    from app.config import get_settings
    from app.util import new_id, normalize_value, sha256_text
    source_id = source_msg_id or new_id()
    forced = {}
    for key in ("due_date", "occurred_date", "valid_from"):
        if args.get(key) and not _date_ms(args[key]):
            raise ValueError("日期格式应为 YYYY-MM-DD")
    source = source_text if source_text is not None else args.get("evidence", "手动记录")
    confidence = 0.95
    if source_text is not None:
        value = str(args.get("value") or args.get("content") or args.get("text") or "")
        if not value or value not in source:
            confidence = 0.7
    if name == "record_transfer":
        import re
        cents = amount_to_cents(args["amount_yuan"])
        if args["direction"] not in ("in", "out"):
            raise ValueError("金额必须为正，方向必须为 in 或 out")
        if source_text is not None:
            mentioned = False
            for value in re.findall(r"\d+(?:\.\d+)?", source):
                try:
                    mentioned |= amount_to_cents(value) == cents
                except ValueError:
                    pass
            confidence = .95 if mentioned and re.search(r"转|给|收|付|借|还|礼金|随礼|报销", source) else .7
    if name == "note":
        obj = {"people": [], "my_notes": [{"text": args["text"], "kind": "semantic", "confidence": confidence}]}
    elif name == "create_promise" and not args.get("person") and not args.get('person_id'):
        due = args.get("due_date") or None
        if due and not _date_ms(due):
            raise ValueError("日期格式应为 YYYY-MM-DD")
        fp = promise_key(None, args["who"], args["content"], due)
        if repo.has_patch(conn, fp) or matching_promise(conn, None, args['who'], args['content'], due, _date_ms(due)):
            return {"ok": True, "receipt": {"type": "promise", "duplicate": True}}
        eid = repo.add_evidence(conn, "text", source_ref=source_id, excerpt=source)
        if confidence < get_settings().confidence_auto:
            pending = repo.add_pending(conn, "promise", {"person_id": None, "who": args["who"],
                "content": args["content"], "due": due, "confidence": confidence,
                "fingerprint": fp, "source_msg_id": source_id, "evidence_id": eid}, "承诺待确认")
            conn.commit()
            return {"ok": True, "receipt": {"type": "promise", "status": "pending", "pending_id": pending}}
        pid = repo.add_promise(conn, args["who"], args["content"], None, _date_ms(due), eid)
        repo.mark_patch(conn, fp, None, "promise", normalize_value(args["content"]), source_id)
        if due:
            repo.add_reminder(conn, _date_ms(due), args["content"], None, pid)
        conn.commit()
        return {"ok": True, "receipt": {"type": "promise", "id": pid, "status": "ok"}}
    else:
        person = args.get("person")
        if not person and not args.get('person_id'):
            raise ValueError("请指定对应人物")
        pid = resolve_person(conn, person, person_id=args.get('person_id'), create=True)
        person = person or repo.get_person(conn, pid)['display_name']
        forced[person] = pid
        p = {"name": person, "canonical_name": person, "fields": [], "aliases": [],
             "relations": [], "facts": [], "promises": [], "ledger": []}
        if name == "save_person_fact":
            p["fields"] = [{"field": args["field"], "value": args["value"],
                "confidence": confidence, "evidence": source, "valid_from": args.get("valid_from")}]
        elif name == "record_transfer":
            day = args.get("occurred_date") or None
            if day and not _date_ms(day):
                raise ValueError("日期格式应为 YYYY-MM-DD")
            p["ledger"] = [{"direction": args["direction"], "amount_cents": cents,
                "note": args.get("note", ""), "occurred": day, "confidence": confidence}]
        else:
            day = args.get("due_date") or None
            if day and not _date_ms(day):
                raise ValueError("日期格式应为 YYYY-MM-DD")
            p["promises"] = [{"who": args["who"], "content": args["content"],
                "due": day, "confidence": confidence}]
        obj = {"people": [p], "my_notes": []}
    result = apply_extraction(conn, obj, source_id, source, forced=forced)
    return {"ok": not any(o["status"] == "failed" for o in result["outcomes"]),
            **result, "receipt": {"type": name, "outcomes": result["outcomes"],
                                  "pending_ids": result["pending_ids"]}}


def _execute_tool(conn, name: str, args: dict) -> dict:
    from app.config import tz
    from datetime import datetime

    if name == "search_people":
        hits = retrieve.search_people(conn, args["query"])
        return {"ok": True, "results": hits}

    if name == "get_person":
        pid = _resolve_person_id(conn, args.get("name"), args.get('person_id'))
        if not pid:
            return {"ok": False, "error": f"没有找到「{args['name']}」的档案"}
        card = compile_card.build_card(conn, pid)
        p = repo.get_person(conn, pid)
        return {"ok": True, "card": card, "person_id": pid,
                "importance": p["importance"], "status": p["status"]}

    if name == "query_memory":
        results = retrieve.hybrid_search(conn, args["query"], args.get("top_k", 8))
        for r in results:
            r["when"] = ms_to_local_str(r["ts"])
        return {"ok": True, "results": results}

    if name == "list_open_promises":
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        rows = repo.open_promises(conn, pid)
        return {"ok": True, "promises": [
            {"id": r["id"], "who": r["who"], "content": r["content"],
             "person": r["display_name"] if "display_name" in r.keys() else args.get("person"),
             "due": ms_to_local_str(r["due_at"])[:10] if r["due_at"] else None} for r in rows]}

    if name == "complete_promise":
        row = conn.execute("SELECT * FROM promise WHERE id=?", (args["promise_id"],)).fetchone()
        if not row:
            raise ValueError("承诺不存在")
        repo.set_promise_status(conn, args["promise_id"], "done")
        conn.execute("UPDATE reminder SET status='cancelled' WHERE promise_id=? AND status='pending'",
                     (args["promise_id"],))
        conn.commit()
        return {"ok": True, "receipt": {"type": "promise_done", "id": args["promise_id"]}}

    if name == "ledger_summary":
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        totals = repo.ledger_totals(conn, pid)
        entries = repo.ledger_entries(conn, pid, include_pending=False)
        return {"ok": True,
                "totals": {"in_yuan": totals["in_cents"] / 100, "out_yuan": totals["out_cents"] / 100,
                           "count": totals["count"]},
                "entries": [{"direction": e["direction"], "amount_yuan": e["amount_cents"] / 100,
                             "note": e["note"], "person": e["display_name"],
                             "when": ms_to_local_str(e["occurred_at"])[:10]} for e in entries[:20]]}

    if name == "save_skill":
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        sid, created = repo.save_skill(conn, args["name"], args["body"], pid,
                                       args.get("trigger_hint"), args.get("style_notes"))
        conn.commit()
        return {"ok": True, "receipt": {"type": "skill", "id": sid, "name": args["name"],
                                        "created": created},
                "how_to_use": f"之后对我说 /{args['name']} 即可唤起"}

    if name == "run_skill":
        sk = repo.get_skill(conn, args["name"].lstrip("/"))
        if not sk:
            return {"ok": False, "error": f"没有找到技能「{args['name']}」"}
        return {"ok": True, "skill": {"name": sk["name"], "style_notes": sk["style_notes"],
                                       "body": sk["body"], "trigger_hint": sk["trigger_hint"]}}

    if name == "list_skills":
        return {"ok": True, "skills": [
            {"name": s["name"], "person": s["display_name"], "trigger_hint": s["trigger_hint"]}
            for s in repo.list_skills(conn)]}

    if name == "deep_research":
        from app.agent import research as research_mod
        result = research_mod.prepare_research(conn, args["topic"])
        return {"ok": True, "receipt": {"type": "research", "id": result["id"]},
                **result, "message": "已生成研究提案，请在研究面板确认后开始"}

    if name == "create_reminder":
        try:
            fire_ms = int(datetime.strptime(args["fire_date"], "%Y-%m-%d")
                          .replace(hour=9, tzinfo=tz()).timestamp() * 1000)
        except ValueError:
            return {"ok": False, "error": "fire_date 格式应为 YYYY-MM-DD"}
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        rid = repo.add_reminder(conn, fire_ms, args["text"], pid)
        conn.commit()
        return {"ok": True, "receipt": {"type": "reminder", "id": rid,
                                        "fire_at": ms_to_local_str(fire_ms)}}

    if name == "who_quiet":
        people = retrieve.quiet_people(conn, args.get("days", 14))
        return {"ok": True, "quiet": [
            {**p, "last_interaction": ms_to_local_str(p["last_interaction"])[:10]
             if p["last_interaction"] else "从未"} for p in people]}

    return {"ok": False, "error": f"未知工具 {name}"}


def arguments_of(tool_call: dict) -> dict:
    try:
        return json.loads(tool_call["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, KeyError):
        return {}
