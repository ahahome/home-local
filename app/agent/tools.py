"""工具集：OpenAI function-calling schema + 执行分发。

所有写操作工具必须返回 receipt（P2：写入必有回执）。
"""
from __future__ import annotations

import json

from app.memory import compile_card, retrieve
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


def _resolve_person_id(conn, name: str | None) -> str | None:
    if not name:
        return None
    p = repo.get_person_by_name(conn, name)
    return p["id"] if p else None


def execute_tool(conn, name: str, args: dict) -> dict:
    from app.config import tz
    from datetime import datetime

    if name == "search_people":
        hits = retrieve.search_people(conn, args["query"])
        return {"ok": True, "results": hits}

    if name == "get_person":
        pid = _resolve_person_id(conn, args["name"])
        if not pid:
            return {"ok": False, "error": f"没有找到「{args['name']}」的档案"}
        card = compile_card.build_card(conn, pid)
        p = repo.get_person(conn, pid)
        return {"ok": True, "card": card, "person_id": pid,
                "importance": p["importance"], "status": p["status"]}

    if name == "save_person_fact":
        pid = _resolve_person_id(conn, args["person"])
        if not pid:
            pid = repo.create_person(conn, args["person"], status="active")
        eid = repo.add_evidence(conn, "text", excerpt=args.get("evidence", ""))
        repo.upsert_field(conn, pid, args["field"], args["value"], 1.0, eid)
        repo.touch_person(conn, pid)
        conn.commit()
        return {"ok": True, "receipt": {"type": "field", "person_id": pid,
                                        "field": args["field"], "value": args["value"]}}

    if name == "query_memory":
        results = retrieve.hybrid_search(conn, args["query"], args.get("top_k", 8))
        for r in results:
            r["when"] = ms_to_local_str(r["ts"])
        return {"ok": True, "results": results}

    if name == "create_promise":
        pid = _resolve_person_id(conn, args.get("person"))
        due_ms = None
        if args.get("due_date"):
            try:
                due_ms = int(datetime.strptime(args["due_date"], "%Y-%m-%d")
                             .replace(hour=9, tzinfo=tz()).timestamp() * 1000)
            except ValueError:
                return {"ok": False, "error": "due_date 格式应为 YYYY-MM-DD"}
        pmid = repo.add_promise(conn, args["who"], args["content"], pid, due_ms)
        rid = None
        if due_ms:
            who_txt = "我答应" if args["who"] == "me" else f"{args.get('person') or '对方'} 答应"
            rid = repo.add_reminder(conn, due_ms, f"{who_txt}：{args['content']}", pid, pmid)
        conn.commit()
        return {"ok": True, "receipt": {"type": "promise", "id": pmid, "reminder_id": rid}}

    if name == "list_open_promises":
        pid = _resolve_person_id(conn, args.get("person"))
        rows = repo.open_promises(conn, pid)
        return {"ok": True, "promises": [
            {"id": r["id"], "who": r["who"], "content": r["content"],
             "person": r["display_name"] if "display_name" in r.keys() else args.get("person"),
             "due": ms_to_local_str(r["due_at"])[:10] if r["due_at"] else None} for r in rows]}

    if name == "complete_promise":
        repo.set_promise_status(conn, args["promise_id"], "done")
        conn.commit()
        return {"ok": True, "receipt": {"type": "promise_done", "id": args["promise_id"]}}

    if name == "record_transfer":
        pid = _resolve_person_id(conn, args.get("person"))
        cents = int(round(float(args["amount_yuan"]) * 100))
        occurred = now_ms()
        if args.get("occurred_date"):
            try:
                occurred = int(datetime.strptime(args["occurred_date"], "%Y-%m-%d")
                               .replace(hour=12, tzinfo=tz()).timestamp() * 1000)
            except ValueError:
                pass
        from app.util import sha256_text
        dh = sha256_text(f"ledger|{pid}|{args['direction']}|{cents}|{args.get('occurred_date') or 'today'}")
        lid, created = repo.add_ledger(conn, args["direction"], cents, pid,
                                       args.get("note"), occurred, "confirmed", dh)
        conn.commit()
        return {"ok": True, "receipt": {"type": "ledger", "id": lid, "duplicate": not created,
                                        "amount_yuan": args["amount_yuan"]}}

    if name == "ledger_summary":
        pid = _resolve_person_id(conn, args.get("person"))
        totals = repo.ledger_totals(conn, pid)
        entries = repo.ledger_entries(conn, pid, include_pending=False)
        return {"ok": True,
                "totals": {"in_yuan": totals["in_cents"] / 100, "out_yuan": totals["out_cents"] / 100,
                           "count": totals["count"]},
                "entries": [{"direction": e["direction"], "amount_yuan": e["amount_cents"] / 100,
                             "note": e["note"], "person": e["display_name"],
                             "when": ms_to_local_str(e["occurred_at"])[:10]} for e in entries[:20]]}

    if name == "save_skill":
        pid = _resolve_person_id(conn, args.get("person"))
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
        result = research_mod.run_research(conn, args["topic"])
        return {"ok": True, "receipt": {"type": "research", "id": result["id"]},
                "summary": result["summary"], "report_md": result["report_md"],
                "sources": result["sources"], "thin": result.get("thin", False)}

    if name == "create_reminder":
        try:
            fire_ms = int(datetime.strptime(args["fire_date"], "%Y-%m-%d")
                          .replace(hour=9, tzinfo=tz()).timestamp() * 1000)
        except ValueError:
            return {"ok": False, "error": "fire_date 格式应为 YYYY-MM-DD"}
        pid = _resolve_person_id(conn, args.get("person"))
        rid = repo.add_reminder(conn, fire_ms, args["text"], pid)
        conn.commit()
        return {"ok": True, "receipt": {"type": "reminder", "id": rid,
                                        "fire_at": ms_to_local_str(fire_ms)}}

    if name == "who_quiet":
        people = retrieve.quiet_people(conn, args.get("days", 14))
        return {"ok": True, "quiet": [
            {**p, "last_interaction": ms_to_local_str(p["last_interaction"])[:10]
             if p["last_interaction"] else "从未"} for p in people]}

    if name == "note":
        mid = repo.add_memory(conn, "semantic", args["text"], None, 1.0)
        conn.commit()
        return {"ok": True, "receipt": {"type": "note", "id": mid}}

    return {"ok": False, "error": f"未知工具 {name}"}


def arguments_of(tool_call: dict) -> dict:
    try:
        return json.loads(tool_call["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, KeyError):
        return {}
