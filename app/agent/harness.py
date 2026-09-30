"""Agent harness：工具循环 + 上下文预算 + 时间间隔感知 + 写后抽取回执。

上下文预算策略（PRD §7.4）：
- 注入：系统提示 + 个人记忆全量(FR-MEM-05) + 相关人档案卡(≤3) + 近7天情节摘要 + 历史消息
- 超预算：旧消息压缩为摘要（kv 缓存，每 +10 条重算）
- 被裁掉的原文留库可回溯，不删除
"""
from __future__ import annotations

import json
import re

from app.agent import prompts, tools
from app.config import get_settings
from app.llm import get_llm
from app.memory import compile_card, extract
from app.store import repo
from app.util import clamp, ms_to_local_str, now_ms, today_local

_MENTION = re.compile(r"@([^\s@，。,.！!？?]{1,20})")
_MAX_TOOL_ROUNDS = 10
_KEEP_RECENT_MSGS = 12


def _detect_mentioned_person_ids(conn, text: str) -> list[str]:
    """@提及 + 已知别名子串命中。"""
    ids: list[str] = []
    seen = set()

    def add(pid):
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)

    for m in _MENTION.findall(text):
        p = repo.get_person_by_name(conn, m)
        if p:
            add(p["id"])
    # 别名子串（长度≥2 避免单字误伤）
    for row in repo.alias_rows(conn):
        alias = row["alias"]
        if len(alias) >= 2 and alias in text:
            add(row["person_id"])
    return ids[:3]


def _context_blocks(conn, person_ids: list[str]) -> str:
    blocks = []
    mine = repo.my_profile_memories(conn)
    if mine:
        blocks.append("【关于用户自己（每轮全量提供）】\n" +
                      "\n".join(f"- {m['text']}" for m in mine[:50]))
    for pid in person_ids:
        card = compile_card.build_card(conn, pid)
        if card:
            blocks.append("【相关人物档案卡】\n" + card)
    recent = repo.memories_since(conn, now_ms() - 7 * 86400000, kinds=["episodic"])
    if recent:
        lines = []
        for m in recent[-10:]:
            who = ""
            if m["person_id"]:
                p = repo.get_person(conn, m["person_id"])
                who = (p["display_name"] + "：") if p else ""
            lines.append(f"- [{ms_to_local_str(m['ts'])[:10]}] {who}{m['text']}")
        blocks.append("【最近 7 天发生的事】\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _history_messages(conn, conversation_id: str) -> list[dict]:
    """历史消息 → chat 格式，超预算则压缩旧段为摘要（kv 缓存）。"""
    s = get_settings()
    rows = repo.list_messages(conn, conversation_id, limit=500)
    msgs = [{"role": r["role"], "content": r["content"] or ""}
            for r in rows if r["role"] in ("user", "assistant")]
    total = sum(len(m["content"]) for m in msgs)
    if total <= s.ctx_budget_chars or len(msgs) <= _KEEP_RECENT_MSGS:
        return msgs

    recent = msgs[-_KEEP_RECENT_MSGS:]
    old = msgs[:-_KEEP_RECENT_MSGS]
    summarized_count = int(repo.get_kv(conn, f"summ_count_{conversation_id}", "0"))
    summary = repo.get_kv(conn, f"summ_text_{conversation_id}", "")
    if len(old) - summarized_count >= 10 or not summary:
        llm = get_llm()
        old_text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in old)
        summary = llm.chat([
            {"role": "system", "content": "把以下对话历史压缩成一段不超过 300 字的摘要，"
             "保留：涉及的人、承诺、关键事实、未决事项。"},
            {"role": "user", "content": old_text[:8000]},
        ], tier="mini", max_tokens=500)["content"]
        repo.set_kv(conn, f"summ_text_{conversation_id}", summary)
        repo.set_kv(conn, f"summ_count_{conversation_id}", str(len(old)))
        conn.commit()
    return [{"role": "user", "content": f"（此前对话摘要）{summary}"},
            {"role": "assistant", "content": "好的，我记得之前的内容。"}] + recent


def run_turn(conn, conversation_id: str, user_text: str) -> dict:
    s = get_settings()
    llm = get_llm()

    conv = repo.get_conversation(conn, conversation_id)
    if not conv:
        repo.create_conversation(conn)
        conv = repo.get_conversation(conn, conversation_id)

    # 时间间隔感知（FR-TIM-03）
    last_ts = repo.last_message_ts(conn, conversation_id)
    gap_days = int((now_ms() - last_ts) / 86400000) if last_ts else None

    user_msg_id = repo.add_message(conn, conversation_id, "user", user_text)
    conn.commit()

    person_ids = _detect_mentioned_person_ids(conn, user_text)
    memory_block = _context_blocks(conn, person_ids)

    messages: list[dict] = [{"role": "system", "content": prompts.build_system(gap_days)}]
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages += _history_messages(conn, conversation_id)
    messages.append({"role": "user", "content": user_text})

    # ---- 工具循环 ----
    receipts: list[dict] = []
    answer = ""
    for _ in range(_MAX_TOOL_ROUNDS):
        msg = llm.chat(messages, tier="mid", tools=tools.TOOLS, max_tokens=3000)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            answer = msg.get("content", "")
            break
        messages.append(msg)
        for tc in tool_calls:
            args = tools.arguments_of(tc)
            try:
                result = tools.execute_tool(conn, tc["function"]["name"], args)
            except Exception as e:  # noqa: BLE001
                result = {"ok": False, "error": str(e)}
            if result.get("receipt"):
                receipts.append(result["receipt"])
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, ensure_ascii=False)[:6000]})
    else:
        answer = msg.get("content", "") or "（已达到最大推理轮数，请换个问法或缩小范围）"

    if not answer.strip():
        answer = "（这一轮没有产出内容，请重试）"  # FR-CHT-06

    repo.add_message(conn, conversation_id, "assistant", answer)

    # 首轮生成标题
    if conv["title"] == "新对话":
        try:
            title = llm.chat([
                {"role": "system", "content": "给这段对话起一个 10 字以内的标题，只输出标题。"},
                {"role": "user", "content": user_text[:300]},
            ], tier="mini", max_tokens=30)["content"].strip().strip('"“”')
            if title:
                repo.set_conversation_title(conn, conversation_id, title[:20])
        except Exception:  # noqa: BLE001
            pass

    # ---- 写后抽取（P2：系统完成写入并给回执，模型不自称"已记录"） ----
    outcomes: list[dict] = []
    pending_ids: list[str] = []
    try:
        known = [p["display_name"] for p in repo.all_active_persons(conn)][:100]
        extraction = extract.extract_from_text(user_text, known, today_local())
        from app.memory import apply as apply_mod
        applied = apply_mod.apply_extraction(conn, extraction, user_msg_id, user_text)
        outcomes = applied["outcomes"]
        pending_ids = applied["pending_ids"]
        # 把这条用户消息记为一次互动，并链接到涉及的人（驱动档案卡/重要度/谁安静了）
        linked = [(pid, "mentioned") for pid in applied.get("person_ids", [])]
        if linked:
            repo.insert_interaction(conn, now_ms(), "chat", clamp(user_text, 4000),
                                    conversation_id, linked)
            for pid, _role in linked:
                repo.recompute_importance(conn, pid)
    except Exception as e:  # noqa: BLE001
        outcomes = [{"kind": "extract", "status": "failed", "detail": str(e)}]

    conn.commit()
    saved = [o for o in outcomes if o["status"] == "ok"]
    pending = [o for o in outcomes if o["status"] == "pending"]
    return {
        "answer": answer,
        "conversation_id": conversation_id,
        "receipts": receipts,
        "saved": saved,
        "pending": pending,
        "pending_ids": pending_ids,
    }
