"""定时任务（M5/M3）：到期提醒、每日 journal（本地时区）、每周反思归纳。

- 提醒每分钟扫一次，触发后写入 notification（推送层由 API 暴露）
- journal 每天 23:50 生成，全部用用户本地时区（FR-MEM-07 / AC-05）
- 反思记忆必须由后台异步归纳，不在对话中实时生成（FR-MEM-01）
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.config import get_settings, tz
from app.llm import get_llm
from app.store import repo
from app.store.db import get_conn
from app.util import clamp, ms_to_local_str, now_ms, sha256_text, today_local


def fire_due_reminders() -> int:
    conn = get_conn()
    fired = 0
    for r in repo.due_reminders(conn, now_ms()):
        who = ""
        if r["person_id"]:
            p = repo.get_person(conn, r["person_id"])
            who = f"（{p['display_name']}）" if p else ""
        repo.add_notification(conn, "reminder", f"提醒{who}：{r['text']}")
        repo.mark_reminder_fired(conn, r["id"])
        fired += 1
    conn.commit()
    return fired


def _day_range_ms(day: str) -> tuple[int, int]:
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=tz())
    return int(d.timestamp() * 1000), int((d + timedelta(days=1)).timestamp() * 1000)


def generate_journal(day: str | None = None) -> str | None:
    conn = get_conn()
    day = day or today_local()
    start, end = _day_range_ms(day)

    interactions = [i for i in repo.interactions_since(conn, start) if i["ts"] < end]
    episodic = [m for m in repo.memories_since(conn, start, kinds=["episodic"]) if m["ts"] < end]
    if not interactions and not episodic:
        return None

    lines = []
    for i in interactions[:30]:
        names = repo.interaction_person_names(conn, i["id"])
        who = f"（涉及：{'、'.join(names)}）" if names else ""
        lines.append(f"- [{ms_to_local_str(i['ts'])}] {who}{clamp(i['content'], 80)}")
    for m in episodic[:30]:
        lines.append(f"- 记忆：{clamp(m['text'], 80)}")

    text = get_llm().chat([
        {"role": "system", "content": "把下面这一天发生的事整理成一段简洁的中文日志。"
         "分三块：今天联系了谁、发生了什么、有哪些承诺/金钱往来。客观记录，不要评论。"},
        {"role": "user", "content": "\n".join(lines)[:6000]},
    ], tier="mini", max_tokens=800)["content"]

    repo.upsert_journal(conn, day, text)
    conn.commit()
    return text


def reflect_weekly() -> int:
    """从近 7 天情节记忆归纳 1-3 条「经验」（reflective），异步、可去重。"""
    conn = get_conn()
    items = repo.memories_since(conn, now_ms() - 7 * 86400000, kinds=["episodic"])
    if len(items) < 3:
        return 0
    blob = "\n".join(f"- {m['text']}" for m in items[:80])
    obj = get_llm().chat_json([
        {"role": "system", "content": "从这些事件里归纳出 1~3 条关于「如何与这些人相处」的经验。"
         "每条经验要可复用、具体到人。只输出 JSON："
         "{\"lessons\":[{\"person\":\"人名或null\",\"text\":\"经验\"}]}"},
        {"role": "user", "content": blob[:6000]},
    ], tier="think", max_tokens=1000)
    n = 0
    for lesson in (obj.get("lessons") if isinstance(obj, dict) else []) or []:
        text = lesson.get("text", "").strip()
        if not text:
            continue
        pid = None
        if lesson.get("person"):
            p = repo.get_person_by_name(conn, lesson["person"])
            pid = p["id"] if p else None
        fp = sha256_text(f"reflective|{pid}|{text}")
        if repo.has_patch(conn, fp):
            continue
        repo.add_memory(conn, "reflective", text, pid, 0.8)
        repo.mark_patch(conn, fp, pid, "reflective", text, "weekly_reflect")
        n += 1
    conn.commit()
    return n


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    sch = BackgroundScheduler(timezone=get_settings().timezone)
    sch.add_job(fire_due_reminders, "interval", minutes=1, id="reminders", replace_existing=True)
    sch.add_job(generate_journal, CronTrigger(hour=23, minute=50), id="journal", replace_existing=True)
    sch.add_job(reflect_weekly, CronTrigger(day_of_week="sun", hour=23, minute=0),
                id="reflect", replace_existing=True)
    sch.start()
    return sch
