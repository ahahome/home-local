"""聊天记录导入：微信 / QQ 导出的纯文本记录 → 逐条消息 → 复用既有抽取管线。

为什么单独做一个模块：截图/文档摄入是「一段文本」，而聊天记录是「一条条带发言人和时间的
消息」。直接整篇丢给抽取器会丢掉两样关键信息——谁说的、什么时候说的。这里先解析成消息流，
再按人切块喂给抽取器。

设计要点：
1. 容错解析：不假设单一导出格式，逐行状态机依次尝试五类行模式
   （`[时间] 名字: 内容` / `名字 时间戳` / `时间戳 名字` / 仅日期行 / `名字: 内容`）。
2. 身份归属：聊天里「对方」常常没有名字，所以按 (对方, 日期) 切块，把对方名字写进片段表头，
   让抽取器知道这批事实属于谁。
3. 成本有界：默认最多处理 20 个片段，超出时只取最近的，并如实回报 truncated，不假装全量。
4. 不假装知道（P7）：无法确定「我」是谁时，用出现最多的一方作为猜测并置 me_guessed=True。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime

from app.config import tz
from app.memory import apply as apply_mod
from app.memory import extract
from app.store import repo
from app.util import clamp, now_ms, sha256_bytes, today_local

# ---------------- 行模式 ----------------

# [2024-05-12 14:23:11] 张三: 在吗
_BRACKET_TS = re.compile(
    r"^\[(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<rest>.*)$")
# 张三 2024-05-12 14:23:11
_NAME_TS = re.compile(
    r"^(?P<name>[^\s:：]{1,24})\s+(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*[:：]?\s*(?P<rest>.*)$")
# 2024-05-12 14:23:11 张三
_TS_REST = re.compile(
    r"^(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*[:：]?\s*(?P<rest>.*)$")
# 2024-05-12 / 2024年5月12日 星期一
_DATE_ONLY = re.compile(
    r"^(?P<date>\d{4}[-/年]\d{1,2}[-/月]\d{1,2})日?(?:\s*星期[一二三四五六日天])?\s*$")
# 张三: 在吗
_SPEAKER_INLINE = re.compile(r"^(?P<name>[^\s:：]{1,12})\s*[:：]\s*(?P<text>.+)$")
# 消息对象:李四
_PARTNER = re.compile(r"^消息对象\s*[:：]\s*(?P<name>.+?)\s*$")

NOISE_PREFIX = ("===", "---", "___", "消息记录", "消息分组", "日期:", "导出时间", "消息类型")
SYSTEM_MARKERS = ("撤回了一条消息", "以上是打招呼的内容", "对方开启了朋友验证",
                  "你已添加了", "已经是好友", "请使用最新版本", "开启了朋友验证")
SELF_NAMES = {"我", "本人", "自己", "me", "Me", "ME", "自己本人"}

MEDIA_RE = re.compile(r"^\[(图片|表情|动画表情|语音|视频|文件|位置|链接|音乐|转账|红包)\]$")


# ---------------- 解码与工具 ----------------


def decode_text(data: bytes) -> str:
    """微信/QQ 导出可能是 utf-8（有时带 BOM）或 GBK，逐个试。"""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    try:
        t = data.decode("utf-8-sig")
        if "\ufffd" not in t:
            return t
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("gb18030")
    except UnicodeDecodeError:
        pass
    return data.decode("utf-8", errors="replace")


def _clean_name(raw: str) -> str:
    """去掉尾部括号（QQ 号/备注）与冒号。"""
    n = (raw or "").strip().strip(":：").strip()
    n = re.sub(r"[（(][^（()）]{0,40}[)）]\s*$", "", n).strip()
    return n


def _norm_date(raw: str) -> str | None:
    m = re.match(r"^(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", (raw or "").strip())
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _ms(datestr: str | None, timestr: str | None) -> int | None:
    """本地时区毫秒时间戳；无日期返回 None，有日期无时间按当日 12:00（与 apply 同约定）。"""
    if not datestr:
        return None
    hh, mm, ss = 12, 0, 0
    if timestr:
        parts = [int(x) for x in re.findall(r"\d+", timestr)[:3]]
        if len(parts) >= 2:
            hh, mm = parts[0], parts[1]
            ss = parts[2] if len(parts) >= 3 else 0
    try:
        y, mo, d = (int(x) for x in datestr.split("-"))
        return int(datetime(y, mo, d, hh, mm, ss, tzinfo=tz()).timestamp() * 1000)
    except ValueError:
        return None


def _hhmm(ts: int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts / 1000, tz()).strftime("%H:%M")


def _is_noise(line: str) -> bool:
    return line.startswith(NOISE_PREFIX) or set(line) <= {"=", "-", "_", " ", "*"}


# ---------------- 解析 ----------------


def parse(text: str, me: str | None = None) -> dict:
    """把导出文本解析成消息流。返回 messages / speakers / me / stats / span 等。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stats = {"lines": len(lines), "parsed": 0, "skipped": 0, "system": 0}
    messages: list[dict] = []
    partner_hint: str | None = None
    seen_speakers: set[str] = set()
    cur_date: str | None = None
    seen_ts = False
    cur: dict | None = None

    def flush() -> None:
        nonlocal cur
        if cur and cur["speaker"]:
            body = "\n".join(cur["lines"]).strip()
            if body:
                messages.append({"ts": cur["ts"], "date": cur["date"],
                                 "speaker": cur["speaker"], "text": body})
        cur = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _PARTNER.match(line)
        if m:
            partner_hint = _clean_name(m.group("name"))
            continue
        if _is_noise(line):
            stats["skipped"] += 1
            continue
        if any(k in line for k in SYSTEM_MARKERS):
            stats["system"] += 1
            continue

        # --- 带时间戳的头部行 ---
        m = _BRACKET_TS.match(line)
        name, body, date_s, time_s = None, "", None, None
        if m:
            date_s, time_s, rest = m.group("date"), m.group("time"), m.group("rest").strip()
        else:
            m = _TS_REST.match(line)
            if m:
                date_s, time_s, rest = m.group("date"), m.group("time"), m.group("rest").strip()
            else:
                m = _NAME_TS.match(line)
                if m:
                    name = _clean_name(m.group("name"))
                    date_s, time_s, rest = m.group("date"), m.group("time"), m.group("rest").strip()

        if date_s:
            flush()
            seen_ts = True
            cur_date = _norm_date(date_s)
            if name is None and rest:
                if len(rest) <= 24 and " " not in rest and not re.search(r"[:：]", rest):
                    name = _clean_name(rest)
                else:
                    parts = re.split(r"[:：]", rest, maxsplit=1)
                    if len(parts) == 2:
                        name, body = _clean_name(parts[0]), parts[1].strip()
                    else:
                        body = rest
            if name:
                seen_speakers.add(name)
            cur = {"date": cur_date, "ts": _ms(cur_date, time_s),
                   "speaker": name, "lines": [body] if body else []}
            continue

        # --- 仅日期行 ---
        m = _DATE_ONLY.match(line)
        if m:
            flush()
            cur_date = _norm_date(m.group("date"))
            continue

        # --- 名字: 内容 ---
        m = _SPEAKER_INLINE.match(line)
        if m:
            cand = _clean_name(m.group("name"))
            accept = (not seen_ts) or (cur is None) or (cand in seen_speakers)
            if accept and cand:
                flush()
                seen_speakers.add(cand)
                cur = {"date": cur_date, "ts": None, "speaker": cand,
                       "lines": [m.group("text").strip()]}
                continue

        # --- 正文行 ---
        if cur is not None and cur["speaker"]:
            cur["lines"].append(line)
        else:
            stats["skipped"] += 1

    flush()

    for msg in messages:
        if msg["ts"] is None:
            msg["ts"] = _ms(msg["date"], None)

    speakers = Counter(m["speaker"] for m in messages)
    stats["parsed"] = len(messages)
    dates = [m["date"] for m in messages if m["date"]]

    # 「我」是谁：显式 > 自称 > 导出里的对话对象反推 > 出现最多（标记为猜测）
    me_name, guessed = None, False
    if me and me in speakers:
        me_name = me
    elif me:
        me_name, guessed = me, True
    elif any(s in SELF_NAMES for s in speakers):
        me_name = next(s for s in speakers if s in SELF_NAMES)
    elif partner_hint and partner_hint in speakers and len(speakers) == 2:
        me_name = next(s for s in speakers if s != partner_hint)
    elif speakers:
        me_name, guessed = speakers.most_common(1)[0][0], True

    return {
        "messages": messages,
        "speakers": dict(speakers),
        "partner_hint": partner_hint,
        "me": me_name,
        "me_guessed": guessed,
        "stats": stats,
        "span": [min(dates), max(dates)] if dates else [None, None],
    }


def looks_like_chatlog(text: str) -> bool:
    """粗判是不是聊天记录，避免把普通文档误当聊天解析。"""
    lines = [l.strip() for l in text.replace("\r\n", "\n").split("\n") if l.strip()]
    probe = lines[:500]
    if not probe:
        return False
    names: set[str] = set()
    ts_hits = 0
    inline_hits = 0
    for line in probe:
        m = _BRACKET_TS.match(line) or _TS_REST.match(line)
        if m:
            ts_hits += 1
            rest = m.group("rest").strip()
            if rest:
                nm = _clean_name(re.split(r"[:：]", rest)[0])
                if 0 < len(nm) <= 24:
                    names.add(nm)
            continue
        m = _NAME_TS.match(line)
        if m:
            ts_hits += 1
            names.add(_clean_name(m.group("name")))
            continue
        m = _SPEAKER_INLINE.match(line)
        if m:
            inline_hits += 1
            names.add(_clean_name(m.group("name")))
    if ts_hits >= 3 and len(names) >= 2:
        return True
    if inline_hits >= 8 and 2 <= len(names) <= 8:
        counts = Counter()
        for line in probe:
            m = _SPEAKER_INLINE.match(line)
            if m:
                counts[_clean_name(m.group("name"))] += 1
        return bool(counts) and counts.most_common(1)[0][1] >= 3
    return False


# ---------------- 导入 ----------------

CHUNK_HEAD = "【对话片段】日期 {day}；与你对话的人是「{partner}」（记录里「{me}」是我自己）"


def _format_chunk(day: str, partner: str, msgs: list[dict], me_name: str) -> str:
    out = [CHUNK_HEAD.format(day=day, partner=partner, me=me_name)]
    for m in msgs:
        t = _hhmm(m["ts"])
        out.append(f"{t} {m['speaker']}：{clamp(m['text'], 300)}".strip())
    return "\n".join(out)


def _known_names(conn) -> list[str]:
    return [p["display_name"] for p in repo.all_active_persons(conn)][:100]


def import_chatlog(conn, filename: str, data: bytes, me: str | None = None,
                   max_chunks: int = 20, source_msg_id: str | None = None) -> dict:
    """导入一份聊天记录导出文件：解析 → 按 (对方, 日期) 切块 → 抽取 → 幂等落库。

    返回结果里 updated 的字段都如实反映「做了什么/没做什么」，包含 truncated 与 me_guessed。
    """
    jid, created = repo.create_job(conn, "chatlog", payload=filename,
                                   idempotency_key=sha256_bytes(data))
    if not created:
        return {"job_id": jid, "status": repo.get_job(conn, jid)["status"], "duplicate": True,
                "message": "相同聊天记录文件已导入过，未重复处理"}
    conn.commit()
    repo.update_job(conn, jid, "parsing")
    conn.commit()

    try:
        parsed = parse(decode_text(data), me=me)
        if not parsed["messages"]:
            raise ValueError("没解析出任何聊天消息，请确认是微信/QQ 导出的纯文本聊天记录")

        me_name = parsed["me"] or "我"
        partners = [s for s in parsed["speakers"] if s != me_name]

        by_day: dict[str, list[dict]] = {}
        for msg in parsed["messages"]:
            by_day.setdefault(msg["date"] or "未知日期", []).append(msg)

        specs: list[tuple[str, str, list[dict]]] = []
        for partner in partners:
            for day, msgs in by_day.items():
                if any(m["speaker"] == partner for m in msgs):
                    specs.append((day, partner, msgs))
        specs.sort(key=lambda t: (t[0], t[1]))

        total = len(specs)
        used = specs[-max_chunks:] if total > max_chunks else specs
        truncated = total > len(used)

        outcomes: list[dict] = []
        person_ids: list[str] = []
        pending_ids: list[str] = []
        for day, partner, msgs in used:
            chunk = _format_chunk(day, partner, msgs, me_name)
            extraction = extract.extract_from_text(chunk, _known_names(conn), today_local())
            applied = apply_mod.apply_extraction(
                conn, extraction, source_msg_id or f"{jid}|{day}|{partner}", chunk,
                channel="chatlog")
            outcomes += applied["outcomes"]
            person_ids += applied["person_ids"]
            pending_ids += applied["pending_ids"]
            repo.insert_interaction(conn, max((m["ts"] or 0) for m in msgs) or now_ms(),
                                    "chatlog", clamp(chunk, 4000), f"{filename}#{day}",
                                    [(pid, "mentioned") for pid in applied.get("person_ids", [])])

        n_ok = sum(1 for o in outcomes if o["status"] == "ok")
        n_pending = sum(1 for o in outcomes if o["status"] == "pending")
        status = "partial" if (n_pending or truncated or any(o["status"] == "failed" for o in outcomes)) else "done"
        repo.update_job(conn, jid, status, result=json.dumps(
            {"ok": n_ok, "pending": n_pending, "messages": len(parsed["messages"]),
             "chunks_total": total, "chunks_used": len(used)}, ensure_ascii=False))
        conn.commit()

        span = parsed["span"]
        summary = (f"解析 {len(parsed['messages'])} 条消息（{len(parsed['speakers'])} 位发言人"
                   f"{'，' + span[0] + ' ~ ' + span[1] if span[0] else ''}）；"
                   f"识别「我」= {me_name}{'（自动判断，可能不准）' if parsed['me_guessed'] and not me else ''}；"
                   f"处理 {len(used)} 个对话片段"
                   f"{f'（共 {total} 个，仅取最近 {len(used)} 个）' if truncated else ''}")
        return {"job_id": jid, "status": status, "channel": "chatlog",
                "transcript": summary, "me": me_name,
                "me_guessed": parsed["me_guessed"] and not bool(me),
                "speakers": parsed["speakers"], "partner_hint": parsed["partner_hint"],
                "messages": len(parsed["messages"]), "span": span,
                "chunks_total": total, "chunks_used": len(used), "truncated": truncated,
                "stats": parsed["stats"],
                "outcomes": outcomes, "person_ids": person_ids, "pending_ids": pending_ids}
    except Exception as e:  # noqa: BLE001
        repo.update_job(conn, jid, "failed", error=str(e))
        conn.commit()
        raise
