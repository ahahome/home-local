"""dossier card 预编译：把一个人压缩成 ≤ dossier_card_chars 的档案卡，注入上下文。"""
from __future__ import annotations

from app.config import get_settings
from app.store import repo
from app.util import clamp, ms_to_local_str


def build_card(conn, person_id: str) -> str:
    p = repo.get_person(conn, person_id)
    if not p or p["deleted_at"]:
        return ""
    lines = []
    aliases = [a for a in repo.aliases_of(conn, person_id) if a != p["display_name"]]
    head = f"人物：{p['display_name']}"
    if aliases:
        head += f"（别名：{'、'.join(aliases[:4])}）"
    lines.append(head)

    fields = repo.current_fields(conn, person_id)
    if fields:
        fl = "；".join(f"{k}={v['value']}" for k, v in list(fields.items())[:10] if v["value"])
        if fl:
            lines.append(f"字段：{fl}")
    if p["value_read"]:
        lines.append(f"价值评估：{p['value_read']}")

    for i in repo.recent_interactions(conn, person_id, 3):
        label = "联系" if i["kind"] == "contact" and i["person_role"] == "participant" else "资料记录"
        lines.append(f"{label}[{ms_to_local_str(i['ts'])[:10]}]：{clamp(i['content'], 60)}")

    for pr in repo.open_promises(conn, person_id)[:5]:
        due = ms_to_local_str(pr["due_at"])[:10] if pr["due_at"] else "无期限"
        who = "我答应" if pr["who"] == "me" else "对方答应"
        lines.append(f"未闭合承诺（{who}·{due}）：{pr['content']}")

    tot = repo.ledger_totals(conn, person_id)
    if tot["count"]:
        lines.append(f"账本：收 {tot['in_cents']/100:.2f} / 支 {tot['out_cents']/100:.2f}（{tot['count']} 笔）")

    return clamp("\n".join(lines), get_settings().dossier_card_chars)


def build_cards_for_names(conn, names: list[str]) -> str:
    """给一组人名出档案卡块（注入 harness 上下文用）。"""
    cards = []
    seen = set()
    for n in names:
        p = repo.get_person_by_name(conn, n)
        if p and p["id"] not in seen:
            seen.add(p["id"])
            card = build_card(conn, p["id"])
            if card:
                cards.append(card)
    return "\n\n".join(cards)
