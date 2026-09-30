"""批量导入（FR-ING-05）：CSV / vCard 名单。同名候选走待确认队列，支持批量确认。"""
from __future__ import annotations

import csv
import io
import re

from app.memory import disambiguate
from app.store import repo


def import_csv_names(conn, text: str) -> dict:
    """每行一个名字（或含 name 列的 CSV）。返回 {created, pending, skipped}。"""
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        return {"created": 0, "pending": 0, "skipped": 0, "items": []}
    # 探测表头
    header = [c.strip().lower() for c in rows[0]]
    name_idx = 0
    if any(h in ("name", "姓名", "名字", "备注") for h in header):
        name_idx = next(i for i, h in enumerate(header) if h in ("name", "姓名", "名字", "备注"))
        rows = rows[1:]

    created = pending = skipped = 0
    items = []
    for r in rows:
        if name_idx >= len(r):
            continue
        name = r[name_idx].strip()
        if not name or len(name) > 30:
            skipped += 1
            continue
        res = disambiguate.resolve(conn, name)
        if res.kind == "bound":
            skipped += 1  # 已存在
            items.append({"name": name, "status": "exists", "person_id": res.person_id})
        elif res.kind == "confirm":
            pid = repo.add_pending(conn, "person_identity", {
                "mention": name, "canonical_name": name,
                "candidates": [{"person_id": c.person_id, "display_name": c.display_name,
                                "score": round(c.score, 3)} for c in res.candidates[:3]],
                "patch": None, "source_msg_id": "import", "channel": "import",
            }, reason="导入名单与现有人物疑似重复")
            pending += 1
            items.append({"name": name, "status": "pending", "pending_id": pid})
        else:
            pid = repo.create_person(conn, name, status="active", alias_source="import")
            created += 1
            items.append({"name": name, "status": "created", "person_id": pid})
    conn.commit()
    return {"created": created, "pending": pending, "skipped": skipped, "items": items}


def import_vcard(conn, text: str) -> dict:
    names = re.findall(r"^FN[;:](.+)$", text, flags=re.M)
    csv_text = "\n".join(n.strip() for n in names if n.strip())
    return import_csv_names(conn, csv_text)
