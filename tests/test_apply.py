"""幂等写入（FR-ING-06/07、AC-01/02/16）。"""
from app.memory import apply as apply_mod
from app.store import repo


def _extraction(fields, person="老陆"):
    return {"people": [{
        "name": person, "canonical_name": person,
        "fields": [{"field": k, "value": v, "confidence": 0.9, "evidence": "原话"}
                   for k, v in fields],
        "aliases": [], "relations": [], "facts": [], "promises": [], "ledger": [],
        "value_read": None,
    }], "my_notes": []}


def test_six_fields_all_land(conn):
    """AC-01：一条消息 6 个字段必须全部落库。"""
    fields = [("company", "A公司"), ("title", "采购负责人"), ("city", "上海"),
              ("birthday", "1985-03"), ("trait.style", "谨慎"), ("need.budget", "收紧")]
    applied = apply_mod.apply_extraction(conn, _extraction(fields), "msg1", "原文")
    ok = [o for o in applied["outcomes"] if o["status"] == "ok" and o["kind"] == "field"]
    assert len(ok) == 6, applied["outcomes"]
    pid = applied["person_ids"][0]
    cur = repo.current_fields(conn, pid)
    assert cur["company"]["value"] == "A公司"
    assert cur["need.budget"]["value"] == "收紧"


def test_replay_is_idempotent(conn):
    """FR-ING-06：重放同一条消息不产生重复。"""
    ext = _extraction([("company", "A公司")])
    apply_mod.apply_extraction(conn, ext, "msg1", "原文")
    applied2 = apply_mod.apply_extraction(conn, ext, "msg1", "原文")
    dups = [o for o in applied2["outcomes"] if o["status"] == "duplicate"]
    assert dups, "重放应被识别为重复"
    pid = applied2["person_ids"][0]
    # 字段只有一个当前版本
    assert len([r for r in repo.field_history(conn, pid, "company")]) == 1


def test_field_versioning_keeps_history(conn):
    """FR-ID-03：字段双时态，旧值可查。"""
    ext = _extraction([("company", "A公司")])
    a1 = apply_mod.apply_extraction(conn, ext, "msg1", "原文")
    pid = a1["person_ids"][0]
    ext2 = _extraction([("company", "B公司")])
    apply_mod.apply_extraction(conn, ext2, "msg2", "原文2")
    cur = repo.current_fields(conn, pid)
    assert cur["company"]["value"] == "B公司"
    hist = repo.field_history(conn, pid, "company")
    assert len(hist) == 2 and hist[0]["value"] == "A公司"


def test_merge_and_split_restores(conn):
    """AC-16：合并→拆分后字段与互动与合并前一致。"""
    a = apply_mod.apply_extraction(conn, _extraction([("company", "甲公司")], "甲"), "m1", "x")
    b = apply_mod.apply_extraction(conn, _extraction([("city", "杭州")], "乙"), "m2", "y")
    pa, pb = a["person_ids"][0], b["person_ids"][0]
    log_id = apply_mod.merge_persons(conn, pa, pb)

    surv = repo.get_person(conn, pa)
    merged = repo.get_person(conn, pb)
    assert merged["status"] == "merged" and merged["merged_into"] == pa
    cur = repo.current_fields(conn, pa)
    assert cur["company"]["value"] == "甲公司" and cur["city"]["value"] == "杭州"

    apply_mod.split_merge(conn, log_id)
    b2 = repo.get_person(conn, pb)
    assert b2["status"] != "merged" and b2["deleted_at"] is None
    assert repo.current_fields(conn, pb)["city"]["value"] == "杭州"
    assert repo.current_fields(conn, pa)["company"]["value"] == "甲公司"


def test_pending_identity_blocks_write(conn):
    """P1：身份存疑时整条 patch 挂起，不落字段。"""
    # 制造两个并列候选
    for nm in ("老陆", "老陆总"):
        repo.create_person(conn, nm, status="active")
    p = repo.get_person_by_name(conn, "老陆总")
    repo.add_alias(conn, p["id"], "老陆", "chat", 0.9)
    conn.commit()
    ext = _extraction([("company", "A公司")], "老陆")
    applied = apply_mod.apply_extraction(conn, ext, "m1", "原文")
    assert any(o["status"] == "pending" and o["kind"] == "person" for o in applied["outcomes"])
    # 字段不得落库
    pendings = repo.list_pending(conn, "person_identity")
    assert pendings, "应产生待确认项"
