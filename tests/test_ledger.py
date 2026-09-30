"""账本（FR-LED-01/02/03、AC-03/04）与时区（AC-05）。"""
from app.store import repo
from app.util import now_ms, sha256_text


def _add(conn, direction, cents, pid=None, status="confirmed", tag="t1", occurred=None):
    dh = sha256_text(f"ledger|{pid}|{direction}|{cents}|{tag}")
    return repo.add_ledger(conn, direction, cents, pid, "测试", occurred or now_ms(), status, dh)


def test_pending_excluded_from_totals(conn):
    """AC-04：待确认不计入总额。"""
    _add(conn, "out", 10000, status="confirmed", tag="a")
    _add(conn, "out", 5000, status="pending", tag="b")
    conn.commit()
    totals = repo.ledger_totals(conn)
    assert totals["out_cents"] == 10000
    assert totals["count"] == 1


def test_dedupe_hash_unique(conn):
    """同一笔转账的指纹唯一，重复插入被拒。"""
    _, created1 = _add(conn, "in", 20000, tag="same")
    _, created2 = _add(conn, "in", 20000, tag="same")
    assert created1 is True and created2 is False


def test_journal_local_day_roundtrip(conn):
    """AC-05：journal 按本地时区存取。"""
    from app.jobs.scheduler import _day_range_ms
    day = "2026-09-29"
    start, end = _day_range_ms(day)
    assert end - start == 86400000
    repo.upsert_journal(conn, day, "今天联系了老陆")
    conn.commit()
    j = repo.get_journal(conn, day)
    assert j["text"] == "今天联系了老陆"
