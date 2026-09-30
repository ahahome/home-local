"""身份消歧三档决策（FR-ID-04 / AC-15）。"""
from app.memory import disambiguate
from app.store import repo


def test_exact_single_bound(conn):
    pid = repo.create_person(conn, "张伟", status="active")
    conn.commit()
    res = disambiguate.resolve(conn, "张伟")
    assert res.kind == "bound"
    assert res.person_id == pid


def test_no_candidate_is_new(conn):
    res = disambiguate.resolve(conn, "不存在的人")
    assert res.kind == "new"


def test_multiple_same_name_never_auto_merge(conn):
    """3 个'小张'：即使最高分别名精确命中，也绝不自动合并（AC-15）。"""
    repo.create_person(conn, "张一", status="active")
    repo.create_person(conn, "张二", status="active")
    p3 = repo.create_person(conn, "张三", status="active")
    repo.add_alias(conn, p3, "小张", "manual", 1.0)
    # 再给另外两人也加"小张"别名，制造并列
    for name in ("张一", "张二"):
        p = repo.get_person_by_name(conn, name)
        repo.add_alias(conn, p["id"], "小张", "manual", 1.0)
    conn.commit()
    res = disambiguate.resolve(conn, "小张")
    assert res.kind == "confirm", f"同名并列必须回问，得到 {res.kind}"
    assert len(res.candidates) >= 2


def test_gap_rule_forces_confirm(conn):
    """领先差值 < 0.10 时，即使分数够也不自动绑定。"""
    p1 = repo.create_person(conn, "王芳", status="active")
    p2 = repo.create_person(conn, "王芳芳", status="active")
    repo.add_alias(conn, p2, "王芳", "chat", 0.9)
    conn.commit()
    # 两个候选都精确命中"王芳"（p1 本名 + p2 别名），得分相同 → ambiguous
    res = disambiguate.resolve(conn, "王芳")
    assert res.kind == "confirm"
