"""产品原则（七不变量）的回归测试 —— 对应评测集的 L2 落库层与 L5 不变量层。

这层不调 LLM、不花钱，跑在每次提交上。原则是硬约束，不是评分项：
违反就是失败，没有「大部分情况对」这种说法。

覆盖情况：
- P1 永不静默合并 → AC-15（5 个同名也不能自动合并）
- FR-ID-01 角色词不建档 → 「前台」不得经 people[] 或 relation 旁路变成人物
- P2 写入必有回执 → 每条 patch 都有 outcome，且数量自洽
- P4 假设优于断言 → 置信度三档：写入 / 待确认 / 明确跳过，不能静默丢弃
- P5 删除是真删除 → 两级语义：**归档**（可恢复、不复活，AC-02 行为层）+ **彻底清除**
  （物理删除，含字段/别名/记忆/账本/关系边/凭证/blob，FR-ID-02 数据层）
- P6 代笔不代发 → 工具面不得存在发送类能力
- P7 不假装知道 → 提示词硬约束（严禁模拟工具）存在
"""
from __future__ import annotations

import re
from pathlib import Path

from app.agent import prompts
from app.config import get_settings
from app.memory import apply as apply_mod
from app.memory import disambiguate, retrieve
from app.store import repo
from app.util import now_ms

ROOT = Path(__file__).resolve().parent.parent


def _field(value, confidence, field="company"):
    return {"field": field, "value": value, "confidence": confidence, "evidence": value}


def _extraction(fields=None, name="张三"):
    return {"people": [{"name": name, "canonical_name": name, "fields": fields or [],
                        "aliases": [], "relations": [], "facts": [], "promises": [],
                        "ledger": [], "value_read": None}], "my_notes": []}


# ---------------- P1 · 永不静默合并（AC-15 加强版） ----------------


def test_p1_five_same_name_never_auto_merge(conn):
    """AC-15：存在 5 个同名候选人时，任何情况下都不得自动合并。"""
    for i in range(5):
        repo.create_person(conn, f"张伟{i}", status="active")
    for i in range(5):
        p = repo.get_person_by_name(conn, f"张伟{i}")
        repo.add_alias(conn, p["id"], "张伟", "chat", 0.95)
    conn.commit()
    res = disambiguate.resolve(conn, "张伟")
    assert res.kind == "confirm", f"同名并列必须回问，实得 {res.kind}"
    assert len(res.candidates) >= 2


def test_p1_single_candidate_binds(conn):
    """只有一个候选且分数够，才允许自动绑定（对照组，避免把规则写成永不绑定）。"""
    pid = repo.create_person(conn, "李娜", status="active")
    conn.commit()
    assert disambiguate.resolve(conn, "李娜").kind == "bound"
    assert disambiguate.resolve(conn, "李娜").person_id == pid


# ---------------- FR-ID-01 · 角色词不建档（含「关系边」这条旁路） ----------------


def test_role_word_never_becomes_a_person_through_relations(conn):
    """角色词不得因为出现在 relation 里就被建成人物。

    apply.py 的 relations 分支在 `get_person_by_name(to)` 找不到人时会
    `create_person(to)` —— 如果闸门只拦 people[]，这条旁路会绕过它，
    于是「前台」还是会变成一个档案。所以这条必须在落库层钉住。
    """
    patch = {"people": [
        {"name": "李娜", "canonical_name": "李娜", "fields": [], "aliases": [],
         "relations": [{"to": "前台", "kind": "know", "confidence": 0.9, "evidence": "前台"}],
         "facts": [], "promises": [], "ledger": [], "value_read": None},
    ], "my_notes": []}
    apply_mod.apply_extraction(conn, patch, "msg-role-1", "前台说李娜今天不在", channel="test")
    conn.commit()
    assert repo.get_person_by_name(conn, "前台") is None, "角色词被建成了人物"
    assert [p["display_name"] for p in repo.all_active_persons(conn)] == ["李娜"]


# ---------------- P2 · 写入必有回执 ----------------


def test_p2_every_patch_returns_receipt(conn):
    """每条 patch 都必须有 outcome，且 outcome 总数与 patch 数自洽，绝不静默丢弃。"""
    fields = [_field("恒远进出口", 0.95), _field("不确定的公司", 0.65)]
    res = apply_mod.apply_extraction(conn, _extraction(fields), "msg-1", "原文片段", channel="test")
    conn.commit()
    assert res["outcomes"], "必须有回执"
    for o in res["outcomes"]:
        assert set(o) >= {"kind", "status", "detail"}, f"回执结构不完整：{o}"
        assert o["status"] in {"ok", "pending", "duplicate", "skipped", "failed"}
    field_receipts = [o for o in res["outcomes"] if o["kind"] == "field"]
    assert len(field_receipts) == len(fields), "每条字段都要有独立回执"
    assert any(o["kind"] == "person" for o in res["outcomes"]), "建档本身也要有回执"


# ---------------- P4 · 假设优于断言（置信度三档） ----------------


def test_p4_low_confidence_never_silently_written(conn):
    """置信度不足时必须进待确认或明确跳过，不得当成事实写进档案。"""
    s = get_settings()
    high = s.confidence_auto + 0.05
    mid = (s.confidence_auto + s.confidence_confirm) / 2
    low = s.confidence_confirm - 0.10
    res = apply_mod.apply_extraction(
        conn, _extraction([_field("高置信公司", high, "company"),
                           _field("中置信公司", mid, "company2"),
                           _field("低置信公司", low, "company3")]),
        "msg-2", "原文", channel="test")
    conn.commit()
    statuses = {o["detail"].split("=")[0]: o["status"] for o in res["outcomes"]}
    assert statuses["company"] == "ok"
    assert statuses["company2"] == "pending", "中置信必须进待确认"
    assert statuses["company3"] == "skipped", "低置信必须显式跳过并告知"
    person = repo.get_person_by_name(conn, "张三")
    fields = set(repo.current_fields(conn, person["id"]))
    assert "company3" not in fields, "低置信不得落库"


def test_p4_pending_does_not_enter_summary(conn):
    """FR-MEM-03：待确认条目不得进入任何汇总。"""
    res = apply_mod.apply_extraction(
        conn, _extraction([_field("待确认公司", 0.65)]), "msg-3", "原文", channel="test")
    conn.commit()
    assert any(o["status"] == "pending" for o in res["outcomes"])
    person = repo.get_person_by_name(conn, "张三")
    assert not repo.current_fields(conn, person["id"]), "待确认字段不得出现在档案里"


# ---------------- P5 · 删除是真删除（AC-02） ----------------


def test_p5_deleted_person_is_not_resurrected(conn):
    """AC-02：归档后继续谈论该人，不得复用旧档案，搜索也不应命中。

    P5 现在是**两级**语义：日常「删除」= 归档（可恢复），行还在；
    真正抹痕迹走 purge。AC-02 说的是「不复活」，归档就该满足它。
    """
    pid = repo.create_person(conn, "赵敏", status="active")
    repo.add_alias(conn, pid, "小赵", "manual", 1.0)
    conn.commit()
    repo.archive_person(conn, pid)
    conn.commit()
    assert repo.get_person_by_name(conn, "赵敏") is None
    assert disambiguate.resolve(conn, "赵敏").kind == "new", "不得复活旧档案"
    assert retrieve.search_people(conn, "赵敏") == []
    # 归档 ≠ 删除：行还在，所以能恢复
    assert repo.archived_persons(conn) and repo.restore_person(conn, pid)
    conn.commit()
    assert repo.get_person_by_name(conn, "赵敏") is not None, "恢复后应重新可见"


def _rich_person(conn, name="钱七", alias="小七"):
    """造一个「什么都有」的人：字段/别名/记忆/承诺/账本/关系边/互动/提醒。"""
    pid = repo.create_person(conn, name, status="active")
    repo.add_alias(conn, pid, alias, "manual", 1.0)
    repo.upsert_field(conn, pid, "company", "敏感公司名", 0.95, None)
    repo.add_memory(conn, "episodic", f"{name}的公司很敏感", pid, 0.9)
    repo.add_promise(conn, "me", f"把{name}的资料删掉", pid, None)
    repo.add_ledger(conn, "out", 12345, pid, "份子钱", now_ms(), "confirmed",
                    f"dedupe-{pid}")
    repo.upsert_edge(conn, pid, repo.create_person(conn, "孙八", status="active"),
                     "colleague", 0.8, None)
    repo.insert_interaction(conn, now_ms(), "chat", f"跟{name}聊了聊", None,
                            [(pid, "mentioned")])
    repo.add_reminder(conn, now_ms() + 86400000, f"提醒{name}的事", pid)
    conn.commit()
    return pid


_P5_TABLES = ("person", "person_alias", "person_field", "memory_item", "promise",
              "ledger_entry", "reminder", "interaction_person", "applied_patch")


def test_p5_purge_removes_every_trace(conn):
    """彻底清除的数据层要求：删完之后一张表都不许剩该人的行（FR-ID-02）。

    这张表清单是**故意独立于实现**写死的——直接抄 repo._PURGE_LINKS 的话，
    以后往库里加表却忘了清，测试会跟着一起漏。
    """
    pid = _rich_person(conn)
    rep = repo.purge_person(conn, pid)
    assert rep["ok"], rep
    conn.commit()
    for table in _P5_TABLES:
        col = "id" if table == "person" else ("owner_id" if table.startswith("vector") else "person_id")
        n = conn.execute(f"SELECT COUNT(*) n FROM {table} WHERE {col}=?", (pid,)).fetchone()["n"]
        assert n == 0, f"{table} 还残留 {n} 行"
    assert conn.execute("SELECT COUNT(*) n FROM relation_edge WHERE from_person=? OR to_person=?",
                        (pid, pid)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM vector_item WHERE owner_id=?",
                        (pid,)).fetchone()["n"] == 0
    # FR-ID-02 的验收口径：全库搜曾用名，再也指不到这个人
    assert repo.verify_purged(conn, pid, rep["aliases"]) == []
    assert [r for r in retrieve.search_people(conn, "钱七") if r["person_id"] == pid] == []
    assert [r for r in retrieve.search_people(conn, "小七") if r["person_id"] == pid] == []
    assert repo.get_person_by_name(conn, "钱七") is None


def test_p5_purge_leaves_other_people_intact(conn):
    """清除一个人不能顺手删掉别人的东西。

    这不是假想的：一次 apply_extraction 只建**一条** evidence，同一批次里
    所有字段都指向它。如果清人时无脑删 evidence，会把别人的凭证一起删掉。
    """
    pid = repo.create_person(conn, "钱七", status="active")
    other = repo.create_person(conn, "周九", status="active")
    shared_ev = repo.add_evidence(conn, "text", source_ref="msg-shared", excerpt="同一批次的原文")
    repo.upsert_field(conn, pid, "company", "甲", 0.95, shared_ev)
    repo.upsert_field(conn, other, "company", "乙", 0.95, shared_ev)
    repo.add_alias(conn, pid, "小七", "manual", 1.0)
    repo.add_alias(conn, other, "小七", "manual", 1.0)   # 别名允许共享（三个「小张」）
    conn.commit()

    repo.purge_person(conn, pid)
    conn.commit()

    assert repo.get_person(conn, other) is not None
    assert repo.current_fields(conn, other), "别人的字段不能被连带删掉"
    assert conn.execute("SELECT COUNT(*) n FROM evidence WHERE id=?",
                        (shared_ev,)).fetchone()["n"] == 1, "共享的凭证要留着"
    assert repo.get_person_by_name(conn, "小七") is not None, "共享别名不能一起删"
    # 口径：别名再也指不到**被清除的那个人**（而不是「全库搜不到这个别名」）
    assert repo.verify_purged(conn, pid, ["小七"]) == []


def test_p5_verifier_actually_reports_leftovers(conn):
    """校验器必须真能报出残留——否则「部分删除」这套话术是空的。"""
    pid = _rich_person(conn)
    rep = repo.purge_person(conn, pid)
    conn.commit()
    assert repo.verify_purged(conn, pid, rep["aliases"]) == []

    # 人为制造残留（模拟删除被打断 / 有并发写回）
    conn.execute("INSERT INTO person(id,display_name,status,created_at,updated_at)"
                 " VALUES(?,?,?,?,?)", (pid, "钱七", "active", now_ms(), now_ms()))
    conn.commit()
    assert "person" in repo.verify_purged(conn, pid, rep["aliases"])


def test_p5_merged_tombstone_is_refused_not_silently_deleted(conn):
    """合并墓碑的行已经迁到存活方了，清除它没有意义 → 明确拒绝，而不是假装删干净。"""
    survivor = repo.create_person(conn, "甲", status="active")
    victim = repo.create_person(conn, "乙", status="active")
    conn.commit()
    apply_mod.merge_persons(conn, survivor, victim)
    rep = repo.purge_person(conn, victim)
    assert rep["ok"] is False and rep["leftovers"], "必须拒绝并说明原因"
    assert repo.get_person(conn, survivor) is not None


# ---------------- P6 · 代笔不代发 ----------------


def test_p6_no_send_capability_in_tools():
    """工具面不得存在任何「代发」能力，否则原则形同虚设。"""
    src = (ROOT / "app" / "agent" / "tools.py").read_text(encoding="utf-8")
    banned = re.findall(r'"name":\s*"([a-z_]*(?:send|post|publish|deliver|reply|forward)[a-z_]*)"', src)
    assert not banned, f"出现疑似发送类工具：{banned}"


def test_p6_prompt_states_draft_only():
    assert "代笔不代发" in prompts.SYSTEM


# ---------------- P7 · 不假装知道 ----------------


def test_p7_prompt_bans_simulated_tools():
    """模型曾角色扮演「正在调用天眼查API」，提示词必须有硬约束阻断。"""
    assert "严禁模拟工具" in prompts.SYSTEM
    assert "不假装知道" in prompts.SYSTEM


def test_p7_research_drops_sourceless_claims():
    """Deep Research 复核阶段必须存在「无来源论断被丢弃」的判定。"""
    src = (ROOT / "app" / "agent" / "research.py").read_text(encoding="utf-8")
    assert re.search(r"(source|sources|来源)", src), "研究模块需处理来源"
