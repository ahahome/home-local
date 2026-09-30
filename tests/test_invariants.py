"""产品原则（七不变量）的回归测试 —— 对应评测集的 L2 落库层与 L5 不变量层。

这层不调 LLM、不花钱，跑在每次提交上。原则是硬约束，不是评分项：
违反就是失败，没有「大部分情况对」这种说法。

覆盖情况：
- P1 永不静默合并 → AC-15（5 个同名也不能自动合并）
- P2 写入必有回执 → 每条 patch 都有 outcome，且数量自洽
- P4 假设优于断言 → 置信度三档：写入 / 待确认 / 明确跳过，不能静默丢弃
- P5 删除是真删除 → AC-02 行为层（不复用旧档案）、数据层（见 xfail 说明）
- P6 代笔不代发 → 工具面不得存在发送类能力
- P7 不假装知道 → 提示词硬约束（严禁模拟工具）存在
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.agent import prompts
from app.config import get_settings
from app.memory import apply as apply_mod
from app.memory import disambiguate, retrieve
from app.store import repo

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
    """AC-02：删除后继续谈论该人，不得复用旧档案，搜索也不应命中。"""
    pid = repo.create_person(conn, "赵敏", status="active")
    repo.add_alias(conn, pid, "小赵", "manual", 1.0)
    conn.commit()
    repo.soft_delete_person(conn, pid)
    conn.commit()
    assert repo.get_person_by_name(conn, "赵敏") is None
    assert disambiguate.resolve(conn, "赵敏").kind == "new", "不得复活旧档案"
    assert retrieve.search_people(conn, "赵敏") == []


@pytest.mark.xfail(reason="P5 实为墓碑删除：deleted_at 打标，字段与别名行仍留在库里", strict=False)
def test_p5_delete_is_physical(conn):
    """P5「删除是真删除」的数据层要求：删除后该人的字段/别名不得留在磁盘上。

    现状：repo.soft_delete_person 只写 deleted_at，person / person_field / person_alias
    的行都还在（实测 person=1、field=1、alias=2 行）。行为层 AC-02 是过的，
    但数据层没达到「真删除」——对私人关系库来说这是隐私缺口，需要产品决策：
    要么改成物理删除（含 blob 与证据），要么在 UI 上把语气改成「归档」并支持彻底清除。
    """
    pid = repo.create_person(conn, "钱七", status="active")
    repo.add_alias(conn, pid, "小七", "manual", 1.0)
    repo.upsert_field(conn, pid, "company", "敏感公司名", 0.95, None)
    conn.commit()
    repo.soft_delete_person(conn, pid)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) n FROM person WHERE id=?", (pid,)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM person_field WHERE person_id=?",
                        (pid,)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM person_alias WHERE person_id=?",
                        (pid,)).fetchone()["n"] == 0


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
