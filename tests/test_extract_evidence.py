"""抽取后处理：evidence 逐字性兜底（缺陷 B）与 sanitize 归一化。

这一层是纯函数，不调 LLM、不碰数据库——正是它把「逐字截取」从 prompt 的
君子协定变成代码保证，所以必须被单测钉住。
"""
from app.memory.extract import sanitize
from app.util import flatten_for_match

import pytest


def _wrap(fields=None, facts=None, value_read=None, source=""):
    p = {
        "name": "老陆", "canonical_name": "老陆",
        "fields": fields or [], "aliases": [], "relations": [],
        "facts": facts or [], "promises": [], "ledger": [], "value_read": value_read,
    }
    return sanitize({"people": [p], "my_notes": []}, source=source)


def test_whole_evidence_in_source_is_kept():
    src = "老陆昨天跟我说他跳槽了，现在在杭州的一家新能源公司做采购总监。"
    got = _wrap(fields=[{"field": "city", "value": "杭州", "confidence": 0.9,
                         "evidence": "现在在杭州的一家新能源公司做采购总监"}], source=src)
    assert got["people"][0]["fields"][0]["evidence"] == "现在在杭州的一家新能源公司做采购总监"


def test_evidence_with_punctuation_difference_still_matches():
    """标点/全半角差异不该误伤——校验走的是 flatten_for_match。"""
    src = "李娜：[图片]\n李娜：这是我新房，在杭州钱江新城那边"
    got = _wrap(fields=[{"field": "city", "value": "杭州", "confidence": 0.9,
                         "evidence": "这是我新房 在杭州钱江新城那边"}], source=src)
    assert got["people"][0]["fields"][0]["evidence"] == "这是我新房 在杭州钱江新城那边"


def test_concatenated_evidence_is_cut_to_one_contiguous_piece():
    """缺陷 B 的核心：用「；」拼两处不相邻原文，必须被切成一段连续片段。"""
    src = "老陆说他公司搬到杭州了。中间还聊了别的。他对公司的事很上心。"
    joined = "老陆说他公司搬到杭州了；他对公司的事很上心"
    # 前提校验：拼起来的整条确实不在原文里（否则这条测试没有意义）
    assert flatten_for_match(joined) not in flatten_for_match(src)
    got = _wrap(value_read={"text": "对公司上心", "confidence": 0.8, "evidence": joined}, source=src)
    assert got["people"][0]["value_read"]["evidence"] == "老陆说他公司搬到杭州了"


def test_fabricated_evidence_is_dropped_not_kept():
    """都对不上就清空——宁可没有凭证，也不留一条伪造的凭证。"""
    src = "张伟说他最近在忙一个新项目。"
    got = _wrap(fields=[{"field": "city", "value": "杭州", "confidence": 0.9,
                         "evidence": "他已经在杭州买房定居了"}], source=src)
    assert got["people"][0]["fields"][0]["evidence"] == ""


def test_facts_evidence_also_enforced():
    src = "王工说他老婆下个月要生了，他最近在医院和公司两头跑。"
    got = _wrap(facts=[{"text": "王工妻子下月生产", "confidence": 0.9,
                        "evidence": "他老婆下个月要生了；医院和公司两头跑"}], source=src)
    assert got["people"][0]["facts"][0]["evidence"] == "他老婆下个月要生了"


def test_no_source_means_no_enforcement():
    """不传 source 时保持旧行为（不校验），避免影响非抽取场景的调用方。"""
    got = _wrap(fields=[{"field": "city", "value": "杭州", "confidence": 0.9,
                         "evidence": "随便编的一段"}])
    assert got["people"][0]["fields"][0]["evidence"] == "随便编的一段"


def test_evidence_drop_does_not_touch_confidence():
    """凭证不可用 ≠ 事实不可信：清空 evidence，但 confidence 原样保留。"""
    src = "张伟说他最近在忙。"
    got = _wrap(fields=[{"field": "title", "value": "总监", "confidence": 0.93,
                         "evidence": "已经是公司总监了"}], source=src)
    f = got["people"][0]["fields"][0]
    assert f["evidence"] == "" and f["confidence"] == 0.93


def test_amount_becomes_cents_and_bad_amount_dropped():
    p = {
        "name": "张伟", "canonical_name": "张伟",
        "fields": [], "aliases": [], "relations": [], "facts": [], "promises": [],
        "ledger": [
            {"direction": "out", "amount": 2000, "note": "份子钱", "confidence": 0.9},
            {"direction": "out", "amount": 0, "note": "零元", "confidence": 0.9},
            {"direction": "out", "amount": "不是数字", "note": "脏数据", "confidence": 0.9},
        ],
        "value_read": None,
    }
    got = sanitize({"people": [p], "my_notes": []})
    led = got["people"][0]["ledger"]
    assert len(led) == 1 and led[0]["amount_cents"] == 200000


def test_unknown_direction_defaults_to_out():
    p = {"name": "甲", "canonical_name": "甲", "fields": [], "aliases": [], "relations": [],
         "facts": [], "promises": [], "ledger": [{"amount": 10, "direction": "sideways"}]}
    got = sanitize({"people": [p], "my_notes": []})
    assert got["people"][0]["ledger"][0]["direction"] == "out"


def test_confidence_is_clamped_and_defaulted():
    p = {"name": "甲", "canonical_name": "甲", "aliases": [], "relations": [],
         "fields": [{"field": "city", "value": "上海", "confidence": 7}],
         "facts": [], "promises": [], "ledger": []}
    got = sanitize({"people": [p], "my_notes": []})
    assert got["people"][0]["fields"][0]["confidence"] == 1.0


def test_non_dict_input_returns_empty_shape():
    assert sanitize(None) == {"people": [], "my_notes": []}
    assert sanitize("不是对象") == {"people": [], "my_notes": []}


def test_person_without_name_is_skipped():
    got = sanitize({"people": [{"canonical_name": "无名字段"}], "my_notes": []})
    assert got["people"] == []


# ---------- 社交客套闸门（缺陷 A 的代码兜底） ----------

def _promises(*contents):
    p = {
        "name": "张伟", "canonical_name": "张伟", "fields": [], "aliases": [], "relations": [],
        "facts": [], "ledger": [], "value_read": None,
        "promises": [{"who": "them", "content": c, "due": None, "confidence": 0.4}
                     for c in contents],
    }
    return sanitize({"people": [p], "my_notes": []})["people"][0]["promises"]


@pytest.mark.parametrize("text", [
    "改天一起吃个饭",
    "以后在杭州能常见面",
    "以后咱们能常见面",
    "回头我联系你",
    "保持联系",
    "以后常联系",
    "有空一起坐坐",
    "哪天聚聚",
])
def test_social_pleasantry_is_never_a_promise(text):
    """客套与置信度无关：哪怕模型只给了 0.4 分，也不该作为承诺出现在抽取结果里。"""
    assert _promises(text) == []


@pytest.mark.parametrize("text", [
    "下周三之前把报价单发给他",
    "9/8 之前给你确认",
    "下次见面把合同签了",
    "以后每周五给你发报表",
    "周末把东西还你",
])
def test_concrete_commitment_survives_the_pleasantry_gate(text):
    """带具体动作/时间的不能误杀——含「以后」「下次」也照常当承诺。"""
    got = _promises(text)
    assert len(got) == 1 and got[0]["content"] == text


def test_pleasantry_gate_mixed_batch_keeps_only_real_ones():
    got = _promises("改天吃饭", "下周三之前把报价单发给他")
    assert [g["content"] for g in got] == ["下周三之前把报价单发给他"]


def test_pleasantry_gate_only_touches_promises():
    """闸门不许影响 facts/ledger/字段——只拦承诺。"""
    p = {
        "name": "张伟", "canonical_name": "张伟", "aliases": [], "relations": [],
        "fields": [{"field": "city", "value": "杭州", "confidence": 0.9}],
        "promises": [{"who": "them", "content": "改天吃饭", "confidence": 0.9}],
        "facts": [{"text": "张伟搬到杭州", "confidence": 0.9}],
        "ledger": [{"direction": "out", "amount": 500, "confidence": 0.9}],
        "value_read": None,
    }
    got = sanitize({"people": [p], "my_notes": []})["people"][0]
    assert got["promises"] == []
    assert len(got["fields"]) == 1 and len(got["facts"]) == 1 and len(got["ledger"]) == 1
