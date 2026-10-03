"""抽取后处理：evidence 逐字性兜底（缺陷 B）与 sanitize 归一化。

这一层是纯函数，不调 LLM、不碰数据库——正是它把「逐字截取」从 prompt 的
君子协定变成代码保证，所以必须被单测钉住。
"""
from app.memory.extract import _locate_span, _week_bounds, _with_weekday, is_role_person, sanitize
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


def test_unknown_direction_requires_confirmation():
    p = {"name": "甲", "canonical_name": "甲", "fields": [], "aliases": [], "relations": [],
         "facts": [], "promises": [], "ledger": [{"amount": 10, "direction": "sideways"}]}
    got = sanitize({"people": [p], "my_notes": []})
    assert got["people"][0]["ledger"][0]["direction"] is None


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


@pytest.mark.parametrize("text", [
    "那个事回头再说吧",
    "改天请你搓一顿",
    "哪天得空喝一杯",
    "你有空再说",
    "改天我请你吃一顿饭",
])
def test_pleasantry_gate_covers_forms_outside_the_old_verb_whitelist(text):
    """回归：旧版要求「触发词 + 动词白名单（吃饭|聚|聊|见面|喝|约…）」，
    于是「回头再说吧」（说）、「请你搓一顿」（搓）都漏掉了——那两条只是
    「模型这次听话」才过的，不是代码兜住的。白名单已去掉，改判「无具体动作/时间」。"""
    assert _promises(text) == []


@pytest.mark.parametrize("text", [
    "回头我把钱转你",
    "下周有空的话把资料发我",
    "哪天你有空我们把合同签了",
])
def test_concrete_action_still_survives_after_dropping_the_whitelist(text):
    """去掉白名单后仍然不能误杀：「把…」是明确的可执行动作。"""
    got = _promises(text)
    assert len(got) == 1 and got[0]["content"] == text


# ---------- 客套闸门要连原句一起判（hold2-n06：模型改写 content 弄丢触发词） ----------

def _promises_with_source(source, *contents):
    p = {
        "name": "李娜", "canonical_name": "李娜", "fields": [], "aliases": [], "relations": [],
        "facts": [], "ledger": [], "value_read": None,
        "promises": [{"who": "them", "content": c, "due": None, "confidence": 0.4}
                     for c in contents],
    }
    return sanitize({"people": [p], "my_notes": []}, source=source)["people"][0]["promises"]


def test_pleasantry_gate_uses_source_span_when_content_is_rewritten():
    """hold2-n06 的修法：模型把「回头我加你微信」改写成 content=「加你微信」，
    触发词「回头」被改没了，只判 content 就漏（实测 3 次全漏）。把 content 摁回
    原文找到它所在的那句一起判——span 里有触发词、全句无具体动作 → 客套，拦掉。"""
    assert _promises_with_source("李娜：回头我加你微信", "加你微信") == []


@pytest.mark.parametrize("source,content", [
    ("改天一起吃饭吧", "一起吃饭"),
    ("李娜：回头我加你微信", "加你微信"),
    ("哪天得空喝一杯", "喝一杯"),
    ("找个时间我们聊聊", "我们聊聊"),
])
def test_pleasantry_forms_caught_via_source_span(source, content):
    """词形变体 + 模型改写，经原句定位后都应被拦（含新触发词「找时间」）。"""
    assert _promises_with_source(source, content) == []


def test_locate_span_ignores_speaker_prefix():
    """「李娜：回头我加你微信」要定位到「回头我加你微信」这句，而不是说话人前缀。"""
    source = "李娜：回头我加你微信"
    assert _locate_span("加你微信", source, flatten_for_match(source)) == "回头我加你微信"


def test_paraphrased_real_promise_survives_when_not_locatable():
    """摁不回去（改写太多）时退化为只判 content，不能因此误杀真承诺：
    content「给老陆转账五百元」摁不回原文（原文是「转五百就行」），
    content 自身带「五百元」（数词+单位）→ 放行。"""
    got = _promises_with_source("老陆:上次那顿饭我付的，你回头给我转五百就行", "给老陆转账五百元")
    assert len(got) == 1 and got[0]["content"] == "给老陆转账五百元"


def test_source_span_does_not_kill_concrete_promise():
    """摁回原句也不能误杀：原句里「把合同签了」是具体动作 → 放行。"""
    got = _promises_with_source("下次见面把合同签了", "把合同签了")
    assert len(got) == 1


# ---------- 相对日期必须由代码给出星期几（缺陷 F） ----------

@pytest.mark.parametrize("day,expect", [
    ("2026-09-30", "2026-09-30（周三）"),
    ("2026-10-01", "2026-10-01（周四）"),
    ("2026-10-02", "2026-10-02（周五）"),
    ("2026-10-05", "2026-10-05（周一）"),
])
def test_today_carries_weekday(day, expect):
    """LLM 的星期运算不可靠（实测「这周五」被算成周六），所以星期几由代码算好喂给它，
    而不是让模型自己推——这一行决定的是承诺提醒建在哪一天（FR-TIM-02）。"""
    assert _with_weekday(day) == expect


@pytest.mark.parametrize("bad", ["", "不是日期", "昨天", "2026/09/30"])
def test_non_iso_today_is_returned_untouched(bad):
    assert _with_weekday(bad) == bad


# ---------- 本周/下周的边界与锚点由代码给出（hold2-p02） ----------

def test_week_bounds_gives_ranges_and_anchors():
    """星期几给了还不够——「这周/下周」以哪天为界也得给（周一起算），
    否则模型按「最近的周五」理解，「下周五」会错成本周五。"""
    s = _week_bounds("2026-09-30")  # 周三
    assert "2026-09-28" in s and "2026-10-04" in s   # 本周一 ~ 本周日
    assert "2026-10-05" in s and "2026-10-11" in s   # 下周一 ~ 下周日
    assert "本周五=2026-10-02" in s
    assert "下周五=2026-10-09" in s


def test_week_bounds_on_sunday():
    """周日属于本周（周一起算）：本周一 = 六天前，不是明天。"""
    s = _week_bounds("2026-10-04")  # 周日
    assert "2026-09-28" in s and "2026-10-04" in s
    assert "本周五=2026-10-02" in s
    assert "下周五=2026-10-09" in s


def test_week_bounds_gives_month_anchor():
    """只给周界会把「现在」锚歪：周界里全是下个月的日期（本周 09-28~10-04），
    模型实测把「下个月 5 号」算成 11-05（hold-p11 回归）。月界必须一并给。"""
    s = _week_bounds("2026-09-30")
    assert "本月是 2026-09" in s and "下个月是 2026-10" in s


def test_week_bounds_month_anchor_at_year_end():
    """跨年：12 月的下个月是次年 01 月，不是 13 月。"""
    s = _week_bounds("2026-12-30")
    assert "本月是 2026-12" in s and "下个月是 2027-01" in s


@pytest.mark.parametrize("bad", ["", "不是日期", "昨天", "2026/09/30"])
def test_week_bounds_bad_input_returns_empty(bad):
    """解析失败返回空串——拼进 prompt 时等于没这句，不硬塞错误信息。"""
    assert _week_bounds(bad) == ""


# ---------- 角色词不建档（缺陷 E / FR-ID-01） ----------

@pytest.mark.parametrize("name", [
    "王芳的老公", "李娜的儿子", "我的对象", "我对象", "那个客户", "前台", "有个人", "她的同事",
])
def test_role_words_are_not_person_names(name):
    assert is_role_person(name)


@pytest.mark.parametrize("name", ["张老师", "老王", "陈工", "王芳", "陆建国", "小张", "王志强"])
def test_ordinary_names_are_not_role_words(name):
    """「老师」是角色词，但「张老师」是称呼/名字——纯角色词只做全等匹配，别误伤。"""
    assert not is_role_person(name)


_FAMILY_SRC = "王芳她老公是苏州一家医院的主任医师。"


def _family_patch():
    """hold-n13 上模型的真实输出：把「王芳的老公」也建成一个人物。"""
    return {
        "people": [
            {"name": "王芳", "canonical_name": "王芳",
             "fields": [{"field": "family", "value": "已婚，配偶为苏州某医院主任医师",
                         "confidence": 0.85, "evidence": _FAMILY_SRC}],
             "aliases": [], "facts": [],
             "relations": [{"to": "王芳的老公", "kind": "family", "confidence": 0.9,
                            "evidence": _FAMILY_SRC}],
             "promises": [], "ledger": [], "value_read": None},
            {"name": "王芳的老公", "canonical_name": "王芳的老公",
             "fields": [{"field": "title", "value": "主任医师", "confidence": 0.9,
                         "evidence": _FAMILY_SRC}],
             "aliases": [{"alias": "她老公", "confidence": 0.9}],
             "relations": [{"to": "王芳", "kind": "family", "confidence": 0.9,
                            "evidence": _FAMILY_SRC}],
             "facts": [{"text": "王芳的老公是苏州一家医院的主任医师。", "confidence": 0.9,
                        "evidence": _FAMILY_SRC}],
             "promises": [], "ledger": [], "value_read": None},
        ],
        "my_notes": [],
    }


def test_family_role_person_is_not_created_and_info_moves_to_referent():
    """FR-ID-01 + 不丢信息：不为「王芳的老公」建档，但家属信息要落到王芳名下。"""
    got = sanitize(_family_patch(), source=_FAMILY_SRC)
    assert [p["canonical_name"] for p in got["people"]] == ["王芳"]
    texts = " ".join(f["text"] for f in got["people"][0]["facts"])
    assert "主任医师" in texts, "家属信息不能因为不建档就丢掉"


def test_relations_and_aliases_pointing_at_role_words_are_filtered():
    """apply.py 在 relations[].to 找不到人时会 create_person —— 不过滤等于绕开闸门又建一个人。"""
    got = sanitize(_family_patch(), source=_FAMILY_SRC)
    wang = got["people"][0]
    assert wang["relations"] == []
    assert all(not is_role_person(a["alias"]) for a in wang["aliases"])


def test_bare_role_word_is_dropped_and_not_pushed_onto_the_sole_person():
    """「前台」没有明确参照人 → 直接丢弃；不能塞给本批唯一的那个人，
    那恰恰是 FR-MEM-06 禁止的「把他人特征归档到当前人物名下」。"""
    src = "前台说王总今天不在，让我明天再来。"
    raw = {"people": [
        {"name": "王总", "canonical_name": "王总", "fields": [], "aliases": [],
         "relations": [], "facts": [], "promises": [], "ledger": [], "value_read": None},
        {"name": "前台", "canonical_name": "前台", "fields": [],
         "aliases": [], "relations": [], "facts": [],
         "promises": [], "ledger": [], "value_read": None},
    ], "my_notes": []}
    got = sanitize(raw, source=src)
    assert [p["canonical_name"] for p in got["people"]] == ["王总"]
    assert got["people"][0]["facts"] == [], "角色词的信息不许挪到王总头上"


def test_family_info_is_not_duplicated_when_evidence_repeats():
    """本人 facts 里已有同一条时不再重复追加（evidence 去重）。"""
    raw = _family_patch()
    raw["people"][0]["facts"] = [{"text": "王芳的老公是苏州一家医院的主任医师。",
                                  "confidence": 0.9, "evidence": _FAMILY_SRC}]
    got = sanitize(raw, source=_FAMILY_SRC)
    texts = [f["text"] for f in got["people"][0]["facts"]]
    assert texts.count("王芳的老公是苏州一家医院的主任医师。") == 1
