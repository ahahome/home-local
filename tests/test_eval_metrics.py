"""评估器自身的单测。

「评估器写错了」比「被测代码错了」更危险，所以比对逻辑必须独立可验证。
这些用例全部用合成数据，不调 LLM。
"""
from eval import metrics


def _person(**kw):
    base = {"name": "张三", "canonical_name": "张三", "fields": [], "aliases": [],
            "facts": [], "promises": [], "ledger": [], "value_read": None}
    base.update(kw)
    return base


def test_subset_matching_allows_extra_content():
    case = {"id": "x", "input": "老陆做采购总监，还喜欢钓鱼", "expect": {"people": [
        {"canonical_name": "老陆", "fields": [{"field": "title", "contains": "采购"}]}]}}
    got = {"people": [_person(canonical_name="老陆", fields=[
        {"field": "title", "value": "采购总监", "evidence": "老陆做采购总监"},
        {"field": "hobby", "value": "钓鱼", "evidence": "还喜欢钓鱼"}]),
        _person(canonical_name="别人", fields=[])]}
    res = metrics.check_case(case, got)
    assert res["passed"], res["failures"]


def test_missing_field_fails_with_reason():
    case = {"id": "x", "input": "老陆", "expect": {"people": [
        {"canonical_name": "老陆", "fields": [{"field": "city", "contains": "杭州"}]}]}}
    res = metrics.check_case(case, {"people": [_person(canonical_name="老陆", fields=[])]})
    assert not res["passed"]
    assert "city" in res["failures"][0]


def test_any_of_accepts_multiple_date_formats():
    case = {"id": "x", "input": "3月12日", "expect": {"people": [
        {"canonical_name": "赵敏", "fields": [
            {"field": "birthday", "any_of": ["03-12", "3月12"]}]}]}}
    got = {"people": [_person(canonical_name="赵敏", fields=[
        {"field": "birthday", "value": "3月12日", "evidence": "3月12日"}])]}
    assert metrics.check_case(case, got)["passed"]


def test_fact_scope_any_searches_fields_too():
    case = {"id": "x", "input": "室友", "expect": {"people": [
        {"canonical_name": "老陆", "facts": [{"contains": ["室友"], "scope": "any"}]}]}}
    got = {"people": [_person(canonical_name="老陆",
                              fields=[{"field": "trait.关系", "value": "大学室友", "evidence": "室友"}])]}
    assert metrics.check_case(case, got)["passed"]


def test_ledger_amount_converts_yuan_to_cents():
    case = {"id": "x", "input": "转了两千", "expect": {"people": [
        {"canonical_name": "张伟", "ledger": [{"direction": "out", "amount": 2000}]}]}}
    good = {"people": [_person(canonical_name="张伟", ledger=[
        {"direction": "out", "amount_cents": 200000, "note": "份子钱"}])]}
    bad = {"people": [_person(canonical_name="张伟", ledger=[
        {"direction": "in", "amount_cents": 200000, "note": "份子钱"}])]}
    assert metrics.check_case(case, good)["passed"]
    assert not metrics.check_case(case, bad)["passed"]


def test_promise_due_must_match_exactly():
    case = {"id": "x", "input": "下周三", "expect": {"people": [
        {"canonical_name": "老陆", "promises": [{"contains": "报价单", "due": "2026-10-07"}]}]}}
    right = {"people": [_person(canonical_name="老陆", promises=[
        {"who": "me", "content": "发报价单", "due": "2026-10-07"}])]}
    wrong = {"people": [_person(canonical_name="老陆", promises=[
        {"who": "me", "content": "发报价单", "due": "2026-10-08"}])]}
    assert metrics.check_case(case, right)["passed"]
    assert not metrics.check_case(case, wrong)["passed"]


def test_must_not_promises_catches_false_positive():
    case = {"id": "x", "input": "改天吃饭", "must_not": ["promises"], "expect": {}}
    bad = {"people": [_person(promises=[{"who": "them", "content": "改天吃饭", "due": None}])]}
    res = metrics.check_case(case, bad)
    assert not res["passed"] and "不该出现承诺" in res["failures"][0]


def test_must_not_people_and_role_words():
    case = {"id": "x", "input": "那个客户说", "must_not": ["name:那个客户"], "expect": {}}
    bad = {"people": [_person(canonical_name="那个客户")]}
    assert not metrics.check_case(case, bad)["passed"]
    assert metrics.check_case(case, {"people": [_person(canonical_name="王总")]})["passed"]


def test_must_not_value_blocks_negation_leak():
    case = {"id": "x", "input": "他没去上海", "must_not": ["value:上海"], "expect": {}}
    bad = {"people": [_person(fields=[{"field": "city", "value": "上海", "evidence": "上海"}])]}
    assert not metrics.check_case(case, bad)["passed"]


def test_evidence_not_in_source_fails():
    case = {"id": "x", "input": "老陆在杭州", "expect": {}}
    bad = {"people": [_person(fields=[
        {"field": "city", "value": "上海", "evidence": "老陆在上海"}]),
        ]}
    res = metrics.check_case(case, bad)
    assert not res["passed"]
    assert "evidence" in res["failures"][0]
    assert res["stats"]["evidence_total"] == 1 and res["stats"]["evidence_hit"] == 0


def test_evidence_tolerates_whitespace_and_punctuation_diff():
    case = {"id": "x", "input": "老陆说：他要搬到杭州了。", "expect": {}}
    got = {"people": [_person(fields=[
        {"field": "city", "value": "杭州", "evidence": "他要搬到杭州了"}])]}
    assert metrics.check_case(case, got)["passed"]


def test_max_people_blocks_role_words():
    case = {"id": "x", "input": "他老婆", "expect": {}, "max_people": 1}
    bad = {"people": [_person(canonical_name="王工"), _person(canonical_name="他老婆")]}
    assert not metrics.check_case(case, bad)["passed"]


def test_find_person_matches_by_alias_and_also_match():
    people = [{"name": "陆建国", "canonical_name": "陆建国", "aliases": [{"alias": "老陆"}]}]
    assert metrics.find_person(people, {"canonical_name": "老陆"}) is not None
    assert metrics.find_person(people, {"canonical_name": "陆建国"}) is not None
    assert metrics.find_person(people, {"canonical_name": "张三"}) is None


def test_aliases_check_accepts_canonical_name():
    case = {"id": "x", "input": "陆建国", "expect": {"people": [
        {"canonical_name": "陆建国", "aliases": ["老陆"]}]}}
    got = {"people": [_person(canonical_name="老陆", name="陆建国", aliases=[{"alias": "老陆"}])]}
    assert metrics.check_case(case, got)["passed"]


def test_aggregate_computes_rates_and_negative_rate():
    cases = [
        {"id": "a", "tags": ["negative-promise"], "input": "改天吃饭", "must_not": ["promises"], "expect": {}},
        {"id": "b", "tags": ["positive"], "input": "老陆", "expect": {"people": [
            {"canonical_name": "老陆", "fields": [{"field": "city", "contains": "杭州"}]}]}},
    ]
    got_a = {"people": [_person(promises=[{"who": "them", "content": "改天吃饭", "due": None}])]}
    got_b = {"people": [_person(canonical_name="老陆", fields=[
        {"field": "city", "value": "杭州", "evidence": "老陆"}])]}
    results = [metrics.check_case(cases[0], got_a), metrics.check_case(cases[1], got_b)]
    agg = metrics.aggregate(cases, results)
    assert agg["cases"] == 2 and agg["passed"] == 1
    assert agg["pass_rate"] == 0.5
    assert agg["negative_promise_cases"] == 1
    assert agg["promise_false_positive_rate"] == 1.0
    assert agg["evidence_hit_rate"] == 1.0
    assert agg["field_recall"] == 1.0


def test_aggregate_handles_empty_denominator():
    agg = metrics.aggregate([], [])
    assert agg["evidence_hit_rate"] is None and agg["pass_rate"] is None


# ---------- 同字段多值：不再由字典覆盖顺序决定结果 ----------

def test_duplicate_field_values_fail_whatever_the_order():
    """城市给了两个取值（杭州 / 滨江）：换顺序都必须失败，否则用例会时过时不过。"""
    case = {"id": "x", "input": "老陆在杭州，公司在滨江", "expect": {"people": [
        {"canonical_name": "老陆", "fields": [{"field": "city", "contains": "杭州"}]}]}}
    hz_first = {"people": [_person(canonical_name="老陆", fields=[
        {"field": "city", "value": "杭州", "evidence": "老陆在杭州"},
        {"field": "city", "value": "滨江", "evidence": "公司在滨江"}])]}
    bj_first = {"people": [_person(canonical_name="老陆", fields=[
        {"field": "city", "value": "滨江", "evidence": "公司在滨江"},
        {"field": "city", "value": "杭州", "evidence": "老陆在杭州"}])]}
    ra, rb = metrics.check_case(case, hz_first), metrics.check_case(case, bj_first)
    assert not ra["passed"] and not rb["passed"]
    assert any("多个取值" in f for f in ra["failures"])
    assert any("多个取值" in f for f in rb["failures"])


def test_duplicate_field_with_identical_value_is_not_flagged():
    case = {"id": "x", "input": "老陆在杭州", "expect": {"people": [
        {"canonical_name": "老陆", "fields": [{"field": "city", "contains": "杭州"}]}]}}
    got = {"people": [_person(canonical_name="老陆", fields=[
        {"field": "city", "value": "杭州", "evidence": "老陆在杭州"},
        {"field": "city", "value": "杭州", "evidence": "老陆在杭州"}])]}
    assert metrics.check_case(case, got)["passed"]


def test_field_hit_when_any_of_multiple_values_matches():
    case = {"id": "x", "input": "老陆在杭州", "expect": {"people": [
        {"canonical_name": "老陆", "fields": [{"field": "city", "contains": "杭州"}]}]}}
    got = {"people": [_person(canonical_name="老陆", fields=[
        {"field": "title", "value": "总监", "evidence": "老陆"},
        {"field": "city", "value": "杭州", "evidence": "老陆在杭州"}])]}
    assert metrics.check_case(case, got)["passed"]


def test_field_mismatch_reports_all_candidate_values():
    case = {"id": "x", "input": "老陆", "expect": {"people": [
        {"canonical_name": "老陆", "fields": [{"field": "city", "contains": "杭州"}]}]}}
    got = {"people": [_person(canonical_name="老陆", fields=[
        {"field": "city", "value": "上海", "evidence": "老陆"}])]}
    res = metrics.check_case(case, got)
    assert not res["passed"] and "不满足" in res["failures"][0]
