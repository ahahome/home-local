"""评测对比逻辑：把抽取结果与用例期望对比，产出可解释的失败原因。

刻意做成纯计算——不调 LLM、不碰数据库。因为「评估器本身写错了」比「被测代码错了」
更危险，所以这一层必须能被普通单测覆盖（见 tests/test_eval_metrics.py）。

三条比较规则：
1. 子集匹配：expect 只声明「必须命中」的，抽取多出来的合理内容不算错。
2. must_not 优先：负样本（客套话、角色词、否定句）必须为空，否则直接判失败。
3. evidence 子串校验：每条抽出的 evidence 必须逐字出现在原文里（免费幻觉探针）。
"""
from __future__ import annotations

from app.util import flatten_for_match


def _flat(s) -> str:
    """去空白与标点后归一化：用于 evidence 子串校验与包含匹配。

    与抽取器共用 app.util.flatten_for_match，避免两边规则漂移——
    否则会出现「评测说 evidence 不在原文，但抽取器认为在」这种自相矛盾。
    """
    return flatten_for_match(s)


def _contains(hay: str, needle: str) -> bool:
    n = _flat(needle)
    return bool(n) and n in _flat(hay)


def _person_keys(p: dict) -> set[str]:
    keys = {_flat(p.get("canonical_name")), _flat(p.get("name"))}
    for a in p.get("aliases") or []:
        if isinstance(a, dict):
            keys.add(_flat(a.get("alias")))
        else:
            keys.add(_flat(a))
    return {k for k in keys if k}


def find_person(people: list[dict], expect: dict) -> dict | None:
    targets = set()
    for key in ("canonical_name", "name"):
        if expect.get(key):
            targets.add(_flat(expect[key]))
    for extra in expect.get("also_match") or []:
        targets.add(_flat(extra))
    targets = {t for t in targets if t}
    if not targets:
        return None
    for p in people:
        if _person_keys(p) & targets:
            return p
    return None


def _field_groups(p: dict) -> dict[str, list[dict]]:
    """按字段名分组，保留该字段的全部取值（一次抽取里同一字段可能给多条）。"""
    groups: dict[str, list[dict]] = {}
    for f in p.get("fields") or []:
        groups.setdefault((f.get("field") or "").strip().lower(), []).append(f)
    return groups


def _fields_of(p: dict) -> dict[str, dict]:
    return {k: v[-1] for k, v in _field_groups(p).items()}


def _spec_ok(f: dict, spec: dict) -> bool:
    value = f.get("value")
    if spec.get("contains") is not None and not _contains(value, spec["contains"]):
        return False
    if spec.get("equals") is not None and _flat(value) != _flat(spec["equals"]):
        return False
    if spec.get("any_of") is not None and not any(_contains(value, opt) for opt in spec["any_of"]):
        return False
    return True


def _match_field(p: dict, spec) -> str | None:
    """返回 None 表示通过，否则返回失败原因。

    同一字段在一次抽取里出现多条时，只要【任一】取值满足要求就算命中——
    不靠「后写覆盖前写」的顺序巧合来定评测结果（那会让同一条用例时过时不过）。
    同字段多值本身是缺陷，由 _dup_field_failures 单独判失败。
    """
    if isinstance(spec, str):
        for f in p.get("fields") or []:
            if _contains(f.get("value"), spec):
                return None
        return f"字段缺失：任何字段值都不含「{spec}」"
    key = (spec.get("field") or "").strip().lower()
    candidates = _field_groups(p).get(key)
    if not candidates:
        return f"字段缺失：{key}（实得 {sorted(_fields_of(p))}）"
    if any(_spec_ok(f, spec) for f in candidates):
        return None
    got = [f.get("value") for f in candidates]
    shown = "、".join(f"{k}={v}" for k, v in spec.items())
    return f"字段 {key} 取值 {got!r} 不满足（期望 {shown}）"


def _dup_field_failures(people: list[dict]) -> list[str]:
    """同一个人、同一字段出现多个【不同】取值 → 判失败。

    这是真实缺陷：一次抽取给出两条 city（杭州 / 滨江），落到 apply 层就由写入顺序
    决定谁生效，结果不可预期。宁可显式判失败，也不要让评测结论依赖字典覆盖顺序。
    """
    bad: list[str] = []
    for p in people:
        who = p.get("canonical_name") or p.get("name")
        for key, group in _field_groups(p).items():
            vals = {_flat(f.get("value")) for f in group}
            if key and len(vals) > 1:
                bad.append(f"{who} 的字段 {key} 一次给了多个取值 "
                           f"{[f.get('value') for f in group]}（落库时谁生效不可预期）")
    return bad


def _match_fact(p: dict, spec) -> str | None:
    facts = [fa.get("text") or "" for fa in p.get("facts") or []]
    needles = [spec] if isinstance(spec, str) else list(spec.get("contains") or [])
    scope = [facts]
    if isinstance(spec, dict) and spec.get("scope") == "any":
        scope.append([f.get("value") or "" for f in p.get("fields") or []])
    for needle in needles:
        if not any(_contains(text, needle) for group in scope for text in group):
            return f"事实缺失：未找到含「{needle}」的事实"
    return None


def _match_promise(p: dict, spec) -> str | None:
    spec = {"contains": spec} if isinstance(spec, str) else spec
    for pr in p.get("promises") or []:
        ok = True
        if spec.get("contains") is not None and not _contains(pr.get("content"), spec["contains"]):
            ok = False
        if spec.get("due") is not None and str(pr.get("due") or "").strip() != str(spec["due"]):
            ok = False
        if spec.get("who") is not None and pr.get("who") != spec["who"]:
            ok = False
        if ok:
            return None
    detail = "、".join(f"{k}={v}" for k, v in spec.items())
    return f"承诺缺失或字段不符（期望 {detail}；实得 {[pr.get('content') for pr in p.get('promises') or []]}）"


def _match_ledger(p: dict, spec) -> str | None:
    for entry in p.get("ledger") or []:
        ok = True
        if spec.get("direction") and entry.get("direction") != spec["direction"]:
            ok = False
        if spec.get("amount") is not None:
            want = int(round(float(spec["amount"]) * 100))
            if int(entry.get("amount_cents") or 0) != want:
                ok = False
        if spec.get("contains") and not _contains(entry.get("note"), spec["contains"]):
            ok = False
        if ok:
            return None
    detail = "、".join(f"{k}={v}" for k, v in spec.items())
    return f"账本缺失或金额不符（期望 {detail}；实得 {[(e.get('direction'), e.get('amount_cents')) for e in p.get('ledger') or []]}）"


def _check_must_not(people: list[dict], spec) -> str | None:
    """负样本检查：命中即失败。返回违反描述或 None。"""
    if isinstance(spec, str):
        if spec == "promises":
            spec = {"type": "promises"}
        elif spec == "ledger":
            spec = {"type": "ledger"}
        elif spec == "people":
            spec = {"type": "people"}
        elif spec.startswith("name:"):
            spec = {"type": "no_person", "contains": spec[5:]}
        elif spec.startswith("field:"):
            spec = {"type": "no_field", "field": spec[6:]}
        elif spec.startswith("value:"):
            spec = {"type": "value", "contains": spec[6:]}
        else:
            spec = {"type": "value", "contains": spec}

    kind = spec.get("type")
    if kind == "promises":
        bad = [p.get("canonical_name") for p in people if p.get("promises")]
        return f"不该出现承诺，却抽到了：{bad}" if bad else None
    if kind == "ledger":
        bad = [p.get("canonical_name") for p in people if p.get("ledger")]
        return f"不该出现账本，却抽到了：{bad}" if bad else None
    if kind == "people":
        return f"不该出现任何人物，却抽到了：{[p.get('canonical_name') for p in people]}" if people else None
    if kind == "no_person":
        bad = [k for p in people for k in _person_keys(p) if _contains(k, spec.get("contains"))]
        return f"不该被当成人名，却抽到了：{bad}" if bad else None
    if kind == "no_field":
        bad = [f"{p.get('canonical_name')}.{spec['field']}" for p in people if spec["field"].lower() in _fields_of(p)]
        return f"不该出现字段 {spec['field']}，却抽到了：{bad}" if bad else None
    if kind == "value":
        bad = [
            f"{p.get('canonical_name')}.{f.get('field')}={f.get('value')}"
            for p in people for f in p.get("fields") or []
            if _contains(f.get("value"), spec.get("contains"))
        ]
        return f"不该出现取值「{spec.get('contains')}」，却抽到了：{bad}" if bad else None
    if kind == "no_bind":
        return None  # 绑定层判定，L1 不适用
    return f"未知的 must_not 类型：{kind}"


def check_evidence(people: list[dict], source_text: str) -> tuple[int, int, list[str]]:
    """(总条数, 命中数, 反例)。evidence 必须逐字出现在原文中。"""
    items: list[tuple[str, str]] = []
    for p in people:
        who = p.get("canonical_name")
        for f in p.get("fields") or []:
            items.append((f"{who}.{f.get('field')}", f.get("evidence") or ""))
        for fa in p.get("facts") or []:
            items.append((f"{who}.fact", fa.get("evidence") or ""))
        vr = p.get("value_read")
        if isinstance(vr, dict):
            items.append((f"{who}.value_read", vr.get("evidence") or ""))
    checked = [(k, e) for k, e in items if e.strip()]
    hits = sum(1 for _, e in checked if _contains(source_text, e))
    bad = [f"{k}: {e[:40]}" for k, e in checked if not _contains(source_text, e)]
    return len(checked), hits, bad[:5]


def check_case(case: dict, got: dict) -> dict:
    """对比单条用例，返回 {passed, failures, stats}。"""
    failures: list[str] = []
    people = got.get("people") or []
    expect = case.get("expect") or {}
    field_total = field_hit = promise_expected = promise_hit = 0

    for ep in expect.get("people") or []:
        who = ep.get("canonical_name") or ep.get("name")
        p = find_person(people, ep)
        if p is None:
            failures.append(f"缺少人物「{who}」（实得 {[q.get('canonical_name') for q in people]}）")
            continue
        for spec in ep.get("aliases") or []:
            got_aliases = " ".join(
                [str(a.get("alias") if isinstance(a, dict) else a) for a in p.get("aliases") or []]
                + [p.get("canonical_name") or "", p.get("name") or ""])
            if not _contains(got_aliases, spec):
                failures.append(f"{who} 缺少别名「{spec}」")
        for spec in ep.get("fields") or []:
            field_total += 1
            err = _match_field(p, spec)
            if err:
                failures.append(f"{who}：{err}")
            else:
                field_hit += 1
        for spec in ep.get("facts") or []:
            err = _match_fact(p, spec)
            if err:
                failures.append(f"{who}：{err}")
        for spec in ep.get("promises") or []:
            promise_expected += 1
            err = _match_promise(p, spec)
            if err:
                failures.append(f"{who}：{err}")
            else:
                promise_hit += 1
        for spec in ep.get("ledger") or []:
            err = _match_ledger(p, spec)
            if err:
                failures.append(f"{who}：{err}")

    failures.extend(_dup_field_failures(people))

    for spec in case.get("must_not") or []:
        err = _check_must_not(people, spec)
        if err:
            failures.append(err)

    max_people = case.get("max_people")
    if max_people is not None and len(people) > max_people:
        failures.append(f"人物数量 {len(people)} 超过上限 {max_people}（疑似把角色词/自己抽成人名）")

    ev_total, ev_hit, ev_bad = check_evidence(people, case.get("input") or "")
    if ev_total and ev_hit < ev_total:
        failures.append(f"evidence 不在原文中（{ev_total - ev_hit}/{ev_total}）：{ev_bad}")

    return {
        "id": case.get("id"),
        "passed": not failures,
        "failures": failures,
        "people_found": len(people),
        "stats": {
            "field_total": field_total, "field_hit": field_hit,
            "promise_expected": promise_expected, "promise_hit": promise_hit,
            "evidence_total": ev_total, "evidence_hit": ev_hit,
        },
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def aggregate(cases: list[dict], results: list[dict]) -> dict:
    """汇总指标。pass_rate 用「全部断言通过」的严格口径。"""
    by_id = {c.get("id"): c for c in cases}
    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    f_tot = sum(r["stats"]["field_total"] for r in results)
    f_hit = sum(r["stats"]["field_hit"] for r in results)
    p_exp = sum(r["stats"]["promise_expected"] for r in results)
    p_hit = sum(r["stats"]["promise_hit"] for r in results)
    e_tot = sum(r["stats"]["evidence_total"] for r in results)
    e_hit = sum(r["stats"]["evidence_hit"] for r in results)

    # 承诺误判率：标了 negative-promise 的用例里，仍然抽出承诺的比例
    neg = [r for r in results if "negative-promise" in (by_id.get(r["id"], {}).get("tags") or [])]
    neg_bad = [r for r in neg if not r["passed"] and any("不该出现承诺" in f for f in r["failures"])]

    return {
        "cases": n,
        "passed": passed,
        "pass_rate": _rate(passed, n),
        "field_recall": _rate(f_hit, f_tot),
        "promise_recall": _rate(p_hit, p_exp),
        "evidence_hit_rate": _rate(e_hit, e_tot),
        "promise_false_positive_rate": _rate(len(neg_bad), len(neg)),
        "negative_promise_cases": len(neg),
    }
