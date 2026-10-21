"""工具集：OpenAI function-calling schema + 执行分发。

所有写操作工具必须返回 receipt（P2：写入必有回执）。
"""
from __future__ import annotations

import json
import re

from app.agent.memory_sources import field_sources, memory_source
from app.memory import compile_card, retrieve
from app.money import amount_to_cents
from app.store import repo
from app.util import ms_to_local_str, now_ms

TOOLS = [
    {"type": "function", "function": {
        "name": "search_people",
        "description": "按名字/昵称模糊搜索人物，返回候选人列表。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "名字、昵称或拼音"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_person",
        "description": "获取某人的完整档案卡：字段、最近互动、未闭合承诺、账本汇总、价值评估。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "save_person_fact",
        "description": "用户明确告知某人信息时，保存一个字段（如 company/title/city/birthday/trait.*）。",
        "parameters": {"type": "object", "properties": {
            "person": {"type": "string"}, "field": {"type": "string"},
            "value": {"type": "string"}, "evidence": {"type": "string", "description": "用户原话"}},
            "required": ["person", "field", "value"]}}},
    {"type": "function", "function": {
        "name": "query_memory",
        "description": "检索记忆与互动，返回带日期、来源与历史背景标记的结果。指定人物可限制范围，同名人物必须用 person_id。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 100},
            "person": {"type": "string", "description": "准确姓名或别名，同名时需要 person_id"},
            "person_id": {"type": "string", "description": "已知人物 ID，限制检索该人物相关记忆与互动"},
            "person_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True,
                           "description": "比较多个人物时指定 ID 集合，不能同时传 person_id 或 person"},
            "retrieval_mode": {"type": "string", "enum": ["hybrid", "fts"], "default": "hybrid",
                               "description": "hybrid 使用已配置通道，fts 只使用本地文字检索"},
            "scope": {"type": "string", "enum": ["all", "personal"], "default": "all",
                      "description": "all 为全部或指定人物，personal 仅个人记忆与未关联人物的资料，不能同时指定人物"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_promise",
        "description": "记录一条承诺（我答应的或对方答应的），可带截止日期，会自动建提醒。",
        "parameters": {"type": "object", "properties": {
            "person": {"type": "string"}, "who": {"type": "string", "enum": ["me", "them"]},
            "content": {"type": "string"}, "due_date": {"type": "string", "description": "YYYY-MM-DD，可空"}},
            "required": ["who", "content"]}}},
    {"type": "function", "function": {
        "name": "list_open_promises",
        "description": "列出未闭合的承诺。不传 person 则列出全部。",
        "parameters": {"type": "object", "properties": {"person": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "complete_promise",
        "description": "把一条承诺标记为已完成。",
        "parameters": {"type": "object", "properties": {
            "promise_id": {"type": "string"}}, "required": ["promise_id"]}}},
    {"type": "function", "function": {
        "name": "record_transfer",
        "description": "记录一笔与某人的资金往来（转账/收款）。",
        "parameters": {"type": "object", "properties": {
            "person": {"type": "string"},
            "direction": {"type": "string", "enum": ["in", "out"], "description": "in=我收到 out=我付出"},
            "amount_yuan": {"type": "number"}, "note": {"type": "string"},
            "occurred_date": {"type": "string", "description": "YYYY-MM-DD，默认今天"}},
            "required": ["direction", "amount_yuan"]}}},
    {"type": "function", "function": {
        "name": "ledger_summary",
        "description": "查看账本汇总与明细（待确认不计入总额）。",
        "parameters": {"type": "object", "properties": {"person": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "save_skill",
        "description": "把一套打法/某人的沟通风格保存为可复用技能，之后用 /技能名 唤起。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "person": {"type": "string"},
            "style_notes": {"type": "string", "description": "沟通风格要点"},
            "body": {"type": "string", "description": "技能正文：步骤/话术/注意事项"},
            "trigger_hint": {"type": "string"}},
            "required": ["name", "body"]}}},
    {"type": "function", "function": {
        "name": "run_skill",
        "description": "取出一个已保存技能的完整内容，用于指导当前回复。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list_skills",
        "description": "列出所有已保存的技能。", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "deep_research",
        "description": "对一个人/公司/行业做深度调研，出一份带来源的报告。耗时较长。",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string"}}, "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "create_reminder",
        "description": "创建一个到点提醒。",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "fire_date": {"type": "string", "description": "YYYY-MM-DD"},
            "person": {"type": "string"}},
            "required": ["text", "fire_date"]}}},
    {"type": "function", "function": {
        "name": "who_quiet",
        "description": "找出最近 N 天没有实际联系、按重要度排序的人。资料记录和草稿不算联系。",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "default": 14}}}}},
    {"type": "function", "function": {
        "name": "note",
        "description": "记录一条关于用户自己的目标/偏好/状态。",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}}, "required": ["text"]}}},
]


for _tool in TOOLS:
    if _tool['function']['name'] in {'get_person', 'save_person_fact', 'create_promise', 'list_open_promises',
                                    'record_transfer', 'ledger_summary', 'save_skill', 'create_reminder'}:
        _tool['function']['parameters']['properties']['person_id'] = {
            'type': 'string', 'description': '已知人物 ID；有同名人物时必须使用检索返回的 ID'}


def _resolve_person_id(conn, name: str | None, person_id: str | None = None) -> str | None:
    if not name and not person_id:
        return None
    from app.memory.writes import resolve_person
    return resolve_person(conn, name, person_id=person_id)


def _query_person_hint(conn, query):
    from app.memory.writes import AmbiguousPerson
    rows = conn.execute('SELECT DISTINCT a.alias,p.id,p.display_name FROM person_alias a '
                        "JOIN person p ON p.id=a.person_id WHERE p.deleted_at IS NULL AND p.status!='merged'").fetchall()
    aliases = {row['alias'] for row in rows if len(row['alias']) >= 2}
    matches = [(match.start(), match.end(), alias) for alias in aliases
               for match in re.finditer(_alias_pattern(alias), query)]
    longest = {alias for start, end, alias in matches if not any(
        other_start <= start and other_end >= end and other_end - other_start > end - start
        for other_start, other_end, _ in matches)}
    people = set()
    for alias in sorted(longest):
        matches = [row for row in rows if row['alias'] == alias]
        if len(matches) > 1:
            raise AmbiguousPerson(alias, matches)
        people.add(matches[0]['id'])
    return next(iter(people)) if len(people) == 1 else None


def _alias_pattern(alias):
    if re.fullmatch(r'[A-Za-z0-9_ ]+', alias):
        return r'(?<![A-Za-z0-9_])' + re.escape(alias) + r'(?![A-Za-z0-9_])'
    return re.escape(alias)


def _alias_in_query(alias, query):
    return re.search(_alias_pattern(alias), query) is not None


def _scoped_query(conn, query, person_id):
    aliases = sorted(repo.aliases_of(conn, person_id), key=len, reverse=True)
    matches = [(match.start(), match.end()) for alias in aliases
               for match in re.finditer(_alias_pattern(alias) if len(alias) > 1 else
                                        r'(?<!\w)' + re.escape(alias) + r'(?!\w)', query)]
    # Apply intervals against the original query, so removing a long name cannot expose a short alias inside a word.
    for start, end in sorted(matches, reverse=True):
        if any(other_start <= start and other_end >= end and (other_start, other_end) != (start, end)
               for other_start, other_end in matches):
            continue
        query = query[:start] + ' ' + query[end:]
    return query.strip()


class ReadScopeError(ValueError):
    def __init__(self, message, candidates=None):
        super().__init__(message)
        self.candidates = candidates or []


def _read_scope_args(conn, name, args, read_scope):
    if read_scope is None:
        return args
    if not isinstance(read_scope, dict):
        raise ReadScopeError('请先确认本次查询的人物。')
    if read_scope.get('blocked') or read_scope.get('mode') == 'clarify':
        raise ReadScopeError(read_scope.get('reason') or '请先确认本次查询的人物。', read_scope.get('candidates'))
    ids = read_scope.get('person_ids', [])
    if not isinstance(ids, list) or any(not isinstance(pid, str) or not pid for pid in ids):
        raise ReadScopeError('人物查询范围无效，请重新指定人物。')
    if not ids and read_scope.get('mode') in {'none', 'global'}:
        return args
    if not ids:
        raise ReadScopeError('当前没有已确认的查询人物，请重新指定。')
    allowed = sorted(set(ids))
    candidates = []
    for pid in allowed:
        person = repo.get_person(conn, pid)
        if not person or person['deleted_at'] is not None or person['status'] == 'merged':
            raise ReadScopeError('本轮查询的人物资料已变化，请重新确认人物。')
        candidates.append({'person_id': pid, 'display_name': person['display_name']})
    args = dict(args)
    if name == 'query_memory' and args.get('scope') == 'personal':
        raise ReadScopeError('本轮已限定人物，不能同时切换为个人记忆范围。', candidates)
    explicit_many = args.get('person_ids')
    if explicit_many is not None:
        if name == 'get_person':
            raise ReadScopeError('请指定要查看的单个人物档案。', candidates)
        selected = retrieve._selected_people(args.get('person_id'), explicit_many)
        if args.get('person') or not selected or not set(selected) <= set(allowed):
            raise ReadScopeError('本次读取超出了已确认的人物范围，请重新指定。', candidates)
        return args
    explicit_name = args.get('name') if name == 'get_person' else args.get('person')
    explicit_id = args.get('person_id')
    if explicit_name and not explicit_id:
        matches = conn.execute('SELECT DISTINCT p.id FROM person p JOIN person_alias a ON a.person_id=p.id '
                               "WHERE a.alias=? AND p.deleted_at IS NULL AND p.status!='merged'",
                               (explicit_name,)).fetchall()
        matches = [row['id'] for row in matches if row['id'] in allowed]
        if len(matches) == 1:
            explicit_id = matches[0]
        elif len(matches) > 1:
            raise ReadScopeError('这个名字对应多位查询人物，请使用人物 ID。', candidates)
        else:
            raise ReadScopeError('本次读取超出了已确认的人物范围，请重新指定。', candidates)
    if explicit_id:
        if explicit_id not in allowed:
            raise ReadScopeError('本次读取超出了已确认的人物范围，请重新指定。', candidates)
        args['person_id'] = explicit_id
    elif len(allowed) == 1:
        args['person_id'] = allowed[0]
    elif name == 'query_memory':
        args['person_ids'] = allowed
    else:
        raise ReadScopeError('请从本次比较的人物中选择要查看的档案。', candidates)
    return args


_WRITE_TOOLS = frozenset({'save_person_fact', 'record_transfer', 'create_promise', 'note',
                          'complete_promise', 'save_skill', 'deep_research', 'create_reminder'})


def _denied_write(name, reason):
    detail = '本轮未授权这项写入，未保存或修改资料。' + reason
    return {'ok': False, 'error': detail, 'write_blocked': True,
            'receipt': {'type': name, 'status': 'skipped', 'detail': detail},
            'outcomes': [{'kind': name, 'status': 'skipped', 'detail': detail}]}


def _person_arg_grounded(conn, args, text, read_scope):
    from app.memory.write_grounding import mentions, scoped_pronoun_reference
    person = args.get('person')
    pid = args.get('person_id')
    if isinstance(read_scope, dict) and read_scope.get('blocked'):
        return False
    if person or pid:
        aliases = repo.aliases_of(conn, pid) if pid else [person]
        if pid and person and person not in aliases:
            return False
        found = mentions(conn, text)
        mentioned = any(ids == {pid} for _, _, _, ids in found) if pid else \
            any(isinstance(alias, str) and alias and alias in text for alias in aliases)
        selected_name = (pid and isinstance(read_scope, dict) and read_scope.get('mode') == 'single'
                         and read_scope.get('person_ids') == [pid] and
                         any(pid in ids for _, _, _, ids in found) and
                         not any(pid not in ids for _, _, _, ids in found))
        scoped_pronoun = (isinstance(read_scope, dict) and read_scope.get('mode') == 'single'
                          and read_scope.get('person_ids') == [pid]
                          and scoped_pronoun_reference(text)
                          and not any(pid not in ids for _, _, _, ids in found))
        if not mentioned and not selected_name and not (pid and pid in text) and not scoped_pronoun:
            return False
    return True


def _literal_arg(args, key, text, *, optional=False):
    value = args.get(key)
    if optional and value in (None, ''):
        return True
    return isinstance(value, str) and bool(value.strip()) and value in text


def _explicit_dates(text):
    from datetime import date
    dates = set()
    for match in re.finditer(r'(?<!\d)(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})(?:日|号)?(?!\d)', text):
        try:
            dates.add(date(*(int(part) for part in match.groups())).isoformat())
        except ValueError:
            pass
    return dates


def _completion_grounded(conn, args, text, read_scope):
    row = conn.execute('SELECT * FROM promise WHERE id=?', (args.get('promise_id'),)).fetchone()
    if row is None:
        return False
    pid = row['person_id']
    if isinstance(read_scope, dict):
        allowed = read_scope.get('person_ids', [])
        if read_scope.get('blocked') or allowed and pid not in allowed:
            return False
    if args.get('person_id') and args['person_id'] != pid:
        return False
    aliases = repo.aliases_of(conn, pid) if pid else []
    if args.get('person') and args['person'] not in aliases:
        return False
    # A literal promise ID identifies the exact operation even when its wording
    # is duplicated. Otherwise require the full content and an unambiguous owner.
    if row['id'] in text:
        return True
    if not row['content'] or row['content'] not in text:
        return False
    owner_selected = (pid is None or any(alias and alias in text for alias in aliases)
                      or isinstance(read_scope, dict) and read_scope.get('mode') == 'single'
                      and read_scope.get('person_ids') == [pid])
    if not owner_selected:
        return False
    matching = conn.execute("SELECT id FROM promise WHERE content=? AND person_id IS ? AND status='open'",
                            (row['content'], pid)).fetchall()
    return len(matching) == 1 and matching[0]['id'] == row['id']


def _action_args_grounded(conn, name, args, text, read_scope):
    if not _person_arg_grounded(conn, args, text, read_scope):
        return False
    if name == 'create_reminder':
        action = re.search(r'提醒|闹钟|\bremind\b', text, re.I)
        if (action is None or not _literal_arg(args, 'text', text[action.end():])
                or re.search(r'如果|要是|假如|\bif\b', text, re.I)):
            return False
        # The current tool accepts a fixed calendar day, not a conditional or
        # relative schedule. Multiple dates require a more precise instruction.
        return isinstance(args.get('fire_date'), str) and _explicit_dates(text) == {args['fire_date']}
    if name == 'save_skill':
        return bool(re.search(r'技能|打法|\bskill\b', text, re.I)
                    and re.search(r'保存|记录|记下|存成|创建|新增|更新|修改|替换|\b(?:save|create|update)\b', text, re.I)
                    and all(_literal_arg(args, key, text) for key in ('name', 'body'))
                    and all(_literal_arg(args, key, text, optional=True)
                            for key in ('style_notes', 'trigger_hint')))
    if name == 'complete_promise':
        return bool(re.search(r'完成|办妥|履行|结清|搞定|\b(?:done|completed|fulfilled)\b', text, re.I)
                    and _completion_grounded(conn, args, text, read_scope))
    if name == 'deep_research':
        action = re.search(r'调研|调查|研究|查.{0,5}背景|\bresearch\b', text, re.I)
        return bool(action and _literal_arg(args, 'topic', text[action.end():]))
    return False


def _shared_dates_grounded(conn, name, args, spans, read_scope):
    supplied = [args[key] for key in ('valid_from', 'due_date', 'occurred_date')
                if args.get(key) not in (None, '')]
    if not supplied:
        return True  # Omitted dates preserve the tool's existing system-time default.
    if any(not isinstance(value, str) for value in supplied) or len(set(supplied)) != 1:
        return False
    day = supplied[0]
    value_key = {'save_person_fact': 'value', 'create_promise': 'content', 'note': 'text'}.get(name)
    for span in spans:
        if _explicit_dates(span) != {day} or not _person_arg_grounded(conn, args, span, read_scope):
            continue
        if value_key and not _literal_arg(args, value_key, span):
            continue
        if name == 'record_transfer':
            cents = amount_to_cents(args.get('amount_yuan'))
            amounts = []
            for literal in re.findall(r'\d+(?:\.\d+)?', span):
                try:
                    amounts.append(amount_to_cents(literal))
                except ValueError:
                    pass
            if cents not in amounts or not re.search(r'转|给|收|付|借|还|礼金|随礼|报销', span):
                continue
        return True
    return False


def _party_names(conn, args, read_scope):
    pid, person = args.get('person_id'), args.get('person')
    names = repo.aliases_of(conn, pid) if pid else [person] if person else []
    if (isinstance(read_scope, dict) and read_scope.get('mode') == 'single'
            and read_scope.get('person_ids') == [pid]):
        names = [*names, '他', '她', '对方']
    return [name for name in names if isinstance(name, str) and name]


def _shared_span_grounded(conn, name, args, span, read_scope):
    if not _person_arg_grounded(conn, args, span, read_scope):
        return False
    value_key = {'save_person_fact': 'value', 'create_promise': 'content', 'note': 'text'}.get(name)
    if value_key and not _literal_arg(args, value_key, span):
        return False
    if not _shared_dates_grounded(conn, name, args, [span], read_scope):
        return False
    if name == 'record_transfer':
        from app.memory.writes import explicit_transfer_direction
        cents = amount_to_cents(args.get('amount_yuan'))
        amounts = []
        for value in re.findall(r'\d+(?:\.\d+)?', span):
            try:
                amounts.append(amount_to_cents(value))
            except ValueError:
                pass
        if (cents not in amounts or not re.search(r'转|给|收|付|借|还|礼金|随礼|报销', span)
                or not _literal_arg(args, 'note', span, optional=True)):
            return False
        # Calendar dates between "我" and "给" do not change the parties or
        # direction. Keep the existing narrow transfer parser for that binding.
        direction_text = re.sub(r'(?:在)?\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日|号)?', '', span)
        direction = explicit_transfer_direction(direction_text, _party_names(conn, args, read_scope), cents)
        # The existing matcher handles only explicit transfer phrasing. Do not
        # claim it proves every possible direction; unsupported phrasing retains
        # apply_extraction's original confidence and confirmation rules.
        return direction is None or direction == args.get('direction')
    return name in {'save_person_fact', 'create_promise', 'note'}


def _bound_write_span(conn, name, args, text, read_scope, record_spans=None):
    # Return the actual supporting span so apply_extraction's direction and
    # confidence checks cannot read a different approved record as its evidence.
    check = (_action_args_grounded if name in
             {'create_reminder', 'save_skill', 'complete_promise', 'deep_research'}
             else _shared_span_grounded)
    return next((span for span in (record_spans or [text])
                 if check(conn, name, args, span, read_scope)), None)


def execute_tool(conn, name: str, args: dict, source_text: str | None = None,
                 source_msg_id: str | None = None, read_scope: dict | None = None,
                 turn_intent: dict | None = None) -> dict:
    from app.memory.writes import AmbiguousPerson
    try:
        if name in _WRITE_TOOLS:
            from app.memory.write_grounding import validate_write_scope
            validate_write_scope(conn, source_text or '', read_scope)
        if name in _WRITE_TOOLS and turn_intent is not None:
            from app.agent.turn_intent import write_text
            grounded = write_text(turn_intent, source_text)
            if grounded is None:
                return _denied_write(name, '查询或无法判定的消息仅作只读处理。')
            # Even in a mixed turn a tool cannot turn the question into a new
            # fact/pending item. Paraphrases still go through extraction's review.
            value_key = {'save_person_fact': 'value', 'create_promise': 'content', 'note': 'text'}.get(name)
            if value_key and (not isinstance(args.get(value_key), str) or not args[value_key]
                              or args[value_key] not in grounded):
                return _denied_write(name, '工具内容不在本轮已确认的记事原话中。')
            bound_span = _bound_write_span(conn, name, args, grounded, read_scope, turn_intent['record_spans'])
            if bound_span is None:
                return _denied_write(name, '人物、内容、金额、日期或操作不在同一段已确认的记事原话中；请明确操作对象与日历日期。')
            source_text = bound_span
            if name in {'save_person_fact', 'record_transfer', 'create_promise'} and args.get('person'):
                from app.memory.write_grounding import scoped_identity
                bound = scoped_identity(conn, {'name': args['person']}, bound_span, read_scope)
                if bound:
                    if args.get('person_id') and args['person_id'] != bound:
                        return _denied_write(name, '人物称呼与选择的人物不一致。')
                    args = {**args, 'person_id': bound}
        if name in {"save_person_fact", "record_transfer", "create_promise", "note"}:
            return _shared_write(conn, name, args, source_text, source_msg_id)
        if name in {'query_memory', 'get_person'}:
            args = _read_scope_args(conn, name, args, read_scope)
        return _execute_tool(conn, name, args)
    except ReadScopeError as e:
        return {'ok': False, 'needs_confirmation': True, 'error': str(e), 'candidates': e.candidates}
    except AmbiguousPerson as e:
        return {"ok": False, "needs_confirmation": True, "error": str(e), "candidates": e.candidates}
    except (ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": str(e)}


def _shared_write(conn, name, args, source_text, source_msg_id):
    from app.memory.apply import apply_extraction, _date_ms
    from app.memory.writes import resolve_person, promise_key, matching_promise
    from app.config import get_settings
    from app.util import new_id, normalize_value, sha256_text
    source_id = source_msg_id or new_id()
    forced = {}
    for key in ("due_date", "occurred_date", "valid_from"):
        if args.get(key) and not _date_ms(args[key]):
            raise ValueError("日期格式应为 YYYY-MM-DD")
    source = source_text if source_text is not None else args.get("evidence", "手动记录")
    confidence = 0.95
    if source_text is not None:
        value = str(args.get("value") or args.get("content") or args.get("text") or "")
        if not value or value not in source:
            confidence = 0.7
    if name == "record_transfer":
        import re
        cents = amount_to_cents(args["amount_yuan"])
        if args["direction"] not in ("in", "out"):
            raise ValueError("金额必须为正，方向必须为 in 或 out")
        if source_text is not None:
            mentioned = False
            for value in re.findall(r"\d+(?:\.\d+)?", source):
                try:
                    mentioned |= amount_to_cents(value) == cents
                except ValueError:
                    pass
            confidence = .95 if mentioned and re.search(r"转|给|收|付|借|还|礼金|随礼|报销", source) else .7
    if name == "note":
        obj = {"people": [], "my_notes": [{"text": args["text"], "kind": "semantic", "confidence": confidence}]}
    elif name == "create_promise" and not args.get("person") and not args.get('person_id'):
        due = args.get("due_date") or None
        if due and not _date_ms(due):
            raise ValueError("日期格式应为 YYYY-MM-DD")
        fp = promise_key(None, args["who"], args["content"], due)
        if repo.has_patch(conn, fp) or matching_promise(conn, None, args['who'], args['content'], due, _date_ms(due)):
            return {"ok": True, "receipt": {"type": "promise", "duplicate": True}}
        eid = repo.add_evidence(conn, "text", source_ref=source_id, excerpt=source)
        if confidence < get_settings().confidence_auto:
            pending = repo.add_pending(conn, "promise", {"person_id": None, "who": args["who"],
                "content": args["content"], "due": due, "confidence": confidence,
                "fingerprint": fp, "source_msg_id": source_id, "evidence_id": eid}, "承诺待确认")
            conn.commit()
            return {"ok": True, "receipt": {"type": "promise", "status": "pending", "pending_id": pending}}
        pid = repo.add_promise(conn, args["who"], args["content"], None, _date_ms(due), eid)
        repo.mark_patch(conn, fp, None, "promise", normalize_value(args["content"]), source_id)
        if due:
            repo.add_reminder(conn, _date_ms(due), args["content"], None, pid)
        conn.commit()
        return {"ok": True, "receipt": {"type": "promise", "id": pid, "status": "ok"}}
    else:
        person = args.get("person")
        if not person and not args.get('person_id'):
            raise ValueError("请指定对应人物")
        if person and args.get('person_id') and person not in repo.aliases_of(conn, args['person_id']):
            raise ValueError('人物称呼与人物ID不一致，请重新选择')
        pid = resolve_person(conn, person, person_id=args.get('person_id'), create=True)
        person = person or repo.get_person(conn, pid)['display_name']
        forced[person] = pid
        p = {"name": person, "canonical_name": person, "fields": [], "aliases": [],
             "relations": [], "facts": [], "promises": [], "ledger": []}
        if name == "save_person_fact":
            p["fields"] = [{"field": args["field"], "value": args["value"],
                "confidence": confidence, "evidence": source, "valid_from": args.get("valid_from")}]
        elif name == "record_transfer":
            day = args.get("occurred_date") or None
            if day and not _date_ms(day):
                raise ValueError("日期格式应为 YYYY-MM-DD")
            p["ledger"] = [{"direction": args["direction"], "amount_cents": cents,
                "note": args.get("note", ""), "occurred": day, "confidence": confidence}]
        else:
            day = args.get("due_date") or None
            if day and not _date_ms(day):
                raise ValueError("日期格式应为 YYYY-MM-DD")
            p["promises"] = [{"who": args["who"], "content": args["content"],
                "due": day, "confidence": confidence}]
        obj = {"people": [p], "my_notes": []}
    result = apply_extraction(conn, obj, source_id, source, forced=forced)
    return {"ok": not any(o["status"] == "failed" for o in result["outcomes"]),
            **result, "receipt": {"type": name, "outcomes": result["outcomes"],
                                  "pending_ids": result["pending_ids"]}}


def _execute_tool(conn, name: str, args: dict) -> dict:
    from app.config import tz
    from datetime import datetime

    if name == "search_people":
        hits = retrieve.search_people(conn, args["query"])
        return {"ok": True, "results": hits}

    if name == "get_person":
        pid = _resolve_person_id(conn, args.get("name"), args.get('person_id'))
        if not pid:
            return {"ok": False, "error": f"没有找到「{args['name']}」的档案"}
        card = compile_card.build_card(conn, pid)
        p = repo.get_person(conn, pid)
        sources = field_sources(conn, pid)
        sources.extend(source for row in repo.memories_of(conn, pid, limit=10)
                       if (source := memory_source(conn, row['id'])) is not None)
        return {"ok": True, "card": card, "person_id": pid,
                "importance": p["importance"], "status": p["status"], "sources": sources}

    if name == "query_memory":
        scope = args.get('scope', 'all')
        pids = args.get('person_ids')
        if pids is not None and (args.get('person') or args.get('person_id')):
            raise ValueError('人物集合不能同时指定单个人物')
        pid = _resolve_person_id(conn, args.get('person'), args.get('person_id'))
        inferred = False
        if pid is None and pids is None and scope == 'all':
            pid = _query_person_hint(conn, args['query'])
            inferred = pid is not None
        query = _scoped_query(conn, args['query'], pid) if pid is not None else args['query']
        if pids is not None:
            for selected in retrieve._selected_people(person_ids=pids):
                query = _scoped_query(conn, query, selected)
        results = retrieve.hybrid_search(conn, query, args.get("top_k", 8), person_id=pid, scope=scope,
                                         person_ids=pids, retrieval_mode=args.get('retrieval_mode', 'hybrid'))
        for r in results:
            r["when"] = ms_to_local_str(r["ts"])
        person = repo.get_person(conn, pid) if pid else None
        return {"ok": True, "results": results, "scope": scope, 'scope_inferred': inferred,
                'scoped_person': {'id': pid, 'display_name': person['display_name']} if person else None,
                'scoped_person_ids': retrieve._selected_people(pid, pids)}

    if name == "list_open_promises":
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        rows = repo.open_promises(conn, pid)
        return {"ok": True, "promises": [
            {"id": r["id"], "who": r["who"], "content": r["content"],
             "person": r["display_name"] if "display_name" in r.keys() else args.get("person"),
             "due": ms_to_local_str(r["due_at"])[:10] if r["due_at"] else None} for r in rows]}

    if name == "complete_promise":
        row = conn.execute("SELECT * FROM promise WHERE id=?", (args["promise_id"],)).fetchone()
        if not row:
            raise ValueError("承诺不存在")
        repo.set_promise_status(conn, args["promise_id"], "done")
        conn.execute("UPDATE reminder SET status='cancelled' WHERE promise_id=? AND status='pending'",
                     (args["promise_id"],))
        conn.commit()
        return {"ok": True, "receipt": {"type": "promise_done", "id": args["promise_id"]}}

    if name == "ledger_summary":
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        totals = repo.ledger_totals(conn, pid)
        entries = repo.ledger_entries(conn, pid, include_pending=False)
        return {"ok": True,
                "totals": {"in_yuan": totals["in_cents"] / 100, "out_yuan": totals["out_cents"] / 100,
                           "count": totals["count"]},
                "entries": [{"direction": e["direction"], "amount_yuan": e["amount_cents"] / 100,
                             "note": e["note"], "person": e["display_name"],
                             "when": ms_to_local_str(e["occurred_at"])[:10]} for e in entries[:20]]}

    if name == "save_skill":
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        sid, created = repo.save_skill(conn, args["name"], args["body"], pid,
                                       args.get("trigger_hint"), args.get("style_notes"))
        conn.commit()
        return {"ok": True, "receipt": {"type": "skill", "id": sid, "name": args["name"],
                                        "created": created},
                "how_to_use": f"之后对我说 /{args['name']} 即可唤起"}

    if name == "run_skill":
        sk = repo.get_skill(conn, args["name"].lstrip("/"))
        if not sk:
            return {"ok": False, "error": f"没有找到技能「{args['name']}」"}
        return {"ok": True, "skill": {"name": sk["name"], "style_notes": sk["style_notes"],
                                       "body": sk["body"], "trigger_hint": sk["trigger_hint"]}}

    if name == "list_skills":
        return {"ok": True, "skills": [
            {"name": s["name"], "person": s["display_name"], "trigger_hint": s["trigger_hint"]}
            for s in repo.list_skills(conn)]}

    if name == "deep_research":
        from app.agent import research as research_mod
        result = research_mod.prepare_research(conn, args["topic"])
        return {"ok": True, "receipt": {"type": "research", "id": result["id"]},
                **result, "message": "已生成研究提案，请在研究面板确认后开始"}

    if name == "create_reminder":
        try:
            fire_ms = int(datetime.strptime(args["fire_date"], "%Y-%m-%d")
                          .replace(hour=9, tzinfo=tz()).timestamp() * 1000)
        except ValueError:
            return {"ok": False, "error": "fire_date 格式应为 YYYY-MM-DD"}
        pid = _resolve_person_id(conn, args.get("person"), args.get("person_id"))
        rid = repo.add_reminder(conn, fire_ms, args["text"], pid)
        conn.commit()
        return {"ok": True, "receipt": {"type": "reminder", "id": rid,
                                        "fire_at": ms_to_local_str(fire_ms)}}

    if name == "who_quiet":
        people = retrieve.quiet_people(conn, args.get("days", 14))
        return {"ok": True, "quiet": [
            {**p, "last_interaction": ms_to_local_str(p["last_interaction"])[:10]
             if p["last_interaction"] else "从未"} for p in people]}

    return {"ok": False, "error": f"未知工具 {name}"}


def arguments_of(tool_call: dict) -> dict:
    try:
        return json.loads(tool_call["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, KeyError):
        return {}
