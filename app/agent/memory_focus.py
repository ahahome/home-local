"""Resolve a follow-up to a persisted quotation or same-person quotation set.

This is reference resolution, not a classifier for arbitrary questions. The
binding lives on the current user row; latest facts are re-read on every retry.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import re

from app.agent import conversation_scope, turn_intent
from app.agent.memory_sources import memory_source, source_state
from app.config import tz
from app.util import sha256_text

VERSION = 1
_REFERENCE = re.compile(r'这条记忆|那条记忆|这条记录|刚才那条记录|刚才那条记忆|上面那条记录|这个偏好|刚才那个偏好')
_SHORT_REFERENCE = re.compile(r'^(?:请问|那|那么)?这条(?=是(?:什么|哪|几)|后来(?:是|在)?(?:什么|哪|几|改|修改|编辑|撤销|恢复)|'
                              r'(?:的)?(?:日期|时间|来源|原文))')
_SET_REFERENCE = re.compile(
    r'(?:这|那)(?:些|几)条?(?:记忆|记录|偏好)|'
    r'(?:这|那)(?:两|2)条(?:记忆|记录|偏好)?|'
    r'(?:刚才|上面)(?:提到|列出|查到)?的?(?:那)?(?:两|2)条(?:记忆|记录|偏好)?')
_PAIR_REFERENCE = re.compile(r'(?:这|那|刚才|上面)(?:提到|列出|查到)?的?(?:那)?(?:两|2)条')
_PREVIOUS_ANSWER = re.compile(r'(?:上|前)(?:一)?(?:条|次)(?:回答|回复)|刚才(?:的)?(?:回答|回复)')
_SOURCE_ID = re.compile(r'(?<![A-Za-z0-9_])S[1-9]\d*(?![A-Za-z0-9_])', re.I)
_LOCAL_SOURCE_ID = re.compile(r'S[1-9]\d*\Z')
_DATE_REFERENCE = r'(?:这条记忆|这条记录|这个偏好|刚才那条记录|这些记录|这些记忆|这两条记录|这两条记忆|刚才两条|刚才那两条|这条|这两条)'
_DATE_COMPARISON = re.compile(
    r'^(?:那|那么|请问)?\s*(?:\d{4}年)?\d{1,2}月\d{1,2}[日号]是'
    r'(?:这条记忆|这条记录|刚才那条记录)(?:所标|标注)(?:的)?日期[，,]?'
    r'还是我(?:实际|人工|手动)?(?:修改|编辑)(?:它|这条记忆|这条记录)的(?:时间|日期)[？?]?$')
_EDIT_DATE = re.compile(
    r'^(?:请问|那|那么)?' + _DATE_REFERENCE + r'(?:各自|分别)?(?:的)?'
    r'(?:(?:实际|最近|最后一次)?(?:人工|手动)?(?:修改|编辑|更正)(?:时间|日期)'
    r'(?:是|在|是在)?(?:哪天|什么时候|多少|几号)?|(?:是|在|是在)?(?:哪天|什么时候|几号)'
    r'(?:被)?(?:我)?(?:实际|人工|手动)?(?:修改|编辑|更正)(?:的)?)[？?]?$')
_RECORD_DATE = re.compile(
    r'^(?:请问|那|那么)?' + _DATE_REFERENCE + r'(?:各自|分别)?(?:的)?'
    r'(?:(?:最初|首次)?(?:录入|保存)(?:时间|日期)(?:是|在)?(?:哪天|什么时候|几号)?|'
    r'(?:最初|首次)?(?:是|在|是在)?(?:哪天|什么时候|几号)(?:被)?(?:我)?(?:录入|保存)(?:的)?|'
    r'(?:所标日期|记录时间|录入时间|修改时间|编辑时间)(?:和|与|、)(?:所标日期|记录时间|录入时间|修改时间|编辑时间)(?:分别)?(?:是|在)?(?:哪天|什么时候|几号)?)[？?]?$')
_TIMELINE_DATE = re.compile(
    r'^(?:请问|那|那么)?' + _DATE_REFERENCE + r'(?:各自|分别)?(?:是)?(?:何时|什么时候|哪天)'
    r'(?:被)?(?:我)?(?:记下|录入|记录|保存)(?:的)?'
    r'(?:[，,](?:后来|之后)(?:又)?(?:是)?(?:哪天|什么时候)(?:被)?(?:我)?(?:改过|修改|编辑)(?:的)?)?[？?]?$')
_LIFECYCLE_DATE = re.compile(
    r'^(?:请问|那|那么)?' + _DATE_REFERENCE + r'(?:后来|之后)?(?:被)?(?:我)?(?:改过|修改过|编辑过)吗'
    r'[，,](?:撤销|恢复)(?:又)?(?:是|在|是在)?(?:哪天|什么时候|几号)[？?]?$')

# Deliberately a small whole-question grammar, not a keyword router. Unknown
# clauses (including drafting, recommendations or newly asserted facts) keep
# their normal model/intent path. It applies only after the set is bound.
_COMPARE_SUBJECT = r'(?:' + _SET_REFERENCE.pattern + r'|(?:上一条|刚才的?)回答(?:中|里)?的?\s*S[1-9]\d*(?:\s*(?:和|与|、)\s*S[1-9]\d*)+)'
_COMPARE_ACTION = (
    r'(?:(?:现在)?(?:再)?(?:比较|对照|核对)(?:一下|下)?|'
    r'(?:有|存在)(?:什么|哪些|何种)(?:区别|差异|不同|共同点)|'
    r'(?:原文|文字|内容)(?:有|存在)(?:什么|哪些)(?:区别|差异|不同)|'
    r'(?:说的|讲的)?是(?:同一种|同一个|相同的?|一样的?)(?:联系安排|安排|偏好|内容|意思|要求|事情|事实|习惯|计划)吗|'
    r'(?:分别)?(?:怎么说|怎么写|怎么讲|说了什么|讲了什么)|'
    r'(?:能|可以)(?:一起)?(?:比较|对照)(?:吗)?)')
_COMPARE_FIRST = re.compile(r'(?:请|麻烦|那|那么)?' + _COMPARE_SUBJECT + r'(?:现在|再|现在再)?' + _COMPARE_ACTION + r'[吗呢]?$', re.I)
_COMPARE_REST = re.compile(
    r'(?:哪些地方(?:对得上|相同|不同|一致|还不能确定|不能确定)|'
    r'(?:原文|文字|内容)(?:(?:和|与)(?:能支持的)?结论)?(?:有(?:没有|无)|是否有|有何|有什么)(?:变化|差异|不同)|'
    r'(?:原文|文字|内容)(?:和|与)(?:能支持的)?结论(?:有(?:没有|无)|是否有|有何|有什么)(?:变化|差异|不同))$')


def _metadata(raw):
    try:
        value = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _memory_snapshot(value):
    if not isinstance(value, dict) or value.get('type') != 'memory' \
            or not isinstance(value.get('entity_id'), str) or not value['entity_id'] \
            or not isinstance(value.get('text'), str) \
            or not isinstance(value.get('id'), str) or not _LOCAL_SOURCE_ID.fullmatch(value['id']):
        return False
    person = value.get('person')
    return person is None or isinstance(person, dict) and isinstance(person.get('id'), str) and bool(person['id'])


def _memory_sources(metadata, key):
    values = metadata.get(key, [])
    return [value for value in values if _memory_snapshot(value)] if isinstance(values, list) else []


def _anchor(conn, conversation_id, before_rowid):
    rows = conn.execute("SELECT rowid AS ordering,* FROM message WHERE conversation_id=? "
                        "AND role='assistant' AND rowid<? ORDER BY rowid DESC",
                        (conversation_id, before_rowid))
    for row in rows:
        try:
            raw = json.loads(row['tool_calls']) if row['tool_calls'] else {}
        except (ValueError, TypeError):
            raw = None
        # Intermediate tool-call assistants are not the previous final answer.
        if isinstance(raw, list):
            continue
        return row, raw if isinstance(raw, dict) else {}
    return None, {}


def _scope(conn, current_user_id, conversation_id, person_id=None, blocked=False):
    mode = 'clarify' if blocked else 'single' if person_id else 'none'
    scope = conversation_scope._scope(conn, mode, [person_id] if person_id else [],
                                      'memory_reference' if not blocked else 'unresolved_memory_reference',
                                      origin=current_user_id)
    scope.update(conversation_id=conversation_id, user_message_id=current_user_id)
    return scope


def _result(status, reason, read_scope, binding=None, sources=()):
    messages = {
        'ambiguous_memory': '上一条回答涉及多条记忆。请指出具体原话，或说明上一条回答中的资料编号。',
        'unanchored_source_id': '资料编号只在各自回答中有效。请说明你指的是上一条回答中的哪条记忆。',
        'unknown_source_id': '上一条回答没有这条可核查的记忆引用，请重新指出具体记忆。',
        'unavailable_memory': '这条记忆或其所属人物目前不可读取，请重新指定可用的记忆。',
        'moved_memory': '这条记忆的归属已变化，请重新确认人物和记忆，不能沿用之前的归属。',
        'invalid_binding': '之前的记忆引用无法核验，请重新指出具体记忆。',
        'missing_memory': '上一条回答没有可以唯一定位的记忆来源。请指出具体记忆或人物。',
        'ambiguous_memory_set': '无法把这组记录与上一条回答的记忆引用完整对应，请明确具体原话或上一条回答中的资料编号。',
        'ambiguous_memory_owner': '上一条回答涉及不同人物的记忆，请先明确要核对哪位人物的哪些记录。',
    }
    sources = list(sources)
    return {'status': status, 'reason': reason, 'read_scope': read_scope, 'binding': deepcopy(binding),
            'source': deepcopy(sources[0]) if len(sources) == 1 else None, 'sources': deepcopy(sources),
            'clarification': messages.get(reason, messages['invalid_binding']) if status == 'clarify' else None}


def _persist(conn, row, binding, scope):
    # Merge with the latest metadata so intent/scope annotations are preserved.
    current = conn.execute('SELECT tool_calls FROM message WHERE id=?', (row['id'],)).fetchone()
    metadata = _metadata(current['tool_calls'])
    metadata['memory_focus'] = binding
    metadata['conversation_scope'] = scope
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?',
                 (json.dumps(metadata, ensure_ascii=False), row['id']))
    conn.commit()


def _live(conn, reference, read_scope):
    state = source_state(conn, reference)
    if state['status'] == 'moved':
        return None, 'moved_memory'
    if state['status'] in {'unavailable', 'withdrawn'}:
        return None, 'unavailable_memory'
    current = memory_source(conn, reference['entity_id'])
    if current is None:
        return None, 'unavailable_memory'
    old_pid = (reference.get('person') or {}).get('id')
    new_pid = (current.get('person') or {}).get('id')
    if old_pid != new_pid:
        return None, 'moved_memory'
    if read_scope.get('mode') in {'single', 'multiple'} and new_pid not in read_scope.get('person_ids', []):
        return None, 'moved_memory'
    return current, None


def _references(binding):
    """Old single-source bindings stay readable, without reselecting an anchor."""
    if 'reference_snapshots' not in binding:
        reference = binding.get('reference_snapshot')
        return [reference] if _memory_snapshot(reference) else []
    references = binding['reference_snapshots']
    if not isinstance(references, list) or not references or any(not _memory_snapshot(s) for s in references):
        return []
    # Reject corruption of either representation; the singular field is legacy.
    if binding.get('reference_snapshot') != (references[0] if len(references) == 1 else None):
        return []
    return references


def _unique(sources):
    values = {}
    for source in sorted(sources, key=lambda s: int(s['id'][1:])):
        key = (source['entity_id'], (source.get('person') or {}).get('id'))
        values.setdefault(key, source)
    return list(values.values())


def for_user_message(conn, conversation_id, user_message_id, text, read_scope, *, persist=True):
    """Return a focus plus a suggested read scope, preserving a retry anchor.

    Root harness registers all ``sources`` before using them, honors ``clarification``
    when status is clarify, and keeps SourceRegistry.save_answer's final guard.
    An explicit new person/global choice takes precedence over a memory anaphor.
    """
    row = conn.execute("SELECT rowid AS ordering,* FROM message WHERE id=? AND conversation_id=? AND role='user'",
                       (user_message_id, conversation_id)).fetchone()
    if row is None or row['content'] != text:
        raise ValueError('记忆追问消息不属于此对话或原文已变化')
    if not isinstance(read_scope, dict):
        raise ValueError('记忆追问需要已解析的人物范围')
    ids = list(dict.fromkeys(match.upper() for match in _SOURCE_ID.findall(text)))
    set_reference = bool(_SET_REFERENCE.search(text)) or len(ids) > 1
    singular_reference = bool(_REFERENCE.search(text))
    new_reference = bool(_SET_REFERENCE.search(text) or _SHORT_REFERENCE.search(text))
    # A new shorthand is optional context resolution, not permission to suppress
    # an explicit note. The normal classifier still decides whether it can save.
    if not singular_reference and turn_intent._EXPLICIT_RULE.search(text.lstrip()):
        new_reference = False
    if not singular_reference and not new_reference and not (_PREVIOUS_ANSWER.search(text) and ids):
        return _result('not_applicable', 'no_explicit_memory_reference', read_scope)
    if read_scope.get('blocked') or read_scope.get('mode') == 'global':
        return _result('not_applicable', 'explicit_scope_precedence', read_scope)

    saved = _metadata(row['tool_calls']).get('memory_focus')
    valid_saved = isinstance(saved, dict) and saved.get('version') == VERSION \
        and saved.get('conversation_id') == conversation_id and saved.get('user_message_id') == user_message_id \
        and saved.get('source_sha256') == sha256_text(text) \
        and saved.get('status') in {'selected', 'clarify'} and isinstance(saved.get('reason'), str)
    binding = deepcopy(saved) if valid_saved else {
        'version': VERSION, 'conversation_id': conversation_id, 'user_message_id': user_message_id,
        'source_sha256': sha256_text(text), 'status': 'clarify',
        'reason': 'invalid_binding' if saved is not None else 'missing_memory',
        'anchor_assistant_message_id': None, 'reference_snapshot': None, 'reference_snapshots': [],
        'selection_mode': 'set' if set_reference else 'single',
    }
    anchor, metadata = _anchor(conn, conversation_id, row['ordering'])
    if valid_saved:
        references = _references(binding)
        cited = _memory_sources(metadata, 'sources')
        expected = _unique([source for source in cited if source['id'] in ids] if ids else
                           cited if set_reference else cited or _memory_sources(metadata, 'source_dependencies'))
        complete_set = (len(expected) >= 2 and len({(s.get('person') or {}).get('id') for s in expected}) == 1
                        and (not _PAIR_REFERENCE.search(text) or len(expected) == 2)
                        and (bool(ids) or len(metadata.get('sources', [])) == len(cited))) if set_reference else len(expected) == 1
        if binding.get('status') == 'selected' and (not anchor \
                or anchor['id'] != binding.get('anchor_assistant_message_id') \
                or not references or references != expected or not complete_set \
                or ids and set(ids) != {s['id'] for s in cited if s['id'] in ids} \
                or binding.get('selection_mode', 'single') != ('set' if set_reference else 'single')):
            binding.update(status='clarify', reason='invalid_binding')
    elif saved is not None:
        # A corrupted retry binding must not silently select a different quote.
        pass
    elif ids and not _PREVIOUS_ANSWER.search(text):
        binding['reason'] = 'unanchored_source_id'
    elif anchor is not None:
        binding['anchor_assistant_message_id'] = anchor['id']
        cited = _memory_sources(metadata, 'sources')
        dependencies = _memory_sources(metadata, 'source_dependencies')
        candidates = []
        if ids:
            # Only cited labels were shown to the user in the preceding answer.
            candidates = [source for source in cited if source['id'] in ids]
            if set(ids) != {source['id'] for source in candidates}:
                binding['reason'] = 'unknown_source_id'
                candidates = []
        elif set_reference:
            # A plural reference cannot silently pick memory rows out of a mixed
            # field/interaction answer or include uncited, unseen dependencies.
            visible = metadata.get('sources', [])
            if not isinstance(visible, list) or len(visible) != len(cited):
                binding['reason'] = 'ambiguous_memory_set'
            else:
                candidates = cited
        else:
            candidates = cited or dependencies
        if candidates:
            selected = _unique(candidates)
            owners = {(source.get('person') or {}).get('id') for source in selected}
            if not set_reference and len(selected) != 1:
                binding['reason'] = 'ambiguous_memory'
            elif set_reference and len(owners) != 1:
                binding['reason'] = 'ambiguous_memory_owner'
            elif set_reference and (len(selected) < 2 or _PAIR_REFERENCE.search(text) and len(selected) != 2):
                binding['reason'] = 'ambiguous_memory_set'
            else:
                binding.update(status='selected', reason='unique_cited_memory' if cited else 'unique_memory_dependency',
                               reference_snapshot=deepcopy(selected[0]) if len(selected) == 1 else None,
                               reference_snapshots=deepcopy(selected))
                if set_reference:
                    binding['reason'] = 'cited_memory_set'
        elif set_reference and binding['reason'] == 'missing_memory':
            binding['reason'] = 'ambiguous_memory_set'

    # A named new subject must never be replaced by the owner of an old quote.
    references = _references(binding)
    owners = {(reference.get('person') or {}).get('id') for reference in references}
    old_pid = next(iter(owners)) if len(owners) == 1 else None
    if read_scope.get('reason') in {'explicit_name', 'explicit_id'} and (
            binding['status'] != 'selected' or read_scope.get('person_ids') != ([old_pid] if old_pid else [])):
        return _result('not_applicable', 'explicit_scope_precedence', read_scope)

    if binding['status'] == 'selected':
        sources, reason = [], None
        for reference in references:
            source, reason = _live(conn, reference, read_scope)
            if source is None:
                break
            sources.append(source)
        if reason is None:
            scope = _scope(conn, user_message_id, conversation_id, old_pid)
            if persist:
                _persist(conn, row, binding, scope)
            return _result('selected', binding['reason'], scope, binding, sources)
        # Keep the entity binding stable while re-checking availability on retry.
        scope = _scope(conn, user_message_id, conversation_id, blocked=True)
        if persist:
            _persist(conn, row, binding, scope)
        return _result('clarify', reason, scope, binding)
    scope = _scope(conn, user_message_id, conversation_id, blocked=True)
    if persist:
        _persist(conn, row, binding, scope)
    return _result('clarify', binding['reason'], scope, binding)


def register_source(focus, registry, budget_chars):
    """Register a whole set only after every compact quotation fits the budget."""
    from app.agent.source_registry import compact_source
    sources = focus.get('sources') or ([focus['source']] if focus.get('source') else [])
    if focus.get('status') != 'selected' or not sources:
        return {'status': 'not_selected', 'source_id': None, 'source_ids': [], 'context': ''}
    if type(budget_chars) is not int or budget_chars < 0:
        raise ValueError('记忆上下文预算必须是非负整数')
    previews = registry.preview_many(sources)
    context = '\n\n'.join(json.dumps(compact_source(source), ensure_ascii=False) for source in previews)
    if len(context) > budget_chars:
        return {'status': 'budget_exceeded', 'source_id': None, 'source_ids': [], 'context': ''}
    registered = [registry.register(source)['id'] for source in sources]
    return {'status': 'registered', 'source_id': registered[0] if len(registered) == 1 else None,
            'source_ids': registered, 'context': context}


def is_date_question(text):
    """Recognize only complete, read-only questions about a quoted date role."""
    return isinstance(text, str) and bool(
        _DATE_COMPARISON.fullmatch(text.strip()) or _EDIT_DATE.fullmatch(text.strip()) or
        _RECORD_DATE.fullmatch(text.strip()) or _TIMELINE_DATE.fullmatch(text.strip()) or
        _LIFECYCLE_DATE.fullmatch(text.strip()))


def date_answer(focus, source_id, text):
    """Project explicit date roles for a bound quotation/set, never infer intent."""
    source_ids = [source_id] if isinstance(source_id, str) else source_id
    sources = focus.get('sources') or ([focus['source']] if focus.get('source') else [])
    if focus.get('status') != 'selected' or not isinstance(source_ids, list) or not sources \
            or len(source_ids) != len(sources) \
            or any(not isinstance(sid, str) or not _LOCAL_SOURCE_ID.fullmatch(sid) for sid in source_ids) \
            or not is_date_question(text):
        return None

    def day(stamp):
        if type(stamp) is not int:
            return '未知'
        try:
            return datetime.fromtimestamp(stamp / 1000, tz()).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return '未知'

    blocks = []
    for index, (source, sid) in enumerate(zip(sources, source_ids), 1):
        context = source.get('record_context') or {}
        labeled, recorded, edited = day(source.get('ts')), day(context.get('recorded_at')), day(context.get('last_user_edit_at'))
        lines = [f'这条记忆所标的日期是 {labeled}，首次录入日期是 {recorded}。[[{sid}]]']
        lines.append(f'实际人工编辑日期是 {edited}。[[{sid}]]' if edited != '未知' else
                     f'没有可核验的人工编辑时间，不能从记忆所标日期推断修改日期。[[{sid}]]')
        lifecycle = context.get('last_lifecycle_change')
        if isinstance(lifecycle, dict) and lifecycle.get('action') in {'restore', 'withdraw'}:
            action = '恢复' if lifecycle['action'] == 'restore' else '撤销'
            lines.append(f'最近一次{action}操作日期是 {day(lifecycle.get("created_at"))}，与人工编辑时间分别记录。[[{sid}]]')
        if len(sources) > 1:
            lines.insert(0, f'第 {index} 条记忆（本轮资料 {sid}）')
        blocks.append('\n\n'.join(lines))
    return '\n\n'.join(blocks)


def is_comparison_question(text):
    """Recognize complete requests for literal comparison, not arbitrary tasks."""
    if not isinstance(text, str):
        return False
    clauses = [part.strip() for part in re.split(r'[，,。.!！?？；;\n\r]+', text.strip()) if part.strip()]
    return bool(clauses and _COMPARE_FIRST.fullmatch(clauses[0])
                and all(_COMPARE_REST.fullmatch(clause) for clause in clauses[1:]))


def comparison_answer(focus, source_ids, text, intent):
    """Compare stored strings only; never infer scopes, motives or replacement.

    The caller must register this complete current set first. Its immutable
    binding is the evidence for comparison with the previous answer; old text
    is not passed off as a quotation of today's source.
    """
    from app.agent.source_registry import _prose_references

    if not isinstance(intent, dict) or intent.get('intent') != 'query' \
            or intent.get('allow_writes') is not False or intent.get('record_spans') \
            or not is_comparison_question(text) or focus.get('status') != 'selected':
        return None
    sources = focus.get('sources')
    binding = focus.get('binding')
    if not isinstance(sources, list) or len(sources) < 2 or not isinstance(binding, dict) \
            or not isinstance(source_ids, list) or len(source_ids) != len(sources) \
            or any(not isinstance(sid, str) or not _LOCAL_SOURCE_ID.fullmatch(sid) for sid in source_ids) \
            or len(set(source_ids)) != len(source_ids):
        return None
    references = _references(binding)
    if any(not _memory_snapshot({**source, 'id': sid}) for source, sid in zip(sources, source_ids)
           if isinstance(source, dict)) or any(not isinstance(source, dict) for source in sources):
        return None

    def identity(source):
        return source['entity_id'], (source.get('person') or {}).get('id')

    old = {identity(source): source for source in references}
    current = {identity(source): source for source in sources}
    if len(old) != len(sources) or len(current) != len(sources) or old.keys() != current.keys() \
            or len({identity(source)[1] for source in sources}) != 1:
        return None
    blocks = ['下面逐条对照当前保存的文字。与上次引用相比的变化仅指文字变化，不能据此认定当事人改变了偏好。']
    for index, (source, sid) in enumerate(zip(sources, source_ids), 1):
        original = source['text']
        quote = original[:1600]
        rendered = '当前保存文字：' + quote + f'\n\n[[{sid}]]'
        # Data can contain Markdown or citation-like strings. Preserve it in
        # the attached snapshot, without allowing it to create/hide buttons.
        if re.search(r'\[\[.*?\]\]|[`<>]', quote, re.DOTALL) or _prose_references(quote) \
                or _prose_references(rendered) != [sid]:
            rendered = f'这条文字含有引用标记或格式，未直接展开；请打开资料查看完整记录。[[{sid}]]'
        elif len(original) > 1600:
            rendered += '\n\n（以上仅显示记录前1600字符；完整文字见资料。）'
        changed = original != old[identity(source)]['text']
        delta = f'与上次引用相比，保存文字{"已变化" if changed else "未变化"}。[[{sid}]]'
        dates = date_answer({'status': 'selected', 'sources': [source]}, sid, '这条是什么时候记下的？')
        blocks.append(f'第 {index} 条记忆（本轮资料 {sid}）\n\n' + rendered + '\n\n' + delta + '\n\n' + dates)
    blocks.append('原文中的限定条件应分别保留。文字差异和日期先后本身不能证明这些安排是否同义、如何同时适用或哪条取代哪条；原文未明确说明的适用范围与关联仍需确认。')
    return '\n\n'.join(blocks)
