"""Resolve an explicit follow-up to one previous, persisted memory quotation.

This is reference resolution, not a classifier for arbitrary questions. The
binding lives on the current user row; latest facts are re-read on every retry.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import re

from app.agent import conversation_scope
from app.agent.memory_sources import memory_source, source_state
from app.config import tz
from app.util import sha256_text

VERSION = 1
_REFERENCE = re.compile(r'这条记忆|那条记忆|这条记录|刚才那条记录|刚才那条记忆|上面那条记录|这个偏好|刚才那个偏好')
_PREVIOUS_ANSWER = re.compile(r'(?:上|前)(?:一)?(?:条|次)(?:回答|回复)|刚才(?:的)?(?:回答|回复)')
_SOURCE_ID = re.compile(r'(?<![A-Za-z0-9_])S[1-9]\d*(?![A-Za-z0-9_])', re.I)
_LOCAL_SOURCE_ID = re.compile(r'S[1-9]\d*\Z')
_DATE_COMPARISON = re.compile(
    r'^(?:那|那么|请问)?\s*(?:\d{4}年)?\d{1,2}月\d{1,2}[日号]是'
    r'(?:这条记忆|这条记录|刚才那条记录)(?:所标|标注)(?:的)?日期[，,]?'
    r'还是我(?:实际|人工|手动)?(?:修改|编辑)(?:它|这条记忆|这条记录)的(?:时间|日期)[？?]?$')
_EDIT_DATE = re.compile(
    r'^(?:请问|那|那么)?(?:这条记忆|这条记录|刚才那条记录|这个偏好)(?:的)?'
    r'(?:(?:实际|最近|最后一次)?(?:人工|手动)?(?:修改|编辑|更正)(?:时间|日期)'
    r'(?:是|在|是在)?(?:哪天|什么时候|多少|几号)?|(?:是|在|是在)?(?:哪天|什么时候|几号)'
    r'(?:被)?(?:我)?(?:实际|人工|手动)?(?:修改|编辑|更正)(?:的)?)[？?]?$')


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


def _result(status, reason, read_scope, binding=None, source=None):
    messages = {
        'ambiguous_memory': '上一条回答涉及多条记忆。请指出具体原话，或说明上一条回答中的资料编号。',
        'unanchored_source_id': '资料编号只在各自回答中有效。请说明你指的是上一条回答中的哪条记忆。',
        'unknown_source_id': '上一条回答没有这条可核查的记忆引用，请重新指出具体记忆。',
        'unavailable_memory': '这条记忆或其所属人物目前不可读取，请重新指定可用的记忆。',
        'moved_memory': '这条记忆的归属已变化，请重新确认人物和记忆，不能沿用之前的归属。',
        'invalid_binding': '之前的记忆引用无法核验，请重新指出具体记忆。',
        'missing_memory': '上一条回答没有可以唯一定位的记忆来源。请指出具体记忆或人物。',
    }
    return {'status': status, 'reason': reason, 'read_scope': read_scope, 'binding': deepcopy(binding),
            'source': deepcopy(source),
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


def _live(conn, binding, read_scope):
    reference = binding.get('reference_snapshot')
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


def for_user_message(conn, conversation_id, user_message_id, text, read_scope, *, persist=True):
    """Return a focus plus a suggested read scope, preserving a retry anchor.

    Root harness registers ``source`` before using it, honors ``clarification``
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
    if not _REFERENCE.search(text) and not (_PREVIOUS_ANSWER.search(text) and ids):
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
        'anchor_assistant_message_id': None, 'reference_snapshot': None,
    }
    anchor, metadata = _anchor(conn, conversation_id, row['ordering'])
    if valid_saved:
        reference = binding.get('reference_snapshot')
        if binding.get('status') == 'selected' and (not anchor \
                or anchor['id'] != binding.get('anchor_assistant_message_id') \
                or not _memory_snapshot(reference) or not any(
                value == reference for value in [*_memory_sources(metadata, 'sources'),
                                                 *_memory_sources(metadata, 'source_dependencies')])):
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
        if ids:
            # Only cited labels were shown to the user in the preceding answer.
            candidates = [source for source in cited if source['id'] in ids]
            if len(ids) != 1 or not candidates:
                binding['reason'] = 'unknown_source_id' if len(ids) == 1 else 'ambiguous_memory'
        else:
            candidates = cited or dependencies
        if not ids or len(ids) == 1 and candidates:
            identities = {(source['entity_id'], (source.get('person') or {}).get('id'))
                          for source in candidates}
            if len(identities) == 1:
                binding.update(status='selected', reason='unique_cited_memory' if cited else 'unique_memory_dependency',
                               reference_snapshot=deepcopy(min(candidates, key=lambda source: int(source['id'][1:]))))
            elif identities:
                binding['reason'] = 'ambiguous_memory'

    # A named new subject must never be replaced by the owner of an old quote.
    reference = binding.get('reference_snapshot')
    old_pid = (reference.get('person') or {}).get('id') if isinstance(reference, dict) else None
    if read_scope.get('reason') in {'explicit_name', 'explicit_id'} and (
            binding['status'] != 'selected' or read_scope.get('person_ids') != ([old_pid] if old_pid else [])):
        return _result('not_applicable', 'explicit_scope_precedence', read_scope)

    if binding['status'] == 'selected':
        source, reason = _live(conn, binding, read_scope)
        if source is not None:
            scope = _scope(conn, user_message_id, conversation_id,
                           (source.get('person') or {}).get('id'))
            if persist:
                _persist(conn, row, binding, scope)
            return _result('selected', binding['reason'], scope, binding, source)
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
    """Register only after its compact quotation fits the provided budget."""
    from app.agent.source_registry import compact_source
    if focus.get('status') != 'selected' or not focus.get('source'):
        return {'status': 'not_selected', 'source_id': None, 'context': ''}
    if type(budget_chars) is not int or budget_chars < 0:
        raise ValueError('记忆上下文预算必须是非负整数')
    source = focus['source']
    line = json.dumps(compact_source(registry.preview(source)), ensure_ascii=False)
    if len(line) > budget_chars:
        return {'status': 'budget_exceeded', 'source_id': None, 'context': ''}
    registered = registry.register(source)
    return {'status': 'registered', 'source_id': registered['id'], 'context': line}


def is_date_question(text):
    """Recognize only complete, read-only questions about a quoted date role."""
    return isinstance(text, str) and bool(
        _DATE_COMPARISON.fullmatch(text.strip()) or _EDIT_DATE.fullmatch(text.strip()))


def date_answer(focus, source_id, text):
    """Template only two explicit date-role question forms, never arbitrary intent."""
    if focus.get('status') != 'selected' or not isinstance(source_id, str) \
            or not _LOCAL_SOURCE_ID.fullmatch(source_id) \
            or not is_date_question(text):
        return None
    source = focus['source']
    context = source.get('record_context') or {}

    def day(stamp):
        if type(stamp) is not int:
            return '未知'
        try:
            return datetime.fromtimestamp(stamp / 1000, tz()).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return '未知'

    labeled, recorded, edited = day(source.get('ts')), day(context.get('recorded_at')), day(context.get('last_user_edit_at'))
    lines = [f'这条记忆所标的日期是 {labeled}，首次录入日期是 {recorded}。[[{source_id}]]']
    lines.append(f'实际人工编辑日期是 {edited}。[[{source_id}]]' if edited != '未知' else
                 f'没有可核验的人工编辑时间，不能从记忆所标日期推断修改日期。[[{source_id}]]')
    lifecycle = context.get('last_lifecycle_change')
    if isinstance(lifecycle, dict) and lifecycle.get('action') in {'restore', 'withdraw'}:
        action = '恢复' if lifecycle['action'] == 'restore' else '撤销'
        lines.append(f'最近一次{action}操作日期是 {day(lifecycle.get("created_at"))}，与人工编辑时间分别记录。[[{source_id}]]')
    return '\n\n'.join(lines)
