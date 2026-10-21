"""Explicitly regenerate an unreviewed query without reclassifying or writing."""
from __future__ import annotations

from app.agent import conversation_scope, turn_intent
from app.agent.json_protocol import JSONProtocolError, parse_json_object
from app.agent.memory_sources import source_state
from app.jobs import tasks
from app.util import sha256_text


def _metadata(raw):
    try:
        return parse_json_object(raw, max_chars=1_000_000, allow_fence=False).value
    except JSONProtocolError:
        raise ValueError('原消息的处理信息无法核验，请重新明确提问') from None


def _write_result(data):
    if data.get('pending_ids'):
        return True
    stack = []
    for key in ('outcomes', 'receipts', 'saved', 'pending'):
        values = data.get(key, [])
        if not isinstance(values, list):
            return True
        stack.extend(values)
    while stack:
        value = stack.pop()
        if not isinstance(value, dict) or value.get('status') in {'ok', 'pending'}:
            return True
        nested = value.get('outcomes', [])
        if not isinstance(nested, list):
            return True
        stack.extend(nested)
    return False


def source_question(conn, conversation_id, answer_id):
    """Validate the saved answer, exact user decision, people and dependencies.

    Updated readable sources may be loaded again. Withdrawn, moved, deleted or
    archived sources cannot silently turn into a different question or person.
    """
    answer = conn.execute("SELECT rowid ordering,* FROM message WHERE id=? AND conversation_id=? AND role='assistant'",
                          (answer_id, conversation_id)).fetchone()
    if answer is None:
        raise LookupError('原回答不属于此对话')
    data = _metadata(answer['tool_calls'])
    review = data.get('answer_review')
    if not isinstance(review, dict) or review.get('status') != 'unavailable':
        raise ValueError('仅资料核对未完成的查询回答可重新生成')
    scope = data.get('conversation_scope')
    if not isinstance(scope, dict) or scope.get('version') != 1 or scope.get('conversation_id') != conversation_id or \
            not isinstance(scope.get('user_message_id'), str) or \
            scope.get('context_message_id') is not None and not isinstance(scope['context_message_id'], str):
        raise ValueError('原问题的人物范围无法核验，请重新明确提问')
    user = conn.execute("SELECT rowid ordering,* FROM message WHERE id=? AND conversation_id=? AND role='user'",
                        (scope.get('user_message_id'), conversation_id)).fetchone()
    if user is None or user['ordering'] >= answer['ordering'] or not isinstance(user['content'], str) or not user['content'].strip():
        raise ValueError('原问题与回答无法关联，请重新明确提问')
    original = _metadata(user['tool_calls'])
    intent = original.get('turn_intent')
    if not isinstance(intent, dict) or intent.get('version') != turn_intent.VERSION or \
            intent.get('intent') != 'query' or not isinstance(intent.get('status'), str) or \
            intent['status'] not in {'classified', 'ok'} or \
            intent.get('allow_writes') is not False or intent.get('record_spans') != [] or \
            intent.get('extraction_text') != '' or intent.get('source_sha256') != sha256_text(user['content']):
        raise ValueError('仅已确认只读的查询可重新生成；记事需重新明确发送')
    checked = turn_intent.validate_intent({key: intent.get(key) for key in ('intent', 'record_spans', 'reason')},
                                         user['content'])
    if checked['status'] != 'classified' or checked['intent'] != 'query' or data.get('turn_intent') != intent:
        raise ValueError('原问题的查询决定无法核验，请重新明确提问')
    saved_scope = original.get('conversation_scope')
    binding_keys = ('version', 'conversation_id', 'user_message_id', 'mode', 'person_ids', 'blocked',
                    'origin_message_id', 'context_message_id')
    if not isinstance(saved_scope, dict) or any(saved_scope.get(key) != scope.get(key) for key in binding_keys) or \
            not isinstance(scope.get('mode'), str) or scope['mode'] not in {'none', 'global', 'single', 'multiple'} or \
            scope.get('blocked') is not False or not isinstance(scope.get('person_ids'), list) or \
            any(not isinstance(pid, str) for pid in scope['person_ids']) or \
            len(scope['person_ids']) != len(set(scope['person_ids'])) or \
            scope['mode'] in {'none', 'global'} and scope['person_ids'] or \
            scope['mode'] == 'single' and len(scope['person_ids']) != 1 or \
            scope['mode'] == 'multiple' and len(scope['person_ids']) < 2:
        raise ValueError('原问题的人物范围无法核验，请重新明确提问')
    if conversation_scope._valid_previous(conn, scope).get('blocked'):
        raise ValueError('原问题的人物已不可用，请重新明确提问')
    if _write_result(data) or conn.execute('SELECT 1 FROM applied_patch WHERE source_msg_id=? LIMIT 1',
                                           (user['id'],)).fetchone():
        raise ValueError('原消息已有保存或待确认结果，请查看回执后重新明确提问')
    # A prior retry may have persisted a write receipt despite corrupt query
    # metadata. A new generation must not replay any such operation.
    for row in conn.execute("SELECT tool_calls FROM message WHERE conversation_id=? AND role='assistant'",
                            (conversation_id,)):
        try:
            other = _metadata(row['tool_calls'])
        except ValueError:
            continue
        other_scope = other.get('conversation_scope')
        if isinstance(other_scope, dict) and other_scope.get('user_message_id') == user['id'] and _write_result(other):
            raise ValueError('原消息已有保存或待确认结果，请查看回执后重新明确提问')
    sources, dependencies = data.get('sources'), data.get('source_dependencies')
    if not isinstance(sources, list) or not isinstance(dependencies, list) or \
            any(source not in dependencies for source in sources):
        raise ValueError('原回答的来源无法核验，请重新明确提问')
    for source in dependencies:
        if not isinstance(source, dict) or source.get('type') not in {'memory', 'field', 'interaction', 'profile_context'} or \
                not isinstance(source.get('entity_id'), str) or not isinstance(source.get('text'), str) or \
                source_state(conn, source)['status'] not in {'current', 'updated'}:
            raise ValueError('原回答的部分来源已不可用，请重新明确提问并读取资料')
    context_id = user['id']
    if original.get('reanalysis') is not None:
        from app.jobs.chat_reanalysis import context_scope
        context_row, current_scope = context_scope(conn, conversation_id, user['id'])
        context_id = context_row['id']
        if scope.get('context_message_id') != context_id:
            raise ValueError('原问题的重新分析上下文无法核验，请重新明确提问')
    elif scope.get('context_message_id') not in {None, user['id']}:
        raise ValueError('原问题的上下文无法核验，请重新明确提问')
    else:
        current_scope = conversation_scope.resolve_user_message(conn, conversation_id, user['id'], user['content'])
    if any(current_scope.get(key) != scope.get(key) for key in ('mode', 'person_ids', 'blocked')):
        raise ValueError('原问题的人物指向已变化，请重新明确提问')
    return user, context_id


def submit(conn, conversation_id, answer_id):
    key = f'chat-regenerate:{conversation_id}:{answer_id}'
    existing = conn.execute('SELECT id FROM task WHERE idempotency_key=?', (key,)).fetchone()
    if existing:
        return tasks.read(conn, existing[0])
    user, _context_id = source_question(conn, conversation_id, answer_id)
    return tasks.submit(conn, 'chat', '重新回答：' + user['content'][:24],
                        {'conversation_id': conversation_id, 'text': user['content'],
                         'user_message_id': user['id'], 'regeneration_of': answer_id,
                         'source_sha256': sha256_text(user['content'])}, key)
