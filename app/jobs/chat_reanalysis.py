"""An explicit new analysis preserves the original decision and is idempotent."""
from __future__ import annotations

import json
from copy import deepcopy

from app.jobs import tasks
from app.util import sha256_text


def source_message(conn, conversation_id, message_id):
    row = conn.execute("SELECT * FROM message WHERE id=? AND conversation_id=? AND role='user'",
                       (message_id, conversation_id)).fetchone()
    if row is None:
        raise LookupError('原消息不属于此对话')
    try:
        metadata = json.loads(row['tool_calls'] or '{}')
        intent = metadata['turn_intent']
    except (ValueError, TypeError, KeyError):
        raise ValueError('原消息尚无可重新分析的只读决定，请在任务中心重试') from None
    if not isinstance(intent, dict) or intent.get('version') != 1 or \
            intent.get('intent') not in {'query', 'unknown'} or intent.get('allow_writes') is not False or \
            intent.get('record_spans') != [] or intent.get('source_sha256') != sha256_text(row['content']):
        raise ValueError('仅未授权记事写入的原消息可重新分析')
    # Also reject any persisted evidence of application, including interrupted
    # tasks. A new analysis must never repeat an already authorised operation.
    if conn.execute('SELECT 1 FROM applied_patch WHERE source_msg_id=? LIMIT 1', (message_id,)).fetchone():
        raise ValueError('原消息已有保存记录，请直接查看回执')
    for answer in conn.execute("SELECT tool_calls FROM message WHERE conversation_id=? AND role='assistant'",
                               (conversation_id,)):
        try:
            data = json.loads(answer[0] or '{}')
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or data.get('conversation_scope', {}).get('user_message_id') != message_id:
            continue
        outcomes = [*data.get('outcomes', []), *data.get('receipts', []),
                    *data.get('saved', []), *data.get('pending', [])]
        if data.get('pending_ids') or any(item.get('status') in {'ok', 'pending'} for item in outcomes):
            raise ValueError('原消息已有保存或待确认记录，请直接查看回执')
    return row


def context_message(conn, conversation_id, message_id):
    """Keep a chain of explicit analyses anchored before the first question."""
    row = source_message(conn, conversation_id, message_id)
    original_text, seen = row['content'], set()
    while True:
        if row['id'] in seen or len(seen) >= 100:
            raise ValueError('重新分析关联无法核验，请重新明确提问')
        seen.add(row['id'])
        metadata = json.loads(row['tool_calls'])
        link = metadata.get('reanalysis')
        if link is None:
            return row
        if not isinstance(link, dict) or link.get('source_sha256') != sha256_text(original_text):
            raise ValueError('重新分析原文关联无法核验')
        row = source_message(conn, conversation_id, link.get('source_user_message_id'))
        if row['content'] != original_text:
            raise ValueError('重新分析原文与原消息不一致')


def context_scope(conn, conversation_id, message_id):
    from app.agent import conversation_scope
    row = context_message(conn, conversation_id, message_id)
    scope = json.loads(row['tool_calls']).get('conversation_scope')
    if not isinstance(scope, dict) or scope.get('version') != 1 or \
            scope.get('user_message_id') != row['id'] or scope.get('conversation_id') != conversation_id:
        raise ValueError('原消息的人物范围无法核验，请重新明确提问')
    return row, conversation_scope._valid_previous(conn, deepcopy(scope))


def submit(conn, conversation_id, message_id):
    # One successor per user message, even after a lost HTTP response. If that
    # successor also fails classification it can itself be explicitly analysed.
    key = f'chat-reanalysis:{conversation_id}:{message_id}'
    existing = conn.execute('SELECT id FROM task WHERE idempotency_key=?', (key,)).fetchone()
    if existing:
        return tasks.read(conn, existing[0])
    row = source_message(conn, conversation_id, message_id)
    return tasks.submit(conn, 'chat', '重新分析：' + row['content'][:24],
                        {'conversation_id': conversation_id, 'text': row['content'],
                         'reanalysis_of': message_id, 'source_sha256': sha256_text(row['content'])}, key)
