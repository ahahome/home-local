"""Immutable quotations of stored memory and their current availability."""
from __future__ import annotations

import json
from datetime import datetime

from app.config import tz
from app.store import repo
from app.util import now_ms

EVIDENCE_KEYS = ('id', 'kind', 'excerpt', 'source_ref', 'blob_hash', 'created_at')
# Additive quotation metadata did not exist in old saved answers. Compare it only
# when it was captured, while continuing to compare every original content key.
_CONTEXT_KEYS = {'record_context'}


def _person(conn, person_id):
    if person_id is None:
        return None
    row = repo.get_person(conn, person_id)
    if not row or row['deleted_at'] is not None or row['status'] == 'merged':
        return None
    return {'id': row['id'], 'display_name': row['display_name']}


def _evidence(conn, evidence_id):
    row = conn.execute('SELECT * FROM evidence WHERE id=?', (evidence_id,)).fetchone()
    return {key: row[key] for key in EVIDENCE_KEYS} if row else None


def _original_evidence(conn, initial):
    source, seen = initial, set()
    while source and source['id'] not in seen:
        seen.add(source['id'])
        if not (source['source_ref'] or '').startswith(('outcome-memory-review:', 'memory-correction:')):
            break
        try:
            record = json.loads(source['excerpt'])
        except (json.JSONDecodeError, TypeError):
            break
        if not isinstance(record, dict):
            break
        reference = record.get('source_evidence_id') or record.get('origin_evidence_id')
        if not isinstance(reference, str) or reference in seen:
            break
        original = _evidence(conn, reference)
        if original is None:
            break
        source = original
    return source


def _memory_snapshot(conn, row):
    person = _person(conn, row['person_id'])
    if row['deleted_at'] is not None or row['person_id'] is not None and person is None:
        return None
    source = {
        'type': 'memory', 'entity_id': row['id'], 'person': person, 'text': row['text'],
        'date': row['ts'], 'temporal_status': 'historical_background', 'revision': row['revision'],
        'ts': row['ts'], 'kind': row['kind'], 'evidence': _evidence(conn, row['evidence_id']),
        'captured_at': now_ms(),
    }
    edited = conn.execute("SELECT created_at FROM memory_correction WHERE memory_id=? AND action='edit' "
                          'ORDER BY revision DESC,created_at DESC,id DESC LIMIT 1', (row['id'],)).fetchone()
    lifecycle = conn.execute("SELECT action,created_at FROM memory_correction WHERE memory_id=? "
                             "AND action IN ('withdraw','restore') "
                             'ORDER BY revision DESC,created_at DESC,id DESC LIMIT 1', (row['id'],)).fetchone()
    source['record_context'] = {
        'status': row['status'], 'recorded_at': row['created_at'],
        # updated_at and revision also change on archive/move/restore. Neither is
        # evidence of a user edit. Only the dedicated edit audit supports this.
        'last_user_edit_at': edited['created_at'] if edited else None,
        'last_lifecycle_change': dict(lifecycle) if lifecycle else None,
    }
    initial = _evidence(conn, row['origin_evidence_id'])
    original = _original_evidence(conn, initial)
    if original:
        source['origin_source'] = original
    if initial and original and initial['id'] != original['id']:
        source['initial_source'] = initial
    return source


def memory_source(conn, memory_id):
    row = conn.execute(f'SELECT m.* FROM memory_item m WHERE m.id=? AND {repo.ACTIVE_MEMORY_WHERE}',
                       (memory_id,)).fetchone()
    return _memory_snapshot(conn, row) if row else None


def _field_snapshot(conn, row):
    person = _person(conn, row['person_id'])
    if person is None:
        return None
    source = {
        'type': 'field',
        'entity_id': json.dumps([row['person_id'], row['field'], row['valid_from']],
                                ensure_ascii=False, separators=(',', ':')),
        'person': person, 'text': row['value'] if row['value'] is not None else '',
        'field': row['field'], 'date': row['valid_from'],
        'valid_from': row['valid_from'], 'valid_to': row['valid_to'], 'recorded_at': row['recorded_at'],
        'temporal_status': 'current_field' if row['valid_to'] is None else 'historical_field',
        'evidence': _evidence(conn, row['evidence_id']), 'captured_at': now_ms(),
        'record_context': {'recorded_at': row['recorded_at']},
    }
    original = _original_evidence(conn, source['evidence'])
    if original:
        source['origin_source'] = original
    return source


def field_sources(conn, person_id, limit=10):
    """Keep every current field and at most ``limit`` recent historical versions."""
    if _person(conn, person_id) is None:
        return []
    current = conn.execute('SELECT * FROM person_field WHERE person_id=? AND valid_to IS NULL '
                           'ORDER BY field,valid_from DESC', (person_id,)).fetchall()
    historical = conn.execute('SELECT * FROM person_field WHERE person_id=? AND valid_to IS NOT NULL '
                              'ORDER BY valid_from DESC,field LIMIT ?', (person_id, max(0, limit))).fetchall()
    return [_field_snapshot(conn, row) for row in [*current, *historical]]


def interaction_source(conn, interaction_id):
    row = conn.execute('SELECT * FROM interaction WHERE id=?', (interaction_id,)).fetchone()
    if row is None:
        return None
    links = conn.execute('SELECT ip.person_id,ip.role FROM interaction_person ip '
                         'WHERE ip.interaction_id=? ORDER BY ip.person_id', (interaction_id,)).fetchall()
    people = []
    for link in links:
        person = _person(conn, link['person_id'])
        if person is None:
            return None
        people.append(person | {'role': link['role']})
    evidence = _evidence(conn, row['source_ref'])
    if evidence is None and row['source_ref']:
        candidates = conn.execute('SELECT * FROM evidence WHERE source_ref=? ORDER BY id LIMIT 2',
                                  (row['source_ref'],)).fetchall()
        if len(candidates) == 1:
            candidate = {key: candidates[0][key] for key in EVIDENCE_KEYS}
            excerpt = candidate['excerpt']
            if isinstance(excerpt, str) and excerpt and (excerpt in row['content'] or row['content'] in excerpt):
                evidence = candidate
    source = {
        'type': 'interaction', 'entity_id': row['id'],
        'person': {key: people[0][key] for key in ('id', 'display_name')} if len(people) == 1 else None,
        'people': people, 'text': row['content'], 'date': row['ts'], 'ts': row['ts'],
        'temporal_status': row['kind'], 'kind': row['kind'], 'channel': row['channel'],
        'source_ref': row['source_ref'], 'evidence': evidence, 'captured_at': now_ms(),
        'record_context': {'recorded_at': row['created_at']},
    }
    original = _original_evidence(conn, evidence)
    if original:
        source['origin_source'] = original
    return source


def _unchanged(snapshot, current):
    # Registry labels and capture time describe the quotation, not its stored content.
    ignored = {'id', 'captured_at'} | (_CONTEXT_KEYS - snapshot.keys())
    return {key: value for key, value in snapshot.items() if key not in ignored} == \
        {key: value for key, value in current.items() if key not in ignored}


def _readable_time(value):
    if type(value) is not int:
        return '未知'
    try:
        return datetime.fromtimestamp(value / 1000, tz()).isoformat(timespec='seconds')
    except (ValueError, OverflowError, OSError):
        return '未知'


def model_evidence_context(source, excerpt_limit=400):
    """Project known audit envelopes without exposing their storage/input keys.

    An ordinary user excerpt can itself contain JSON. Interpret only envelopes
    produced by the two known review workflows, never similarly named keys in a
    plain excerpt. Exact text comparisons describe stored strings, not whether
    somebody actually uttered those words or whether an import is complete.
    """
    evidence = source.get('evidence')
    if not isinstance(evidence, dict):
        return {'文字核对': '没有可见来源摘录，不能判断是否逐字记录。'}
    excerpt = evidence.get('excerpt')
    excerpt = excerpt if isinstance(excerpt, str) else ''
    reference = evidence.get('source_ref') or ''
    audit_type = ('correction' if reference.startswith('memory-correction:') else
                  'proposal' if reference.startswith('outcome-memory-review:') else None)
    if audit_type:
        try:
            audit = json.loads(excerpt)
        except (ValueError, TypeError):
            audit = None
        result = {'来源说明': '用户人工修订审计记录' if audit_type == 'correction' else '用户确认提案的审计记录',
                  '文字核对': '审计记录不等于当事人的逐字原话；当前文字与旧来源可能不同。'}
        if isinstance(audit, dict):
            if audit_type == 'correction':
                edited = audit.get('user_correction')
                if isinstance(edited, dict) and isinstance(edited.get('text'), str):
                    result['用户修订文字'] = edited['text'][:excerpt_limit]
            else:
                confirmed = audit.get('user_confirmed')
                if isinstance(confirmed, dict) and isinstance(confirmed.get('content'), str):
                    result['用户确认文字'] = confirmed['content'][:excerpt_limit]
                if isinstance(audit.get('original_quote'), str):
                    result['提案所引摘录'] = audit['original_quote'][:excerpt_limit]
            result['摘录范围说明'] = '仅显示审计中已有的文字，可能因展示预算截断；日期含义只看时间信息。'
        return result
    text = source.get('text')
    if not excerpt:
        comparison = '没有可见来源摘录，不能判断是否逐字记录。'
    elif isinstance(text, str) and text == excerpt:
        comparison = '当前文字与来源摘录逐字一致。'
    elif isinstance(text, str) and text and text in excerpt:
        comparison = '当前文字是来源摘录中的连续片段。'
    else:
        comparison = '当前文字与来源摘录不同；不能仅据此判断是提炼、改写还是来源不完整。'
    return {'来源摘录': excerpt[:excerpt_limit], '摘录已截断': len(excerpt) > excerpt_limit,
            '文字核对': comparison,
            '核对边界': '只比较系统内保存的文字，不证明是当事人逐字原话或原始资料的完整内容。'}


def model_source_context(source):
    """Explain database availability separately from the truth/date of its claim.

    This projection is model input only. Original snapshot keys remain intact for
    old answer rendering, evidence inspection and optimistic concurrency checks.
    """
    kind = source.get('type')
    context = source.get('record_context') or {}
    times = {'最近人工编辑时间': _readable_time(context.get('last_user_edit_at'))}
    if kind == 'memory':
        status = context.get('status')
        state = {'active': '记录可读取，未撤销', 'withdrawn': '记录已撤销，不可作为可用记忆采用'}.get(
            status, '未提供撤销状态，不可据此推断记录已撤销或已恢复')
        state += '；所属人物当前可读取' if source.get('person') else '；属于用户自己的记忆'
        times.update({'记忆所标日期（不代表录入或修改日期）': _readable_time(source.get('ts', source.get('date'))),
                      '首次录入时间': _readable_time(context.get('recorded_at'))})
        lifecycle = context.get('last_lifecycle_change')
        if isinstance(lifecycle, dict) and lifecycle.get('action') in {'restore', 'withdraw'}:
            action = '恢复' if lifecycle['action'] == 'restore' else '撤销'
            times['最近撤销或恢复操作'] = {'操作': action, '操作时间': _readable_time(lifecycle.get('created_at'))}
        return {'记录状态': state, '事实时效': '有日期的背景记录；当前事实是否仍成立尚未核实。背景记录不表示已撤销。',
                '时间信息': times,
                '时间表述边界': '所标日期不能用来断言当事人何时说过、用户何时纠正或何时修改。'
                          '叙述“某日纠正/修改”也只能使用最近人工编辑时间；未知就不能给出日期。'}
    if kind == 'field':
        current = source.get('valid_to') is None
        times.update({'字段生效时间': _readable_time(source.get('valid_from')),
                      '录入时间': _readable_time(source.get('recorded_at'))})
        if not current:
            times['字段结束时间'] = _readable_time(source.get('valid_to'))
        return {'记录状态': '档案可读取；所属人物当前可读取',
                '事实时效': '当前档案字段，尚无结束记录' if current else '已结束的历史档案字段，不代表现状',
                '时间信息': times}
    if kind == 'interaction':
        times.update({'发生时间': _readable_time(source.get('ts', source.get('date'))),
                      '录入时间': _readable_time(context.get('recorded_at'))})
        return {'记录状态': '互动记录及其关联人物当前可读取',
                '事实时效': '所标发生时间的记录，不证明其中描述的情况现在仍成立', '时间信息': times}
    return {}


def source_state(conn, snapshot):
    if not isinstance(snapshot, dict):
        return {'status': 'unavailable', 'current': None}
    kind, entity_id = snapshot.get('type'), snapshot.get('entity_id')
    current = None
    if kind == 'profile_context' and isinstance(entity_id, str):
        from app.memory.compile_card import build_card
        if _person(conn, entity_id):
            current = {'type': kind, 'entity_id': entity_id, 'text': build_card(conn, entity_id)}
    elif kind == 'memory' and isinstance(entity_id, str):
        row = conn.execute('SELECT * FROM memory_item WHERE id=?', (entity_id,)).fetchone()
        current = _memory_snapshot(conn, row) if row else None
        if current is not None and row['status'] == 'withdrawn':
            return {'status': 'withdrawn', 'current': current}
        quoted_person = snapshot.get('person')
        quoted_person_id = quoted_person.get('id') if isinstance(quoted_person, dict) else None
        if current is not None and (current.get('person') or {}).get('id') != quoted_person_id:
            return {'status': 'moved', 'current': current}
    elif kind == 'field' and isinstance(entity_id, str):
        try:
            identity = json.loads(entity_id)
        except (json.JSONDecodeError, TypeError):
            identity = None
        if isinstance(identity, list) and len(identity) == 3 and isinstance(identity[0], str) \
                and isinstance(identity[1], str) and type(identity[2]) is int:
            row = conn.execute('SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from=?',
                               identity).fetchone()
            current = _field_snapshot(conn, row) if row else None
    elif kind == 'interaction' and isinstance(entity_id, str):
        current = interaction_source(conn, entity_id)
    if current is None:
        return {'status': 'unavailable', 'current': None}
    return {'status': 'current' if _unchanged(snapshot, current) else 'updated', 'current': current}
