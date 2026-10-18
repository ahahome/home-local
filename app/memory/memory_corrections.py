"""Versioned manual memory corrections, independent of source and person archival."""
from __future__ import annotations

from datetime import date, datetime, time
import json

from app.agent.followup import _atomic
from app.config import tz
from app.memory.fts import tokenize
from app.memory.vector import get_index
from app.store import repo
from app.util import new_id, now_ms, sha256_text, today_local


class MemoryCorrectionError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def get_memory(conn, memory_id):
    row = conn.execute('SELECT m.*,p.display_name AS person_name FROM memory_item m '
        'LEFT JOIN person p ON p.id=m.person_id WHERE m.id=? AND m.deleted_at IS NULL '
        "AND (m.person_id IS NULL OR (p.id IS NOT NULL AND p.deleted_at IS NULL AND p.status!='merged'))",
        (memory_id,)).fetchone()
    if not row:
        raise MemoryCorrectionError('记忆不存在或已不可见', 404)
    return dict(row)


def _target(conn, person_id):
    if person_id is None:
        return
    person = repo.get_person(conn, person_id)
    if not person or person['deleted_at'] is not None or person['status'] == 'merged':
        raise MemoryCorrectionError('目标人物不存在或已归档', 404)


def _snapshot(conn, row, count=None):
    record = dict(row)
    record['person'] = {'id': row['person_id'], 'display_name': row['person_name']} \
        if row['person_id'] is not None else None
    record['valid_from'] = datetime.fromtimestamp(row['ts'] / 1000, tz()).date().isoformat()
    record['correction_count'] = count if count is not None else conn.execute(
        'SELECT COUNT(*) FROM memory_correction WHERE memory_id=?', (row['id'],)).fetchone()[0]
    return record


def _evidence(conn, evidence_id):
    row = conn.execute('SELECT * FROM evidence WHERE id=?', (evidence_id,)).fetchone()
    return {key: row[key] for key in ('id', 'kind', 'source_ref', 'excerpt', 'blob_hash', 'created_at')} \
        if row else None


def _audit(conn, row):
    before, after = json.loads(row['before_json']), json.loads(row['after_json'])
    return {key: row[key] for key in ('id', 'memory_id', 'action', 'reason', 'request_id',
        'revision', 'evidence_id', 'created_at')} | {
        'before': before, 'after': after, 'source': _evidence(conn, row['evidence_id']),
        'before_source': _evidence(conn, before['evidence_id']),
        'after_source': _evidence(conn, after['evidence_id'])}


def history(conn, memory_id):
    get_memory(conn, memory_id)
    return [_audit(conn, row) for row in conn.execute('SELECT * FROM memory_correction WHERE memory_id=? '
        'ORDER BY revision DESC,created_at DESC,id DESC', (memory_id,)).fetchall()]


def _validate(body):
    allowed = {'action', 'expected_revision', 'request_id', 'reason', 'text', 'valid_from', 'person_id'}
    if set(body) - allowed or body.get('action') not in {'edit', 'withdraw', 'restore'}:
        raise MemoryCorrectionError('请提交有效的记忆操作')
    if type(body.get('expected_revision')) is not int or body['expected_revision'] < 0:
        raise MemoryCorrectionError('请提交有效的记忆版本')
    if not isinstance(body.get('request_id'), str) or not body['request_id'].strip() \
            or len(body['request_id']) > 128:
        raise MemoryCorrectionError('请提交有效的请求标识')
    if not isinstance(body.get('reason', ''), str) or len(body.get('reason', '')) > 1000:
        raise MemoryCorrectionError('纠错原因不能超过1000字')
    edits = set(body) & {'text', 'valid_from', 'person_id'}
    if (body['action'] != 'edit' and edits) or (body['action'] == 'edit' and not edits):
        raise MemoryCorrectionError('编辑请至少提交内容、日期或归属之一；撤销和恢复不能附带修改')
    if 'text' in body and (not isinstance(body['text'], str) or not body['text'].strip()
                           or len(body['text']) > 20000):
        raise MemoryCorrectionError('记忆内容不能为空且不能超过20000字')
    if 'person_id' in body and body['person_id'] is not None \
            and (not isinstance(body['person_id'], str) or not body['person_id'].strip()):
        raise MemoryCorrectionError('人物归属必须是有效人物或个人记忆')


def _timestamp(value, current_ts):
    if not isinstance(value, str):
        raise MemoryCorrectionError('日期必须使用 YYYY-MM-DD')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise MemoryCorrectionError('日期必须使用 YYYY-MM-DD') from exc
    if parsed.isoformat() != value or value > today_local():
        raise MemoryCorrectionError('记忆日期不能晚于今天，且须使用 YYYY-MM-DD')
    if datetime.fromtimestamp(current_ts / 1000, tz()).date() == parsed:
        return current_ts
    return int(datetime.combine(parsed, time(12), tz()).timestamp() * 1000)


def _index(conn, row):
    conn.execute("DELETE FROM memory_fts WHERE owner_type='memory' AND owner_id=?", (row['id'],))
    if row['status'] == 'active':
        conn.execute('INSERT INTO memory_fts(tokens,owner_type,owner_id) VALUES(?,?,?)',
                     (tokenize(row['text']), 'memory', row['id']))
    repo.delete_vector(conn, 'memory', row['id'])


def correct_memory(conn, memory_id, body):
    body = dict(body)
    _validate(body)
    request_hash = sha256_text(_json(body))
    with _atomic(conn):
        row = get_memory(conn, memory_id)
        existing = conn.execute('SELECT * FROM memory_correction WHERE memory_id=? AND request_id=?',
                               (memory_id, body['request_id'])).fetchone()
        if existing:
            if existing['request_hash'] != request_hash:
                raise MemoryCorrectionError('该请求标识已用于其他记忆修改', 409)
            audit = _audit(conn, existing)
            return {'memory': _snapshot(conn, row), 'correction': audit, 'idempotent': True}
        if row['revision'] != body['expected_revision']:
            raise MemoryCorrectionError('记忆已发生变化，请刷新后重新核查', 409)
        before = _snapshot(conn, row)
        correction_id, stamp = new_id(), now_ms()
        action = body['action']
        text, person_id, timestamp = row['text'], row['person_id'], row['ts']
        evidence_id = None
        if action == 'edit':
            text = body.get('text', text).strip()
            person_id = body.get('person_id', person_id)
            _target(conn, person_id)
            timestamp = _timestamp(body['valid_from'], timestamp) if 'valid_from' in body else timestamp
            if (text, person_id, timestamp) == (row['text'], row['person_id'], row['ts']):
                raise MemoryCorrectionError('记忆内容、日期和归属没有变化')
            evidence_id = repo.add_evidence(conn, 'text', source_ref='memory-correction:' + correction_id,
                excerpt=_json({'memory_id': memory_id, 'correction_id': correction_id,
                    'origin_evidence_id': row['origin_evidence_id'], 'previous_evidence_id': row['evidence_id'],
                    'user_correction': {'text': text, 'person_id': person_id,
                        'valid_from': datetime.fromtimestamp(timestamp / 1000, tz()).date().isoformat()},
                    'edited_by_user': True, 'reason': body.get('reason', '').strip()}))
            conn.execute('UPDATE memory_item SET text=?,person_id=?,ts=?,evidence_id=?,confidence=1.0,'
                'assignment_revision=assignment_revision+?,revision=revision+1,updated_at=? WHERE id=?',
                (text, person_id, timestamp, evidence_id, int(person_id != row['person_id']), stamp, memory_id))
        else:
            expected_status = 'active' if action == 'withdraw' else 'withdrawn'
            if row['status'] != expected_status:
                raise MemoryCorrectionError('仅有效记忆可撤销，仅已撤销记忆可恢复', 409)
            status = 'withdrawn' if action == 'withdraw' else 'active'
            conn.execute('UPDATE memory_item SET status=?,revision=revision+1,updated_at=? WHERE id=?',
                         (status, stamp, memory_id))
        after_row = get_memory(conn, memory_id)
        after = _snapshot(conn, after_row, before['correction_count'] + 1)
        _index(conn, after_row)
        conn.execute('INSERT INTO memory_correction(id,memory_id,action,before_json,after_json,reason,'
            'request_id,request_hash,revision,evidence_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (correction_id, memory_id, action, _json(before), _json(after), body.get('reason', '').strip(),
             body['request_id'], request_hash, after_row['revision'], evidence_id, stamp))
        audit = _audit(conn, conn.execute('SELECT * FROM memory_correction WHERE id=?', (correction_id,)).fetchone())
    # Invalidate after commit so a concurrent pre-commit read cannot repopulate stale vectors.
    get_index().invalidate('memory')
    return {'memory': after, 'correction': audit, 'idempotent': False}
