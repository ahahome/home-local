"""Manual corrections. Call inside a write transaction; never replace provenance."""
from __future__ import annotations

import json

from app.memory.writes import resolve_person
from app.store import repo
from app.util import new_id, now_ms


class Conflict(ValueError):
    pass


def audit(c, entity, entity_id, person_id, field, action, before, after,
          reason='', related_person_id=None):
    cid = new_id()
    c.execute('INSERT INTO correction_log VALUES(?,?,?,?,?,?,?,?,?,?,?)',
              (cid, entity, entity_id, person_id, related_person_id, field, action,
               json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False),
               reason.strip(), now_ms()))
    return cid


def current(c, pid, field):
    resolve_person(c, person_id=pid)
    row = repo.current_fields(c, pid).get(field)
    if row:
        row['person_name'] = repo.get_person(c, pid)['display_name']
    return row


def check_version(row, expected):
    if (row['valid_from'] if row else None) != expected:
        raise Conflict('信息已发生变化，请重新打开档案后再操作')


def close_field(c, row):
    end = max(now_ms(), row['valid_from'] + 1)
    c.execute('UPDATE person_field SET valid_to=? WHERE person_id=? AND field=? AND valid_from=?',
              (end, row['person_id'], row['field'], row['valid_from']))


def set_field(c, pid, field, value, expected, reason='', check=True):
    before = current(c, pid, field)
    if check:
        check_version(before, expected)
    value = value.strip()
    if not value:
        raise ValueError('字段内容不能为空')
    if before and before['value'] == value:
        return {'ok': True, 'unchanged': True}
    # A manual correction is itself evidence; an old quote must not attest a new value.
    eid = repo.add_evidence(c, 'text', excerpt=f'用户手动录入：{field} = {value}')
    repo.upsert_field(c, pid, field, value, 1.0, eid)
    after = current(c, pid, field)
    aid = audit(c, 'field', pid, pid, field, 'edit', before, after, reason)
    return {'ok': True, 'audit_id': aid, 'field': after}


def remove_field(c, pid, field, expected, reason=''):
    before = current(c, pid, field)
    check_version(before, expected)
    if not before:
        raise ValueError('当前字段不存在')
    close_field(c, before)
    aid = audit(c, 'field', pid, pid, field, 'delete', before, None, reason)
    return {'ok': True, 'audit_id': aid}


def restore_field(c, pid, field, version, expected, reason=''):
    before = current(c, pid, field)
    check_version(before, expected)
    source = c.execute('SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from=?',
                       (pid, field, version)).fetchone()
    if not source:
        raise ValueError('历史版本不存在')
    source = dict(source)
    if before and before['valid_from'] == version:
        return {'ok': True, 'unchanged': True}
    # Explicit valid_from forces a new version even when the values are equal.
    latest = c.execute('SELECT MAX(valid_from) FROM person_field WHERE person_id=? AND field=?',
                       (pid, field)).fetchone()[0]
    vf = max(now_ms(), (latest or 0) + 1)
    repo.upsert_field(c, pid, field, source['value'], source['confidence'], source['evidence_id'], vf)
    after = current(c, pid, field)
    aid = audit(c, 'field', pid, pid, field, 'restore',
                {'current': before, 'selected': source}, after, reason)
    return {'ok': True, 'audit_id': aid, 'field': after}


def transfer_field(c, pid, field, target, expected, target_expected, replace=False, reason=''):
    if target == pid:
        raise ValueError('请选择另一个人物')
    before = current(c, pid, field)
    check_version(before, expected)
    if not before:
        raise ValueError('当前字段不存在')
    target_before = current(c, target, field)
    check_version(target_before, target_expected)
    if target_before and not replace:
        raise Conflict('目标人物已有该字段，请明确确认替换')
    # Preserve source quote and confidence, but mark the assignment as a manual action.
    latest = c.execute('SELECT MAX(valid_from) FROM person_field WHERE person_id=? AND field=?',
                       (target, field)).fetchone()[0]
    vf = max(now_ms(), (latest or 0) + 1)
    close_field(c, before)
    repo.upsert_field(c, target, field, before['value'], before['confidence'], before['evidence_id'], vf)
    after = current(c, target, field)
    aid = audit(c, 'field', pid, pid, field, 'transfer',
                {'source': before, 'target': target_before}, {'source': None, 'target': after},
                reason, target)
    return {'ok': True, 'audit_id': aid, 'target_person_id': target, 'field': after}


def ledger_row(c, lid):
    r = c.execute('SELECT * FROM ledger_entry WHERE id=?', (lid,)).fetchone()
    if not r:
        raise ValueError('账目不存在')
    r = dict(r)
    if r['person_id']:
        resolve_person(c, person_id=r['person_id'])
        r['person_name'] = repo.get_person(c, r['person_id'])['display_name']
    else:
        r['person_name'] = '未关联人物'
    return r


def change_ledger(c, lid, expected, values=None, void=False, reason=''):
    before = ledger_row(c, lid)
    if before['revision'] != expected:
        raise Conflict('账目已发生变化，请刷新后再操作')
    if before['status'] != 'confirmed':
        raise Conflict('只能修改或撤销已确认账目')
    if void:
        c.execute("UPDATE ledger_entry SET status='void',revision=revision+1 WHERE id=?", (lid,))
        target = None
    else:
        target = values['person_id']
        if target:
            resolve_person(c, person_id=target)
        if (values['direction'] not in {'in', 'out'} or
                type(values['amount_cents']) is not int or values['amount_cents'] <= 0 or
                type(values['occurred_at']) is not int or values['occurred_at'] <= 0):
            raise ValueError('金额、日期或收支方向无效')
        # Keep the original dedupe key so replaying its source never recreates the old amount.
        c.execute('UPDATE ledger_entry SET person_id=?,direction=?,amount_cents=?,note=?,occurred_at=?,revision=revision+1 WHERE id=?',
                  (target, values['direction'], values['amount_cents'], values['note'], values['occurred_at'], lid))
    after = ledger_row(c, lid)
    aid = audit(c, 'ledger', lid, before['person_id'], None, 'void' if void else 'edit',
                before, after, reason, target if target != before['person_id'] else None)
    return {'ok': True, 'audit_id': aid, 'entry': after}


def logs(c, entity, entity_id):
    if entity == 'person':
        rows = c.execute('WITH RECURSIVE ids(id) AS (SELECT ? UNION SELECT p.id FROM person p '
                         "JOIN ids ON p.merged_into=ids.id WHERE p.status='merged') "
                         'SELECT * FROM correction_log WHERE person_id IN (SELECT id FROM ids) '
                         'OR related_person_id IN (SELECT id FROM ids) ORDER BY created_at DESC,rowid DESC',
                         (entity_id,))
    else:
        rows = c.execute('SELECT * FROM correction_log WHERE entity=? AND entity_id=? ORDER BY created_at DESC,rowid DESC',
                         (entity, entity_id))
    result = []
    for r in rows:
        item = dict(r)
        item['before'] = json.loads(item.pop('before_json'))
        item['after'] = json.loads(item.pop('after_json'))
        result.append(item)
    return result
