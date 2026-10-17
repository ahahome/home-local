"""Deterministic daily relationship agenda; dispositions never mutate source events."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, time, timedelta

from app.config import get_settings, tz
from app.store import repo
from app.util import new_id, now_ms, sha256_text

DAY_MS = 86400000
MAX_SNOOZE_MS = 365 * DAY_MS


class AgendaError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


@contextmanager
def _write(conn):
    nested = conn.in_transaction
    savepoint = 'agenda_' + new_id().replace('-', '')
    conn.execute(f'SAVEPOINT {savepoint}' if nested else 'BEGIN IMMEDIATE')
    try:
        yield
        if nested:
            conn.execute(f'RELEASE SAVEPOINT {savepoint}')
        else:
            conn.commit()
    except BaseException:
        if nested:
            conn.execute(f'ROLLBACK TO SAVEPOINT {savepoint}')
            conn.execute(f'RELEASE SAVEPOINT {savepoint}')
        else:
            conn.rollback()
        raise


def _person(conn, person_id):
    person = repo.get_person(conn, person_id)
    if not person or person['deleted_at'] is not None or person['status'] == 'merged':
        raise AgendaError('人物不存在或已归档', 404)
    return person


def get_contact_policy(conn, person_id):
    _person(conn, person_id)
    row = conn.execute('SELECT * FROM contact_policy WHERE person_id=?', (person_id,)).fetchone()
    return {'person_id': person_id, 'enabled': bool(row['enabled']) if row else True,
            'interval_days': row['interval_days'] if row else 14, 'revision': row['revision'] if row else 0}


def update_contact_policy(conn, person_id, enabled, interval_days, expected_revision):
    if type(enabled) is not bool or type(interval_days) is not int or not 1 <= interval_days <= 365 \
            or type(expected_revision) is not int or expected_revision < 0:
        raise AgendaError('联系周期设置无效')
    with _write(conn):
        current = get_contact_policy(conn, person_id)
        if current['revision'] != expected_revision:
            raise AgendaError('联系周期已更新，请刷新后重试', 409)
        conn.execute('INSERT INTO contact_policy(person_id,enabled,interval_days,revision,updated_at) '
            'VALUES(?,?,?,?,?) ON CONFLICT(person_id) DO UPDATE SET enabled=excluded.enabled,'
            'interval_days=excluded.interval_days,revision=excluded.revision,updated_at=excluded.updated_at',
            (person_id, int(enabled), interval_days, expected_revision + 1, now_ms()))
    return get_contact_policy(conn, person_id)


def _person_info(conn, person):
    fields = repo.current_fields(conn, person['id'])
    return {'id': person['id'], 'display_name': person['display_name'], 'summary': ' · '.join(
        str(fields[field]['value']) for field in ('company', 'title', 'city')
        if field in fields and fields[field]['value'])}


def _source(conn, kind, record_id, timestamp, excerpt, evidence_id=None, source_ref=None):
    evidence = conn.execute('SELECT * FROM evidence WHERE id=?', (evidence_id,)).fetchone() \
        if evidence_id else None
    return {'kind': kind, 'id': record_id, 'date': timestamp, 'evidence_id': evidence_id,
            'blob_hash': evidence['blob_hash'] if evidence else None,
            'source_ref': evidence['source_ref'] if evidence else source_ref,
            'excerpt': evidence['excerpt'] if evidence and evidence['excerpt'] else excerpt}


def _contact_event(contact):
    if contact is None:
        return None
    return {key: contact[key] for key in ('id', 'ts', 'channel', 'content')}


def _item(item_id, kind, person, title, reason, occurred_at, due_at, priority, source, goal,
          followup_id=None, event=None):
    stable = {'id': item_id, 'kind': kind, 'person_id': person['id'], 'title': title,
              'occurred_at': occurred_at, 'due_at': due_at, 'source': source,
              'followup_id': followup_id, 'event': event}
    fingerprint = sha256_text(json.dumps(stable, ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':')))
    return {'id': item_id, 'fingerprint': fingerprint, 'kind': kind, 'person': person,
            'title': title, 'reason': reason, 'occurred_at': occurred_at, 'due_at': due_at,
            'priority': priority, 'source': source, 'goal': (goal.strip() or title.strip() or '问候近况')[:1000],
            'followup_id': followup_id,
            'disposition': {'state': 'active', 'until': None}}


def _collect(conn, clock):
    zone = tz()
    local_now = datetime.fromtimestamp(clock / 1000, zone)
    today = local_now.date()
    start = int(datetime.combine(today, time.min, zone).timestamp() * 1000)
    end = int(datetime.combine(today + timedelta(days=1), time.min, zone).timestamp() * 1000)
    people = {p['id']: _person_info(conn, p) for p in repo.all_active_persons(conn)}
    latest_contacts = {}
    for contact in conn.execute("SELECT i.*,i.rowid AS event_order,ip.person_id FROM interaction i "
        "JOIN interaction_person ip ON ip.interaction_id=i.id WHERE i.kind='contact' "
        "AND ip.role='participant' ORDER BY i.ts DESC,i.created_at DESC,i.rowid DESC").fetchall():
        if contact['person_id'] in people:
            latest_contacts.setdefault(contact['person_id'], contact)
    promises = {p['id']: p for p in conn.execute("SELECT * FROM promise WHERE status='open'").fetchall()
                if p['person_id'] in people}
    followups = {f['id']: f for f in conn.execute('SELECT * FROM followup').fetchall()
                 if f['person_id'] in people}
    latest_outcomes = {}
    reminder_followups = {}
    for outcome in conn.execute('SELECT * FROM followup_outcome ORDER BY created_at DESC,rowid DESC').fetchall():
        if outcome['followup_id'] in followups:
            latest_outcomes.setdefault(outcome['followup_id'], outcome)
            if outcome['reminder_id']:
                reminder_followups[outcome['reminder_id']] = followups[outcome['followup_id']]
    for followup in followups.values():
        if followup['reminder_id']:
            reminder_followups[followup['reminder_id']] = followup
    reminders = {}
    for reminder in conn.execute("SELECT * FROM reminder WHERE status IN ('pending','fired')").fetchall():
        pid = reminder['person_id']
        if pid not in people:
            continue
        if reminder['promise_id']:
            promise = promises.get(reminder['promise_id'])
            if not promise or promise['person_id'] != pid:
                continue
        linked = reminder_followups.get(reminder['id'])
        if linked and (linked['status'] == 'completed' or linked['reminder_id'] != reminder['id']
                       or linked['person_id'] != pid):
            continue
        reminders[reminder['id']] = reminder

    items = []
    due_promises = set()
    for promise in promises.values():
        if promise['due_at'] is None or promise['due_at'] >= end:
            continue
        pid = promise['person_id']
        due_promises.add(promise['id'])
        overdue = promise['due_at'] < start
        date_text = datetime.fromtimestamp(promise['due_at'] / 1000, zone).strftime('%Y-%m-%d')
        who = '我答应的' if promise['who'] == 'me' else '对方答应的'
        reason = f"{who}约定{'已逾期' if overdue else '今天到期'}，约定日期 {date_text}"
        source = _source(conn, 'promise', promise['id'], promise['created_at'], promise['content'],
                         promise['evidence_id'])
        items.append(_item('promise:' + promise['id'], 'promise', people[pid], promise['content'],
            reason, promise['created_at'], promise['due_at'], 0 if overdue else 1, source,
            '跟进约定：' + promise['content'], event={'who': promise['who'],
                'last_contact': _contact_event(latest_contacts.get(pid))}))

    for reminder in reminders.values():
        if reminder['fire_at'] >= end or reminder['promise_id'] in due_promises:
            continue
        pid = reminder['person_id']
        linked = reminder_followups.get(reminder['id'])
        promise = promises.get(reminder['promise_id'])
        outcome = latest_outcomes.get(linked['id']) if linked else None
        date_text = datetime.fromtimestamp(reminder['fire_at'] / 1000, zone).strftime('%Y-%m-%d %H:%M')
        source = _source(conn, 'reminder', reminder['id'], reminder['created_at'], reminder['text'],
                         promise['evidence_id'] if promise else None)
        items.append(_item('reminder:' + reminder['id'], 'reminder', people[pid], reminder['text'],
            f"你设置的提醒{'已到期' if reminder['fire_at'] <= clock else '将在今天到期'}：{date_text}",
            reminder['created_at'], reminder['fire_at'], 2, source,
            linked['goal'] if linked else reminder['text'], linked['id'] if linked else None,
            {'last_contact': _contact_event(latest_contacts.get(pid)),
             'promise': {'id': promise['id'], 'content': promise['content'], 'due_at': promise['due_at']}
             if promise else None, 'outcome_id': outcome['id'] if outcome else None}))

    for followup in followups.values():
        if followup['status'] not in {'awaiting_reply', 'follow_up'}:
            continue
        outcome = latest_outcomes.get(followup['id'])
        if not outcome:
            continue
        payload = json.loads(outcome['payload'])
        contacted_at = payload['contacted_at']
        awaiting = followup['status'] == 'awaiting_reply'
        if awaiting and clock - contacted_at < 3 * DAY_MS:
            continue
        pid = followup['person_id']
        contact = latest_contacts.get(pid)
        if awaiting and contact and contact['id'] != outcome['contact_id'] and contact['ts'] >= contacted_at:
            continue
        # A concrete followup reminder takes precedence, including one later today.
        if followup['reminder_id'] in reminders:
            continue
        source = _source(conn, 'contact', outcome['contact_id'], contacted_at, payload['notes'],
                         source_ref='followup-outcome:' + outcome['id'])
        marked_at = datetime.fromtimestamp(outcome['created_at'] / 1000, zone).strftime('%Y-%m-%d %H:%M')
        items.append(_item(('awaiting:' if awaiting else 'followup:') + followup['id'],
            'awaiting_reply' if awaiting else 'follow_up', people[pid],
            ('跟进回复：' if awaiting else '继续跟进：') + followup['goal'],
            f"已等待回复 {(clock - contacted_at) // DAY_MS} 天，尚未记录后续实际联系" if awaiting
            else '你已标记待跟进，记录于 ' + marked_at,
            contacted_at if awaiting else outcome['created_at'], contacted_at + 3 * DAY_MS if awaiting else None,
            3, source, followup['goal'], followup['id'],
            {'outcome_id': outcome['id'], 'payload': payload, 'last_contact': _contact_event(contact)}))

    occupied_people = {item['person']['id'] for item in items}
    # Respect an explicit future followup schedule before offering a generic contact nudge.
    occupied_people.update(reminder['person_id'] for reminder in reminders.values()
        if reminder['fire_at'] >= end and reminder['id'] in reminder_followups)
    policies = {p['person_id']: p for p in conn.execute('SELECT * FROM contact_policy').fetchall()}
    for pid, contact in latest_contacts.items():
        if pid in occupied_people:
            continue
        policy = policies.get(pid)
        if policy and not policy['enabled']:
            continue
        interval = policy['interval_days'] if policy else 14
        local_contact = datetime.fromtimestamp(contact['ts'] / 1000, zone)
        due_at = int((local_contact + timedelta(days=interval)).timestamp() * 1000)
        if clock < due_at:
            continue
        elapsed = (today - local_contact.date()).days
        source = _source(conn, 'contact', contact['id'], contact['ts'], contact['content'],
                         source_ref=contact['source_ref'])
        items.append(_item('contact:' + pid, 'contact_gap', people[pid],
            '联系一下' + people[pid]['display_name'], f'距上次实际联系已 {elapsed} 天，联系周期为 {interval} 天',
            contact['ts'], due_at, 4, source, '问候近况，接着上次话题聊聊',
            event={'contact': _contact_event(contact), 'interval_days': interval}))
    items.sort(key=lambda item: (item['priority'], item['due_at'] is None,
                                item['due_at'] if item['due_at'] is not None else item['occurred_at'], item['id']))
    unknown = sorted((person for pid, person in people.items() if pid not in latest_contacts),
                     key=lambda p: (p['display_name'].casefold(), p['id']))
    return today.isoformat(), items, unknown


def get_agenda(conn, include_hidden=False):
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute('BEGIN')
    try:
        clock = now_ms()
        today, candidates, unknown = _collect(conn, clock)
        actions = {(a['item_id'], a['fingerprint']): a for a in conn.execute('SELECT * FROM agenda_action')}
        active, hidden = [], []
        counts = {'active': 0, 'snoozed': 0, 'dismissed': 0, 'unknown': len(unknown)}
        for item in candidates:
            action = actions.get((item['id'], item['fingerprint']))
            if action and (action['state'] == 'dismissed' or action['until'] > clock):
                item['disposition'] = {'state': action['state'], 'until': action['until']}
                hidden.append(item)
                counts[action['state']] += 1
            else:
                active.append(item)
        counts['active'] = len(active)
        result = {'date': today, 'timezone': get_settings().timezone, 'items': active,
                  'unknown_people': unknown, 'counts': counts}
        if include_hidden:
            result['hidden_items'] = hidden
        if owns_transaction:
            conn.commit()
        return result
    except BaseException:
        if owns_transaction:
            conn.rollback()
        raise


def _source_person(conn, item_id):
    prefix, _, record_id = item_id.partition(':')
    tables = {'promise': ('promise', 'person_id'), 'reminder': ('reminder', 'person_id'),
              'awaiting': ('followup', 'person_id'), 'followup': ('followup', 'person_id'),
              'contact': ('person', 'id')}
    if prefix not in tables or not record_id:
        raise AgendaError('跟进事项不存在', 404)
    table, column = tables[prefix]
    row = conn.execute(f'SELECT {column} AS person_id FROM {table} WHERE id=?', (record_id,)).fetchone()
    if not row or row['person_id'] is None:
        raise AgendaError('跟进事项不存在', 404)
    _person(conn, row['person_id'])


def apply_action(conn, item_id, fingerprint, action, until=None):
    if action not in {'snooze', 'dismiss', 'restore'}:
        raise AgendaError('跟进操作无效')
    if action != 'snooze' and until is not None:
        raise AgendaError('此操作不能设置暂缓时间')
    with _write(conn):
        clock = now_ms()
        if action == 'snooze' and (type(until) is not int or not clock < until <= clock + MAX_SNOOZE_MS):
            raise AgendaError('暂缓时间必须晚于现在且不超过一年')
        _source_person(conn, item_id)
        _, candidates, _ = _collect(conn, clock)
        item = next((item for item in candidates if item['id'] == item_id), None)
        if not item or item['fingerprint'] != fingerprint:
            raise AgendaError('跟进事项已发生变化，请刷新后重试', 409)
        if action == 'restore':
            conn.execute('DELETE FROM agenda_action WHERE item_id=? AND fingerprint=?', (item_id, fingerprint))
        else:
            conn.execute('INSERT INTO agenda_action(person_id,item_id,fingerprint,state,until,updated_at) '
                'VALUES(?,?,?,?,?,?) ON CONFLICT(item_id,fingerprint) DO UPDATE SET state=excluded.state,'
                'until=excluded.until,updated_at=excluded.updated_at,person_id=excluded.person_id',
                (item['person']['id'], item_id, fingerprint, 'snoozed' if action == 'snooze' else 'dismissed',
                 until, clock))
    return {'ok': True}
