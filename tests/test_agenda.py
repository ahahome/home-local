"""Daily suggestions use real contacts, local date boundaries and durable source versions."""
import json
from contextlib import contextmanager
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import agenda
from app.store import repo
from app.util import new_id


def ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


@pytest.fixture
def clock(monkeypatch):
    value = ms('2026-10-16T04:00:00+00:00')
    monkeypatch.setattr(agenda, 'now_ms', lambda: value)
    return value


@pytest.fixture
def client(conn, clock, monkeypatch):
    import app.llm
    from app.api import agenda as api
    monkeypatch.setattr(app.llm, 'get_llm', lambda: pytest.fail('agenda must never call models'))
    monkeypatch.setattr(api, 'get_conn', lambda: conn)
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


@pytest.fixture
def pid(conn):
    person_id = repo.create_person(conn, '林舟', status='active')
    conn.commit()
    return person_id


def contact(conn, person_id, timestamp, text='实际聊了近况', role='participant', kind='contact'):
    return repo.insert_interaction(conn, timestamp, 'phone', text, 'source-' + new_id(),
        [(person_id, role)], kind=kind, dedupe_key=new_id())[0]


def waiting(conn, person_id, timestamp, state='awaiting_reply'):
    fid, oid = new_id(), new_id()
    cid = contact(conn, person_id, timestamp)
    conn.execute('INSERT INTO followup(id,person_id,origin_person_id,goal,status,request_id,request_hash,'
        'created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
        (fid, person_id, person_id, '确认见面安排', state, new_id(), new_id(), timestamp, timestamp))
    payload = {'request_id': new_id(), 'expected_revision': 1, 'contacted_at': timestamp,
               'channel': 'phone', 'notes': '已经联系，等待答复', 'state': state,
               'next_followup_at': None, 'complete_promise_ids': [], 'new_promises': []}
    conn.execute('INSERT INTO followup_outcome(id,followup_id,request_id,request_hash,payload,contact_id,'
        'receipts,created_at) VALUES(?,?,?,?,?,?,?,?)',
        (oid, fid, payload['request_id'], new_id(), json.dumps(payload), cid, '[]', timestamp))
    return fid, oid, cid


def action(client, item, command, **kwargs):
    return client.post('/api/agenda/actions', json={'item_id': item['id'],
        'fingerprint': item['fingerprint'], 'action': command, **kwargs})


def items(client):
    response = client.get('/api/agenda')
    assert response.status_code == 200, response.text
    return response.json()['items']


def test_no_contact_is_unknown_not_a_contact_gap_even_with_material_mentions(client, conn, pid, clock):
    contact(conn, pid, clock - 30 * agenda.DAY_MS, role='mentioned')
    contact(conn, pid, clock - 25 * agenda.DAY_MS, kind='note')
    repo.add_promise(conn, 'me', '未定日期的承诺', pid)
    repo.add_reminder(conn, clock - 1000, '没有人物的全局提醒')
    conn.commit()
    response = client.get('/api/agenda').json()
    assert response['date'] == '2026-10-16' and response['timezone'] == 'Asia/Shanghai'
    assert response['items'] == []
    assert response['unknown_people'] == [{'id': pid, 'display_name': '林舟', 'summary': ''}]
    assert response['counts'] == {'active': 0, 'snoozed': 0, 'dismissed': 0, 'unknown': 1}
    assert 'hidden_items' not in response


def test_due_promises_use_sources_and_deduplicate_their_reminders(client, conn, pid, clock):
    evidence_id = repo.add_evidence(conn, 'blob', blob_hash='a' * 64,
        source_ref='original-source', excerpt='我答应今天发资料')
    due = repo.add_promise(conn, 'me', '发送资料', pid, clock + 3600000, evidence_id)
    overdue = repo.add_promise(conn, 'them', '回复日期', pid, clock - 2 * agenda.DAY_MS)
    future = repo.add_promise(conn, 'me', '明天发送', pid, clock + agenda.DAY_MS)
    repo.add_reminder(conn, clock - 1000, '重复承诺提醒', pid, due)
    repo.add_reminder(conn, clock + 1000, '重复逾期提醒', pid, overdue)
    repo.add_reminder(conn, clock + 1000, '未来承诺的提前提醒', pid, future)
    conn.commit()
    result = items(client)
    assert [r['id'] for r in result[:2]] == ['promise:' + overdue, 'promise:' + due]
    assert len(result) == 3
    assert result[2]['kind'] == 'reminder'
    assert '已逾期' in result[0]['reason'] and '今天到期' in result[1]['reason']
    source = result[1]['source']
    assert source['id'] == due and source['kind'] == 'promise'
    assert source['evidence_id'] == evidence_id and source['blob_hash'] == 'a' * 64
    assert source['source_ref'] == 'original-source' and source['excerpt'] == '我答应今天发资料'
    assert all(r['person']['id'] == pid for r in result)


def test_long_source_keeps_full_title_but_goal_fits_communication_preparation(client, conn, pid, clock):
    from app.api.followups import PrepareIn
    content = '很长的明确约定' * 250
    repo.add_promise(conn, 'me', content, pid, clock)
    conn.commit()
    item = items(client)[0]
    assert item['title'] == item['source']['excerpt'] == content
    assert len(item['goal']) == 1000
    assert PrepareIn(goal=item['goal'], request_id='agenda-test').goal == item['goal']


@pytest.mark.parametrize('zone,current,end', [
    ('Asia/Shanghai', '2026-10-15T23:30:00+00:00', '2026-10-16T16:00:00+00:00'),
    ('America/New_York', '2026-03-08T16:00:00+00:00', '2026-03-09T04:00:00+00:00'),
    ('America/New_York', '2026-11-01T17:00:00+00:00', '2026-11-02T05:00:00+00:00'),
])
def test_local_day_end_is_exclusive_across_timezones_and_dst(client, conn, pid, monkeypatch, zone, current, end):
    from app.config import get_settings
    monkeypatch.setenv('HOME_TIMEZONE', zone)
    get_settings.cache_clear()
    monkeypatch.setattr(agenda, 'now_ms', lambda: ms(current))
    last = repo.add_promise(conn, 'me', '今天最后一刻', pid, ms(end) - 1)
    tomorrow = repo.add_promise(conn, 'me', '明天零点', pid, ms(end))
    last_reminder = repo.add_reminder(conn, ms(end) - 1, '今天最后一刻提醒', pid)
    tomorrow_reminder = repo.add_reminder(conn, ms(end), '明天零点提醒', pid)
    conn.commit()
    result = client.get('/api/agenda').json()
    assert {r['id'] for r in result['items']} == {'promise:' + last, 'reminder:' + last_reminder}
    assert 'promise:' + tomorrow not in {r['id'] for r in result['items']}
    assert 'reminder:' + tomorrow_reminder not in {r['id'] for r in result['items']}
    assert result['timezone'] == zone


def test_awaiting_reply_requires_three_days_and_no_later_actual_contact(client, conn, clock):
    first = repo.create_person(conn, '等了三天', status='active')
    second = repo.create_person(conn, '尚未三天', status='active')
    third = repo.create_person(conn, '后来联系过', status='active')
    fid, _, cid = waiting(conn, first, clock - 3 * agenda.DAY_MS)
    waiting(conn, second, clock - 3 * agenda.DAY_MS + 1)
    waiting(conn, third, clock - 4 * agenda.DAY_MS)
    contact(conn, third, clock - agenda.DAY_MS)
    conn.commit()
    result = items(client)
    assert len(result) == 1 and result[0]['id'] == 'awaiting:' + fid
    assert result[0]['followup_id'] == fid
    assert result[0]['source']['kind'] == 'contact' and result[0]['source']['id'] == cid
    assert '3 天' in result[0]['reason']


def test_explicit_followup_reminder_precedes_awaiting_and_replaced_or_completed_reminders_are_hidden(client, conn, pid, clock):
    fid, oid, _ = waiting(conn, pid, clock - 5 * agenda.DAY_MS)
    old = repo.add_reminder(conn, clock - agenda.DAY_MS, '已被替换的旧提醒', pid)
    current = repo.add_reminder(conn, clock + 3600000, '今天的新提醒', pid)
    conn.execute("UPDATE reminder SET status='fired' WHERE id=?", (old,))
    conn.execute('UPDATE followup_outcome SET reminder_id=? WHERE id=?', (old, oid))
    conn.execute('UPDATE followup SET reminder_id=? WHERE id=?', (current, fid))
    conn.commit()
    assert [r['id'] for r in items(client)] == ['reminder:' + current]
    conn.execute("UPDATE followup SET status='completed' WHERE id=?", (fid,))
    conn.commit()
    assert items(client) == []


def test_future_followup_reminder_defers_waiting_without_showing_future_date(client, conn, pid, clock):
    fid, _, _ = waiting(conn, pid, clock - 5 * agenda.DAY_MS)
    rid = repo.add_reminder(conn, clock + 2 * agenda.DAY_MS, '后天跟进', pid)
    conn.execute('UPDATE followup SET reminder_id=? WHERE id=?', (rid, fid))
    conn.commit()
    assert items(client) == []
    conn.execute("UPDATE reminder SET status='cancelled' WHERE id=?", (rid,))
    conn.commit()
    assert [r['id'] for r in items(client)] == ['awaiting:' + fid]


@pytest.mark.parametrize('scenario,suppressed', [
    ('current', True), ('ordinary', False), ('cancelled', False),
    ('completed', False), ('superseded', False),
])
def test_only_current_future_followup_schedule_defers_periodic_contact(client, conn, pid, clock,
                                                                      scenario, suppressed):
    fid, oid, _ = waiting(conn, pid, clock - 31 * agenda.DAY_MS)
    # A newer actual contact resolves waiting, while still being old enough for a periodic nudge.
    contact(conn, pid, clock - 30 * agenda.DAY_MS)
    rid = repo.add_reminder(conn, clock + 2 * agenda.DAY_MS, '后天联系', pid)
    if scenario != 'ordinary':
        conn.execute('UPDATE followup SET reminder_id=? WHERE id=?', (rid, fid))
        conn.execute('UPDATE followup_outcome SET reminder_id=? WHERE id=?', (rid, oid))
    if scenario == 'cancelled':
        conn.execute("UPDATE reminder SET status='cancelled' WHERE id=?", (rid,))
    elif scenario == 'completed':
        conn.execute("UPDATE followup SET status='completed' WHERE id=?", (fid,))
    elif scenario == 'superseded':
        conn.execute('UPDATE followup SET reminder_id=NULL WHERE id=?', (fid,))
    conn.commit()
    result = items(client)
    assert [item['id'] for item in result] == ([] if suppressed else ['contact:' + pid])


def test_explicit_follow_up_is_immediate_and_later_contact_does_not_complete_it(client, conn, pid, clock):
    fid, _, _ = waiting(conn, pid, clock - 1000, state='follow_up')
    contact(conn, pid, clock, text='又聊了别的话题')
    conn.commit()
    result = items(client)
    assert len(result) == 1 and result[0]['id'] == 'followup:' + fid
    assert result[0]['kind'] == 'follow_up' and result[0]['followup_id'] == fid
    assert result[0]['priority'] == 3 and result[0]['due_at'] is None
    assert '你已标记待跟进' in result[0]['reason']
    assert action(client, result[0], 'dismiss').status_code == 200
    conn.execute("UPDATE followup SET draft='改了草稿',revision=revision+1 WHERE id=?", (fid,))
    conn.commit()
    hidden = client.get('/api/agenda?include_hidden=true').json()
    assert hidden['items'] == [] and hidden['hidden_items'][0]['fingerprint'] == result[0]['fingerprint']
    assert action(client, result[0], 'restore').status_code == 200


def test_explicit_follow_up_obeys_current_reminder_schedule(client, conn, pid, clock):
    fid, _, _ = waiting(conn, pid, clock - 1000, state='follow_up')
    rid = repo.add_reminder(conn, clock + agenda.DAY_MS, '明天继续跟进', pid)
    conn.execute('UPDATE followup SET reminder_id=? WHERE id=?', (rid, fid))
    conn.commit()
    assert items(client) == []
    conn.execute('UPDATE reminder SET fire_at=? WHERE id=?', (clock + 60000, rid))
    conn.commit()
    assert [r['id'] for r in items(client)] == ['reminder:' + rid]
    conn.execute("UPDATE reminder SET status='cancelled' WHERE id=?", (rid,))
    conn.commit()
    assert [r['id'] for r in items(client)] == ['followup:' + fid]


def test_closed_promise_reminders_are_excluded_even_if_pending_or_fired(client, conn, pid, clock):
    for status in ('done', 'cancelled', 'expired'):
        pmid = repo.add_promise(conn, 'me', status, pid, clock - agenda.DAY_MS)
        repo.add_reminder(conn, clock - 1000, status, pid, pmid)
        conn.execute('UPDATE promise SET status=? WHERE id=?', (status, pmid))
    conn.execute("UPDATE reminder SET status='fired' WHERE text='done'")
    conn.commit()
    assert items(client) == []


def test_periodic_uses_only_actual_participation_and_person_policy(client, conn, pid, clock):
    cid = contact(conn, pid, clock - 14 * agenda.DAY_MS)
    contact(conn, pid, clock - 1000, role='mentioned')
    conn.commit()
    result = items(client)
    assert len(result) == 1 and result[0]['id'] == 'contact:' + pid
    assert result[0]['kind'] == 'contact_gap'
    assert result[0]['occurred_at'] == clock - 14 * agenda.DAY_MS
    assert result[0]['source']['id'] == cid and result[0]['due_at'] == clock
    assert client.get(f'/api/persons/{pid}/contact-policy').json() == {
        'person_id': pid, 'enabled': True, 'interval_days': 14, 'revision': 0}
    policy = {'enabled': True, 'interval_days': 20, 'expected_revision': 0}
    assert client.patch(f'/api/persons/{pid}/contact-policy', json=policy).status_code == 200
    assert items(client) == []
    policy = {'enabled': False, 'interval_days': 1, 'expected_revision': 1}
    assert client.patch(f'/api/persons/{pid}/contact-policy', json=policy).status_code == 200
    assert items(client) == []
    due = repo.add_promise(conn, 'me', '关闭周期仍需完成的承诺', pid, clock)
    conn.commit()
    assert [r['id'] for r in items(client)] == ['promise:' + due]


def test_other_current_items_prevent_periodic_even_when_dismissed(client, conn, pid, clock):
    contact(conn, pid, clock - 30 * agenda.DAY_MS)
    repo.add_promise(conn, 'me', '今天承诺', pid, clock)
    conn.commit()
    result = items(client)
    assert len(result) == 1 and result[0]['kind'] == 'promise'
    assert action(client, result[0], 'dismiss').status_code == 200
    assert items(client) == []


def test_snooze_dismiss_restore_do_not_mutate_sources_and_expiry_reactivates(client, conn, pid, clock, monkeypatch):
    pmid = repo.add_promise(conn, 'me', '发资料', pid, clock - agenda.DAY_MS)
    rid = repo.add_reminder(conn, clock, '独立提醒', pid)
    conn.commit()
    promise, reminder = items(client)
    assert action(client, promise, 'snooze', until=clock + 60000).json() == {'ok': True}
    assert action(client, reminder, 'dismiss').json() == {'ok': True}
    response = client.get('/api/agenda?include_hidden=true').json()
    assert response['items'] == []
    assert response['counts'] == {'active': 0, 'snoozed': 1, 'dismissed': 1, 'unknown': 1}
    assert response['hidden_items'][0]['disposition'] == {'state': 'snoozed', 'until': clock + 60000}
    assert response['hidden_items'][1]['disposition'] == {'state': 'dismissed', 'until': None}
    assert conn.execute('SELECT status FROM promise WHERE id=?', (pmid,)).fetchone()[0] == 'open'
    assert conn.execute('SELECT status FROM reminder WHERE id=?', (rid,)).fetchone()[0] == 'pending'
    assert conn.execute('SELECT COUNT(*) FROM interaction').fetchone()[0] == 0
    monkeypatch.setattr(agenda, 'now_ms', lambda: clock + 60000)
    assert [r['id'] for r in items(client)] == [promise['id']]
    assert action(client, reminder, 'restore').status_code == 200
    assert len(items(client)) == 2


def test_fingerprint_ignores_time_delivery_state_and_draft_edits(client, conn, pid, clock, monkeypatch):
    fid, _, _ = waiting(conn, pid, clock - 5 * agenda.DAY_MS)
    rid = repo.add_reminder(conn, clock - 1000, '今天跟进', pid)
    conn.execute('UPDATE followup SET reminder_id=? WHERE id=?', (rid, fid))
    conn.commit()
    original = items(client)[0]
    assert action(client, original, 'dismiss').status_code == 200
    conn.execute("UPDATE reminder SET status='fired',fired_at=? WHERE id=?", (clock, rid))
    conn.execute("UPDATE followup SET draft='改后的草稿',revision=revision+1,updated_at=? WHERE id=?", (clock, fid))
    conn.commit()
    monkeypatch.setattr(agenda, 'now_ms', lambda: clock + agenda.DAY_MS)
    response = client.get('/api/agenda?include_hidden=true').json()
    assert response['items'] == []
    assert response['hidden_items'][0]['fingerprint'] == original['fingerprint']


def test_changed_source_or_real_contact_reappears_and_rejects_stale_actions(client, conn, pid, clock):
    pmid = repo.add_promise(conn, 'me', '原承诺', pid, clock)
    conn.commit()
    original = items(client)[0]
    action(client, original, 'dismiss')
    conn.execute('UPDATE promise SET content=? WHERE id=?', ('改后的承诺', pmid))
    conn.commit()
    updated = items(client)[0]
    assert updated['fingerprint'] != original['fingerprint']
    assert action(client, original, 'restore').status_code == 409
    action(client, updated, 'dismiss')
    contact(conn, pid, clock - 1000)
    conn.commit()
    latest = items(client)[0]
    assert latest['fingerprint'] not in {original['fingerprint'], updated['fingerprint']}


def test_completed_source_is_stale_and_archived_person_actions_are_hidden(client, conn, pid, clock):
    pmid = repo.add_promise(conn, 'me', '原承诺', pid, clock)
    conn.commit()
    original = items(client)[0]
    repo.set_promise_status(conn, pmid, 'done')
    conn.commit()
    assert action(client, original, 'dismiss').status_code == 409
    repo.archive_person(conn, pid)
    conn.commit()
    assert action(client, original, 'dismiss').status_code == 404
    assert client.get(f'/api/persons/{pid}/contact-policy').status_code == 404
    assert client.patch(f'/api/persons/{pid}/contact-policy', json={
        'enabled': True, 'interval_days': 7, 'expected_revision': 0}).status_code == 404
    response = client.get('/api/agenda?include_hidden=true').json()
    assert response['items'] == response['unknown_people'] == response['hidden_items'] == []


@pytest.mark.parametrize('until', [None, True, 0, -1, '2026-10-17', 999999999999999])
def test_invalid_snoozes_never_persist(client, conn, pid, clock, until):
    repo.add_promise(conn, 'me', '今天承诺', pid, clock)
    conn.commit()
    original = items(client)[0]
    assert action(client, original, 'snooze', until=until).status_code in {400, 422}
    assert conn.execute('SELECT COUNT(*) FROM agenda_action').fetchone()[0] == 0


def test_snooze_time_is_validated_after_waiting_for_write_lock(client, conn, pid, clock, monkeypatch):
    repo.add_promise(conn, 'me', '今天承诺', pid, clock)
    conn.commit()
    item = items(client)[0]
    original_write = agenda._write
    @contextmanager
    def delayed_write(connection):
        with original_write(connection):
            monkeypatch.setattr(agenda, 'now_ms', lambda: clock + 1000)
            yield
    monkeypatch.setattr(agenda, '_write', delayed_write)
    assert action(client, item, 'snooze', until=clock + 500).status_code == 400
    assert conn.execute('SELECT COUNT(*) FROM agenda_action').fetchone()[0] == 0


@pytest.mark.parametrize('patch', [
    {'enabled': 1, 'interval_days': 14, 'expected_revision': 0},
    {'enabled': True, 'interval_days': True, 'expected_revision': 0},
    {'enabled': True, 'interval_days': 0, 'expected_revision': 0},
    {'enabled': True, 'interval_days': 366, 'expected_revision': 0},
    {'enabled': True, 'interval_days': 14, 'expected_revision': True},
])
def test_policy_rejects_invalid_types_and_limits(client, conn, pid, patch):
    assert client.patch(f'/api/persons/{pid}/contact-policy', json=patch).status_code == 422
    assert conn.execute('SELECT COUNT(*) FROM contact_policy').fetchone()[0] == 0


def test_contact_policy_requires_matching_revision_and_preserves_current_value(client, conn, pid):
    url = f'/api/persons/{pid}/contact-policy'
    body = {'enabled': False, 'interval_days': 30, 'expected_revision': 0}
    response = client.patch(url, json=body)
    assert response.json() == {'person_id': pid, 'enabled': False, 'interval_days': 30, 'revision': 1}
    assert client.patch(url, json={**body, 'enabled': True}).status_code == 409
    assert client.get(url).json() == response.json()


def test_get_agenda_does_not_persist_any_derived_state(client, conn, pid, clock):
    repo.add_promise(conn, 'me', '今天承诺', pid, clock)
    conn.commit()
    before = conn.total_changes
    assert items(client)
    assert client.get('/api/agenda?include_hidden=true').status_code == 200
    assert conn.total_changes == before
    assert conn.execute('SELECT COUNT(*) FROM contact_policy').fetchone()[0] == 0
