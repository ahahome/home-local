"""Source-bound proposals, full-history review and atomic confirmed writes."""
from copy import deepcopy
from datetime import datetime
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import followup, outcome_memory as memory
from app.config import tz
from app.jobs import tasks
from app.store import repo
from app.util import new_id, now_ms

DAY = 86400000
NOTES = '他已经搬到深圳，现在在星河公司任职，喜欢周末骑车。'


@pytest.fixture
def client(conn, monkeypatch):
    from app.api import outcome_memory
    app = FastAPI()
    app.include_router(outcome_memory.router)
    monkeypatch.setattr(outcome_memory, 'get_conn', lambda: conn)
    monkeypatch.setattr(tasks, 'start', lambda task_id: None)
    result = TestClient(app)
    yield result
    result.close()


def make_outcome(conn, notes=NOTES, contacted_at=None, promises=None):
    pid = repo.create_person(conn, '林舟', status='active')
    fid, stamp = new_id(), now_ms()
    conn.execute('INSERT INTO followup(id,person_id,origin_person_id,goal,status,draft,prepare_status,'
        'request_id,request_hash,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
        (fid, pid, pid, '了解近况', 'ready', '最近怎么样？', 'done', new_id(), 'test', stamp, stamp))
    conn.commit()
    body = {'request_id': new_id(), 'expected_revision': 1,
        'contacted_at': contacted_at or stamp - DAY, 'channel': '微信', 'notes': notes,
        'state': 'completed', 'next_followup_at': None, 'complete_promise_ids': [],
        'new_promises': promises or []}
    record = followup.record_outcome(conn, fid, body)['record']
    return SimpleNamespace(pid=pid, fid=fid, oid=record['outcomes'][0]['id'],
                           contacted_at=body['contacted_at'])


def candidate(record, **changes):
    return {'person_id': record.pid, 'kind': 'field', 'field': 'city', 'content': '深圳',
            'memory_kind': None, 'valid_from': None, 'evidence': '他已经搬到深圳', **changes}


def fake_model(monkeypatch, conn, proposals, during=None):
    calls = []
    def chat_json(messages, **kwargs):
        assert not conn.in_transaction, 'model calls must not hold the SQLite write lock'
        calls.append(json.loads(messages[-1]['content']))
        if during:
            during()
        return deepcopy(proposals)
    monkeypatch.setattr(memory, 'get_llm', lambda: SimpleNamespace(chat_json=chat_json))
    return calls


def analyze(conn, client, record, monkeypatch, proposals=None, during=None):
    task = client.post(f'/api/followup-outcomes/{record.oid}/memory-analysis', json={})
    assert task.status_code == 202, task.text
    calls = fake_model(monkeypatch, conn,
        {'proposals': proposals if proposals is not None else [candidate(record)]}, during)
    result = memory.run_analysis(conn, record.oid)
    return task.json(), result['proposals'], calls


def resolve(client, proposal, action='confirm', **changes):
    return client.post(f"/api/memory-updates/{proposal['id']}/resolve", json={
        'action': action, 'expected_revision': proposal['revision'], **changes})


def counts(conn):
    return {table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            for table in ('person_field', 'memory_item', 'interaction', 'promise', 'pending_item')}


def test_analysis_is_atomic_durable_idempotent_and_provenance_is_original_notes(client, conn, monkeypatch):
    record = make_outcome(conn, promises=[{'who': 'me', 'content': '发送材料', 'due_date': None}])
    before = counts(conn)
    task, proposals, calls = analyze(conn, client, record, monkeypatch)
    assert len(calls) == len(proposals) == 1
    assert counts(conn) == {**before, 'pending_item': before['pending_item'] + 1}
    assert set(calls[0]) == {'person', 'person_id', 'notes', 'contacted_at', 'anchor_date'}
    analysis = conn.execute('SELECT * FROM outcome_memory_analysis').fetchone()
    evidence = conn.execute('SELECT * FROM evidence WHERE id=?', (analysis['evidence_id'],)).fetchone()
    assert evidence['excerpt'] == NOTES
    assert evidence['source_ref'] == 'followup-outcome:' + record.oid
    assert client.post(f'/api/followup-outcomes/{record.oid}/memory-analysis', json={}).json()['id'] == task['id']
    assert memory.run_analysis(conn, record.oid)['proposals'] == proposals
    assert len(calls) == 1


def test_creation_rolls_back_evidence_task_and_events_when_analysis_insert_fails(client, conn):
    record = make_outcome(conn)
    conn.execute("CREATE TRIGGER fail_analysis BEFORE INSERT ON outcome_memory_analysis BEGIN "
                 "SELECT RAISE(ABORT,'analysis storage unavailable'); END")
    conn.commit()
    before = {table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
              for table in ('task', 'task_event', 'evidence')}
    with pytest.raises(sqlite3.IntegrityError):
        client.post(f'/api/followup-outcomes/{record.oid}/memory-analysis', json={})
    assert before == {table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in before}


def test_model_runs_outside_lock_but_late_field_change_creates_reviewable_conflict(client, conn, monkeypatch):
    record = make_outcome(conn)
    def change():
        repo.upsert_field(conn, record.pid, 'city', '北京', 1, None)
        conn.commit()
    _, proposals, _ = analyze(conn, client, record, monkeypatch, during=change)
    proposal = proposals[0]
    assert proposal['original'] is None and proposal['conflict']
    assert proposal['current']['value'] == '北京'
    assert resolve(client, proposal).status_code == 409


def test_complete_history_change_conflicts_even_when_current_row_is_identical(client, conn, monkeypatch):
    record = make_outcome(conn)
    repo.upsert_field(conn, record.pid, 'city', '上海', 1, None, record.contacted_at + 1000)
    conn.commit()
    _, proposals, _ = analyze(conn, client, record, monkeypatch)
    current = repo.current_fields(conn, record.pid)['city']
    repo.upsert_field(conn, record.pid, 'city', '杭州', 1, None, record.contacted_at - DAY)
    conn.commit()
    assert repo.current_fields(conn, record.pid)['city'] == current
    assert resolve(client, proposals[0]).status_code == 409
    refreshed = client.get('/api/memory-updates/' + proposals[0]['id']).json()
    assert refreshed['conflict'] and len(refreshed['current_timeline']) == 2


def test_rebase_keeps_pending_then_edited_confirmation_records_user_provenance(client, conn, monkeypatch):
    record = make_outcome(conn)
    _, proposals, _ = analyze(conn, client, record, monkeypatch)
    proposal = proposals[0]
    repo.upsert_field(conn, record.pid, 'city', '北京', 1, None)
    conn.commit()
    rebased = resolve(client, proposal, 'rebase')
    assert rebased.status_code == 200, rebased.text
    rebased = rebased.json()
    assert rebased['revision'] == 1 and rebased['status'] == 'pending'
    assert rebased['original']['value'] == '北京' and not rebased['conflict']
    confirmed = resolve(client, rebased, content='广州')
    assert confirmed.status_code == 200, confirmed.text
    receipt = confirmed.json()['receipt']
    evidence = conn.execute('SELECT * FROM evidence WHERE id=?', (receipt['evidence_id'],)).fetchone()
    edited = json.loads(evidence['excerpt'])
    assert edited['edited_by_user'] is True
    assert edited['original_quote'] == '他已经搬到深圳'
    assert edited['user_confirmed']['content'] == '广州'
    assert evidence['source_ref'] == 'outcome-memory-review:' + proposal['id']
    assert repo.current_fields(conn, record.pid)['city']['value'] == '北京'


@pytest.mark.parametrize('action', ['confirm', 'discard'])
def test_resolve_replay_is_idempotent_and_conflicting_terminal_action_fails(client, conn, monkeypatch, action):
    record = make_outcome(conn)
    _, proposals, calls = analyze(conn, client, record, monkeypatch)
    proposal = proposals[0]
    first = resolve(client, proposal, action)
    assert first.status_code == 200, first.text
    before = counts(conn)
    assert resolve(client, proposal, action).json() == first.json()
    assert counts(conn) == before
    other = 'confirm' if action == 'discard' else 'discard'
    assert resolve(client, proposal, other).status_code == 409
    repeated = memory.run_analysis(conn, record.oid)
    assert len(calls) == len(repeated['proposals']) == 1
    assert repeated['proposals'][0]['status'] == ('confirmed' if action == 'confirm' else 'discarded')


def test_zero_proposals_marks_analysis_complete_and_retries_do_not_call_model(client, conn, monkeypatch):
    record = make_outcome(conn)
    _, proposals, calls = analyze(conn, client, record, monkeypatch, proposals=[])
    assert proposals == []
    assert memory.run_analysis(conn, record.oid)['proposal_count'] == 0
    assert len(calls) == 1
    assert conn.execute('SELECT analyzed_at FROM outcome_memory_analysis').fetchone()[0] is not None


def test_same_day_outcome_uses_actual_contact_time_after_previous_field(client, conn, monkeypatch):
    stamp = int(datetime.fromisoformat('2026-01-10T15:00:00+08:00').timestamp() * 1000)
    record = make_outcome(conn, contacted_at=stamp)
    repo.upsert_field(conn, record.pid, 'city', '北京', 1, None, stamp - 3600000)
    conn.commit()
    _, proposals, _ = analyze(conn, client, record, monkeypatch)
    assert resolve(client, proposals[0]).status_code == 200
    field = repo.current_fields(conn, record.pid)['city']
    assert field['value'] == '深圳' and field['valid_from'] == stamp


def test_morning_outcome_precedes_same_day_later_current_field_and_memory_timestamp(client, conn, monkeypatch):
    stamp = int(datetime.fromisoformat('2026-01-10T09:00:00+08:00').timestamp() * 1000)
    record = make_outcome(conn, contacted_at=stamp)
    repo.upsert_field(conn, record.pid, 'city', '北京', 1, None, stamp + 3600000)
    conn.commit()
    _, proposals, _ = analyze(conn, client, record, monkeypatch, proposals=[candidate(record),
        candidate(record, kind='memory', field=None, content='喜欢周末骑车',
                  memory_kind='semantic', evidence='喜欢周末骑车')])
    for proposal in proposals:
        assert resolve(client, proposal).status_code == 200
    assert repo.current_fields(conn, record.pid)['city']['value'] == '北京'
    assert repo.field_at(conn, record.pid, 'city', stamp)['value'] == '深圳'
    assert repo.memories_of(conn, record.pid)[0]['ts'] == stamp


def test_date_defaults_use_contact_local_day_at_midnight_boundary(client, conn, monkeypatch):
    stamp = int(datetime.fromisoformat('2026-01-10T00:01:00+08:00').timestamp() * 1000)
    record = make_outcome(conn, contacted_at=stamp)
    _, proposals, calls = analyze(conn, client, record, monkeypatch)
    expected = datetime.fromtimestamp(stamp / 1000, tz()).date().isoformat()
    assert proposals[0]['valid_from'] == calls[0]['anchor_date'] == expected
    assert proposals[0]['contacted_at'] == stamp and proposals[0]['max_valid_from'] == expected


def test_two_reviewable_values_for_same_field_at_explicit_historical_dates(client, conn, monkeypatch):
    record = make_outcome(conn, notes='2026-01-01住在杭州，2026-02-01搬到深圳。')
    _, proposals, _ = analyze(conn, client, record, monkeypatch, proposals=[
        candidate(record, content='杭州', valid_from='2026-01-01', evidence='2026-01-01住在杭州'),
        candidate(record, valid_from='2026-02-01', evidence='2026-02-01搬到深圳')])
    assert len(proposals) == 2
    first, second = sorted(proposals, key=lambda p: p['valid_from'])
    assert resolve(client, first).status_code == 200
    assert resolve(client, second).status_code == 409
    second = resolve(client, second, 'rebase').json()
    assert resolve(client, second).status_code == 200
    assert [row['value'] for row in repo.field_history(conn, record.pid, 'city')] == ['杭州', '深圳']


def test_month_day_in_quote_uses_original_contact_year_not_analysis_year(client, conn, monkeypatch):
    stamp = int(datetime.fromisoformat('2025-11-10T15:00:00+08:00').timestamp() * 1000)
    record = make_outcome(conn, contacted_at=stamp, notes='我在10月10日搬到了深圳。')
    _, proposals, _ = analyze(conn, client, record, monkeypatch, proposals=[
        candidate(record, evidence='我在10月10日搬到了深圳', valid_from='2025-10-10')])
    assert proposals[0]['valid_from'] == '2025-10-10'
    assert resolve(client, proposals[0]).status_code == 200


def test_current_communication_preference_with_future_wording_is_valid_semantic_memory(client, conn, monkeypatch):
    notes = '以后联系我请优先发邮件，我习惯上午沟通。'
    record = make_outcome(conn, notes=notes)
    _, proposals, _ = analyze(conn, client, record, monkeypatch, proposals=[candidate(record,
        kind='memory', field=None, content='以后联系我请优先发邮件', memory_kind='semantic',
        evidence='以后联系我请优先发邮件')])
    assert resolve(client, proposals[0]).status_code == 200
    assert repo.memories_of(conn, record.pid)[0]['text'] == '以后联系我请优先发邮件'


def test_birthday_month_day_after_contact_is_profile_data_not_a_future_event(client, conn, monkeypatch):
    stamp = int(datetime.fromisoformat('2026-01-10T15:00:00+08:00').timestamp() * 1000)
    record = make_outcome(conn, contacted_at=stamp, notes='他的生日是11月10日。')
    _, proposals, _ = analyze(conn, client, record, monkeypatch, proposals=[candidate(record,
        field='birthday', content='11月10日', evidence='他的生日是11月10日')])
    assert proposals[0]['valid_from'] == '2026-01-10'
    assert resolve(client, proposals[0]).status_code == 200
    assert repo.current_fields(conn, record.pid)['birthday']['value'] == '11月10日'


@pytest.mark.parametrize('operation', ['edit', 'remove'])
def test_evidence_changes_mark_proposal_stale_and_prevent_confirmation(client, conn, monkeypatch, operation):
    record = make_outcome(conn)
    _, proposals, _ = analyze(conn, client, record, monkeypatch)
    evidence_id = conn.execute('SELECT evidence_id FROM outcome_memory_analysis').fetchone()[0]
    if operation == 'edit':
        conn.execute('UPDATE evidence SET excerpt=? WHERE id=?', ('被改变的来源', evidence_id))
    else:
        conn.execute('DELETE FROM evidence WHERE id=?', (evidence_id,))
    conn.commit()
    assert memory.read_proposal(conn, proposals[0]['id'])['stale']
    assert resolve(client, proposals[0]).status_code == 409
    assert repo.current_fields(conn, record.pid) == {}


def test_duplicate_candidates_deduplicate_without_losing_different_historical_dates(client, conn, monkeypatch):
    record = make_outcome(conn)
    _, proposals, _ = analyze(conn, client, record, monkeypatch, proposals=[candidate(record),
        candidate(record, evidence='已经搬到深圳')])
    assert len(proposals) == 1


@pytest.mark.parametrize('changes', [
    {'person_id': 'foreign-id'}, {'kind': 'promise'}, {'field': 'status'},
    {'memory_kind': 'semantic'}, {'content': 42}, {'content': True}, {'content': '推测出来的值'},
    {'evidence': '他已经\n搬到深圳'}, {'evidence': '   '}, {'content': ' '},
    {'valid_from': '2099-01-01'}, {'valid_from': '2026-1-1'}, {'valid_from': ''},
    {'valid_from': '2020-01-01'}, {'confidence': 1},
])
def test_invalid_model_rejects_entire_batch_without_any_proposals(client, conn, monkeypatch, changes):
    record = make_outcome(conn)
    client.post(f'/api/followup-outcomes/{record.oid}/memory-analysis', json={})
    fake_model(monkeypatch, conn, {'proposals': [candidate(record), candidate(record, **changes)]})
    before = counts(conn)
    with pytest.raises(memory.OutcomeMemoryError):
        memory.run_analysis(conn, record.oid)
    assert counts(conn) == before
    assert conn.execute('SELECT analyzed_at FROM outcome_memory_analysis').fetchone()[0] is None


@pytest.mark.parametrize('notes', ['他打算搬到深圳。', '他答应搬到深圳。', '他转账给深圳的朋友。',
                                   '他2027-01-01搬到深圳。'])
def test_future_promises_and_ledger_content_cannot_become_memory(client, conn, monkeypatch, notes):
    record = make_outcome(conn, notes=notes)
    client.post(f'/api/followup-outcomes/{record.oid}/memory-analysis', json={})
    fake_model(monkeypatch, conn, {'proposals': [candidate(record, evidence=notes)]})
    with pytest.raises(memory.OutcomeMemoryError):
        memory.run_analysis(conn, record.oid)
    assert repo.list_pending(conn) == []


@pytest.mark.parametrize('body', [
    {'action': 'confirm', 'expected_revision': True},
    {'action': 'confirm', 'expected_revision': '0'},
    {'action': 'confirm', 'expected_revision': -1},
    {'action': 'confirm', 'expected_revision': 0, 'content': 123},
    {'action': 'confirm', 'expected_revision': 0, 'content': None},
    {'action': 'confirm', 'expected_revision': 0, 'valid_from': 123},
    {'action': 'rebase', 'expected_revision': 0, 'content': '北京'},
    {'action': 'discard', 'expected_revision': 0, 'valid_from': '2026-01-01'},
    {'action': 'confirm', 'expected_revision': 0, 'person_id': 'foreign'},
])
def test_resolution_http_contract_is_strict(client, conn, monkeypatch, body):
    record = make_outcome(conn)
    _, proposals, _ = analyze(conn, client, record, monkeypatch)
    assert client.post('/api/memory-updates/' + proposals[0]['id'] + '/resolve', json=body).status_code == 422
    assert repo.current_fields(conn, record.pid) == {}


def test_future_manual_date_is_rejected_but_explicit_manual_backdate_is_allowed(client, conn, monkeypatch):
    record = make_outcome(conn)
    _, proposals, _ = analyze(conn, client, record, monkeypatch)
    assert resolve(client, proposals[0], valid_from='2099-01-01').status_code == 400
    confirmed = resolve(client, proposals[0], valid_from='2020-01-01')
    assert confirmed.status_code == 200, confirmed.text
    evidence = conn.execute('SELECT excerpt FROM evidence WHERE id=?',
                            (confirmed.json()['receipt']['evidence_id'],)).fetchone()[0]
    assert json.loads(evidence)['edited_by_user']


def test_cancel_during_model_prevents_proposals_and_retry_can_resume(client, conn, monkeypatch):
    record = make_outcome(conn)
    task = client.post(f'/api/followup-outcomes/{record.oid}/memory-analysis', json={}).json()
    def cancel():
        conn.execute('UPDATE task SET cancel_requested=1 WHERE id=?', (task['id'],))
        conn.commit()
    fake_model(monkeypatch, conn, {'proposals': [candidate(record)]}, during=cancel)
    with pytest.raises(tasks.Cancelled):
        memory.run_analysis(conn, record.oid)
    assert not repo.list_pending(conn)
    conn.execute("UPDATE task SET status='cancelled' WHERE id=?", (task['id'],))
    conn.commit()
    tasks.retry(conn, task['id'])
    fake_model(monkeypatch, conn, {'proposals': [candidate(record)]})
    assert memory.run_analysis(conn, record.oid)['proposal_count'] == 1


@pytest.mark.parametrize('kind', ['field', 'memory'])
def test_confirmation_storage_failure_rolls_back_writes_evidence_and_resolution(client, conn, monkeypatch, kind):
    record = make_outcome(conn)
    proposal = candidate(record) if kind == 'field' else candidate(record, kind='memory', field=None,
        content='喜欢周末骑车', memory_kind='semantic', evidence='喜欢周末骑车')
    _, proposals, _ = analyze(conn, client, record, monkeypatch, proposals=[proposal])
    conn.execute("CREATE TRIGGER fail_resolution BEFORE UPDATE OF resolved_at ON pending_item BEGIN "
                 "SELECT RAISE(ABORT,'pending storage unavailable'); END")
    conn.commit()
    before = counts(conn)
    evidence_count = conn.execute('SELECT COUNT(*) FROM evidence').fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        resolve(client, proposals[0], content='用户修改的内容')
    assert counts(conn) == before
    assert conn.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == evidence_count
    assert repo.get_pending(conn, proposals[0]['id'])['resolved_at'] is None
    assert memory.read_proposal(conn, proposals[0]['id'])['revision'] == 0


def test_proposal_batch_failure_rolls_back_every_candidate_and_completion_stamp(client, conn, monkeypatch):
    record = make_outcome(conn)
    client.post(f'/api/followup-outcomes/{record.oid}/memory-analysis', json={})
    fake_model(monkeypatch, conn, {'proposals': [candidate(record), candidate(record, field='company',
        content='星河公司', evidence='现在在星河公司任职')]})
    conn.execute("CREATE TRIGGER fail_second_proposal BEFORE INSERT ON pending_item "
        "WHEN json_extract(NEW.payload,'$.field')='company' BEGIN "
        "SELECT RAISE(ABORT,'proposal storage unavailable'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        memory.run_analysis(conn, record.oid)
    assert not repo.list_pending(conn)
    assert conn.execute('SELECT analyzed_at FROM outcome_memory_analysis').fetchone()[0] is None
