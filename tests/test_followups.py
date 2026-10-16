"""Preparation never records contact; explicit results are atomic and replay safe."""
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import followup
from app.jobs import tasks
from app.store import repo
from app.util import now_ms


@pytest.fixture
def client(conn, monkeypatch):
    from app.api import followups
    app = FastAPI()
    app.include_router(followups.router)
    monkeypatch.setattr(followups, 'get_conn', lambda: conn)
    monkeypatch.setattr(tasks, 'start', lambda tid: None)
    return TestClient(app)


@pytest.fixture
def pid(conn):
    person_id = repo.create_person(conn, '林舟', status='active')
    conn.commit()
    return person_id


def create(client, pid, **changes):
    return client.post(f'/api/persons/{pid}/followups', json={
        'goal': '聊聊近况并确认下次见面时间', 'request_id': 'prepare-1', **changes})


def task_fid(response):
    assert response.status_code == 202, response.text
    task = response.json()
    assert task['kind'] == 'followup_prepare'
    return task['id'], task['payload']['followup_id']


def fake_llm(monkeypatch, reply=None, action=None):
    seen = []
    def chat_json(messages, **kwargs):
        data = json.loads(messages[-1]['content'])
        seen.append(data)
        if action:
            action()
        return reply if reply is not None else {
            'draft': '最近怎么样？想找时间和你聊聊。',
            'source_ids': [s['id'] for s in data['brief']['sources'][:1]]}
    monkeypatch.setattr(followup, 'get_llm', lambda: SimpleNamespace(chat_json=chat_json))
    return seen


def ready(client, conn, pid, monkeypatch):
    tid, fid = task_fid(create(client, pid))
    fake_llm(monkeypatch)
    result = followup.prepare_followup(conn, fid)
    return tid, fid, result


def result_body(record, **changes):
    return {'request_id': 'outcome-1', 'expected_revision': record['revision'],
        'contacted_at': now_ms() - 60000, 'channel': '微信', 'notes': '实际聊过，约下周再确认时间。',
        'state': 'awaiting_reply', 'next_followup_at': None,
        'complete_promise_ids': [], 'new_promises': [], **changes}


def count(conn, table):
    return conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]


def test_create_is_atomic_idempotent_and_queues_durable_task(client, conn, pid):
    tid, fid = task_fid(create(client, pid))
    assert task_fid(create(client, pid)) == (tid, fid)
    assert create(client, pid, goal='另一个目标').status_code == 409
    assert count(conn, 'task') == count(conn, 'followup') == 1
    assert count(conn, 'interaction') == 0
    record = client.get(f'/api/followups/{fid}').json()
    assert record['status'] == 'preparing' and record['prepare_status'] == 'queued'
    assert client.get(f'/api/persons/{pid}/followups').json()['followups'][0]['id'] == fid


def test_create_rolls_back_task_if_followup_insert_fails(client, conn, pid):
    conn.execute("CREATE TRIGGER fail_prepare BEFORE INSERT ON followup BEGIN "
                 "SELECT RAISE(ABORT,'followup storage unavailable'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        create(client, pid)
    assert count(conn, 'followup') == count(conn, 'task') == count(conn, 'task_event') == 0


def test_worker_saves_ready_record_and_task_result(client, conn, pid, monkeypatch):
    tid, fid = task_fid(create(client, pid))
    fake_llm(monkeypatch)
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    tasks._run(tid)
    task = tasks.read(conn, tid)
    assert task['status'] == 'done', task['error']
    assert followup.read_followup(conn, fid)['status'] == 'ready'
    assert count(conn, 'interaction') == 0


def test_merge_and_undo_preserve_independent_request_scopes(client, conn, pid):
    from app.memory.apply import merge_persons, split_merge
    other = repo.create_person(conn, '另一位林舟', status='active')
    conn.commit()
    _, fid = task_fid(create(client, pid))
    _, second = task_fid(create(client, other))
    result = merge_persons(conn, pid, other)
    assert {r['id'] for r in followup.list_followups(conn, pid)} == {fid, second}
    assert task_fid(create(client, pid))[1] == fid
    split_merge(conn, result)
    assert [r['id'] for r in followup.list_followups(conn, pid)] == [fid]
    assert task_fid(create(client, other))[1] == second


def test_prepare_uses_exact_person_sources_and_never_writes_memory(client, conn, pid, monkeypatch):
    other = repo.create_person(conn, '林舟', status='active')
    eid = repo.add_evidence(conn, 'text', source_ref='source-A', excerpt='林舟在杭州工作')
    repo.upsert_field(conn, pid, 'city', '杭州', 1, eid)
    repo.upsert_field(conn, other, 'city', '不能混入的同名人物城市', 1, None)
    repo.insert_interaction(conn, now_ms() - 100000, 'manual', '导入材料不算联系', None,
                            [(pid, 'mentioned')])
    repo.insert_interaction(conn, now_ms() - 90000, '微信', '另一人联系中只提及林舟', None,
                            [(pid, 'mentioned'), (other, 'participant')], kind='contact')
    contact_id, _ = repo.insert_interaction(conn, now_ms() - 80000, '电话', '实际谈了杭州近况', None,
                                           [(pid, 'participant')], kind='contact')
    promise_id = repo.add_promise(conn, 'me', '发会议资料', pid, evidence_id=eid)
    repo.add_promise(conn, 'them', '其他人的承诺', other)
    repo.add_pending(conn, 'field', {'person_id': pid, 'field': 'city', 'value': '待确认城市'}, '待确认')
    repo.add_memory(conn, 'semantic', '偏好邮件沟通', pid, 1, eid)
    repo.add_memory(conn, 'episodic', '参加过杭州的会议', pid, 1, eid)
    repo.add_memory(conn, 'reflective', '模型猜测更喜欢电话', pid, 1, eid)
    repo.add_memory(conn, 'semantic', '没有证据的背景', pid, 1)
    repo.add_memory(conn, 'semantic', '另一个人的偏好', other, 1, eid)
    deleted_memory = repo.add_memory(conn, 'semantic', '已删除的背景', pid, 1, eid)
    conn.execute('UPDATE memory_item SET deleted_at=? WHERE id=?', (now_ms(), deleted_memory))
    conn.commit()
    before = {table: count(conn, table) for table in ('interaction', 'person_field', 'memory_item',
                                                   'promise', 'pending_item', 'evidence', 'reminder')}
    tid, fid = task_fid(create(client, pid))
    seen = fake_llm(monkeypatch)
    record = followup.prepare_followup(conn, fid)
    assert record['status'] == 'ready' and record['revision'] == 2
    brief = record['brief']
    assert brief['current_fields'][0]['value'] == '杭州'
    assert brief['last_contact']['id'] == contact_id
    assert len(brief['recent_contacts']) == 1
    assert [p['id'] for p in brief['open_promises']] == [promise_id]
    assert any(s['excerpt'] == '林舟在杭州工作' and s['source_ref'] == 'source-A' for s in brief['sources'])
    assert '不能混入' not in json.dumps(brief, ensure_ascii=False)
    assert '待确认城市' not in json.dumps(brief, ensure_ascii=False)
    assert '其他人的承诺' not in json.dumps(brief, ensure_ascii=False)
    assert {m['text'] for m in brief['memories']} == {'偏好邮件沟通', '参加过杭州的会议'}
    assert {s['kind'] for s in brief['sources']} >= {'field', 'contact', 'promise', 'memory'}
    assert brief['last_contact']['date'] == brief['last_contact']['ts']
    assert seen[0]['goal'] == record['goal']
    assert {table: count(conn, table) for table in before} == before
    assert followup.prepare_followup(conn, fid)['revision'] == 2
    assert len(seen) == 1


def test_unknown_draft_source_rejects_without_counting_contact(client, conn, pid, monkeypatch):
    _, fid = task_fid(create(client, pid))
    fake_llm(monkeypatch, {'draft': '你好', 'source_ids': ['field:nonexistent']})
    with pytest.raises(followup.FollowupError, match='来源'):
        followup.prepare_followup(conn, fid)
    assert followup.read_followup(conn, fid)['status'] == 'preparing'
    assert count(conn, 'interaction') == count(conn, 'promise') == 0


def test_outcome_unavailable_until_preparation_finishes(client, conn, pid):
    _, fid = task_fid(create(client, pid))
    record = followup.read_followup(conn, fid)
    assert client.post(f'/api/followups/{fid}/outcomes', json=result_body(record)).status_code == 409
    assert count(conn, 'interaction') == 0


def test_recording_actual_contact_does_not_extract_promises_from_notes(client, conn, pid, monkeypatch):
    _, fid, record = ready(client, conn, pid, monkeypatch)
    # Recording results must work without calling any model, including embeddings.
    import app.llm
    monkeypatch.setattr(app.llm, 'get_llm', lambda: pytest.fail('outcome must not call models'))
    body = result_body(record, notes='我答应下周发资料，对方说以后再谈。')
    response = client.post(f'/api/followups/{fid}/outcomes', json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    contact = conn.execute('SELECT i.*,ip.role FROM interaction i JOIN interaction_person ip '
                           'ON i.id=ip.interaction_id WHERE ip.person_id=?', (pid,)).fetchone()
    assert contact['kind'] == 'contact' and contact['role'] == 'participant'
    assert contact['ts'] == repo.last_contact(conn, pid) == body['contacted_at']
    assert result['record']['status'] == 'awaiting_reply'
    assert result['receipts'][0]['kind'] == 'contact'
    assert count(conn, 'promise') == count(conn, 'memory_item') == 0


def test_outcome_promises_and_followup_reminder_are_atomic_and_replay_safe(client, conn, pid, monkeypatch):
    old = repo.add_promise(conn, 'me', '旧承诺', pid)
    old_reminder = repo.add_reminder(conn, now_ms() + 60000, '旧承诺提醒', pid, old)
    conn.commit()
    _, fid, record = ready(client, conn, pid, monkeypatch)
    import app.llm
    monkeypatch.setattr(app.llm, 'get_llm', lambda: pytest.fail('explicit promises must not call models'))
    body = result_body(record, state='follow_up', next_followup_at=now_ms() + 86400000,
        complete_promise_ids=[old], new_promises=[{'who': 'them', 'content': '发送项目时间表',
                                                 'due_date': '2026-12-20'}])
    response = client.post(f'/api/followups/{fid}/outcomes', json=body)
    assert response.status_code == 200, response.text
    first = response.json()
    assert first['record']['status'] == 'follow_up'
    assert conn.execute('SELECT status FROM promise WHERE id=?', (old,)).fetchone()[0] == 'done'
    assert conn.execute('SELECT status FROM reminder WHERE id=?', (old_reminder,)).fetchone()[0] == 'cancelled'
    promises = repo.open_promises(conn, pid)
    assert len(promises) == 1 and promises[0]['content'] == '发送项目时间表'
    assert promises[0]['evidence_id'] is not None
    assert len(conn.execute("SELECT * FROM reminder WHERE status='pending'").fetchall()) == 2
    assert client.post(f'/api/followups/{fid}/outcomes', json=body).json() == first
    assert count(conn, 'interaction') == count(conn, 'followup_outcome') == 1
    assert count(conn, 'promise') == 2
    conflict = client.post(f'/api/followups/{fid}/outcomes', json={**body, 'notes': '不同内容'})
    assert conflict.status_code == 409


def test_cross_person_promise_and_stale_revision_reject_before_writes(client, conn, pid, monkeypatch):
    other = repo.create_person(conn, '林舟', status='active')
    pmid = repo.add_promise(conn, 'me', '另一位林舟的承诺', other)
    conn.commit()
    _, fid, record = ready(client, conn, pid, monkeypatch)
    assert client.post(f'/api/followups/{fid}/outcomes', json=result_body(record,
        complete_promise_ids=[pmid])).status_code == 409
    assert client.post(f'/api/followups/{fid}/outcomes', json=result_body(record,
        expected_revision=1)).status_code == 409
    assert count(conn, 'interaction') == count(conn, 'followup_outcome') == 0
    assert repo.open_promises(conn, other)[0]['id'] == pmid


def test_later_contact_can_use_same_notes_and_replaces_only_own_reminder(client, conn, pid, monkeypatch):
    unrelated = repo.add_reminder(conn, now_ms() + 900000, '无关提醒', pid)
    conn.commit()
    _, fid, record = ready(client, conn, pid, monkeypatch)
    body = result_body(record, next_followup_at=now_ms() + 86400000)
    first = client.post(f'/api/followups/{fid}/outcomes', json=body).json()['record']
    old_reminder = first['reminder_id']
    second = client.post(f'/api/followups/{fid}/outcomes', json={**body, 'request_id': 'outcome-2',
        'expected_revision': first['revision'], 'contacted_at': now_ms() - 10000,
        'state': 'completed', 'next_followup_at': None})
    assert second.status_code == 200, second.text
    record = second.json()['record']
    assert record['status'] == 'completed' and record['reminder_id'] is None
    assert len(record['outcomes']) == count(conn, 'interaction') == 2
    assert conn.execute('SELECT status FROM reminder WHERE id=?', (old_reminder,)).fetchone()[0] == 'cancelled'
    assert conn.execute('SELECT status FROM reminder WHERE id=?', (unrelated,)).fetchone()[0] == 'pending'
    assert client.post(f'/api/followups/{fid}/outcomes', json=result_body(record,
        request_id='outcome-3')).status_code == 409
    assert client.patch(f'/api/followups/{fid}/draft', json={'draft': '不能修改已结束记录',
        'expected_revision': record['revision']}).status_code == 409


@pytest.mark.parametrize('changes', [
    {'contacted_at': True}, {'contacted_at': -1}, {'contacted_at': 9999999999999},
    {'next_followup_at': 1}, {'state': 'completed', 'next_followup_at': 9999999999999},
    {'new_promises': [{'who': 'me', 'content': '资料', 'due_date': '2026-02-30'}]},
    {'new_promises': [{'who': 'me', 'content': '资料', 'due_date': '20261220'}]},
    {'new_promises': [{'who': 'other', 'content': '资料'}]},
    {'notes': ' '}, {'channel': ' '}, {'complete_promise_ids': ['x', 'x']},
])
def test_invalid_outcome_has_no_partial_writes(client, conn, pid, monkeypatch, changes):
    _, fid, record = ready(client, conn, pid, monkeypatch)
    response = client.post(f'/api/followups/{fid}/outcomes', json=result_body(record, **changes))
    assert response.status_code in {400, 422}, response.text
    assert count(conn, 'interaction') == count(conn, 'promise') == count(conn, 'reminder') == 0


def test_outcome_storage_failure_rolls_back_every_mutation(client, conn, pid, monkeypatch):
    pmid = repo.add_promise(conn, 'me', '既有承诺', pid)
    rid = repo.add_reminder(conn, now_ms() + 60000, '提醒', pid, pmid)
    conn.commit()
    _, fid, record = ready(client, conn, pid, monkeypatch)
    before_importance = repo.get_person(conn, pid)['importance']
    conn.execute("CREATE TRIGGER fail_outcome BEFORE INSERT ON followup_outcome BEGIN "
                 "SELECT RAISE(ABORT,'outcome storage unavailable'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        client.post(f'/api/followups/{fid}/outcomes', json=result_body(record,
            complete_promise_ids=[pmid], next_followup_at=now_ms() + 86400000,
            new_promises=[{'who': 'me', 'content': '新的承诺', 'due_date': '2026-12-20'}]))
    assert count(conn, 'interaction') == count(conn, 'followup_outcome') == count(conn, 'evidence') == 0
    assert count(conn, 'promise') == count(conn, 'reminder') == 1
    assert repo.open_promises(conn, pid)[0]['id'] == pmid
    assert conn.execute('SELECT status FROM reminder WHERE id=?', (rid,)).fetchone()[0] == 'pending'
    assert repo.get_person(conn, pid)['importance'] == before_importance
    assert followup.read_followup(conn, fid)['revision'] == record['revision']


def test_draft_edits_persist_and_reject_stale_revision_without_recording_contact(client, conn, pid, monkeypatch):
    _, fid, record = ready(client, conn, pid, monkeypatch)
    body = {'draft': '我改好的消息', 'expected_revision': record['revision']}
    response = client.patch(f'/api/followups/{fid}/draft', json=body)
    assert response.status_code == 200
    assert response.json()['draft'] == '我改好的消息'
    assert response.json()['revision'] == record['revision'] + 1
    assert client.patch(f'/api/followups/{fid}/draft', json=body).status_code == 409
    assert client.get(f'/api/followups/{fid}').json()['draft'] == '我改好的消息'
    assert count(conn, 'interaction') == 0


def test_archived_person_hidden_and_archive_during_generation_blocks_late_result(client, conn, pid, monkeypatch):
    tid, fid = task_fid(create(client, pid))
    def archive():
        repo.archive_person(conn, pid)
        conn.commit()
    fake_llm(monkeypatch, action=archive)
    with pytest.raises(followup.FollowupError) as exc:
        followup.prepare_followup(conn, fid)
    assert exc.value.status_code == 404
    assert conn.execute('SELECT status,draft FROM followup WHERE id=?', (fid,)).fetchone()[:] == ('preparing', '')
    assert client.get(f'/api/followups/{fid}').status_code == 404
    assert client.get(f'/api/persons/{pid}/followups').status_code == 404
    assert create(client, pid, request_id='prepare-2').status_code == 404
    assert count(conn, 'interaction') == 0


def test_cancel_during_generation_prevents_ready_and_retry_can_resume(client, conn, pid, monkeypatch):
    tid, fid = task_fid(create(client, pid))
    def cancel():
        conn.execute("UPDATE task SET cancel_requested=1,status='cancelled' WHERE id=?", (tid,))
        conn.commit()
    fake_llm(monkeypatch, action=cancel)
    with pytest.raises(tasks.Cancelled):
        followup.prepare_followup(conn, fid)
    followup.mark_prepare_failed(conn, fid, 'cancelled')
    assert followup.read_followup(conn, fid)['prepare_status'] == 'cancelled'
    tasks.retry(conn, tid)
    assert followup.read_followup(conn, fid)['prepare_status'] == 'queued'
    fake_llm(monkeypatch)
    assert followup.prepare_followup(conn, fid)['status'] == 'ready'
    assert count(conn, 'followup') == count(conn, 'task') == 1
    assert count(conn, 'interaction') == 0


def test_merge_during_generation_does_not_publish_stale_brief(client, conn, pid, monkeypatch):
    _, fid = task_fid(create(client, pid))
    other = repo.create_person(conn, '其他人物', status='active')
    conn.commit()
    def move():
        conn.execute('UPDATE followup SET person_id=? WHERE id=?', (other, fid))
        conn.commit()
    fake_llm(monkeypatch, action=move)
    with pytest.raises(followup.FollowupError, match='人物关联'):
        followup.prepare_followup(conn, fid)
    assert followup.read_followup(conn, fid)['status'] == 'preparing'
    assert count(conn, 'interaction') == 0


def test_purge_removes_followup_tasks_and_snapshot_events_without_touching_other_people(client, conn, pid, monkeypatch):
    other = repo.create_person(conn, '另一位林舟', status='active')
    evidence_id = repo.add_evidence(conn, 'text', excerpt='仅属于要清除人物的秘密资料')
    repo.upsert_field(conn, pid, 'city', '秘密城市', 1, evidence_id)
    conn.commit()
    tid, fid = task_fid(create(client, pid))
    other_tid, other_fid = task_fid(create(client, other))
    fake_llm(monkeypatch)
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    tasks._run(tid)
    assert '秘密城市' in json.dumps(tasks.read(conn, tid)['result'], ensure_ascii=False)
    assert conn.execute("SELECT 1 FROM task_event WHERE task_id=? AND kind='result'", (tid,)).fetchone()
    result = repo.purge_person(conn, pid)
    conn.commit()
    assert result['ok']
    assert not conn.execute('SELECT 1 FROM followup WHERE id=?', (fid,)).fetchone()
    assert not conn.execute('SELECT 1 FROM task WHERE id=?', (tid,)).fetchone()
    assert not conn.execute('SELECT 1 FROM task_event WHERE task_id=?', (tid,)).fetchone()
    assert tasks.read(conn, other_tid)['payload']['followup_id'] == other_fid
    assert client.get(f'/api/followups/{other_fid}').status_code == 200
    with pytest.raises(ValueError, match='任务不存在'):
        tasks.retry(conn, tid)


def test_purge_also_removes_orphaned_task_linked_only_by_payload(client, conn, pid):
    tid, fid = task_fid(create(client, pid))
    conn.execute('UPDATE followup SET task_id=NULL WHERE id=?', (fid,))
    conn.commit()
    assert repo.purge_person(conn, pid)['ok']
    conn.commit()
    assert count(conn, 'followup') == count(conn, 'task') == count(conn, 'task_event') == 0


def test_purge_during_generation_blocks_late_snapshot(client, conn, pid, monkeypatch):
    tid, fid = task_fid(create(client, pid))
    def purge():
        assert repo.purge_person(conn, pid)['ok']
        conn.commit()
    fake_llm(monkeypatch, action=purge)
    with pytest.raises(followup.FollowupError) as error:
        followup.prepare_followup(conn, fid)
    assert error.value.status_code == 404
    assert count(conn, 'person') == count(conn, 'followup') == count(conn, 'task') == 0
    assert count(conn, 'task_event') == count(conn, 'interaction') == 0


def test_archived_person_task_retry_never_recreates_contact_or_calls_model(client, conn, pid, monkeypatch):
    tid, fid = task_fid(create(client, pid))
    repo.archive_person(conn, pid)
    conn.execute("UPDATE task SET status='cancelled',cancel_requested=1 WHERE id=?", (tid,))
    conn.commit()
    tasks.retry(conn, tid)
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    monkeypatch.setattr(followup, 'get_llm', lambda: pytest.fail('archived preparation must not call a model'))
    tasks._run(tid)
    assert tasks.read(conn, tid)['status'] == 'failed'
    assert repo.get_person(conn, pid)['deleted_at'] is not None
    assert conn.execute('SELECT status,draft FROM followup WHERE id=?', (fid,)).fetchone()[:] == ('preparing', '')
    assert count(conn, 'interaction') == count(conn, 'promise') == 0


def test_worker_purged_during_model_call_returns_without_recreating_task_events(client, conn, pid, monkeypatch):
    tid, _ = task_fid(create(client, pid))
    def purge():
        assert repo.purge_person(conn, pid)['ok']
        conn.commit()
    fake_llm(monkeypatch, action=purge)
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    tasks._run(tid)
    assert count(conn, 'person') == count(conn, 'followup') == count(conn, 'task') == 0
    assert count(conn, 'task_event') == count(conn, 'interaction') == 0


def test_worker_retry_after_ready_with_saved_outcome_does_not_interpret_history_as_receipts(client, conn, pid, monkeypatch):
    tid, fid, record = ready(client, conn, pid, monkeypatch)
    response = client.post(f'/api/followups/{fid}/outcomes', json=result_body(record))
    assert response.status_code == 200
    saved = response.json()['record']
    assert saved['outcomes'][0]['state'] == 'awaiting_reply'
    assert 'status' not in saved['outcomes'][0]
    # Simulate a shutdown after preparation committed but before the worker saved success.
    conn.execute("UPDATE task SET status='interrupted' WHERE id=?", (tid,))
    conn.commit()
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    monkeypatch.setattr(followup, 'get_llm', lambda: pytest.fail('ready retry must reuse its saved draft'))
    tasks.retry(conn, tid)
    tasks._run(tid)
    task = tasks.read(conn, tid)
    assert task['status'] == 'done', task['error']
    assert task['result']['followup_id'] == fid
    assert task['result']['record']['outcomes'] == saved['outcomes']
    assert followup.read_followup(conn, fid)['revision'] == saved['revision']
    assert count(conn, 'interaction') == count(conn, 'followup_outcome') == 1
