"""Explicit query-only generation: SQLite provenance, idempotency and write guards."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import conversation_scope, harness, memory_sources, turn_intent
from app.jobs import chat_regeneration, tasks
from app.memory import memory_corrections
from app.store import repo
from app.util import new_id


@pytest.fixture
def client(conn, monkeypatch):
    from app.api import main, workspace
    monkeypatch.setattr(workspace, 'get_conn', lambda: conn)
    monkeypatch.setattr(main, 'get_conn', lambda: conn)
    monkeypatch.setattr(tasks, 'start', lambda tid: None)
    return TestClient(main.app)


def seed(conn, cid, text='顾念的沟通偏好有哪些？', intent='query', sources=()):
    uid = repo.add_message(conn, cid, 'user', text)
    scope = conversation_scope.for_user_message(conn, cid, uid, text)
    value = {'intent': intent, 'record_spans': [text] if intent in {'record', 'mixed'} else [], 'reason': 'fixture'}
    decision = turn_intent.for_user_message(conn, cid, uid, text,
        llm=SimpleNamespace(chat=lambda *args, **kwargs: {'content': json.dumps(value, ensure_ascii=False)}))
    data = {'answer_review': {'status': 'unavailable', 'unavailable_reason': 'invalid_json'},
            'conversation_scope': scope, 'turn_intent': decision, 'sources': list(sources),
            'source_dependencies': list(sources), 'outcomes': [], 'receipts': [], 'pending_ids': []}
    aid = repo.add_message(conn, cid, 'assistant', '资料核对未完成。', json.dumps(data, ensure_ascii=False))
    conn.commit()
    return uid, aid


def metadata(conn, mid):
    return json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0])


def update(conn, mid, mutate):
    data = metadata(conn, mid)
    mutate(data)
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?', (json.dumps(data, ensure_ascii=False), mid))
    conn.commit()


def test_query_regeneration_is_idempotent_and_keeps_the_exact_original_question(client, conn):
    cid, other = repo.create_conversation(conn), repo.create_conversation(conn)
    uid, aid = seed(conn, cid)
    rows = [tuple(row) for row in conn.execute('SELECT * FROM message')]
    url = f'/api/conversations/{cid}/messages/{aid}/regenerate'
    first = client.post(url)
    assert first.status_code == 202
    task = first.json()
    assert task['payload']['user_message_id'] == uid and task['payload']['regeneration_of'] == aid
    assert task['payload']['text'] == '顾念的沟通偏好有哪些？'
    assert 'reanalysis_of' not in task['payload']
    assert client.post(url).json()['id'] == task['id']
    conn.execute("UPDATE task SET status='done' WHERE id=?", (task['id'],))
    conn.commit()
    assert client.post(url).json()['id'] == task['id']
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 1
    assert rows == [tuple(row) for row in conn.execute('SELECT * FROM message')]
    assert client.post(f'/api/conversations/{other}/messages/{aid}/regenerate').status_code == 404
    assert client.post(f'/api/conversations/missing/messages/{aid}/regenerate').status_code == 404
    assert client.post(f'/api/conversations/{cid}/messages/{uid}/regenerate').status_code == 404


@pytest.mark.parametrize('intent', ['unknown', 'record', 'mixed'])
def test_unknown_and_record_messages_cannot_enter_query_only_regeneration(client, conn, intent):
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid, '顾念住青岛。', intent=intent)
    assert client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').status_code == 409
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


@pytest.mark.parametrize('change', [
    lambda data: data.pop('turn_intent'),
    lambda data: data['turn_intent'].update(status='failed'),
    lambda data: data['turn_intent'].update(source_sha256='invalid'),
    lambda data: data['turn_intent'].update(reason=[]),
    lambda data: data['turn_intent'].update(extraction_text='unexpected record'),
    lambda data: data['conversation_scope'].update(conversation_id='other'),
    lambda data: data['conversation_scope'].update(person_ids=['other']),
])
def test_corrupt_or_missing_query_audit_fails_closed(client, conn, change):
    cid = repo.create_conversation(conn)
    uid, aid = seed(conn, cid)
    update(conn, uid, change)
    assert client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').status_code == 409
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


@pytest.mark.parametrize('change', [
    lambda data: data['answer_review'].update(status='completed'),
    lambda data: data['conversation_scope'].update(user_message_id='missing'),
    lambda data: data.update(source_dependencies=None),
    lambda data: data.update(sources=[{'id': 'S1'}]),
    lambda data: data.update(pending_ids=['pending']),
    lambda data: data.update(outcomes=[{'status': 'ok'}]),
    lambda data: data.update(receipts=[{'outcomes': [{'status': 'pending'}]}]),
])
def test_invalid_answer_association_or_runtime_result_never_replays_a_write(client, conn, change):
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid)
    update(conn, aid, change)
    assert client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').status_code == 409
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


def test_applied_database_write_blocks_regeneration_even_with_readonly_metadata(client, conn):
    cid = repo.create_conversation(conn)
    uid, aid = seed(conn, cid)
    repo.mark_patch(conn, 'applied', None, 'memory', 'value', uid)
    conn.commit()
    assert client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').status_code == 409


def test_another_answer_for_the_same_user_with_a_saved_receipt_blocks_regeneration(client, conn):
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid)
    data = metadata(conn, aid)
    data['outcomes'] = [{'status': 'ok'}]
    repo.add_message(conn, cid, 'assistant', 'saved', json.dumps(data))
    conn.commit()
    assert client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').status_code == 409


@pytest.mark.parametrize('action', ['withdraw', 'move', 'delete', 'archive'])
def test_unavailable_or_reassigned_sources_refuse_regeneration(client, conn, action):
    pid = repo.create_person(conn, '顾念', status='active')
    target = repo.create_person(conn, '许栩', status='active')
    eid = repo.add_evidence(conn, 'text', source_ref=new_id(), excerpt='顾念只在约好的时间接电话。')
    mid = repo.add_memory(conn, 'semantic', '顾念只在约好的时间接电话。', pid, .9, eid)
    source = memory_sources.memory_source(conn, mid) | {'id': 'S1'}
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid, sources=[source])
    if action == 'withdraw':
        memory_corrections.correct_memory(conn, mid,
            {'action': 'withdraw', 'expected_revision': source['revision'], 'request_id': new_id()})
    elif action == 'move':
        memory_corrections.correct_memory(conn, mid,
            {'action': 'edit', 'person_id': target, 'expected_revision': source['revision'], 'request_id': new_id()})
    elif action == 'delete':
        conn.execute('DELETE FROM memory_item WHERE id=?', (mid,))
    else:
        conn.execute('UPDATE person SET deleted_at=1 WHERE id=?', (pid,))
    conn.commit()
    response = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate')
    assert response.status_code == 409 and '许栩' not in response.text
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


def test_readable_edited_source_is_loaded_again_without_granting_writes(client, conn):
    pid = repo.create_person(conn, '顾念', status='active')
    mid = repo.add_memory(conn, 'semantic', 'old', pid)
    source = memory_sources.memory_source(conn, mid) | {'id': 'S1'}
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid, sources=[source])
    memory_corrections.correct_memory(conn, mid,
        {'action': 'edit', 'text': 'new', 'expected_revision': source['revision'], 'request_id': new_id()})
    assert client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').status_code == 202


def test_source_loss_after_submission_is_rechecked_before_model_or_domain_writes(client, conn, monkeypatch):
    pid = repo.create_person(conn, '顾念', status='active')
    mid = repo.add_memory(conn, 'semantic', 'old', pid)
    source = memory_sources.memory_source(conn, mid) | {'id': 'S1'}
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid, sources=[source])
    task = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').json()
    memory_corrections.correct_memory(conn, mid,
        {'action': 'withdraw', 'expected_revision': source['revision'], 'request_id': new_id()})
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    monkeypatch.setattr(harness, 'run_turn', lambda *args, **kwargs: pytest.fail('stale source reached model'))
    tasks._run(task['id'])
    assert tasks.read(conn, task['id'])['status'] == 'failed'
    assert conn.execute("SELECT COUNT(*) FROM message WHERE role='user'").fetchone()[0] == 1


def test_worker_retry_reuses_original_user_and_marks_new_answer_without_reanalysis(client, conn, monkeypatch):
    cid = repo.create_conversation(conn)
    uid, aid = seed(conn, cid)
    task = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').json()
    seen = []
    def run(connection, cid, text, **kwargs):
        seen.append(kwargs)
        mid = repo.add_message(connection, cid, 'assistant', 'new answer', json.dumps(metadata(connection, aid)))
        connection.commit()
        return {'assistant_message_id': mid, 'answer': 'new answer', 'outcomes': []}
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    monkeypatch.setattr(harness, 'run_turn', run)
    tasks._run(task['id'])
    first = tasks.read(conn, task['id'])
    assert first['status'] == 'done'
    assert first['result']['regeneration']['source_assistant_message_id'] == aid
    saved = metadata(conn, first['result']['assistant_message_id'])
    assert saved['regeneration']['source_user_message_id'] == uid
    conn.execute("UPDATE task SET status='failed' WHERE id=?", (task['id'],))
    conn.commit()
    tasks.retry(conn, task['id'])
    tasks._run(task['id'])
    assert seen[0]['user_message_id'] == seen[1]['user_message_id'] == uid
    assert 'context_message_id' not in seen[0]
    assert conn.execute("SELECT COUNT(*) FROM message WHERE role='user'").fetchone()[0] == 1


def test_real_harness_uses_original_person_and_history_after_later_subject_switch(client, conn, monkeypatch):
    first, second = repo.create_person(conn, '顾念', status='active'), repo.create_person(conn, '许栩', status='active')
    cid = repo.create_conversation(conn)
    prior = repo.add_message(conn, cid, 'user', '查询顾念')
    conversation_scope.for_user_message(conn, cid, prior, '查询顾念')
    uid, aid = seed(conn, cid, '他住哪里？')
    later = repo.add_message(conn, cid, 'user', '查询许栩')
    conversation_scope.for_user_message(conn, cid, later, '查询许栩')
    task = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate').json()
    requests = []
    def chat(messages, **kwargs):
        requests.extend(messages)
        return {'content': '资料不足。'}
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(turn_intent, 'classify_turn', lambda *args, **kwargs: pytest.fail('must reuse query decision'))
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args, **kwargs: pytest.fail('query must not extract'))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'unavailable', 'semantic_verified': False}, 'warnings': []})
    before = {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table}')]
              for table in ('person_field', 'memory_item', 'interaction', 'applied_patch')}
    tasks._run(task['id'])
    result = tasks.read(conn, task['id'])
    assert result['status'] == 'done', result['error']
    assert result['result']['conversation_scope']['person_ids'] == [first]
    assert second not in result['result']['conversation_scope']['person_ids']
    assert result['result']['regeneration']['source_user_message_id'] == uid
    assert all('查询许栩' not in item['content'] for item in requests)
    assert before == {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table}')] for table in before}


def test_concurrent_submissions_make_one_task_and_busy_dialogue_stays_locked(client, conn):
    from app.store.db import _connect
    from app.config import get_settings
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid)
    barrier = Barrier(2)
    def submit():
        connection = _connect(get_settings().abs_db_path)
        try:
            barrier.wait(timeout=5)
            try:
                return chat_regeneration.submit(connection, cid, aid)['id']
            except sqlite3.IntegrityError:
                connection.rollback()
                return chat_regeneration.submit(connection, cid, aid)['id']
        finally:
            connection.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _: submit(), range(2)))
    assert len(set(ids)) == 1
    conn.execute("UPDATE task SET status='done' WHERE id=?", (ids[0],))
    _uid, other = seed(conn, cid, '再查一次顾念。')
    tasks.submit(conn, 'chat', 'active', {'conversation_id': cid, 'text': 'other'}, dispatch=False)
    assert client.post(f'/api/conversations/{cid}/messages/{other}/regenerate').status_code == 409


@pytest.mark.parametrize('change', ['move_alias', 'ambiguous_alias'])
def test_changed_alias_ownership_never_silently_answers_about_another_person(client, conn, change):
    first, second = repo.create_person(conn, '顾念', status='active'), repo.create_person(conn, '许栩', status='active')
    repo.add_alias(conn, first, '小顾', 'user')
    cid = repo.create_conversation(conn)
    uid, aid = seed(conn, cid, '小顾住哪里？')
    before = conn.execute('SELECT tool_calls FROM message WHERE id=?', (uid,)).fetchone()[0]
    if change == 'move_alias':
        conn.execute('DELETE FROM person_alias WHERE person_id=? AND alias=?', (first, '小顾'))
    repo.add_alias(conn, second, '小顾', 'user')
    conn.commit()
    response = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate')
    assert response.status_code == 409
    assert '人物指向已变化' in response.json()['detail']
    assert conn.execute('SELECT tool_calls FROM message WHERE id=?', (uid,)).fetchone()[0] == before
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


def test_scope_preview_is_readonly_and_preserves_the_exact_old_binding(conn):
    repo.create_person(conn, '顾念', status='active')
    cid = repo.create_conversation(conn)
    uid, _aid = seed(conn, cid)
    before = [tuple(row) for row in conn.execute('SELECT * FROM message')]
    preview = conversation_scope.resolve_user_message(conn, cid, uid, '顾念的沟通偏好有哪些？')
    assert preview == metadata(conn, uid)['conversation_scope']
    assert before == [tuple(row) for row in conn.execute('SELECT * FROM message')]


@pytest.mark.parametrize('invalid', [[], {}])
def test_nontext_user_message_link_is_rejected_before_sql_binding(client, conn, invalid):
    cid = repo.create_conversation(conn)
    _uid, aid = seed(conn, cid)
    update(conn, aid, lambda data: data['conversation_scope'].update(user_message_id=invalid))
    response = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate')
    assert response.status_code == 409
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


@pytest.mark.parametrize('invalid', [[], {}])
def test_nontext_query_status_is_rejected_without_a_membership_exception(client, conn, invalid):
    cid = repo.create_conversation(conn)
    uid, aid = seed(conn, cid)
    update(conn, uid, lambda data: data['turn_intent'].update(status=invalid))
    response = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate')
    assert response.status_code == 409
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


@pytest.mark.parametrize('field', ['mode', 'context_message_id'])
@pytest.mark.parametrize('invalid', [[], {}])
def test_nontext_scope_binding_is_rejected_without_a_membership_exception(client, conn, field, invalid):
    cid = repo.create_conversation(conn)
    uid, aid = seed(conn, cid)
    for mid in (uid, aid):
        update(conn, mid, lambda data: data['conversation_scope'].update({field: invalid}))
    response = client.post(f'/api/conversations/{cid}/messages/{aid}/regenerate')
    assert response.status_code == 409
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0
