"""Explicit reanalysis creates one successor; ordinary retries keep decisions."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import conversation_scope, harness, turn_intent
from app.jobs import chat_reanalysis, tasks
from app.store import repo
from app.util import sha256_text


@pytest.fixture
def client(conn, monkeypatch):
    from app.api import main, workspace
    monkeypatch.setattr(workspace, 'get_conn', lambda: conn)
    monkeypatch.setattr(main, 'get_conn', lambda: conn)
    monkeypatch.setattr(tasks, 'start', lambda tid: None)
    return TestClient(main.app)


def seed(conn, monkeypatch, cid, text='请记住林舟住杭州。', intent='unknown'):
    mid = repo.add_message(conn, cid, 'user', text)
    conversation_scope.for_user_message(conn, cid, mid, text)
    monkeypatch.setattr(turn_intent, 'classify_turn', lambda text, **kwargs:
        turn_intent.validate_intent({'intent': intent, 'record_spans': [text] if intent in {'record', 'mixed'} else [],
                                     'reason': 'scripted'}, text))
    turn_intent.for_user_message(conn, cid, mid, text)
    return mid


def test_reanalysis_is_idempotent_preserves_original_and_checks_ownership(client, conn, monkeypatch):
    cid, other = repo.create_conversation(conn), repo.create_conversation(conn)
    mid = seed(conn, monkeypatch, cid)
    old = tuple(conn.execute('SELECT * FROM message WHERE id=?', (mid,)).fetchone())
    url = f'/api/conversations/{cid}/messages/{mid}/reanalyze'
    first = client.post(url)
    assert first.status_code == 202
    second = client.post(url)
    assert second.json()['id'] == first.json()['id']
    assert first.json()['payload']['reanalysis_of'] == mid
    assert first.json()['payload']['source_sha256'] == sha256_text(old[3])
    conn.execute("UPDATE task SET status='done' WHERE id=?", (first.json()['id'],))
    conn.commit()
    assert client.post(url).json()['id'] == first.json()['id']
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 1
    assert tuple(conn.execute('SELECT * FROM message WHERE id=?', (mid,)).fetchone()) == old
    assert client.post(f'/api/conversations/{other}/messages/{mid}/reanalyze').status_code == 404
    assert client.post(f'/api/conversations/missing/messages/{mid}/reanalyze').status_code == 404


@pytest.mark.parametrize('intent', ['record', 'mixed'])
def test_authorised_record_messages_cannot_be_reanalysed(client, conn, monkeypatch, intent):
    cid = repo.create_conversation(conn)
    mid = seed(conn, monkeypatch, cid, intent=intent)
    assert client.post(f'/api/conversations/{cid}/messages/{mid}/reanalyze').status_code == 409
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 0


def test_reanalysis_rejects_busy_unclassified_changed_or_applied_messages(client, conn, monkeypatch):
    cid = repo.create_conversation(conn)
    mid = seed(conn, monkeypatch, cid)
    active = tasks.submit(conn, 'chat', 'other', {'conversation_id': cid, 'text': 'other'}, dispatch=False)
    assert client.post(f'/api/conversations/{cid}/messages/{mid}/reanalyze').status_code == 409
    conn.execute("UPDATE task SET status='done' WHERE id=?", (active['id'],))
    empty = repo.add_message(conn, cid, 'user', '尚未分类')
    conn.commit()
    assert client.post(f'/api/conversations/{cid}/messages/{empty}/reanalyze').status_code == 409
    conn.execute('UPDATE message SET content=? WHERE id=?', ('changed', mid))
    conn.commit()
    assert client.post(f'/api/conversations/{cid}/messages/{mid}/reanalyze').status_code == 409
    mid2 = seed(conn, monkeypatch, cid)
    repo.mark_patch(conn, 'applied', None, 'memory', 'value', mid2)
    conn.commit()
    assert client.post(f'/api/conversations/{cid}/messages/{mid2}/reanalyze').status_code == 409


def test_pending_receipts_prevent_reanalysis_even_if_decision_claims_readonly(client, conn, monkeypatch):
    cid = repo.create_conversation(conn)
    mid = seed(conn, monkeypatch, cid)
    repo.add_message(conn, cid, 'assistant', '待确认', json.dumps({
        'conversation_scope': {'user_message_id': mid}, 'pending_ids': ['pending']}))
    conn.commit()
    assert client.post(f'/api/conversations/{cid}/messages/{mid}/reanalyze').status_code == 409


def test_worker_retry_reuses_successor_user_and_original_context(conn, monkeypatch):
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    monkeypatch.setattr(tasks, 'start', lambda tid: None)
    cid = repo.create_conversation(conn)
    mid = seed(conn, monkeypatch, cid)
    task = chat_reanalysis.submit(conn, cid, mid)
    seen = []
    def run(conn, cid, text, **kwargs):
        seen.append(kwargs)
        return {'answer': 'done', 'outcomes': []}
    monkeypatch.setattr(harness, 'run_turn', run)
    tasks._run(task['id'])
    conn.execute("UPDATE task SET status='failed' WHERE id=?", (task['id'],))
    conn.commit()
    tasks.retry(conn, task['id'])
    tasks._run(task['id'])
    assert len(seen) == 2 and seen[0]['user_message_id'] == seen[1]['user_message_id'] != mid
    assert seen[0]['context_message_id'] == seen[1]['context_message_id'] == mid
    rows = conn.execute("SELECT * FROM message WHERE conversation_id=? AND role='user'", (cid,)).fetchall()
    assert len(rows) == 2
    successor = next(row for row in rows if row['id'] != mid)
    assert json.loads(successor['tool_calls'])['reanalysis']['source_user_message_id'] == mid


def test_reanalysis_after_subject_switch_uses_original_history_and_person(conn, monkeypatch):
    first = repo.create_person(conn, '林舟', status='active')
    second = repo.create_person(conn, '何川', status='active')
    cid = repo.create_conversation(conn)
    prior = repo.add_message(conn, cid, 'user', '查询林舟')
    conversation_scope.for_user_message(conn, cid, prior, '查询林舟')
    text = '他住哪座城市？'
    mid = seed(conn, monkeypatch, cid, text)
    later = repo.add_message(conn, cid, 'user', '查询何川')
    conversation_scope.for_user_message(conn, cid, later, '查询何川')
    successor = repo.add_message(conn, cid, 'user', text, json.dumps({'reanalysis': {'source_user_message_id': mid}}))
    seen = []
    def chat(messages, **kwargs):
        seen.extend(messages)
        return {'content': '资料不足。'}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test', 'semantic_verified': False}, 'warnings': []})
    result = harness.run_turn(conn, cid, text, user_message_id=successor, context_message_id=mid)
    assert result['conversation_scope']['person_ids'] == [first]
    assert second not in result['conversation_scope']['person_ids']
    assert all('查询何川' not in item['content'] for item in seen)
    assert result['conversation_scope']['user_message_id'] == successor
    assert json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0])['turn_intent']['intent'] == 'unknown'


def test_chained_analysis_stays_at_original_question_and_keeps_all_original_metadata(conn, monkeypatch):
    cid = repo.create_conversation(conn)
    pid = repo.create_person(conn, '林舟', status='active')
    other = repo.create_person(conn, '何川', status='active')
    first = repo.add_message(conn, cid, 'user', '查询林舟')
    conversation_scope.for_user_message(conn, cid, first, '查询林舟')
    text = '他住哪座城市？'
    root = seed(conn, monkeypatch, cid, text)
    original = conn.execute('SELECT tool_calls FROM message WHERE id=?', (root,)).fetchone()[0]
    switch = repo.add_message(conn, cid, 'user', '查询何川')
    conversation_scope.for_user_message(conn, cid, switch, '查询何川')
    child = seed(conn, monkeypatch, cid, text)
    data = json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?', (child,)).fetchone()[0])
    assert data['conversation_scope']['person_ids'] == [other]
    data['reanalysis'] = {'source_user_message_id': root, 'source_sha256': sha256_text(text)}
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?', (json.dumps(data), child))
    conn.commit()
    result, scope = chat_reanalysis.context_scope(conn, cid, child)
    assert result['id'] == root and scope['person_ids'] == [pid]
    assert conn.execute('SELECT tool_calls FROM message WHERE id=?', (root,)).fetchone()[0] == original


def test_concurrent_reanalysis_creates_one_task_and_retry_cannot_overlap(conn, client, monkeypatch):
    from app.store.db import _connect
    from app.config import get_settings
    cid = repo.create_conversation(conn)
    mid = seed(conn, monkeypatch, cid)
    barrier = Barrier(2)
    def submit():
        connection = _connect(get_settings().abs_db_path)
        try:
            barrier.wait(timeout=5)
            try:
                return chat_reanalysis.submit(connection, cid, mid)['id']
            except sqlite3.IntegrityError:
                connection.rollback()
                return chat_reanalysis.submit(connection, cid, mid)['id']
        finally:
            connection.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert len(set(results)) == 1
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == 1
    conn.execute("UPDATE task SET status='failed' WHERE id=?", (results[0],))
    conn.commit()
    tasks.submit(conn, 'chat', 'new', {'conversation_id': cid, 'text': 'other'}, dispatch=False)
    assert client.post(f'/api/tasks/{results[0]}/retry').status_code == 409
    assert tasks.read(conn, results[0])['status'] == 'failed'


def test_explicit_analysis_can_write_but_repeated_request_and_worker_retry_cannot_duplicate(conn, client, monkeypatch):
    from app.memory import extract
    pid = repo.create_person(conn, '林舟', status='active')
    cid = repo.create_conversation(conn)
    text = '请记住林舟现在住杭州。'
    mid = seed(conn, monkeypatch, cid, text)
    original = conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0]
    monkeypatch.setattr(tasks, 'get_conn', lambda: conn)
    classification_calls = []
    def classify(text, **kwargs):
        classification_calls.append(text)
        return turn_intent.validate_intent({'intent': 'record', 'record_spans': [text], 'reason': 'explicit'}, text)
    monkeypatch.setattr(turn_intent, 'classify_turn', classify)
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=lambda *args, **kwargs: {'content': '已处理。'}))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test', 'semantic_verified': False}, 'warnings': []})
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda text, *args: extract.sanitize({
        'people': [{'name': '林舟', 'canonical_name': '林舟', 'aliases': [], 'fields': [
            {'field': 'city', 'value': '杭州', 'confidence': .99, 'evidence': text}],
            'facts': [], 'promises': [], 'ledger': [], 'relations': []}], 'my_notes': []}, source=text))
    url = f'/api/conversations/{cid}/messages/{mid}/reanalyze'
    tid = client.post(url).json()['id']
    tasks._run(tid)
    assert tasks.read(conn, tid)['status'] == 'done'
    assert repo.current_fields(conn, pid)['city']['value'] == '杭州'
    before = {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table}')]
              for table in ('person_field', 'memory_item', 'interaction', 'applied_patch')}
    assert client.post(url).json()['id'] == tid
    conn.execute("UPDATE task SET status='failed' WHERE id=?", (tid,))
    conn.commit()
    tasks.retry(conn, tid)
    tasks._run(tid)
    assert before == {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table}')] for table in before}
    assert classification_calls == [text]
    assert conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0] == original
    successor = tasks.read(conn, tid)['payload']['user_message_id']
    assert client.post(f'/api/conversations/{cid}/messages/{successor}/reanalyze').status_code == 409
