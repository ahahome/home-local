"""Explicit memory corrections preserve identity, provenance, history and index safety."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from threading import Barrier

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import tz
from app.memory import memory_corrections as corrections
from app.memory.vector import VectorIndex
from app.store import db, repo
from app.util import new_id, now_ms


@pytest.fixture
def api(conn, monkeypatch):
    from app.api import memories
    from app import llm
    app = FastAPI()
    app.include_router(memories.router)
    monkeypatch.setattr(memories, 'get_conn', lambda: conn)
    monkeypatch.setattr(llm, 'get_llm', lambda: pytest.fail('manual corrections must not call a model'))
    index = VectorIndex()
    monkeypatch.setattr(corrections, 'get_index', lambda: index)
    client = TestClient(app)
    yield client, memories, index, app
    client.close()


def person(conn, name='林舟', status='active'):
    pid = repo.create_person(conn, name, status=status)
    conn.commit()
    return pid


def memory(conn, pid=None, text='喜欢清晨骑车', evidence=True):
    eid = repo.add_evidence(conn, 'text', source_ref='test-source:' + new_id(),
                            excerpt='原始资料：' + text) if evidence else None
    ts = int(datetime.fromisoformat('2026-01-10T15:23:45+08:00').timestamp() * 1000)
    mid = repo.add_memory(conn, 'semantic', text, pid, .7, eid, ts)
    conn.commit()
    return mid, eid


def edit(client, mid, revision=0, request_id=None, **changes):
    return client.post('/api/memories/' + mid + '/correct', json={
        'action': 'edit', 'expected_revision': revision, 'request_id': request_id or new_id(), **changes})


def action(client, mid, name, revision, request_id=None):
    return client.post('/api/memories/' + mid + '/correct', json={
        'action': name, 'expected_revision': revision, 'request_id': request_id or new_id()})


def detail(client, mid):
    response = client.get('/api/memories/' + mid)
    assert response.status_code == 200, response.text
    return response.json()


def test_v8_upgrade_preserves_rows_and_backfills_source_time_and_independent_state():
    root = Path(__file__).resolve().parents[1]
    c = sqlite3.connect(':memory:')
    c.row_factory = sqlite3.Row
    c.executescript((root / 'app/store/schema.sql').read_text())
    for migration in sorted((root / 'migrations').glob('*.sql')):
        if 1 < int(migration.name.split('_')[0]) <= 8:
            c.executescript(migration.read_text())
    c.execute('PRAGMA user_version=8')
    c.execute('INSERT INTO memory_item(id,kind,text,confidence,evidence_id,ts,created_at,deleted_at) '
              'VALUES(?,?,?,?,?,?,?,?)', ('old', 'semantic', '历史资料', .5, 'origin', 100, 200, 300))
    c.commit()
    db.migrate(c)
    row = dict(c.execute('SELECT * FROM memory_item WHERE id=\'old\'').fetchone())
    assert c.execute('PRAGMA user_version').fetchone()[0] == 9
    assert row['status'] == 'active' and row['revision'] == row['assignment_revision'] == 0
    assert row['origin_evidence_id'] == 'origin' and row['updated_at'] == 200
    assert row['deleted_at'] == 300 and row['text'] == '历史资料'
    assert c.execute('SELECT COUNT(*) FROM memory_correction').fetchone()[0] == 0
    c.close()


def test_edit_keeps_id_and_original_evidence_creates_user_source_and_immutable_history(api, conn):
    client, _, _, _ = api
    pid = person(conn)
    mid, eid = memory(conn, pid)
    original = detail(client, mid)
    response = edit(client, mid, text='更喜欢傍晚骑车', reason='更正沟通中记错的偏好')
    assert response.status_code == 200, response.text
    result = response.json()
    current, audit = result['memory'], result['correction']
    assert not result['idempotent'] and current['id'] == mid
    assert current['text'] == '更喜欢傍晚骑车' and current['kind'] == 'semantic'
    assert current['revision'] == current['correction_count'] == 1
    assert current['assignment_revision'] == 0 and current['updated_at'] >= original['updated_at']
    assert current['origin_source']['id'] == eid and current['source']['id'] != eid
    assert conn.execute('SELECT excerpt FROM evidence WHERE id=?', (eid,)).fetchone()[0] == '原始资料：喜欢清晨骑车'
    source = json.loads(current['source']['excerpt'])
    assert source['edited_by_user'] and source['user_correction']['text'] == current['text']
    assert source['previous_evidence_id'] == source['origin_evidence_id'] == eid
    assert audit['before']['text'] == '喜欢清晨骑车' and audit['after']['text'] == current['text']
    assert audit['before']['person'] == {'id': pid, 'display_name': '林舟'}
    assert audit['before_source']['id'] == eid and audit['after_source']['id'] == current['source']['id']
    assert audit['source']['source_ref'] == 'memory-correction:' + audit['id']
    assert client.get('/api/memories/' + mid + '/history').json()['history'] == [audit]
    assert conn.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 1


def test_old_request_replay_returns_same_audit_and_current_detail_without_extra_writes(api, conn):
    client, _, _, _ = api
    mid, _ = memory(conn)
    first = edit(client, mid, request_id='first-edit', text='第一次纠正').json()
    second = edit(client, mid, revision=1, request_id='second-edit', text='第二次纠正').json()
    before = conn.total_changes
    replay = edit(client, mid, request_id='first-edit', text='第一次纠正')
    assert replay.status_code == 200, replay.text
    assert replay.json()['idempotent']
    assert replay.json()['correction'] == first['correction']
    assert replay.json()['memory'] == second['memory']
    assert conn.total_changes == before
    assert edit(client, mid, request_id='first-edit', text='改了同请求内容').status_code == 409
    assert edit(client, mid, revision=2, request_id='first-edit', text='第一次纠正').status_code == 409


def test_cas_conflict_cannot_mutate_memory_history_evidence_or_indexes(api, conn):
    client, _, _, _ = api
    mid, _ = memory(conn)
    assert edit(client, mid, text='先修改成功').status_code == 200
    before = conn.total_changes
    assert edit(client, mid, text='旧表单不能覆盖').status_code == 409
    assert conn.total_changes == before
    assert detail(client, mid)['text'] == '先修改成功'


def test_assignment_explicit_null_unconfirmed_target_and_assignment_revision(api, conn):
    client, _, _, _ = api
    source, target = person(conn), person(conn, '新认识的人', 'unconfirmed')
    mid, _ = memory(conn, source)
    moved = edit(client, mid, person_id=target).json()['memory']
    assert moved['person'] == {'id': target, 'display_name': '新认识的人'}
    assert moved['assignment_revision'] == 1 and moved['revision'] == 1
    text_only = edit(client, mid, revision=1, person_id=target, text='同一归属下纠错').json()['memory']
    assert text_only['assignment_revision'] == 1
    personal = edit(client, mid, revision=2, person_id=None).json()['memory']
    assert personal['person'] is None and personal['assignment_revision'] == 2
    assert client.get('/api/memories', params={'scope': 'personal'}).json()['memories'][0]['id'] == mid
    entries = client.get('/api/memories/' + mid + '/history').json()['history']
    assert entries[0]['before']['person']['id'] == target and entries[0]['after']['person'] is None
    assert entries[-1]['before']['person']['id'] == source


def test_same_day_date_preserves_exact_timestamp_other_day_uses_configured_local_noon(api, conn):
    client, _, _, _ = api
    mid, _ = memory(conn)
    initial = detail(client, mid)
    same_day = datetime.fromtimestamp(initial['ts'] / 1000, tz()).date().isoformat()
    same = edit(client, mid, text='同日纠错', valid_from=same_day).json()['memory']
    assert same['ts'] == initial['ts']
    old = edit(client, mid, revision=1, valid_from='2025-12-20').json()['memory']
    expected = datetime(2025, 12, 20, 12, tzinfo=tz())
    assert old['ts'] == int(expected.timestamp() * 1000)


@pytest.mark.parametrize('value', ['2099-01-01', '2026-1-1', '2026-02-30', '20260101', ''])
def test_invalid_or_future_dates_rejected_without_history(api, conn, value):
    client, _, _, _ = api
    mid, _ = memory(conn)
    response = edit(client, mid, valid_from=value)
    assert response.status_code == 400, response.text
    assert detail(client, mid)['correction_count'] == 0


def test_withdraw_restore_are_independent_of_archive_and_replay_does_not_change_status(api, conn):
    client, _, _, _ = api
    pid = person(conn)
    mid, _ = memory(conn, pid)
    withdrawn = action(client, mid, 'withdraw', 0, 'withdraw-once')
    assert withdrawn.status_code == 200, withdrawn.text
    assert withdrawn.json()['memory']['status'] == 'withdrawn'
    assert client.get('/api/memories').json()['total'] == 0
    assert client.get('/api/memories', params={'status': 'withdrawn'}).json()['total'] == 1
    assert client.get('/api/memories', params={'status': 'all'}).json()['total'] == 1
    assert action(client, mid, 'withdraw', 1).status_code == 409
    repo.archive_person(conn, pid)
    conn.commit()
    assert client.get('/api/memories/' + mid).status_code == 404
    assert client.get('/api/memories/' + mid + '/history').status_code == 404
    assert repo.restore_person(conn, pid)
    conn.commit()
    assert detail(client, mid)['status'] == 'withdrawn'
    assert client.get('/api/memories').json()['total'] == 0
    current_revision = detail(client, mid)['revision']
    if current_revision != 1:
        assert action(client, mid, 'restore', 1).status_code == 409
    restored = action(client, mid, 'restore', current_revision)
    assert restored.status_code == 200 and restored.json()['memory']['status'] == 'active'
    assert action(client, mid, 'restore', restored.json()['memory']['revision']).status_code == 409
    replay = action(client, mid, 'withdraw', 0, 'withdraw-once').json()
    assert replay['idempotent'] and replay['memory']['status'] == 'active'
    assert replay['correction']['after']['status'] == 'withdrawn'


def test_edit_and_restore_only_index_active_current_text_and_remove_stale_vectors(api, conn):
    client, _, index, _ = api
    mid, _ = memory(conn, text='oldterm')
    index.add(conn, 'memory', mid, 'oldterm', [1., 0.])
    conn.commit()
    assert index.search(conn, 'memory', [1., 0.])[0][0] == mid
    assert edit(client, mid, text='newterm').status_code == 200
    assert index.search(conn, 'memory', [1., 0.]) == []
    assert not conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'oldterm'").fetchone()
    assert conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'newterm'").fetchone()
    assert action(client, mid, 'withdraw', 1).status_code == 200
    assert not conn.execute("SELECT 1 FROM memory_fts WHERE owner_type='memory' AND owner_id=?", (mid,)).fetchone()
    assert edit(client, mid, revision=2, text='restoredterm').status_code == 200
    assert detail(client, mid)['status'] == 'withdrawn'
    assert not conn.execute("SELECT 1 FROM memory_fts WHERE owner_type='memory' AND owner_id=?", (mid,)).fetchone()
    assert action(client, mid, 'restore', 3).status_code == 200
    assert conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'restoredterm'").fetchone()
    assert not conn.execute("SELECT 1 FROM vector_item WHERE owner_type='memory' AND owner_id=?", (mid,)).fetchone()


@pytest.mark.parametrize('target_state', ['unknown', 'archived', 'merged'])
def test_invalid_target_is_rejected_without_silent_identity_redirection(api, conn, target_state):
    client, _, _, _ = api
    mid, _ = memory(conn)
    pid = 'unknown' if target_state == 'unknown' else person(conn)
    if target_state == 'archived':
        repo.archive_person(conn, pid)
    elif target_state == 'merged':
        conn.execute("UPDATE person SET status='merged' WHERE id=?", (pid,))
    conn.commit()
    assert edit(client, mid, person_id=pid).status_code == 404
    assert detail(client, mid)['person'] is None and detail(client, mid)['revision'] == 0


@pytest.mark.parametrize('body', [
    {'action': 'edit', 'text': None}, {'action': 'edit', 'text': ' '},
    {'action': 'edit', 'text': 'x' * 20001}, {'action': 'edit', 'text': 123},
    {'action': 'edit', 'kind': 'episodic', 'text': '新内容'},
    {'action': 'edit'}, {'action': 'edit', 'person_id': ''},
    {'action': 'edit', 'valid_from': None}, {'action': 'edit', 'reason': 'x' * 1001, 'text': '新内容'},
    {'action': 'withdraw', 'text': '不能附带内容'}, {'action': 'restore', 'person_id': None},
    {'action': 'edit', 'expected_revision': True, 'text': '新内容'},
    {'action': 'edit', 'expected_revision': '0', 'text': '新内容'},
    {'action': 'edit', 'request_id': ' ', 'text': '新内容'},
])
def test_strict_http_contract(api, conn, body):
    client, _, _, _ = api
    mid, _ = memory(conn)
    response = client.post('/api/memories/' + mid + '/correct', json={
        'expected_revision': 0, 'request_id': new_id(), **body})
    assert response.status_code == 422, response.text
    assert detail(client, mid)['revision'] == 0


def test_noop_edit_is_rejected_and_no_origin_is_invented_for_initially_unsourced_memory(api, conn):
    client, _, _, _ = api
    mid, _ = memory(conn, evidence=False)
    assert edit(client, mid, text='喜欢清晨骑车').status_code == 400
    result = edit(client, mid, text='实际喜欢傍晚骑车').json()['memory']
    assert result['origin_source'] is None and result['source'] is not None


def test_storage_failure_rolls_back_memory_evidence_history_fts_and_vector(api, conn):
    client, _, index, _ = api
    mid, _ = memory(conn, text='oldterm')
    index.add(conn, 'memory', mid, 'oldterm', [1., 0.])
    conn.commit()
    original = detail(client, mid)
    evidence_count = conn.execute('SELECT COUNT(*) FROM evidence').fetchone()[0]
    conn.execute("CREATE TRIGGER fail_memory_history BEFORE INSERT ON memory_correction BEGIN "
                 "SELECT RAISE(ABORT,'history storage unavailable'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        edit(client, mid, text='newterm')
    assert detail(client, mid) == original
    assert conn.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == evidence_count
    assert conn.execute('SELECT COUNT(*) FROM memory_correction').fetchone()[0] == 0
    assert conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'oldterm'").fetchone()
    assert not conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'newterm'").fetchone()
    assert index.search(conn, 'memory', [1., 0.])[0][1] == 'oldterm'


def test_correction_keeps_original_communication_after_proposal_and_analysis_are_gone(api, conn):
    client, _, _, _ = api
    original_id = repo.add_evidence(conn, 'text', source_ref='followup-outcome:old', excerpt='原沟通完整notes')
    reviewed_id = repo.add_evidence(conn, 'text', source_ref='outcome-memory-review:old', excerpt=json.dumps({
        'source_evidence_id': original_id, 'original_quote': '原沟通', 'edited_by_user': True}))
    mid, _ = memory(conn, evidence=False)
    conn.execute('UPDATE memory_item SET evidence_id=?,origin_evidence_id=? WHERE id=?',
                 (reviewed_id, reviewed_id, mid))
    conn.commit()
    current = edit(client, mid, text='再一次人工纠正').json()['memory']
    assert current['proposal_id'] is None
    assert current['origin_source']['id'] == reviewed_id
    assert current['original_source']['id'] == original_id
    assert current['original_source']['excerpt'] == '原沟通完整notes'


def test_two_connections_race_same_revision_one_edit_wins_and_other_conflicts(api, conn, monkeypatch):
    _, module, _, app = api
    mid, _ = memory(conn)
    database = Path(conn.execute('PRAGMA database_list').fetchone()['file'])
    connections = []
    def connection():
        c = db._connect(database)
        connections.append(c)
        return c
    monkeypatch.setattr(module, 'get_conn', connection)
    clients = [TestClient(app), TestClient(app)]
    start = Barrier(2)
    def worker(client, text):
        start.wait(timeout=5)
        return edit(client, mid, text=text)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [f.result(timeout=10) for f in [pool.submit(worker, c, str(i)) for i,c in enumerate(clients)]]
    assert sorted(r.status_code for r in results) == [200, 409]
    assert conn.execute('SELECT COUNT(*) FROM memory_correction').fetchone()[0] == 1
    assert conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0] == 1
    for client in clients:
        client.close()
    for c in connections:
        c.close()
