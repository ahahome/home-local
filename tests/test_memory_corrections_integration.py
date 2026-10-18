"""Saved-memory corrections through the registered API and shared consumers."""
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import followup
from app.memory import retrieve
from app.memory.vector import VectorIndex
from app.store import db, repo
from app.util import new_id, now_ms


@pytest.fixture
def api(conn, monkeypatch):
    from app import llm
    from app.api import main, memories, workspace

    active = {'conn': conn}
    for module in (main, memories, workspace):
        monkeypatch.setattr(module, 'get_conn', lambda: active['conn'])
    monkeypatch.setattr(llm, 'get_llm', lambda: pytest.fail('manual correction must not call a model'))
    monkeypatch.setattr(followup, 'get_llm', lambda: pytest.fail('manual correction must not call a model'))
    client = TestClient(main.app)
    yield SimpleNamespace(client=client, active=active)
    client.close()


def person(conn, name='Lin'):
    pid = repo.create_person(conn, name, status='active')
    conn.commit()
    return pid


def seed_memory(conn, pid=None, text='prefers telephone', evidence_id=None):
    evidence_id = evidence_id or repo.add_evidence(conn, 'text', source_ref='source:' + new_id(), excerpt=text)
    mid = repo.add_memory(conn, 'semantic', text, pid, .8, evidence_id)
    conn.commit()
    return mid, evidence_id


def detail(api, mid):
    response = api.client.get(f'/api/memories/{mid}')
    assert response.status_code == 200, response.text
    return response.json()


def correct(api, mid, action='edit', revision=None, request_id=None, **changes):
    if revision is None:
        revision = detail(api, mid)['revision']
    return api.client.post(f'/api/memories/{mid}/correct', json={
        'action': action, 'expected_revision': revision, 'request_id': request_id or new_id(), **changes})


def saved(api, **params):
    response = api.client.get('/api/memories', params=params)
    assert response.status_code == 200, response.text
    return response.json()['memories']


def history(api, mid):
    response = api.client.get(f'/api/memories/{mid}/history')
    assert response.status_code == 200, response.text
    return response.json()['history']


def merge(api, survivor, original):
    response = api.client.post('/api/persons/merge', json={'survivor_id': survivor, 'merged_id': original})
    assert response.status_code == 200, response.text
    return response.json()['merge_log_id']


def split(api, log_id):
    response = api.client.post('/api/persons/split', json={'merge_log_id': log_id})
    assert response.status_code == 200, response.text


def test_edit_changes_content_date_and_owner_with_immutable_origin_and_audit(api, conn):
    original, target = person(conn), person(conn, 'Chen')
    mid, original_eid = seed_memory(conn, original)
    original_row = dict(conn.execute('SELECT * FROM memory_item WHERE id=?', (mid,)).fetchone())
    request_id = new_id()
    changes = {'text': 'prefers email', 'valid_from': '2026-01-03', 'person_id': target,
               'reason': 'corrected after checking the conversation'}

    response = correct(api, mid, revision=0, request_id=request_id, **changes)

    assert response.status_code == 200, response.text
    result = response.json()
    memory = result['memory']
    assert memory['text'] == changes['text'] and memory['person']['id'] == target
    assert memory['revision'] == memory['assignment_revision'] == memory['correction_count'] == 1
    assert memory['origin_source']['id'] == original_eid
    assert memory['source']['id'] != original_eid
    assert memory['source']['source_ref'] == 'memory-correction:' + result['correction']['id']
    proof = json.loads(memory['source']['excerpt'])
    assert proof['user_correction']['valid_from'] == '2026-01-03'
    assert proof['origin_evidence_id'] == proof['previous_evidence_id'] == original_eid
    audit = history(api, mid)[0]
    assert audit['before']['person']['id'] == original
    assert audit['before']['text'] == original_row['text']
    assert audit['after']['person']['id'] == target and audit['after']['text'] == changes['text']
    assert audit['reason'] == changes['reason'] and audit['request_id'] == request_id
    assert not repo.memories_of(conn, original)
    assert [row['id'] for row in repo.memories_of(conn, target)] == [mid]

    replay = correct(api, mid, revision=0, request_id=request_id, **changes)
    assert replay.status_code == 200 and replay.json()['idempotent'] is True
    assert len(history(api, mid)) == 1
    conflict = correct(api, mid, revision=0, request_id=request_id, **{**changes, 'text': 'different'})
    assert conflict.status_code == 409
    assert detail(api, mid)['revision'] == 1


def test_stale_correction_from_an_independent_connection_cannot_overwrite_commit(api, conn):
    pid = person(conn)
    mid, _ = seed_memory(conn, pid)
    database = conn.execute('PRAGMA database_list').fetchone()['file']
    other = db._connect(database)
    try:
        api.active['conn'] = other
        stale_revision = detail(api, mid)['revision']
        api.active['conn'] = conn
        assert correct(api, mid, revision=stale_revision, text='new verified preference').status_code == 200
        api.active['conn'] = other

        rejected = correct(api, mid, revision=stale_revision, text='stale overwrite')

        assert rejected.status_code == 409
        assert detail(api, mid)['text'] == 'new verified preference'
        assert len(history(api, mid)) == 1
    finally:
        api.active['conn'] = conn
        other.close()


def test_edit_withdraw_restore_keep_search_profile_and_new_briefs_consistent(api, conn, monkeypatch):
    pid = person(conn)
    mid, _ = seed_memory(conn, pid, text='oldcobalt memory')
    personal, _ = seed_memory(conn, text='personal background')
    stale_index = VectorIndex()
    stale_index.add(conn, 'memory', mid, 'oldcobalt memory', [1., 0.])
    conn.commit()
    assert stale_index.search(conn, 'memory', [1., 0.])[0][0] == mid
    original = detail(api, mid)

    response = correct(api, mid, text='newquartz memory')

    assert response.status_code == 200, response.text
    assert saved(api, q='oldcobalt') == []
    assert saved(api, q='newquartz')[0]['id'] == mid
    assert not conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'oldcobalt'").fetchone()
    assert conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'newquartz'").fetchone()
    assert not repo.all_vectors(conn, 'memory')
    assert retrieve._fetch_items(conn, [('memory', mid)])[('memory', mid)]['text'] == 'newquartz memory'
    assert followup.build_brief(conn, pid)['memories'][0]['text'] == 'newquartz memory'
    current_source = detail(api, mid)['source']['id']

    assert correct(api, mid, action='withdraw').status_code == 200
    assert correct(api, personal, action='withdraw').status_code == 200

    assert saved(api) == []
    assert len(saved(api, status='withdrawn')) == 2
    assert not repo.memories_of(conn, pid)
    assert not repo.memories_of(conn, pid, kinds=['semantic'])
    assert not repo.memories_since(conn, 0)
    assert not repo.memories_since(conn, 0, kinds=['semantic'])
    assert not repo.my_profile_memories(conn)
    assert not followup.build_brief(conn, pid)['memories']
    stale_keys = [('memory', row[0]) for row in stale_index.search(conn, 'memory', [1., 0.])]
    assert stale_keys == [('memory', mid)]
    assert retrieve._fetch_items(conn, stale_keys) == {}
    from app import llm
    from app.memory import vector
    monkeypatch.setattr(llm, 'get_llm', lambda: SimpleNamespace(embed=lambda texts: [[1., 0.]]))
    monkeypatch.setattr(vector, 'get_index', lambda: stale_index)
    assert retrieve.hybrid_search(conn, 'newquartz') == []

    assert correct(api, mid, action='restore').status_code == 200
    assert correct(api, personal, action='restore').status_code == 200

    restored = detail(api, mid)
    assert restored['source']['id'] == current_source
    assert restored['origin_source']['id'] == original['source']['id']
    assert restored['revision'] == 3 and restored['assignment_revision'] == 0
    assert {row['id'] for row in repo.memories_since(conn, 0)} == {mid, personal}
    assert [row['id'] for row in repo.my_profile_memories(conn)] == [personal]
    assert followup.build_brief(conn, pid)['memories'][0]['id'] == mid
    assert retrieve.hybrid_search(conn, 'newquartz')[0]['id'] == mid
    assert [row['action'] for row in history(api, mid)] == ['restore', 'withdraw', 'edit']


@pytest.mark.parametrize('changes', [
    {'action': 'edit'}, {'text': ''}, {'text': '   '}, {'text': None},
    {'valid_from': '2026-02-30'}, {'valid_from': '2999-01-01'}, {'valid_from': None},
    {'person_id': ''}, {'person_id': 'does-not-exist'},
    {'action': 'withdraw', 'text': 'cannot edit and withdraw'},
    {'expected_revision': True, 'text': 'invalid version'},
    {'text': 'new preference', 'unknown': 'not allowed'},
])
def test_invalid_requests_have_no_memory_or_evidence_side_effects(api, conn, changes):
    pid = person(conn)
    mid, eid = seed_memory(conn, pid)
    before = dict(conn.execute('SELECT * FROM memory_item WHERE id=?', (mid,)).fetchone())
    payload = {'action': 'edit', 'expected_revision': 0, 'request_id': new_id(), **changes}

    response = api.client.post(f'/api/memories/{mid}/correct', json=payload)

    assert response.status_code in (400, 404, 422), response.text
    assert dict(conn.execute('SELECT * FROM memory_item WHERE id=?', (mid,)).fetchone()) == before
    assert not history(api, mid)
    assert [row['id'] for row in conn.execute('SELECT id FROM evidence')] == [eid]


def test_archive_restore_invalidates_old_forms_without_reactivating_withdrawn_memory(api, conn):
    pid, hidden_target = person(conn), person(conn, 'Archived')
    mid, _ = seed_memory(conn, pid)
    withdrawn, _ = seed_memory(conn, pid, text='withdrawn background')
    assert correct(api, withdrawn, action='withdraw').status_code == 200
    assert api.client.delete(f'/api/persons/{hidden_target}').status_code == 200
    assert correct(api, mid, person_id=hidden_target).status_code == 404
    old_revision = detail(api, mid)['revision']

    assert api.client.delete(f'/api/persons/{pid}').status_code == 200

    assert api.client.get(f'/api/memories/{mid}').status_code == 404
    assert api.client.get(f'/api/memories/{mid}/history').status_code == 404
    assert correct(api, mid, revision=old_revision, text='archived edit').status_code == 404
    assert not repo.memories_since(conn, 0)
    assert retrieve._fetch_items(conn, [('memory', mid)]) == {}
    archive_revision = conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0]
    assert archive_revision == old_revision + 1
    assert api.client.delete(f'/api/persons/{pid}').status_code == 200
    assert conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0] == archive_revision

    assert api.client.post(f'/api/persons/{pid}/restore').status_code == 200

    assert detail(api, mid)['revision'] == old_revision + 2
    assert detail(api, withdrawn)['status'] == 'withdrawn'
    assert detail(api, withdrawn)['revision'] == 3
    assert [row['id'] for row in saved(api)] == [mid]
    assert correct(api, mid, revision=old_revision, text='stale edit after restore').status_code == 409
    assert [row['id'] for row in followup.build_brief(conn, pid)['memories']] == [mid]


def test_visible_memory_consumers_recheck_archived_and_merged_owners(api, conn):
    pid = person(conn)
    mid, _ = seed_memory(conn, pid)
    for status, deleted_at in [('active', now_ms()), ('merged', None)]:
        conn.execute('UPDATE person SET status=?,deleted_at=? WHERE id=?', (status, deleted_at, pid))
        conn.commit()
        assert not repo.memories_of(conn, pid)
        assert not repo.memories_since(conn, 0)
        assert retrieve._fetch_items(conn, [('memory', mid)]) == {}


@pytest.mark.parametrize('personal_target', [False, True])
def test_split_preserves_manual_assignment_but_returns_content_edits_and_withdrawals(api, conn, personal_target):
    survivor, original, third = person(conn, 'Survivor'), person(conn, 'Original'), person(conn, 'Third')
    edited, _ = seed_memory(conn, original)
    withdrawn, _ = seed_memory(conn, original, text='withdraw before split')
    transferred, _ = seed_memory(conn, original, text='manual transfer')
    moved_back, _ = seed_memory(conn, original, text='manual move away and back')
    previously_assigned, _ = seed_memory(conn, third, text='already assigned before merge')
    assert correct(api, previously_assigned, person_id=original).status_code == 200
    before_merge = {mid: detail(api, mid)['revision'] for mid in (edited, withdrawn, transferred, moved_back, previously_assigned)}

    log_id = merge(api, survivor, original)

    snapshot = json.loads(repo.get_merge_log(conn, log_id)['snapshot'])
    assert snapshot['memory_assignment_revisions'][previously_assigned] == 1
    for mid, revision in before_merge.items():
        assert detail(api, mid)['revision'] == revision + 1
    assert correct(api, edited, revision=before_merge[edited], text='stale before merge').status_code == 409
    assert correct(api, edited, text='verified after merge').status_code == 200
    assert correct(api, withdrawn, action='withdraw').status_code == 200
    target = None if personal_target else third
    assert correct(api, transferred, person_id=target).status_code == 200
    assert correct(api, moved_back, person_id=third).status_code == 200
    assert correct(api, moved_back, person_id=survivor).status_code == 200
    edited_revision = detail(api, edited)['revision']

    split(api, log_id)

    assert detail(api, edited)['person']['id'] == original
    assert detail(api, edited)['text'] == 'verified after merge'
    assert detail(api, edited)['revision'] == edited_revision + 1
    assert detail(api, edited)['assignment_revision'] == 0
    assert detail(api, withdrawn)['person']['id'] == original
    assert detail(api, withdrawn)['status'] == 'withdrawn'
    assert detail(api, transferred)['person'] == (None if personal_target else {'id': third, 'display_name': 'Third'})
    assert detail(api, transferred)['assignment_revision'] == 1
    assert detail(api, moved_back)['person']['id'] == survivor
    assert detail(api, moved_back)['assignment_revision'] == 2
    assert detail(api, previously_assigned)['person']['id'] == original
    assert detail(api, previously_assigned)['assignment_revision'] == 1
    assert correct(api, edited, revision=edited_revision, text='stale before split').status_code == 409


def test_legacy_split_restores_unassigned_memory_and_clears_survivor_archival(api, conn):
    survivor, original = person(conn, 'Survivor'), person(conn, 'Original')
    mid, _ = seed_memory(conn, original)
    withdrawn, _ = seed_memory(conn, original, text='withdrawn before merge')
    assert correct(api, withdrawn, action='withdraw').status_code == 200
    log_id = merge(api, survivor, original)
    snapshot = json.loads(repo.get_merge_log(conn, log_id)['snapshot'])
    snapshot.pop('memory_assignment_revisions')
    conn.execute('UPDATE merge_log SET snapshot=? WHERE id=?', (json.dumps(snapshot), log_id))
    conn.commit()
    assert api.client.delete(f'/api/persons/{survivor}').status_code == 200

    split(api, log_id)

    assert detail(api, mid)['person']['id'] == original
    assert detail(api, withdrawn)['status'] == 'withdrawn'
    assert [row['id'] for row in repo.memories_of(conn, original)] == [mid]


def test_audit_write_failure_rolls_back_memory_evidence_fts_and_vector(api, conn):
    pid = person(conn)
    mid, eid = seed_memory(conn, pid, text='oldcobalt memory')
    repo.upsert_vector(conn, 'memory', mid, 'oldcobalt memory', [1., 0.])
    conn.execute("CREATE TRIGGER fail_memory_correction BEFORE INSERT ON memory_correction "
                 "BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    conn.commit()
    before = dict(conn.execute('SELECT * FROM memory_item WHERE id=?', (mid,)).fetchone())

    with pytest.raises(sqlite3.IntegrityError, match='audit unavailable'):
        correct(api, mid, text='newquartz memory')

    assert dict(conn.execute('SELECT * FROM memory_item WHERE id=?', (mid,)).fetchone()) == before
    assert [row['id'] for row in conn.execute('SELECT id FROM evidence')] == [eid]
    assert conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'oldcobalt'").fetchone()
    assert not conn.execute("SELECT 1 FROM memory_fts WHERE memory_fts MATCH 'newquartz'").fetchone()
    assert repo.all_vectors(conn, 'memory')[0]['owner_id'] == mid
    assert not history(api, mid)


def test_purge_deletes_exclusive_origin_and_history_but_protects_other_memory_history(api, conn):
    purged, retained = person(conn, 'Purge'), person(conn, 'Keep')
    retained_mid, retained_origin = seed_memory(conn, retained)
    first = correct(api, retained_mid, text='first retained correction')
    assert first.status_code == 200, first.text
    historical_eid = first.json()['memory']['source']['id']
    assert correct(api, retained_mid, text='second retained correction').status_code == 200
    shared_mid, _ = seed_memory(conn, purged, evidence_id=historical_eid)
    exclusive, exclusive_origin = seed_memory(conn, purged, text='exclusive origin')
    assert correct(api, exclusive, text='first exclusive correction').status_code == 200
    first_exclusive = detail(api, exclusive)['source']['id']
    assert correct(api, exclusive, text='second exclusive correction').status_code == 200
    latest_exclusive = detail(api, exclusive)['source']['id']
    assert correct(api, exclusive, action='withdraw').status_code == 200

    response = api.client.post(f'/api/persons/{purged}/purge')

    assert response.status_code == 200, response.text
    assert not conn.execute('SELECT 1 FROM memory_item WHERE id IN (?,?)', (exclusive, shared_mid)).fetchone()
    assert not conn.execute('SELECT 1 FROM memory_correction WHERE memory_id=?', (exclusive,)).fetchone()
    for eid in (exclusive_origin, first_exclusive, latest_exclusive):
        assert not conn.execute('SELECT 1 FROM evidence WHERE id=?', (eid,)).fetchone()
    for eid in (retained_origin, historical_eid, detail(api, retained_mid)['source']['id']):
        assert conn.execute('SELECT 1 FROM evidence WHERE id=?', (eid,)).fetchone()
    assert len(history(api, retained_mid)) == 2


@pytest.mark.parametrize('target_personal', [False, True])
def test_purge_old_owner_preserves_transferred_memory_history_and_nested_original_source(api, conn, target_personal):
    original = person(conn, 'Original')
    target = None if target_personal else person(conn, 'Target')
    source_eid = repo.add_evidence(conn, 'text', source_ref='followup-outcome:old', excerpt='original communication')
    origin_eid = repo.add_evidence(conn, 'text', source_ref='outcome-memory-review:old',
        excerpt=json.dumps({'source_evidence_id': source_eid, 'original_quote': 'original communication'}))
    repo.upsert_field(conn, original, 'company', 'Old company', 1., source_eid, now_ms())
    mid, _ = seed_memory(conn, original, evidence_id=origin_eid)
    assert correct(api, mid, text='first correction').status_code == 200
    first_correction = detail(api, mid)['source']['id']
    assert correct(api, mid, person_id=target).status_code == 200
    current_eid = detail(api, mid)['source']['id']

    response = api.client.post(f'/api/persons/{original}/purge')

    assert response.status_code == 200, response.text
    memory = detail(api, mid)
    assert memory['text'] == 'first correction'
    assert memory['person'] == (None if target_personal else {'id': target, 'display_name': 'Target'})
    assert memory['source']['id'] == current_eid
    assert memory['origin_source']['id'] == origin_eid
    assert memory['original_source']['id'] == source_eid
    assert memory['original_source']['excerpt'] == 'original communication'
    assert len(history(api, mid)) == 2
    assert conn.execute('SELECT 1 FROM evidence WHERE id=?', (first_correction,)).fetchone()


def test_purge_follows_cyclic_explicit_evidence_links_once(api, conn):
    pid = person(conn)
    first = repo.add_evidence(conn, 'text', excerpt='first')
    second = repo.add_evidence(conn, 'text', excerpt=json.dumps({'source_evidence_id': first}))
    conn.execute('UPDATE evidence SET excerpt=? WHERE id=?', (json.dumps({'source_evidence_id': second}), first))
    seed_memory(conn, pid, evidence_id=first)

    response = api.client.post(f'/api/persons/{pid}/purge')

    assert response.status_code == 200, response.text
    assert not conn.execute('SELECT 1 FROM evidence WHERE id IN (?,?)', (first, second)).fetchone()
