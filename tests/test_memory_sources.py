"""Stored quotations, source state, and retrieval scoping at candidate selection."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import memory_sources, tools
from app.memory import compile_card, memory_corrections, retrieve
from app.memory.vector import VectorIndex
from app.store import repo
from app.util import new_id, now_ms


def person(conn, name):
    return repo.create_person(conn, name, status='active')


def memory(conn, pid=None, text='quartz preference', evidence=None):
    evidence = evidence or repo.add_evidence(conn, 'text', source_ref=new_id(), excerpt=text)
    return repo.add_memory(conn, 'semantic', text, pid, .9, evidence)


def correct(conn, mid, action='edit', **changes):
    revision = conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0]
    conn.commit()
    return memory_corrections.correct_memory(conn, mid, {
        'action': action, 'expected_revision': revision, 'request_id': new_id(), **changes})


def no_vectors(monkeypatch):
    from app import llm
    def disabled():
        raise ValueError('embedding disabled for deterministic FTS tests')
    monkeypatch.setattr(llm, 'get_llm', disabled)


def test_memory_snapshot_and_state_survive_edit_transfer_withdraw_and_archive(conn):
    source_person, target = person(conn, 'Lin'), person(conn, 'Chen')
    mid = memory(conn, source_person)
    original = memory_sources.memory_source(conn, mid)
    assert original['person'] == {'id': source_person, 'display_name': 'Lin'}
    assert original['text'] == original['evidence']['excerpt'] == 'quartz preference'
    assert original['temporal_status'] == 'historical_background'
    assert memory_sources.source_state(conn, original | {'id': 'S1'})['status'] == 'current'

    correct(conn, mid, text='quartz prefers email')
    state = memory_sources.source_state(conn, original)
    assert state['status'] == 'updated' and state['current']['text'] == 'quartz prefers email'
    assert state['current']['origin_source'] == original['evidence']
    assert original['text'] == 'quartz preference'

    correct(conn, mid, person_id=target)
    state = memory_sources.source_state(conn, original)
    assert state['status'] == 'moved' and state['current']['person']['id'] == target
    correct(conn, mid, action='withdraw')
    assert memory_sources.memory_source(conn, mid) is None
    assert memory_sources.source_state(conn, original)['status'] == 'withdrawn'
    conn.execute('UPDATE person SET deleted_at=? WHERE id=?', (now_ms(), target))
    assert memory_sources.source_state(conn, original) == {'status': 'unavailable', 'current': None}


def test_review_chain_keeps_true_original_and_initial_evidence(conn):
    pid = person(conn, 'Lin')
    original = repo.add_evidence(conn, 'text', source_ref='followup-outcome:1', excerpt='原始沟通内容')
    review = repo.add_evidence(conn, 'text', source_ref='outcome-memory-review:1',
        excerpt=json.dumps({'source_evidence_id': original, 'user_confirmed': {'content': '确认偏好'}}))
    mid = memory(conn, pid, evidence=review)
    correct(conn, mid, text='人工修订偏好')
    snapshot = memory_sources.memory_source(conn, mid)
    assert snapshot['evidence']['source_ref'].startswith('memory-correction:')
    assert snapshot['initial_source']['id'] == review
    assert snapshot['origin_source']['id'] == original
    assert snapshot['origin_source']['excerpt'] == '原始沟通内容'


def test_plain_json_evidence_cannot_replace_origin_via_an_unrelated_id(conn):
    unrelated = repo.add_evidence(conn, 'text', source_ref='other', excerpt='another person')
    original = repo.add_evidence(conn, 'text', source_ref='user-message',
                                 excerpt=json.dumps({'source_evidence_id': unrelated}))
    mid = memory(conn, evidence=original)
    assert memory_sources.memory_source(conn, mid)['origin_source']['id'] == original


def test_fields_separate_current_and_history_and_snapshot_closure_is_updated(conn):
    pid = person(conn, 'Lin')
    evidence = repo.add_evidence(conn, 'text', source_ref='first', excerpt='old company')
    repo.upsert_field(conn, pid, 'company', 'Old Co', .9, evidence, valid_from=1000)
    old = memory_sources.field_sources(conn, pid)[0]
    repo.upsert_field(conn, pid, 'company', 'New Co', .9, None, valid_from=2000)
    sources = memory_sources.field_sources(conn, pid)
    assert [(row['text'], row['temporal_status']) for row in sources] == [
        ('New Co', 'current_field'), ('Old Co', 'historical_field')]
    assert sources[1]['valid_to'] == 2000
    assert memory_sources.source_state(conn, old)['status'] == 'updated'
    assert memory_sources.source_state(conn, sources[1])['status'] == 'current'
    assert [row['text'] for row in memory_sources.field_sources(conn, pid, limit=0)] == ['New Co']
    conn.execute('UPDATE person SET status=\'merged\' WHERE id=?', (pid,))
    assert memory_sources.field_sources(conn, pid) == []
    assert memory_sources.source_state(conn, old)['status'] == 'unavailable'


def test_interaction_source_preserves_people_role_and_avoids_ambiguous_file_evidence(conn):
    first, second = person(conn, 'Lin'), person(conn, 'Chen')
    iid, _ = repo.insert_interaction(conn, 1000, 'chat', 'quartz discussion', 'same-file',
                                     [(first, 'participant'), (second, 'mentioned')], kind='contact')
    source = memory_sources.interaction_source(conn, iid)
    assert source['person'] is None
    assert {p['id']: p['role'] for p in source['people']} == {first: 'participant', second: 'mentioned'}
    assert source['temporal_status'] == 'contact' and source['evidence'] is None
    repo.add_evidence(conn, 'text', source_ref='same-file', excerpt='quartz discussion')
    assert memory_sources.interaction_source(conn, iid)['evidence']['excerpt'] == 'quartz discussion'
    repo.add_evidence(conn, 'blob', blob_hash='another-blob', source_ref='same-file', excerpt='other import')
    assert memory_sources.interaction_source(conn, iid)['evidence'] is None
    conn.execute('UPDATE person SET deleted_at=1 WHERE id=?', (second,))
    assert memory_sources.interaction_source(conn, iid) is None
    assert memory_sources.source_state(conn, source)['status'] == 'unavailable'


def test_get_person_preserves_card_and_supplies_only_active_background(conn):
    pid = person(conn, 'Lin')
    for number in range(12):
        memory(conn, pid, f'quartz {number}')
    withdrawn = memory(conn, pid, 'withdrawn preference')
    correct(conn, withdrawn, action='withdraw')
    repo.upsert_field(conn, pid, 'company', 'Old', .9, None, valid_from=1000)
    repo.upsert_field(conn, pid, 'company', 'New', .9, None, valid_from=2000)
    card = compile_card.build_card(conn, pid)
    result = tools.execute_tool(conn, 'get_person', {'person_id': pid})
    assert result['ok'] and result['card'] == card
    memories = [source for source in result['sources'] if source['type'] == 'memory']
    assert len(memories) == 10 and withdrawn not in {source['entity_id'] for source in memories}
    assert {source['temporal_status'] for source in result['sources']} == {
        'current_field', 'historical_field', 'historical_background'}


def test_person_and_personal_fts_scopes_filter_before_limit(conn, monkeypatch):
    no_vectors(monkeypatch)
    target, other = person(conn, 'Lin'), person(conn, 'Chen')
    for _ in range(12):
        memory(conn, other, 'quartz')
    desired = memory(conn, target, 'quartz with several extra words')
    personal = memory(conn, None, 'quartz personal background')
    contact, _ = repo.insert_interaction(conn, now_ms(), 'chat', 'quartz target contact', new_id(),
                                         [(target, 'participant')], kind='contact')
    note, _ = repo.insert_interaction(conn, now_ms(), 'chat', 'quartz personal note', new_id())
    assert {row['id'] for row in retrieve.hybrid_search(conn, 'quartz', 2, person_id=target)} == {desired, contact}
    assert {row['id'] for row in retrieve.hybrid_search(conn, 'quartz', 2, scope='personal')} == {personal, note}
    with pytest.raises(ValueError):
        retrieve.hybrid_search(conn, 'quartz', person_id=target, scope='personal')


def test_hidden_and_withdrawn_fts_candidates_cannot_starve_visible_memory(conn, monkeypatch):
    no_vectors(monkeypatch)
    pid = person(conn, 'Lin')
    for _ in range(12):
        memory(conn, pid, 'quartz')
    conn.execute('UPDATE person SET deleted_at=1 WHERE id=?', (pid,))
    for _ in range(12):
        mid = memory(conn, None, 'quartz')
        conn.execute("UPDATE memory_item SET status='withdrawn' WHERE id=?", (mid,))
    desired = memory(conn, None, 'quartz with several extra words')
    results = retrieve.hybrid_search(conn, 'quartz', 1)
    assert [row['id'] for row in results] == [desired]
    assert results[0]['source']['entity_id'] == desired


def test_vector_scope_mask_precedes_similarity_limit_even_with_cached_hidden_rows(conn, monkeypatch):
    from app import llm
    from app.memory import vector
    class FakeLLM:
        def embed(self, texts):
            return [[1., 0.]]
    monkeypatch.setattr(llm, 'get_llm', FakeLLM)
    index = VectorIndex()
    monkeypatch.setattr(vector, 'get_index', lambda: index)
    target, other = person(conn, 'Lin'), person(conn, 'Chen')
    for number in range(8):
        mid = memory(conn, other, f'unrelated item {number}')
        index.add(conn, 'memory', mid, f'unrelated item {number}', [1., 0.])
    desired = memory(conn, target, 'target preference')
    index.add(conn, 'memory', desired, 'target preference', [.8, .2])
    assert index.search(conn, 'memory', [1., 0.], top_k=1)[0][0] != desired
    assert index.search(conn, 'memory', [1., 0.], top_k=1, allowed_ids={desired})[0][0] == desired
    assert retrieve.hybrid_search(conn, 'unmatchedword', 1, person_id=target)[0]['id'] == desired
    conn.execute('UPDATE person SET deleted_at=1 WHERE id=?', (other,))
    assert retrieve.hybrid_search(conn, 'unmatchedword', 1)[0]['id'] == desired


def test_tool_person_name_ambiguity_and_plain_query_inference(conn, monkeypatch):
    no_vectors(monkeypatch)
    first, second = person(conn, '张三'), person(conn, '张三')
    memory(conn, first, 'quartz first person')
    memory(conn, second, 'quartz second person')
    for args in ({'query': 'quartz', 'person': '张三'}, {'query': '张三 quartz'}):
        result = tools.execute_tool(conn, 'query_memory', args)
        assert not result['ok'] and result['needs_confirmation']
        assert {p['person_id'] for p in result['candidates']} == {first, second}
    result = tools.execute_tool(conn, 'query_memory', {'query': 'quartz', 'person_id': first})
    assert result['ok'] and {r['source']['person']['id'] for r in result['results']} == {first}
    unique = person(conn, '陈宁')
    mid = memory(conn, unique, 'quartz unique person')
    result = tools.execute_tool(conn, 'query_memory', {'query': '陈宁 quartz'})
    assert result['ok'] and result['scope_inferred'] and result['scoped_person']['id'] == unique
    assert [row['id'] for row in result['results']] == [mid]
    result = tools.execute_tool(conn, 'query_memory', {'query': '陈宁'})
    assert result['ok'] and [row['id'] for row in result['results']] == [mid]
    person(conn, 'Ann')
    result = tools.execute_tool(conn, 'query_memory', {'query': 'Annual quartz'})
    assert result['ok'] and result['scoped_person'] is None


def test_source_endpoint_only_reads_persisted_assistant_snapshot(conn, monkeypatch):
    from app.api import message_sources
    monkeypatch.setattr(message_sources, 'get_conn', lambda: conn)
    app = FastAPI()
    app.include_router(message_sources.router)
    client = TestClient(app)
    cid, other_cid = repo.create_conversation(conn), repo.create_conversation(conn)
    mid = memory(conn)
    source = memory_sources.memory_source(conn, mid) | {'id': 'S1'}
    assistant = repo.add_message(conn, cid, 'assistant', 'answer [S1]', json.dumps({'sources': [source]}))
    user = repo.add_message(conn, cid, 'user', 'user', json.dumps({'sources': [source]}))
    malformed = repo.add_message(conn, cid, 'assistant', 'legacy', '[{}]')
    conn.commit()
    path = f'/api/conversations/{cid}/messages/{assistant}/sources/S1'
    response = client.get(path)
    assert response.status_code == 200 and response.json()['source'] == source
    assert response.json()['state']['status'] == 'current'
    correct(conn, mid, text='quartz changed after answering')
    response = client.get(path)
    assert response.json()['source'] == source
    assert response.json()['state']['status'] == 'updated'
    assert response.json()['state']['current']['text'] == 'quartz changed after answering'
    for bad in (
        f'/api/conversations/{other_cid}/messages/{assistant}/sources/S1',
        f'/api/conversations/{cid}/messages/{user}/sources/S1',
        f'/api/conversations/{cid}/messages/{malformed}/sources/S1', path[:-2] + 'S2'):
        assert client.get(bad).status_code == 404
    assert client.get(path[:-2] + 'S2', params={'snapshot': json.dumps(source)}).status_code == 404
    client.close()


@pytest.mark.parametrize('snapshot', [None, [], {}, {'type': 'field', 'entity_id': '{}'},
                                    {'type': 'field', 'entity_id': '["person","city",true]'}])
def test_malformed_source_state_is_unavailable(conn, snapshot):
    assert memory_sources.source_state(conn, snapshot) == {'status': 'unavailable', 'current': None}
