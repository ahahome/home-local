"""Read-only memory browsing, literal search, provenance and lifecycle boundaries."""
from copy import deepcopy
from datetime import datetime
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import followup, outcome_memory
from app.jobs import tasks
from app.memory import apply
from app.store import repo
from app.util import new_id, now_ms


@pytest.fixture
def client(conn, monkeypatch):
    from app.api import memories, outcome_memory as updates_api
    app = FastAPI()
    app.include_router(memories.router)
    app.include_router(updates_api.router)
    monkeypatch.setattr(memories, 'get_conn', lambda: conn)
    monkeypatch.setattr(updates_api, 'get_conn', lambda: conn)
    monkeypatch.setattr(tasks, 'start', lambda task_id: None)
    monkeypatch.setattr(outcome_memory, 'get_llm', lambda: pytest.fail('browsing must not call a model'))
    result = TestClient(app)
    yield result
    result.close()


def person(conn, name='林舟', status='active'):
    pid = repo.create_person(conn, name, status=status)
    conn.commit()
    return pid


def memory(conn, pid=None, text='习惯上午沟通', kind='semantic', source=None, ts=1000, created_at=2000):
    mid = repo.add_memory(conn, kind, text, person_id=pid, confidence=0.85, evidence_id=source, ts=ts)
    conn.execute('UPDATE memory_item SET created_at=? WHERE id=?', (created_at, mid))
    conn.commit()
    return mid


def saved(client, **params):
    response = client.get('/api/memories', params=params)
    assert response.status_code == 200, response.text
    return response.json()


def proposals(client, **params):
    response = client.get('/api/memory-updates', params=params)
    assert response.status_code == 200, response.text
    return response.json()


def proposal_fixture(conn, monkeypatch, pid=None, notes='现在住在深圳，习惯上午沟通。', items=None):
    pid = pid or person(conn)
    fid, stamp = new_id(), now_ms()
    conn.execute('INSERT INTO followup(id,person_id,origin_person_id,goal,status,draft,prepare_status,'
        'request_id,request_hash,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
        (fid, pid, pid, '了解近况', 'ready', '最近怎么样？', 'done', new_id(), 'test', stamp, stamp))
    conn.commit()
    record = followup.record_outcome(conn, fid, {'request_id': new_id(), 'expected_revision': 1,
        'contacted_at': stamp - 86400000, 'channel': '电话', 'notes': notes, 'state': 'completed',
        'next_followup_at': None, 'complete_promise_ids': [], 'new_promises': []})['record']
    oid = record['outcomes'][0]['id']
    outcome_memory.create_analysis(conn, oid)
    items = items if items is not None else [
        {'person_id': pid, 'kind': 'memory', 'field': None, 'content': '习惯上午沟通',
         'memory_kind': 'semantic', 'valid_from': None, 'evidence': '习惯上午沟通'},
        {'person_id': pid, 'kind': 'field', 'field': 'city', 'content': '深圳',
         'memory_kind': None, 'valid_from': None, 'evidence': '现在住在深圳'}]
    monkeypatch.setattr(outcome_memory, 'get_llm', lambda: SimpleNamespace(
        chat_json=lambda *a, **k: deepcopy({'proposals': items})))
    result = outcome_memory.run_analysis(conn, oid)
    return SimpleNamespace(pid=pid, fid=fid, oid=oid, proposals=result['proposals'], notes=notes)


def resolve(conn, proposal, action='confirm', **changes):
    return outcome_memory.resolve_proposal(conn, proposal['id'], {
        'action': action, 'expected_revision': proposal['revision'], **changes})


def test_lists_saved_memories_with_person_or_personal_scope_and_includes_unconfirmed(client, conn):
    pid = person(conn)
    unconfirmed = person(conn, '新认识的同事', status='unconfirmed')
    owned = memory(conn, pid)
    own = memory(conn, None, '我的年度目标')
    old_extraction = memory(conn, unconfirmed, '通过资料抽取保留的背景')

    result = saved(client)

    assert result['total'] == 3 and not result['has_more']
    assert {row['id'] for row in result['memories']} == {owned, own, old_extraction}
    assert saved(client, person_id=pid)['memories'][0]['person'] == {'id': pid, 'display_name': '林舟'}
    personal = saved(client, scope='personal')
    assert personal['total'] == 1 and personal['memories'][0]['id'] == own
    assert personal['memories'][0]['person'] is None
    assert personal['memories'][0]['proposal_id'] is None
    assert personal['memories'][0]['followup_id'] is None
    assert personal['memories'][0]['source'] is None


def test_search_matches_literal_content_current_person_name_and_alias_without_duplicates(client, conn):
    pid, other = person(conn, '林舟'), person(conn, '王宁')
    repo.add_alias(conn, pid, 'LZ_100%', 'manual')
    repo.add_alias(conn, pid, 'LZ_100%', 'chat')
    conn.commit()
    target = memory(conn, pid, '项目完成度是100%，请走A_B流程。')
    unrelated = memory(conn, other, '普通摘要')
    personal = memory(conn, None, 'MY_NOTE with 100% readiness')

    assert {row['id'] for row in saved(client, q='%')['memories']} == {target, personal}
    assert {row['id'] for row in saved(client, q='_')['memories']} == {target, personal}
    assert saved(client, q='LZ_100%')['total'] == 1
    assert saved(client, q='林舟')['memories'][0]['id'] == target
    assert saved(client, q='my_note')['memories'][0]['id'] == personal
    assert saved(client, q="' OR 1=1 --")['total'] == 0
    assert saved(client, person_id=other, q='100%')['total'] == 0
    assert unrelated not in {row['id'] for row in saved(client, q='林舟')['memories']}


@pytest.mark.parametrize('kind', ['semantic', 'episodic', 'reflective', 'journal', 'skill'])
def test_each_memory_kind_is_queryable_and_preserves_timestamps_confidence(client, conn, kind):
    mid = memory(conn, kind=kind, ts=1234, created_at=5678)
    row = saved(client, kind=kind)['memories'][0]
    assert row['id'] == mid and row['kind'] == kind
    assert row['ts'] == 1234 and row['created_at'] == 5678
    assert row['confidence'] == 0.85


def test_pagination_has_total_has_more_and_stable_order_for_same_timestamp(client, conn):
    ids = [memory(conn, text=str(i), ts=5000, created_at=6000) for i in range(5)]
    first = memory(conn, text='newer timestamp', ts=5001, created_at=1)
    second = memory(conn, text='newer recorded date', ts=5000, created_at=6001)
    expected = [first, second, *sorted(ids, reverse=True)]
    rows = []
    for offset in (0, 3, 6):
        page = saved(client, limit=3, offset=offset)
        assert page['total'] == 7 and page['limit'] == 3 and page['offset'] == offset
        assert page['has_more'] == (offset < 6)
        rows.extend(row['id'] for row in page['memories'])
    assert rows == expected
    exhausted = saved(client, limit=3, offset=100)
    assert exhausted['total'] == 7 and exhausted['memories'] == [] and not exhausted['has_more']


@pytest.mark.parametrize('params', [
    {'limit': 0}, {'limit': 101}, {'offset': -1}, {'limit': 'abc'},
    {'scope': 'other'}, {'kind': 'contact'}, {'scope': 'personal', 'person_id': 'unknown'},
    {'q': 'x' * 201},
])
def test_invalid_memory_filters_return_422(client, params):
    assert client.get('/api/memories', params=params).status_code == 422


def test_invalid_or_archived_person_filter_is_404_without_identity_redirect(client, conn):
    pid, survivor = person(conn), person(conn, '存活人物')
    memory(conn, pid)
    assert client.get('/api/memories', params={'person_id': 'unknown'}).status_code == 404
    assert client.get('/api/memory-updates', params={'person_id': 'unknown'}).status_code == 404
    repo.archive_person(conn, pid)
    conn.commit()
    assert client.get('/api/memories', params={'person_id': pid}).status_code == 404
    assert client.get('/api/memory-updates', params={'person_id': pid}).status_code == 404
    assert repo.restore_person(conn, pid)
    conn.commit()
    apply.merge_persons(conn, survivor, pid)
    assert client.get('/api/memories', params={'person_id': pid}).status_code == 404
    assert client.get('/api/memory-updates', params={'person_id': pid}).status_code == 404
    assert saved(client, person_id=survivor)['total'] == 1


def test_hidden_deleted_missing_person_and_merged_rows_never_leak_through_search_or_detail(client, conn):
    valid, archived, merged = person(conn), person(conn, '归档人物'), person(conn, '合并墓碑')
    visible = memory(conn, valid, '唯一可见共同搜索词')
    deleted = memory(conn, valid, '共同搜索词')
    archived_memory = memory(conn, archived, '共同搜索词')
    merged_memory = memory(conn, merged, '共同搜索词')
    orphan = memory(conn, 'missing-person', '共同搜索词')
    conn.execute('UPDATE memory_item SET deleted_at=? WHERE id=?', (now_ms(), deleted))
    # Keep memory.deleted_at NULL to independently exercise the owner visibility guard.
    conn.execute('UPDATE person SET deleted_at=? WHERE id=?', (now_ms(), archived))
    conn.execute("UPDATE person SET status='merged' WHERE id=?", (merged,))
    conn.commit()
    assert saved(client, q='共同搜索词')['memories'][0]['id'] == visible
    assert saved(client, q='共同搜索词')['total'] == 1
    for mid in (deleted, archived_memory, merged_memory, orphan, 'missing-memory'):
        assert client.get('/api/memories/' + mid).status_code == 404


def test_list_truncates_evidence_but_detail_preserves_full_original_source(client, conn):
    excerpt = '原始沟通内容：' + '确认已经发生的事实。' * 100
    eid = repo.add_evidence(conn, 'text', source_ref='manual-note:source', excerpt=excerpt)
    mid = memory(conn, source=eid)
    listed = saved(client)['memories'][0]
    detail = client.get('/api/memories/' + mid)
    assert detail.status_code == 200, detail.text
    detail = detail.json()
    assert listed['source']['excerpt'] == excerpt[:300]
    assert detail['source']['excerpt'] == excerpt
    assert detail['source']['id'] == eid and detail['source']['kind'] == 'text'
    assert detail['source']['source_ref'] == 'manual-note:source'
    assert detail['source']['blob_hash'] is None
    assert detail['original_source'] is None


def test_missing_evidence_remains_a_visible_memory_with_null_source(client, conn):
    mid = memory(conn, source='missing-evidence')
    row = client.get('/api/memories/' + mid).json()
    assert row['id'] == mid and row['source'] is None and row['original_source'] is None


def test_confirmed_memory_is_immediately_searchable_and_links_to_original_proposal(client, conn, monkeypatch):
    context = proposal_fixture(conn, monkeypatch)
    proposal = next(p for p in context.proposals if p['kind'] == 'memory')
    assert saved(client)['total'] == 0
    confirmed = resolve(conn, proposal)
    mid = confirmed['receipt']['id']

    row = saved(client, q='习惯上午沟通')['memories'][0]

    assert row['id'] == mid and row['proposal_id'] == proposal['id']
    assert row['followup_id'] == context.fid
    assert row['source']['source_ref'] == 'followup-outcome:' + context.oid
    assert row['source']['excerpt'] == context.notes
    assert row['original_source'] is None
    assert client.get('/api/memories/' + mid).json() == row


def test_edited_confirmation_keeps_review_evidence_and_full_original_communication(client, conn, monkeypatch):
    notes = '现在住在深圳，习惯上午沟通。' + '沟通中补充的历史细节。' * 70
    context = proposal_fixture(conn, monkeypatch, notes=notes)
    proposal = next(p for p in context.proposals if p['kind'] == 'memory')
    confirmed = resolve(conn, proposal, content='优先在工作日上午联系')
    mid = confirmed['receipt']['id']

    listed = saved(client, q='优先在工作日上午联系')['memories'][0]
    detail = client.get('/api/memories/' + mid).json()

    assert listed['source']['excerpt'] == detail['source']['excerpt'][:300]
    assert len(listed['original_source']['excerpt']) == 300
    assert detail['source']['source_ref'] == 'outcome-memory-review:' + proposal['id']
    review = json.loads(detail['source']['excerpt'])
    assert review['edited_by_user'] is True and review['original_quote'] == '习惯上午沟通'
    assert review['user_confirmed']['content'] == detail['text']
    assert review['source_evidence_id'] == detail['original_source']['id']
    assert detail['original_source']['excerpt'] == notes
    assert detail['original_source']['source_ref'] == 'followup-outcome:' + context.oid


def test_pending_discarded_and_field_proposals_do_not_appear_as_saved_memories(client, conn, monkeypatch):
    context = proposal_fixture(conn, monkeypatch)
    memory_proposal = next(p for p in context.proposals if p['kind'] == 'memory')
    field_proposal = next(p for p in context.proposals if p['kind'] == 'field')
    assert saved(client)['total'] == 0
    resolve(conn, field_proposal)
    resolve(conn, memory_proposal, 'discard')
    assert saved(client)['total'] == 0
    assert proposals(client, status='confirmed')['proposals'][0]['kind'] == 'field'
    assert proposals(client, status='discarded')['proposals'][0]['kind'] == 'memory'


def test_proposal_collection_filters_status_kind_person_and_preserves_review_shape(client, conn, monkeypatch):
    first = proposal_fixture(conn, monkeypatch)
    second = proposal_fixture(conn, monkeypatch, pid=person(conn, '另一位同事', status='unconfirmed'))
    target = next(p for p in first.proposals if p['kind'] == 'memory')
    confirmed = resolve(conn, target)
    discarded = next(p for p in second.proposals if p['kind'] == 'field')
    resolve(conn, discarded, 'discard')
    assert proposals(client)['total'] == 4
    assert proposals(client, status='pending')['total'] == 2
    assert proposals(client, status='confirmed', person_id=first.pid, kind='memory')['proposals'] == [confirmed]
    assert proposals(client, status='discarded', person_id=second.pid)['total'] == 1
    assert proposals(client, kind='field')['total'] == 2
    assert proposals(client, person_id=second.pid, kind='memory')['total'] == 1


def test_proposal_search_is_literal_across_content_evidence_person_and_alias(client, conn, monkeypatch):
    notes = '现在住在深圳，习惯上午沟通，完成了100%的任务，登记代号是A_B。'
    context = proposal_fixture(conn, monkeypatch, notes=notes)
    repo.add_alias(conn, context.pid, '林_100%', 'manual')
    conn.commit()
    assert proposals(client, q='%')['total'] == 2
    assert proposals(client, q='_')['total'] == 2
    assert proposals(client, q='林舟')['total'] == 2
    assert proposals(client, q='习惯上午沟通')['total'] == 1
    assert proposals(client, q='不存在')['total'] == 0
    assert proposals(client, q="' OR 1=1 --")['total'] == 0


@pytest.mark.parametrize('params', [
    {'status': 'closed'}, {'kind': 'semantic'}, {'limit': 0}, {'limit': 101}, {'offset': -1},
    {'q': 'x' * 201},
])
def test_invalid_proposal_filters_return_422(client, params):
    assert client.get('/api/memory-updates', params=params).status_code == 422


def test_proposal_collection_pagination_uses_stable_created_at_and_id(client, conn, monkeypatch):
    first = proposal_fixture(conn, monkeypatch)
    second = proposal_fixture(conn, monkeypatch)
    ids = [p['id'] for p in first.proposals + second.proposals]
    conn.execute("UPDATE pending_item SET created_at=1000 WHERE kind='outcome_memory'")
    conn.commit()
    page1, page2 = proposals(client, limit=2), proposals(client, limit=2, offset=2)
    assert page1['total'] == page2['total'] == 4
    assert page1['has_more'] and not page2['has_more']
    assert [p['id'] for p in page1['proposals'] + page2['proposals']] == sorted(ids, reverse=True)


def test_stale_source_proposals_stay_visible_but_existing_reviewer_still_rejects_them(client, conn, monkeypatch):
    context = proposal_fixture(conn, monkeypatch)
    row = conn.execute('SELECT payload,contact_id FROM followup_outcome WHERE id=?', (context.oid,)).fetchone()
    payload = json.loads(row['payload'])
    payload['notes'] += ' 修改后的来源。'
    conn.execute('UPDATE followup_outcome SET payload=? WHERE id=?', (json.dumps(payload), context.oid))
    conn.execute('UPDATE interaction SET content=? WHERE id=?', (payload['notes'], row['contact_id']))
    conn.commit()

    visible = proposals(client, person_id=context.pid)

    assert visible['total'] == 2 and all(p['stale'] for p in visible['proposals'])
    proposal = visible['proposals'][0]
    response = client.post('/api/memory-updates/' + proposal['id'] + '/resolve',
        json={'action': 'confirm', 'expected_revision': proposal['revision']})
    assert response.status_code == 409
    assert saved(client)['total'] == 0


def test_archive_restore_and_merge_split_keep_current_memory_ownership_and_proposal_subject(client, conn, monkeypatch):
    context = proposal_fixture(conn, monkeypatch)
    proposal = next(p for p in context.proposals if p['kind'] == 'memory')
    mid = resolve(conn, proposal)['receipt']['id']
    other = person(conn, '存活人物')
    other_mid = memory(conn, other)
    repo.archive_person(conn, context.pid)
    conn.commit()
    assert saved(client)['total'] == 1
    assert client.get('/api/memories/' + mid).status_code == 404
    assert proposals(client)['total'] == 0
    assert repo.restore_person(conn, context.pid)
    conn.commit()
    assert saved(client)['total'] == 2 and proposals(client)['total'] == 2

    merge_id = apply.merge_persons(conn, other, context.pid)
    merged = client.get('/api/memories/' + mid).json()
    assert merged['person'] == {'id': other, 'display_name': '存活人物'}
    assert merged['proposal_id'] == proposal['id'] and merged['followup_id'] == context.fid
    assert proposals(client)['total'] == 0
    original = client.get('/api/memory-updates/' + proposal['id']).json()
    assert original['person']['id'] == context.pid and original['stale']
    assert {row['id'] for row in saved(client, person_id=other)['memories']} == {mid, other_mid}
    apply.split_merge(conn, merge_id)
    assert client.get('/api/memories/' + mid).json()['person']['id'] == context.pid
    assert saved(client, person_id=other)['total'] == 1
    assert proposals(client, person_id=context.pid)['total'] == 2


def test_purge_removes_person_memories_sources_and_proposals_without_hiding_other_people(client, conn, monkeypatch):
    first = proposal_fixture(conn, monkeypatch)
    second = proposal_fixture(conn, monkeypatch, pid=person(conn, '保留的人'))
    first_memory = resolve(conn, next(p for p in first.proposals if p['kind'] == 'memory'))['receipt']['id']
    second_memory = resolve(conn, next(p for p in second.proposals if p['kind'] == 'memory'))['receipt']['id']
    personal = memory(conn, text='个人日志', kind='journal')

    result = repo.purge_person(conn, first.pid)
    conn.commit()

    assert result['ok']
    assert client.get('/api/memories/' + first_memory).status_code == 404
    assert saved(client)['total'] == 2
    assert {row['id'] for row in saved(client)['memories']} == {second_memory, personal}
    assert proposals(client)['total'] == 2
    assert proposals(client, person_id=second.pid)['total'] == 2


def test_reads_do_not_write_or_analyze_and_leave_pending_state_untouched(client, conn, monkeypatch):
    context = proposal_fixture(conn, monkeypatch)
    mid = memory(conn, context.pid)
    monkeypatch.setattr(outcome_memory, 'get_llm', lambda: pytest.fail('read triggered model'))
    before = conn.total_changes
    task_count = conn.execute('SELECT COUNT(*) FROM task').fetchone()[0]
    for response in (client.get('/api/memories'), client.get('/api/memories/' + mid),
                     client.get('/api/memory-updates'),
                     client.get('/api/memory-updates', params={'status': 'pending'})):
        assert response.status_code == 200, response.text
    assert conn.total_changes == before
    assert conn.execute('SELECT COUNT(*) FROM task').fetchone()[0] == task_count
    assert len(repo.list_pending(conn, kind='outcome_memory')) == 2
