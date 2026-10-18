"""Historical briefs stay readable while new drafts require current memory sources."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import followup
from app.jobs import tasks
from app.store import repo
from app.util import new_id


@pytest.fixture
def scenario(conn, monkeypatch):
    from app.api import followups, memories
    app = FastAPI()
    app.include_router(followups.router)
    app.include_router(memories.router)
    monkeypatch.setattr(followups, 'get_conn', lambda: conn)
    monkeypatch.setattr(memories, 'get_conn', lambda: conn)
    monkeypatch.setattr(tasks, 'start', lambda task_id: None)
    pid = repo.create_person(conn, '林舟', status='active')
    evidence_id = repo.add_evidence(conn, 'text', excerpt='偏好电话沟通')
    mid = repo.add_memory(conn, 'semantic', '偏好电话沟通', pid, 1, evidence_id)
    conn.commit()
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, pid=pid, mid=mid)


def correct(scenario, action='edit', **changes):
    client, mid = scenario.client, scenario.mid
    revision = client.get(f'/api/memories/{mid}').json()['revision']
    response = client.post(f'/api/memories/{mid}/correct', json={
        'action': action, 'request_id': new_id(), 'expected_revision': revision,
        'reason': '核对了本人最新说明', **changes})
    assert response.status_code == 200, response.text
    return response.json()


def prepare(conn, monkeypatch, scenario, during=None):
    response = scenario.client.post(f'/api/persons/{scenario.pid}/followups', json={
        'goal': '确认下次沟通方式', 'request_id': new_id()})
    assert response.status_code == 202, response.text
    fid = response.json()['payload']['followup_id']
    seen = []

    def chat_json(messages, **kwargs):
        assert not conn.in_transaction
        brief = json.loads(messages[-1]['content'])['brief']
        seen.append(brief)
        if during:
            during()
        return {'draft': '下次什么时候方便联系？',
                'source_ids': [item['source_id'] for item in brief['memories']]}

    monkeypatch.setattr(followup, 'get_llm', lambda: SimpleNamespace(chat_json=chat_json))
    return fid, seen


def test_edited_source_warns_without_rewriting_old_brief_or_draft(conn, monkeypatch, scenario):
    fid, _ = prepare(conn, monkeypatch, scenario)
    before = followup.prepare_followup(conn, fid)
    assert before['brief_memory_changes'] == []
    stored = conn.execute('SELECT brief,draft FROM followup WHERE id=?', (fid,)).fetchone()
    correct(scenario, text='偏好邮件沟通', valid_from='2026-01-15')
    after = scenario.client.get(f'/api/followups/{fid}').json()
    assert after['brief'] == before['brief']
    assert after['draft'] == before['draft']
    assert after['brief_memory_changes'] == [{'id': scenario.mid,
        'source_id': 'memory:' + scenario.mid, 'reason': 'updated', 'text': '偏好电话沟通'}]
    assert tuple(conn.execute('SELECT brief,draft FROM followup WHERE id=?', (fid,)).fetchone()) == tuple(stored)
    fresh_id, seen = prepare(conn, monkeypatch, scenario)
    fresh = followup.prepare_followup(conn, fresh_id)
    assert seen[0]['memories'][0]['text'] == '偏好邮件沟通'
    assert fresh['brief_memory_changes'] == []


def test_withdrawal_and_restoration_affect_only_new_briefs(conn, monkeypatch, scenario):
    fid, _ = prepare(conn, monkeypatch, scenario)
    original = followup.prepare_followup(conn, fid)
    correct(scenario, 'withdraw')
    assert followup.read_followup(conn, fid)['brief_memory_changes'][0]['reason'] == 'withdrawn'
    fresh_id, seen = prepare(conn, monkeypatch, scenario)
    assert followup.prepare_followup(conn, fresh_id)['brief']['memories'] == []
    assert seen[0]['memories'] == []
    correct(scenario, 'restore')
    restored_id, _ = prepare(conn, monkeypatch, scenario)
    assert followup.prepare_followup(conn, restored_id)['brief']['memories'][0]['id'] == scenario.mid
    assert followup.read_followup(conn, fid)['brief'] == original['brief']
    assert followup.read_followup(conn, fid)['brief_memory_changes'][0]['reason'] == 'updated'


@pytest.mark.parametrize('change,reason', [('move', 'moved'), ('delete', 'unavailable'),
                                         ('archive', 'unavailable')])
def test_owner_and_lifecycle_changes_are_visible_in_old_briefs(conn, monkeypatch, scenario, change, reason):
    fid, _ = prepare(conn, monkeypatch, scenario)
    before = followup.prepare_followup(conn, fid)
    if change == 'move':
        other = repo.create_person(conn, '许宁', status='active')
        conn.commit()
        correct(scenario, person_id=other)
    elif change == 'archive':
        conn.execute('UPDATE memory_item SET deleted_at=1 WHERE id=?', (scenario.mid,))
        conn.commit()
    else:
        conn.execute('DELETE FROM memory_item WHERE id=?', (scenario.mid,))
        conn.commit()
    after = followup.read_followup(conn, fid)
    assert after['brief'] == before['brief']
    assert after['brief_memory_changes'][0]['reason'] == reason


def test_legacy_brief_without_revisions_compares_saved_content_and_source(conn, monkeypatch, scenario):
    fid, _ = prepare(conn, monkeypatch, scenario)
    original = followup.prepare_followup(conn, fid)
    for memory in original['brief']['memories']:
        memory.pop('revision')
        memory.pop('person_id')
    conn.execute('UPDATE followup SET brief=? WHERE id=?', (json.dumps(original['brief']), fid))
    conn.commit()
    assert followup.read_followup(conn, fid)['brief_memory_changes'] == []
    correct(scenario, text='偏好邮件沟通')
    assert followup.read_followup(conn, fid)['brief_memory_changes'][0]['reason'] == 'updated'


@pytest.mark.parametrize('action', ['edit', 'withdraw', 'move'])
def test_source_changes_during_model_call_reject_stale_draft_and_retry_uses_current(conn, monkeypatch, scenario, action):
    def during():
        if action == 'move':
            correct(scenario, person_id=None)
        elif action == 'edit':
            correct(scenario, text='偏好邮件沟通')
        else:
            correct(scenario, 'withdraw')

    fid, _ = prepare(conn, monkeypatch, scenario, during)
    with pytest.raises(followup.FollowupError, match='记忆发生变化') as error:
        followup.prepare_followup(conn, fid)
    assert error.value.status_code == 409
    stale = followup.read_followup(conn, fid)
    assert stale['status'] == 'preparing' and not stale['draft']
    monkeypatch.setattr(followup, 'get_llm', lambda: SimpleNamespace(chat_json=lambda *args, **kwargs:
        {'draft': '现在方便聊聊吗？', 'source_ids': []}))
    fresh = followup.prepare_followup(conn, fid)
    assert fresh['status'] == 'ready' and fresh['brief_memory_changes'] == []
    assert [item['text'] for item in fresh['brief']['memories']] == (
        ['偏好邮件沟通'] if action == 'edit' else [])


def test_unrelated_new_memory_during_generation_does_not_reject_existing_sources(conn, monkeypatch, scenario):
    def during():
        repo.add_memory(conn, 'episodic', '新增的独立历史', scenario.pid)
        conn.commit()
    fid, _ = prepare(conn, monkeypatch, scenario, during)
    result = followup.prepare_followup(conn, fid)
    assert result['status'] == 'ready' and result['brief_memory_changes'] == []
    assert len(result['brief']['memories']) == 1
