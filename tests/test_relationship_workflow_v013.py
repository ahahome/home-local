"""The v0.13 runner traverses real HTTP services; only model replies are scripted."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from eval.memory_questions import fixture_hash, load_fixture as load_questions
from eval.relationship_workflow_v013 import (
    DEFAULT_FIXTURE, _audit_model_responses, _user_review_selection, load_fixture,
    run_workflow, workflow_source_checks,
)


def test_v013_new_question_and_workflow_fixtures_remain_frozen():
    questions = DEFAULT_FIXTURE.parent / 'answer_evidence_v013.json'
    corpus = load_questions(questions)
    workflow = load_fixture()
    assert fixture_hash(corpus) == '7862601da12a9b2c7b1d4d3fd57724936c994aa33baca330fbf1fd106759916a'
    assert hashlib.sha256(questions.read_bytes()).hexdigest() == 'feea152760dd3035d65e87e68292d3ef630810502e6c3cb39ecbf087c75dfe78'
    assert fixture_hash(workflow) == 'fe6d223d909236cf284b4449d47ad3136374af2c75b7acbfd5b9f090b1d7ca13'
    assert len(corpus['dialogues']) == 9
    assert sum(len(case['turns']) for case in corpus['dialogues']) == 20
    assert {person['name'] for person in corpus['persons']}.isdisjoint({
        person['name'] for person in load_questions(DEFAULT_FIXTURE.parent / 'memory_dialogue_v012.json')['persons']})


class ScriptedModel:
    def __init__(self, *, empty_analysis=False, invalid_evidence=False, invalid_draft_sources=False,
                 partial_proposals=False):
        self.empty_analysis = empty_analysis
        self.invalid_evidence = invalid_evidence
        self.invalid_draft_sources = invalid_draft_sources
        self.partial_proposals = partial_proposals
        self.phases = []

    def chat(self, messages, **kwargs):
        system = messages[0]['content']
        if '写入意图分类器' in system:
            self.phases.append('classifier')
            return {'content': json.dumps({'intent': 'query', 'record_spans': [], 'reason': '只查资料'})}
        if '本轮回答的资料编辑' in system:
            self.phases.append('review')
            data = json.loads(messages[-1]['content'])
            return {'content': json.dumps({'answer_type': 'factual', 'segments': [
                {'kind': 'source_quote', 'text': '', 'source_ids': [source['id']]}
                for source in data['sources']], 'issues': []}, ensure_ascii=False)}
        self.phases.append('answer')
        return {'content': '测试模型草稿，以真实来源为准。 [[S1]]'}

    def chat_json(self, messages, **kwargs):
        data = json.loads(messages[-1]['content'])
        if 'brief' in data:
            self.phases.append('followup_draft')
            ids = ['unknown:source'] if self.invalid_draft_sources else [
                source['id'] for source in data['brief']['sources']]
            return {'draft': '你好，我会先把技术合作的问题清单发邮件，收到后再约电话方便吗？', 'source_ids': ids}
        self.phases.append('outcome_analysis')
        if self.empty_analysis:
            return {'proposals': []}
        person_id = data['person_id']
        proposals = [
            {'person_id': person_id, 'kind': 'field', 'field': 'company', 'content': '澄光传感',
             'memory_kind': None, 'valid_from': None,
             'evidence': '不在原文的公司' if self.invalid_evidence else '白芮说目前在澄光传感工作'},
            {'person_id': person_id, 'kind': 'field', 'field': 'city', 'content': '合肥',
             'memory_kind': None, 'valid_from': None, 'evidence': '住在合肥'},
            {'person_id': person_id, 'kind': 'memory', 'field': None,
             'content': '讨论报价时先发邮件，确认清单后再约电话', 'memory_kind': 'semantic', 'valid_from': None,
             'evidence': '白芮说讨论报价时先发邮件，确认清单后再约电话'}]
        if self.partial_proposals:
            proposals[0]['content'] = '在澄光传感工作'
            proposals[1]['content'] = '住在合肥'
            proposals[2].update(content='讨论报价时先发邮件', evidence='白芮说讨论报价时先发邮件')
        return {'proposals': proposals}


@pytest.fixture
def workflow_env(conn, monkeypatch):
    from app.agent import harness, followup, outcome_memory
    from app.config import get_settings
    from app.store import repo

    def execute(*, user_review=False, **flags):
        model = ScriptedModel(**flags)
        for module in (harness, followup, outcome_memory):
            monkeypatch.setattr(module, 'get_llm', lambda: model)
        result = run_workflow({'repo': repo, 'settings': get_settings()}, conn, deepcopy(load_fixture()),
                              user_review=user_review)
        return result, model
    return execute


def test_real_route_task_outcome_proposal_confirmation_and_requery_contract(workflow_env, conn):
    result, model = workflow_env()
    assert 'error' not in result, result
    assert result['summary']['failed_checks'] == []
    assert result['summary']['workflow_contract_passed']
    assert result['summary']['steps_completed'] == 7
    assert model.phases.count('outcome_analysis') == 1
    assert model.phases.count('followup_draft') == 2
    assert conn.execute('SELECT COUNT(*) FROM followup_outcome').fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM interaction WHERE kind='contact'").fetchone()[0] == 1
    assert conn.execute('SELECT COUNT(*) FROM promise').fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM pending_item WHERE resolved_at IS NOT NULL").fetchone()[0] == 3
    assert result['steps'][3]['counts_before']['memory_item'] == result['steps'][3]['counts_after']['memory_item'] == 1
    assert result['steps'][4]['counts_after']['memory_item'] == 2
    assert result['semantic_review']['semantic_accuracy'] is None
    assert {step['queued']['response']['kind'] for step in result['steps'] if 'queued' in step} == {
        'chat', 'followup_prepare', 'outcome_memory'}


def test_empty_real_model_proposals_do_not_get_replaced_with_expected_answers(workflow_env, conn):
    result, _ = workflow_env(empty_analysis=True)
    assert 'error' not in result
    assert not result['summary']['workflow_contract_passed']
    assert 'model_proposed_required_memory' in result['summary']['failed_checks']
    assert 'confirmed_company_is_current' in result['summary']['failed_checks']
    assert conn.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 1
    assert conn.execute('SELECT COUNT(*) FROM person_field').fetchone()[0] == 1
    assert conn.execute('SELECT COUNT(*) FROM pending_item').fetchone()[0] == 0
    assert result['steps'][4]['resolutions'] == []


def test_invalid_analysis_evidence_fails_real_task_without_memory_changes(workflow_env, conn):
    result, _ = workflow_env(invalid_evidence=True)
    assert result.get('error')
    assert not result['summary']['workflow_contract_passed']
    assert result['steps'][-1]['name'] == 'analyze_memory_proposals'
    assert conn.execute('SELECT COUNT(*) FROM pending_item').fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 1
    assert conn.execute("SELECT status FROM task WHERE kind='outcome_memory'").fetchone()[0] == 'failed'


def test_unknown_draft_reference_stops_before_any_contact_or_confirmation(workflow_env, conn):
    result, _ = workflow_env(invalid_draft_sources=True)
    assert result.get('error')
    assert result['steps'][-1]['name'] == 'draft_communication'
    assert not result['summary']['workflow_contract_passed']
    assert conn.execute('SELECT COUNT(*) FROM interaction').fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM followup_outcome').fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM pending_item').fetchone()[0] == 0


def test_fixture_loader_rejects_missing_provenance(tmp_path):
    path = tmp_path / 'missing.json'
    path.write_text(json.dumps({'version': 'test'}), encoding='utf-8')
    with pytest.raises(ValueError, match='Fictional provenance'):
        load_fixture(path)


def test_explicit_user_review_keeps_quality_failures_and_confirms_real_source_bound_edits(workflow_env, conn):
    result, model = workflow_env(user_review=True, partial_proposals=True)
    summary = result['summary']
    assert 'error' not in result, result
    assert summary['workflow_contract_passed']
    assert not summary['all_checks_passed']
    assert not summary['model_proposal_quality_passed']
    assert summary['failed_checks'] == ['model_proposed_required_company', 'model_proposed_required_city',
                                        'model_proposed_required_memory']
    assert summary['model_proposal_quality_checks_passed'] == 0
    assert model.phases.count('outcome_analysis') == 1
    analysis, confirmation = result['steps'][3:5]
    assert [p['content'] for p in analysis['selected_proposals']] == [
        '在澄光传感工作', '住在合肥', '讨论报价时先发邮件']
    assert all(item['accepted'] and item['edited'] for item in analysis['user_review_decisions'])
    assert len(confirmation['revision_guards']) == len(confirmation['resolutions']) == len(confirmation['replays']) == 3
    assert all(item['status_code'] == 409 for item in confirmation['revision_guards'])
    assert [item['submitted_body']['content'] for item in confirmation['resolutions']] == [
        '澄光传感', '合肥', load_fixture()['expected']['memory_evidence']]
    for item in confirmation['confirmation_source_audit']:
        envelope = json.loads(item['evidence']['excerpt'])
        original = next(p for p in analysis['proposals'] if p['id'] == item['proposal_id'])
        assert envelope['original_quote'] == original['evidence']
        assert envelope['edited_by_user'] is True
        assert envelope['user_confirmed']['content'] != original['content']
    memory_id = confirmation['confirmed_memory_ids'][0]
    stored = conn.execute('SELECT text FROM memory_item WHERE id=?', (memory_id,)).fetchone()
    assert stored['text'] == load_fixture()['expected']['memory_evidence']
    visible = confirmation['visible_memory_details'][0]['response']
    assert visible['source']['source_ref'].startswith('outcome-memory-review:')
    assert visible['original_source']['source_ref'].startswith('followup-outcome:')
    assert any(check.get('validation') == 'followup_outcome_backlink' and check['passed']
               for check in result['steps'][2]['checks'])


def test_default_mode_does_not_silently_repair_partial_model_proposals(workflow_env):
    result, _ = workflow_env(partial_proposals=True)
    assert not result['summary']['workflow_contract_passed']
    assert result['steps'][3]['selected_proposals'] == []
    assert result['steps'][3]['user_review_decisions'] == []
    assert result['steps'][4]['resolutions'] == []


def test_user_review_never_manufactures_missing_proposals(workflow_env, conn):
    result, _ = workflow_env(user_review=True, empty_analysis=True)
    assert not result['summary']['workflow_contract_passed']
    assert all(item['reason'] == 'no_unique_existing_proposal' for item in result['steps'][3]['user_review_decisions'])
    assert result['steps'][4]['resolutions'] == []
    assert conn.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 1
    assert conn.execute('SELECT COUNT(*) FROM person_field').fetchone()[0] == 1


@pytest.mark.parametrize('change,reason', [
    ('duplicate', 'no_unique_existing_proposal'),
    ('wrong_person', 'proposal_owner_source_or_state_mismatch'),
    ('wrong_outcome', 'proposal_owner_source_or_state_mismatch'),
    ('stale', 'proposal_owner_source_or_state_mismatch'),
    ('unsupported', 'proposal_or_reviewed_text_not_supported_by_original_notes'),
    ('unrelated_memory', 'proposal_evidence_is_unrelated_to_reviewed_text'),
])
def test_user_review_rejects_ambiguous_unrelated_and_changed_candidates(change, reason):
    fixture = load_fixture()
    proposals = []
    for kind, field, key in [('field', 'company', 'company'), ('field', 'city', 'city'),
                             ('memory', None, 'memory_evidence')]:
        proposals.append({'id': key, 'kind': kind, 'field': field, 'person': {'id': 'person'},
                          'outcome_id': 'outcome', 'followup_id': 'followup', 'status': 'pending',
                          'stale': False, 'conflict': False,
                          'content': fixture['expected'][key], 'evidence': fixture['expected'][key]})
    target = proposals[2] if change == 'unrelated_memory' else proposals[0]
    if change == 'duplicate':
        proposals.append(deepcopy(target))
    elif change == 'wrong_person':
        target['person']['id'] = 'someone-else'
    elif change == 'wrong_outcome':
        target['outcome_id'] = 'another-outcome'
    elif change == 'stale':
        target['stale'] = True
    elif change == 'unsupported':
        target['evidence'] = '未发生的公司变更'
    elif change == 'unrelated_memory':
        target.update(content=fixture['expected']['company'], evidence=fixture['expected']['company'])
    selected, decisions = _user_review_selection(proposals, 'person', 'outcome', 'followup', fixture)
    decision = next(item for item in decisions if item['field'] == target['field'])
    assert not decision['accepted']
    assert decision['reason'] == reason
    assert target['id'] not in {item['id'] for item in selected}


@pytest.mark.parametrize('corruption', [None, 'snapshot_channel', 'snapshot_role', 'snapshot_recorded_at',
                                       'outcome_notes', 'outcome_channel', 'outcome_contact', 'followup_owner',
                                       'contact_role', 'forged_evidence'])
def test_outcome_source_requires_complete_real_database_backlink(workflow_env, conn, corruption):
    from app.agent.memory_sources import interaction_source
    from app.store import repo
    from eval.memory_questions import source_checks

    result, _ = workflow_env()
    outcome = result['steps'][2]['outcome']
    source = interaction_source(conn, outcome['contact_id'])
    assert source['evidence'] is None  # Promise + full-notes evidence make ID selection ambiguous.
    assert 'source evidence does not exist' in source_checks(conn, source)
    if corruption == 'snapshot_channel':
        source['channel'] = '邮件'
    elif corruption == 'snapshot_role':
        source['people'][0]['role'] = 'mentioned'
    elif corruption == 'snapshot_recorded_at':
        source['record_context']['recorded_at'] += 1
    elif corruption in {'outcome_notes', 'outcome_channel'}:
        row = conn.execute('SELECT payload FROM followup_outcome WHERE id=?', (outcome['id'],)).fetchone()
        payload = json.loads(row['payload'])
        payload['notes' if corruption == 'outcome_notes' else 'channel'] = '已被修改'
        conn.execute('UPDATE followup_outcome SET payload=? WHERE id=?', (json.dumps(payload), outcome['id']))
    elif corruption == 'outcome_contact':
        conn.execute('UPDATE followup_outcome SET contact_id=? WHERE id=?', ('missing', outcome['id']))
    elif corruption == 'followup_owner':
        other = repo.create_person(conn, '另一人物', status='active')
        followup_id = result['steps'][1]['record']['id']
        conn.execute('UPDATE followup SET person_id=? WHERE id=?', (other, followup_id))
    elif corruption == 'contact_role':
        conn.execute('UPDATE interaction_person SET role=? WHERE interaction_id=?', ('mentioned', outcome['contact_id']))
    elif corruption == 'forged_evidence':
        source['evidence'] = {'id': 'missing'}
    errors = workflow_source_checks(conn, source)
    if corruption is None:
        assert errors == []
    else:
        assert errors, corruption
        assert 'source evidence does not exist' in errors


def test_model_response_audit_observes_original_bytes_and_json_retries_without_replacing_calls(monkeypatch):
    from app.llm import LLM

    replies = [{'content': 'invalid json'}, {'content': '```json\n{"proposals": []}\n```'}]
    calls = []

    def original(client, messages, **kwargs):
        calls.append(deepcopy(messages))
        return replies[len(calls) - 1]

    monkeypatch.setattr(LLM, 'chat', original)
    records = []
    llm = object.__new__(LLM)
    with _audit_model_responses(records):
        assert llm.chat_json([{'role': 'system', 'content': 'fixture model request'}]) == {'proposals': []}
    assert LLM.chat is original
    assert len(calls) == len(records) == 2
    assert calls[1][1]['content'] == replies[0]['content']
    assert records[1]['response'] == replies[1]
    assert records[1]['content_sha256'] == hashlib.sha256(replies[1]['content'].encode()).hexdigest()
    assert records[1]['response'] is not replies[1]


def test_model_response_audit_does_not_store_exception_bodies_or_retry(monkeypatch):
    from app.llm import LLM

    calls = []

    def original(*args, **kwargs):
        calls.append(True)
        raise RuntimeError('private provider response')

    monkeypatch.setattr(LLM, 'chat', original)
    records = []
    with _audit_model_responses(records), pytest.raises(RuntimeError):
        object.__new__(LLM).chat([])
    assert len(calls) == len(records) == 1
    assert 'private provider response' not in json.dumps(records)


@pytest.mark.parametrize('phase', ['stale', 'replay'])
def test_revision_and_replay_checks_detect_same_count_updates(workflow_env, conn, monkeypatch, phase):
    from app.agent import outcome_memory

    original = outcome_memory.resolve_proposal
    injected = []

    def mutate_existing_row_before_resolve(database, proposal_id, body):
        payload = json.loads(database.execute('SELECT payload FROM pending_item WHERE id=?', (proposal_id,)).fetchone()[0])
        selected = (phase == 'stale' and payload['status'] == 'pending'
                    and body['expected_revision'] != payload['revision']) or \
                   (phase == 'replay' and payload['status'] == 'confirmed')
        if selected and not injected:
            injected.append(True)
            database.execute('UPDATE followup SET goal=goal||? WHERE id=?', (' changed', payload['followup_id']))
            database.commit()
        return original(database, proposal_id, body)

    monkeypatch.setattr(outcome_memory, 'resolve_proposal', mutate_existing_row_before_resolve)
    result, _ = workflow_env(user_review=True)
    assert injected and 'error' not in result
    assert not result['summary']['workflow_contract_passed']
    prefix = 'stale_revision_does_not_write:' if phase == 'stale' else 'confirm_replay_is_idempotent:'
    assert any(name.startswith(prefix) for name in result['summary']['failed_checks'])
    assert conn.execute('SELECT COUNT(*) FROM followup').fetchone()[0] == 2
