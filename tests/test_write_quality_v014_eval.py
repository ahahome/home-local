"""Evaluator negative controls; these tests do not measure model quality."""
from copy import deepcopy
import hashlib
import json
import subprocess

import pytest

from eval import memory_questions as base
from eval import write_quality_v014 as evaluation


def fixture():
    return {
        'version': 'test', 'provenance': base.PROVENANCE, 'description': 'fictional development corpus',
        'persons': [{'key': 'one', 'name': '测试甲', 'aliases': []},
                    {'key': 'two', 'name': '测试乙', 'aliases': []}],
        'fields': [], 'memories': [], 'interactions': [],
        'human_review_rubric': ['Check actual task completion separately'],
        'scenarios': [{'id': 'write-retry', 'cohort': 'known_regression', 'steps': [
            {'op': 'chat', 'text': '记下测试甲住在杭州', 'expect': {
                'intent': 'record', 'saved_fields': [{'person': 'one', 'field': 'city', 'value': '杭州'}],
                'review_points': ['Verify actual owner']}},
            {'op': 'chat', 'text': '换成测试乙，查一下', 'expect': {
                'intent': 'query', 'no_memory_writes': True, 'review_points': ['Verify subject switch']}},
            {'op': 'retry', 'source_step': 1, 'expect': {
                'intent': 'record', 'no_memory_writes': True, 'no_pending': True,
                'review_points': ['Retry original message, not latest subject']}}
        ]}]}


def app_modules():
    from app.agent import harness, memory_sources
    from app.config import get_settings
    from app.store import repo
    return {'repo': repo, 'harness': harness, 'sources': memory_sources, 'settings': get_settings()}


def test_frozen_fixture_separates_cohorts_and_covers_production_operations():
    corpus = evaluation.load_fixture()
    assert base.fixture_hash(corpus) == '868eb486b76775807c0ca60a85dd1080e10b5a309aa8f534390b75e8d3de99f1'
    assert hashlib.sha256(evaluation.DEFAULT_FIXTURE.read_bytes()).hexdigest() == \
        'c36b55e7337dbce9d88e49ad1910c890a1cd015d8099ec3de64645467f65e498'
    cases = corpus['scenarios']
    assert len(cases) == 9
    assert {case['cohort'] for case in cases} == evaluation.COHORTS
    assert sum(len(case['steps']) for case in cases) == 18
    assert {step['op'] for case in cases for step in case['steps']} == {'chat', 'retry', 'prepare'}
    assert '非外部盲测' in corpus['description']
    assert all(step['expect']['review_points'] for case in cases for step in case['steps'])


@pytest.mark.parametrize('mutate,message', [
    (lambda f: f.update(provenance='real_user_data'), 'Fictional'),
    (lambda f: f['scenarios'][0].update(cohort='unseen_external_benchmark'), 'cohort'),
    (lambda f: f['scenarios'][0]['steps'][0].update(person_id='one'), 'Hidden'),
    (lambda f: f['scenarios'][0]['steps'][2].update(source_step=3), 'earlier'),
    (lambda f: f['scenarios'][0]['steps'][2].update(source_step=True), 'earlier'),
    (lambda f: f['scenarios'][0]['steps'][1].update(op='confirm_model_proposal'), 'Unsupported'),
    (lambda f: f['scenarios'][0]['steps'][0]['expect'].update(saved_fields=[{'person': 'missing', 'field': 'city', 'value': '杭州'}]), 'field'),
])
def test_fixture_rejects_hidden_scope_unreviewed_expectations_and_invalid_retry(mutate, message):
    corpus = fixture()
    mutate(corpus)
    with pytest.raises(ValueError, match=message):
        evaluation.validate_fixture(corpus)


def test_same_count_foreign_person_changes_and_pending_are_detected(conn):
    app = app_modules()
    pid = app['repo'].create_person(conn, '档案隔离测试', status='active')
    eid = app['repo'].add_evidence(conn, 'text', excerpt='杭州')
    app['repo'].upsert_field(conn, pid, 'city', '杭州', 1., eid, 1000)
    conn.commit()
    owned = {'one': evaluation._owned_state(conn, pid)}
    conn.execute("UPDATE person_field SET value='错误城市' WHERE person_id=?", (pid,))
    conn.execute("INSERT INTO pending_item(id,kind,payload,created_at) VALUES('pending','field','{}',1)")
    checks = evaluation._extra_checks(app, conn, {}, {
        'unchanged_persons': ['one'], 'absent_fields': [{'person': 'one', 'field': 'city'}],
        'no_pending': True}, {'one': pid}, owned)
    assert len(checks) == 3 and not any(check['passed'] for check in checks)
    # A resolved historical audit row is not an unhandled proposal.
    conn.execute("UPDATE pending_item SET resolved_at=2,resolution='rejected'")
    assert evaluation._extra_checks(app, conn, {}, {'no_pending': True}, {'one': pid}, owned)[0]['passed']


def test_pending_expected_value_cannot_substitute_for_real_current_field(conn):
    app = app_modules()
    pid = app['repo'].create_person(conn, '字段未落库测试', status='active')
    conn.execute("INSERT INTO pending_item(id,kind,payload,created_at) VALUES('p','field',?,1)",
                 (json.dumps({'person_id': pid, 'field': 'city', 'value': '杭州'}),))
    checks = base._write_checks(app, conn, {}, {
        'saved_fields': [{'person': 'one', 'field': 'city', 'value': '杭州'}]}, {'one': pid}, {})
    assert not checks[0]['passed'] and checks[0]['actual'] is None


def test_empty_sources_cannot_hide_a_wrong_resolved_person(conn):
    app = app_modules()
    checks = evaluation._extra_checks(app, conn, {'sources': [], 'source_dependencies': [],
        'conversation_scope': {'blocked': False, 'person_ids': ['wrong-person']}},
        {'allowed_persons': ['one']}, {'one': 'expected-person'}, {})
    assert checks[0]['name'] == 'resolved_scope_persons' and not checks[0]['passed']


def test_correct_target_value_does_not_hide_writes_to_an_unexpected_person(conn):
    app = app_modules()
    checks = evaluation._extra_checks(app, conn, {}, {'saved_fields': [
        {'person': 'one', 'field': 'city', 'value': '杭州'}]}, {'one': 'expected-person'}, {},
        {'person': {'added': [{'id': 'unexpected-person'}]},
         'person_field': {'updated': [{'before': {'person_id': 'expected-person'},
                                      'after': {'person_id': 'unexpected-person'}}]}})
    assert checks[0]['name'] == 'write_person_scope' and not checks[0]['passed']
    assert len(checks[0]['foreign_changes']) == 2


def test_followup_helper_traverses_real_http_and_durable_task_without_domain_writes(conn, monkeypatch):
    from app.agent import followup
    app = app_modules()
    pid = app['repo'].create_person(conn, '草稿路径测试', status='active')
    eid = app['repo'].add_evidence(conn, 'text', excerpt='先邮件后电话')
    app['repo'].add_memory(conn, 'semantic', '先邮件后电话', pid, 1., eid)
    conn.commit()

    class ScriptedDraft:
        def chat_json(self, messages, **kwargs):
            body = json.loads(messages[-1]['content'])
            return {'draft': '你好，我想讨论[问题待补]。\n发送前：补问题清单。',
                    'source_ids': [source['id'] for source in body['brief']['sources']]}

    monkeypatch.setattr(followup, 'get_llm', lambda: ScriptedDraft())
    before = base._data_snapshot(conn)
    result = evaluation._prepare(conn, pid, '起草一封邮件，问题还没整理', 'test-v014-prepare')
    assert result['queued']['kind'] == 'followup_prepare'
    assert result['task']['status'] == 'done' and result['followup']['status'] == 'ready'
    assert result['followup']['person']['id'] == pid and result['answer'].startswith('你好')
    assert base._changes(before, base._data_snapshot(conn)) == {}


def test_retry_calls_real_harness_entry_with_original_user_id_after_subject_switch(conn, monkeypatch):
    app = app_modules()
    seen = []

    def scripted(connection, cid, text, **kwargs):
        seen.append((text, kwargs.get('user_message_id')))
        mid = kwargs.get('user_message_id') or app['repo'].add_message(connection, cid, 'user', text)
        if text.startswith('记下') and not kwargs.get('user_message_id'):
            pid = next(row['id'] for row in app['repo'].all_active_persons(connection)
                       if row['display_name'] == '测试甲')
            eid = app['repo'].add_evidence(connection, 'text', excerpt=text)
            app['repo'].upsert_field(connection, pid, 'city', '杭州', 1., eid, 1000)
        app['repo'].add_message(connection, cid, 'assistant', 'test answer')
        connection.commit()
        return {'answer': 'test answer', 'conversation_id': cid,
                'turn_intent': {'intent': 'record' if text.startswith('记下') else 'query'},
                'conversation_scope': {'user_message_id': mid}, 'sources': [], 'source_dependencies': []}

    monkeypatch.setattr(app['harness'], 'run_turn', scripted)
    corpus = fixture()
    result = evaluation.run_scenario(app, conn, corpus, corpus['scenarios'][0])
    first, second, retry = result['steps']
    assert seen[0][1] is None and seen[1][1] is None
    assert seen[2][1] == first['user_message_id'] != second['user_message_id']
    assert retry['source_step'] == 1 and retry['data_changes'] == {}
    assert retry['stored_before'] == retry['stored_after']
    assert evaluation.summarize([result])['all_checks_passed']


def test_runtime_error_keeps_actual_side_effects_and_does_not_call_it_a_pass(conn, monkeypatch):
    app = app_modules()
    corpus = fixture()
    corpus['scenarios'][0]['steps'] = corpus['scenarios'][0]['steps'][:1]

    def partial_write(connection, cid, text, **kwargs):
        app['repo'].add_message(connection, cid, 'user', text)
        pid = app['repo'].all_active_persons(connection)[0]['id']
        eid = app['repo'].add_evidence(connection, 'text', excerpt='side effect')
        app['repo'].upsert_field(connection, pid, 'city', '错误值', 1., eid, 1000)
        connection.commit()
        raise RuntimeError('answer failed after real write')

    monkeypatch.setattr(app['harness'], 'run_turn', partial_write)
    result = evaluation.run_scenario(app, conn, corpus, corpus['scenarios'][0])
    step = result['steps'][0]
    assert step['error']['type'] == 'RuntimeError'
    assert 'person_field' in step['data_changes'] and step['user_message_id']
    assert step['semantic_review']['status'] == 'not_generated'
    summary = evaluation.summarize([result])
    assert summary['runtime_errors'] == 1 and summary['generated_steps'] == 0
    assert summary['checks'] == 0 and not summary['all_checks_passed']


def test_worker_failure_retains_every_expected_step_and_checkpoints(monkeypatch):
    corpus = fixture()
    monkeypatch.setattr(evaluation, '_execute', lambda job, timeout: {
        'error': {'type': 'WorkerTimeout', 'message': 'deliberate'}})
    checkpoints = []
    report = evaluation.evaluate(corpus, checkpoint=lambda value: checkpoints.append(
        (value['run_status'], len(value['details']))))
    assert checkpoints == [('running', 0), ('running', 1), ('complete', 1)]
    assert report['summary']['steps'] == report['summary']['runtime_errors'] == 3
    assert report['summary']['checks'] == 0 and not report['summary']['all_checks_passed']
    assert report['summary']['semantic_review']['score'] is None


def test_cohort_selection_is_explicit_and_no_sampling_occurs_during_validation(monkeypatch):
    corpus = fixture()
    fresh = deepcopy(corpus['scenarios'][0])
    fresh.update(id='fresh', cohort='new_wording')
    corpus['scenarios'].append(fresh)
    seen = []
    monkeypatch.setattr(evaluation, '_execute', lambda job, timeout: seen.append(job['case']['id']) or {
        'error': {'type': 'NotRun'}})
    evaluation.validate_fixture(corpus)
    assert seen == []
    report = evaluation.evaluate(corpus, cohort='new_wording')
    assert seen == ['fresh'] and report['configuration']['repeat'] == 1
    assert len(report['details']) == 1 and report['details'][0]['cohort'] == 'new_wording'


def test_timeout_never_parses_or_prints_sensitive_subprocess_output(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == .01
        raise subprocess.TimeoutExpired(args[0], .01, output='sensitive raw transport text')
    monkeypatch.setattr(subprocess, 'run', timeout)
    result = evaluation._execute({}, .01)
    assert result['error']['type'] == 'WorkerTimeout'
    assert 'sensitive' not in json.dumps(result)


def test_report_retains_original_semantic_pending_status_and_failure(tmp_path):
    report = {'report_id': 'test', 'app_source_hash': 'a' * 64, 'fixture_hash': 'b' * 64,
              'runner_source_hash': 'c' * 64, 'summary': {'semantic_accuracy': None},
              'details': [{'id': 'failed', 'cohort': 'known_regression', 'steps': [
                  {'step': 1, 'text': 'question', 'expected': {'review_points': ['review']},
                   'error': {'type': 'KnownFailure'}, 'semantic_review': {'status': 'not_generated'}}]}]}
    path, review = evaluation.write_report(report, tmp_path)
    saved = json.loads(path.read_text())
    assert saved['details'][0]['steps'][0]['error']['type'] == 'KnownFailure'
    assert saved['details'][0]['steps'][0]['semantic_review']['status'] == 'not_generated'
    assert 'PENDING' in review.read_text() and '不计算准确率' in review.read_text()
