"""Evaluator negative controls and isolated-run contracts, not LLM quality tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from eval import memory_questions as evaluation


def small_fixture():
    fixture = evaluation.load_fixture()
    fixture['retrieval_cases'] = [{
        'id': 'food', 'tags': ['food'], 'question': '香菜', 'args': {'person': '老陈'},
        'expect': {'relevant': ['chen_food', 'chen_lunch'], 'excluded_persons': ['zhou']}}]
    fixture['dialogues'] = [fixture['dialogues'][0]]
    return fixture


def test_fixture_explicitly_discloses_fictional_provenance_and_rejects_hidden_identity():
    fixture = small_fixture()
    fixture['provenance'] = 'real_users'
    with pytest.raises(ValueError, match='fictional'):
        evaluation.validate_fixture(fixture)
    fixture['provenance'] = evaluation.PROVENANCE
    fixture['retrieval_cases'][0]['args']['person_id'] = 'chen'
    with pytest.raises(ValueError, match='hidden person IDs'):
        evaluation.validate_fixture(fixture)


def test_fixture_rejects_missing_or_duplicated_relevance_labels():
    fixture = small_fixture()
    fixture['retrieval_cases'][0]['expect']['relevant'] = ['made-up-record']
    with pytest.raises(ValueError, match='Unknown relevance'):
        evaluation.validate_fixture(fixture)
    fixture = small_fixture()
    fixture['memories'].append(deepcopy(fixture['memories'][0]))
    with pytest.raises(ValueError, match='unique'):
        evaluation.validate_fixture(fixture)


def test_v010_acceptance_uses_old_records_new_questions_and_explicit_turn_contracts():
    directory = evaluation.RUNNER_ROOT / 'eval/fixtures'
    fixture = evaluation.load_fixture(directory / 'answer_stability_v010.json')
    baseline = evaluation.load_fixture(directory / 'answer_trust_v09.json')
    assert fixture['provenance'] == evaluation.PROVENANCE
    assert '虚构开发验收语料' in fixture['description']
    assert '语义逐轮待人工审阅' in fixture['description']
    for key in ('persons', 'fields', 'memories', 'interactions'):
        assert fixture[key] == baseline[key]
    turns = [turn for case in fixture['dialogues'] for turn in case['turns']]
    assert len(fixture['dialogues']) == 7 and len(turns) == 12
    old_questions = {turn['question'] for case in baseline['dialogues'] for turn in case['turns']}
    assert not old_questions & {turn['question'] for turn in turns}
    assert sum(turn['question'] == '这条记忆是什么时候被我修改的？' for turn in turns) == 2
    for turn in turns:
        expect = turn['expect']
        assert expect['intent'] in {'query', 'record', 'mixed'}
        assert expect['no_memory_writes'] is (expect['intent'] == 'query')
        assert 'source_records' in expect and expect['allowed_persons'] and expect['review_points']
        if expect['intent'] != 'query':
            assert expect['saved_fields']
    focuses = [turn['expect']['memory_focus'] for turn in turns if 'memory_focus' in turn['expect']]
    assert focuses == [{'status': 'selected', 'registration_status': 'registered'},
                       {'status': 'clarify', 'reason': 'ambiguous_memory'}]


@pytest.mark.parametrize('focus', [
    'selected', {}, {'status': 'succeeded'}, {'status': 'clarify', 'reason': ''},
    {'status': 'selected', 'registration_status': 'made_up'},
    {'status': 'selected', 'source_id': 'S1'},
])
def test_fixture_rejects_invalid_focus_contracts(focus):
    fixture = small_fixture()
    fixture['dialogues'][0]['turns'][0]['expect']['memory_focus'] = focus
    with pytest.raises(ValueError, match='memory_focus'):
        evaluation.validate_fixture(fixture)


def test_ranking_miss_and_wrong_first_result_reduce_recall_and_mrr():
    metrics = evaluation.ranking_metrics(['foreign', 'other', 'a', 'more', 'b'], ['a', 'b'])
    assert metrics == {'recall_at_1': 0, 'recall_at_3': .5, 'recall_at_5': 1., 'mrr_at_5': 1 / 3}
    missing = evaluation.ranking_metrics(['foreign'], ['a', 'b'])
    assert missing['recall_at_5'] == 0 and missing['mrr_at_5'] == 0
    assert all(value is None for value in evaluation.ranking_metrics([], []).values())


def test_limited_string_rules_reject_old_fact_but_do_not_claim_semantic_understanding():
    rules = [{'id': 'corrected-channel', 'kind': 'contains_any', 'values': ['邮件']}]
    assert not evaluation.limited_fact_checks('急事打电话。', rules)[0]['passed']
    assert evaluation.limited_fact_checks('现在发邮件。', rules)[0]['passed']
    # The probe also accepts a negated sentence. That is why semantics remain pending.
    assert evaluation.limited_fact_checks('不能发邮件。', rules)[0]['passed']
    assert 'human semantic review' in evaluation.limited_fact_checks('邮件', rules)[0]['limitation']


def test_empty_or_fallback_answer_cannot_pass_an_absence_only_rule():
    rules = [{'id': 'no-fabricated-number', 'kind': 'not_contains_any', 'values': ['12345']}]
    for answer in ('', '  ', None, '（这一轮没有产出内容，请重试）'):
        assert not evaluation.limited_fact_checks(answer, rules)[0]['passed']


def seeded_source(conn):
    from app.agent import memory_sources
    from app.store import repo
    pid = repo.create_person(conn, '引用测试', status='active')
    eid = repo.add_evidence(conn, 'text', source_ref='eval:test', excerpt='偏好邮件')
    mid = repo.add_memory(conn, 'semantic', '偏好邮件', pid, .95, eid)
    conn.commit()
    return memory_sources.memory_source(conn, mid), pid


def test_actual_evidence_id_with_forged_excerpt_is_rejected(conn):
    source, _ = seeded_source(conn)
    assert evaluation.source_checks(conn, source) == []
    source['evidence']['excerpt'] = '不存在的原文'
    assert 'source evidence differs from stored evidence' in evaluation.source_checks(conn, source)


def test_wrong_source_owner_and_withdrawn_memory_are_rejected(conn):
    source, _ = seeded_source(conn)
    source['person']['id'] = 'someone-else'
    assert 'memory owner or evidence mismatch' in evaluation.source_checks(conn, source)
    conn.execute("UPDATE memory_item SET status='withdrawn' WHERE id=?", (source['entity_id'],))
    assert 'withdrawn or deleted memory exposed' in evaluation.source_checks(conn, source)


def test_real_source_attached_to_a_different_hit_does_not_forge_recall(conn):
    source, _ = seeded_source(conn)
    app = {'tools': SimpleNamespace(execute_tool=lambda *args, **kwargs: {
        'ok': True, 'results': [{'type': 'memory', 'id': 'wrong-hit-id', 'text': '偏好邮件', 'source': source}]})}
    case = {'id': 'wrong-hit', 'question': '邮件', 'expect': {'relevant': ['expected']}}
    result = evaluation._retrieval_case(app, conn, case, {}, {
        'expected': {'type': 'memory', 'id': source['entity_id']}})
    assert result['metrics']['recall_at_5'] == 0
    assert any('source differs from retrieved hit' in check.get('failures', [])
               for check in result['deterministic_checks'])


def test_historical_field_cannot_be_labelled_current(conn):
    from app.agent import memory_sources
    from app.store import repo
    pid = repo.create_person(conn, '城市测试', status='active')
    eid = repo.add_evidence(conn, 'text', excerpt='杭州，然后上海')
    repo.upsert_field(conn, pid, 'city', '杭州', 1., eid, valid_from=1000)
    repo.upsert_field(conn, pid, 'city', '上海', 1., eid, valid_from=2000)
    source = next(source for source in memory_sources.field_sources(conn, pid) if source['text'] == '杭州')
    assert evaluation.source_checks(conn, source) == []
    source['temporal_status'] = 'current_field'
    assert 'field value, owner, validity or evidence mismatch' in evaluation.source_checks(conn, source)


def test_worker_error_remains_in_positive_retrieval_denominator(monkeypatch):
    monkeypatch.setattr(evaluation, '_execute', lambda job, **kwargs: {
        'detail': {'id': job['case']['id'], 'error': {'type': 'DeliberateFailure'}}})
    report = evaluation.evaluate(small_fixture(), repeat=2)
    assert report['summary']['runtime_errors'] == 2
    assert report['summary']['scenario_errors'] == 2
    assert report['summary']['failed_turns'] is None
    assert report['summary']['retrieval']['positive_queries'] == 2
    assert report['summary']['retrieval']['recall_at_5'] == 0


def test_answers_worker_error_counts_all_expected_turns_and_rules(monkeypatch):
    monkeypatch.setattr(evaluation, '_execute', lambda job, **kwargs: {
        'detail': {'id': job['case']['id'], 'error': {'type': 'DeliberateFailure'}}})
    report = evaluation.evaluate(small_fixture(), track='answers')
    assert report['summary']['units'] == 2
    assert report['summary']['runtime_errors'] == 2
    assert report['summary']['scenario_errors'] == 1
    assert report['summary']['failed_turns'] == 2
    assert report['summary']['limited_fact_checks']['total'] == 3
    assert report['summary']['limited_fact_checks']['passed'] == 0
    assert report['summary']['semantic_review']['score'] is None


def test_checkpoints_preserve_every_completed_scenario(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(evaluation, '_execute', lambda job, **kwargs: {
        'detail': {'id': job['case']['id'], 'metrics': evaluation.ranking_metrics([], ['chen_food']),
                   'deterministic_checks': []}})
    def checkpoint(report):
        seen.append((report['run_status'], len(report['details'])))
        evaluation.write_report(report, tmp_path)
    report = evaluation.evaluate(small_fixture(), repeat=2, checkpoint=checkpoint)
    assert seen == [('running', 0), ('running', 1), ('running', 2), ('complete', 2)]
    reports = list(tmp_path.glob('*.json'))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['run_status'] == 'complete'
    assert len(list(tmp_path.glob('*-review.md'))) == 1
    assert 'No semantic accuracy score' in next(tmp_path.glob('*-review.md')).read_text()
    assert report['fixture_hash'] == evaluation.fixture_hash(small_fixture())


def test_timeout_is_reported_and_worker_is_not_left_running(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == .1
        raise subprocess.TimeoutExpired(args[0], .1)
    monkeypatch.setattr(subprocess, 'run', timeout)
    result = evaluation._execute({'case': {'id': 'timeout'}}, timeout=.1)
    assert result['detail']['error']['type'] == 'WorkerTimeout'


def test_implementation_hash_excludes_env_and_tracks_production_sources(tmp_path):
    (tmp_path / 'app/agent').mkdir(parents=True)
    source = tmp_path / 'app/agent/harness.py'
    source.write_text('# version one\n')
    initial = evaluation.app_source_hash(tmp_path)
    (tmp_path / '.env').write_text('HOME_LLM_API_KEY=not-a-real-key\n')
    assert evaluation.app_source_hash(tmp_path) == initial
    source.write_text('# version two\n')
    assert evaluation.app_source_hash(tmp_path) != initial
    assert evaluation._git_info(evaluation.RUNNER_ROOT / 'eval') == {'sha': None, 'dirty': None}


def test_repetitions_use_new_processes_and_never_open_user_database(monkeypatch, tmp_path):
    sentinel = tmp_path / 'user.db'
    sentinel.write_bytes(b'not a sqlite database; must remain untouched')
    monkeypatch.setenv('HOME_DB_PATH', str(sentinel))
    monkeypatch.setenv('HOME_LLM_API_KEY', 'not-a-real-key')
    report = evaluation.evaluate(small_fixture(), repeat=2)
    assert report['summary']['runtime_errors'] == 0
    assert report['summary']['retrieval']['recall_at_5'] == 1
    assert sentinel.read_bytes() == b'not a sqlite database; must remain untouched'
    first, second = report['details']
    assert first['result']['scoped_person']['id'] != second['result']['scoped_person']['id']
    assert report['configuration']['cache'] is False
    assert report['configuration']['embedding'] == 'disabled'
    assert 'not-a-real-key' not in json.dumps(report)


def test_dialogue_instrumentation_preserves_harness_extraction_and_captures_side_effects(conn, monkeypatch):
    from app.agent import harness, memory_sources
    from app.config import get_settings
    from app.memory import memory_corrections
    from app.store import db, repo
    from app import util
    app = {'harness': harness, 'repo': repo, 'sources': memory_sources,
           'corrections': memory_corrections, 'util': util, 'db': db, 'settings': get_settings()}
    fixture = small_fixture()
    people, records = evaluation._seed(app, conn, fixture)
    extracted = []
    def extraction(text, *args):
        extracted.append(text)
        eid = repo.add_evidence(conn, 'text', excerpt='instrumentation-only addition')
        repo.add_memory(conn, 'semantic', 'instrumentation-only addition', people['chen'], .95, eid)
        return {'people': [], 'my_notes': []}
    monkeypatch.setattr(harness.extract, 'extract_from_text', extraction)
    # A stub checks recording behavior only. Its response is never a quality score.
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=lambda *args, **kwargs: {
        'role': 'assistant', 'content': '仅用于检查评测记录行为。'}))
    monkeypatch.setattr(harness.turn_intent, 'classify_turn', lambda text, **kwargs:
        harness.turn_intent.validate_intent({'intent': 'record', 'record_spans': [text], 'reason': 'test'}, text))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test_double', 'semantic_verified': False}, 'warnings': []})
    case = {'id': 'instrumentation', 'turns': [{'question': '请记住老陈这次提供了新资料。',
             'expect': {'source_records': ['chen_food'], 'review_points': ['人工审阅仍待完成']}}]}
    result = evaluation._dialogue(app, conn, case, people, records)
    turn = result['turns'][0]
    assert extracted == ['请记住老陈这次提供了新资料。']
    assert 'error' not in turn, turn
    assert turn['semantic_review']['status'] == 'review_pending'
    assert turn['semantic_review']['ratings'] is None
    assert turn['data_changes']['memory_item']['added'][0]['text'] == 'instrumentation-only addition'
    assert all(value['stored'].get('text') != 'instrumentation-only addition'
               for value in turn['expected_material'].values())
    assert turn['tool_trace'][-1]['role'] == 'assistant'
    assert 'answer_accuracy' not in result


def test_report_never_converts_semantic_pending_or_absent_sources_to_accuracy():
    summary = evaluation.summarize('answers', [{'id': 'a', 'turns': [{
        'semantic_review': {'status': 'review_pending'}, 'deterministic_checks': [],
        'limited_fact_checks': []}]}])
    assert summary['semantic_review'] == {'status': 'review_pending', 'pending_turns': 1,
                                           'reviewed_turns': 0, 'score': None}
    assert summary['limited_fact_checks']['total'] == 0
    assert 'accuracy' not in summary


def test_summary_separates_classifier_failures_review_and_focus_from_domain_checks():
    units = [
        {'expected': {'intent': 'query'}, 'result': {
            'turn_intent': {'intent': 'query', 'status': 'classified', 'classifier': {
                'request_count': 2, 'failure_category': None, 'attempts': [
                    {'finish_reason': 'length', 'failure_category': 'truncated',
                     'usage': {'total_tokens': 20, 'reasoning_tokens': 10}},
                    {'finish_reason': 'stop', 'failure_category': None,
                     'usage': {'total_tokens': 40, 'prompt_tokens': 15, 'completion_tokens': 25}},
                ]}},
            'answer_review': {'status': 'completed', 'changed': True, 'issues': [
                {'quote_status': 'unmatched'}]},
            'memory_focus': {'status': 'selected', 'reason': 'unique_cited_memory',
                             'binding': {'status': 'selected'}, 'registration': {'status': 'registered'}},
        }, 'deterministic_checks': [{'name': 'query_has_no_memory_writes', 'passed': False}]},
        {'expected': {'intent': 'mixed'}, 'result': {
            'turn_intent': {'intent': 'unknown', 'status': 'failed', 'classifier': {
                'request_count': 1, 'failure_category': 'authentication', 'attempts': [
                    {'http_status': 403, 'failure_category': 'authentication'},
                ]}},
            'answer_review': {'status': 'unavailable', 'unavailable_reason': 'review_call_failed'},
            'memory_focus': {'status': 'clarify', 'reason': 'ambiguous_memory',
                             'registration': {'status': 'not_selected'}},
        }},
        {'expected': {'intent': 'record'}, 'error': {'type': 'WorkerTimeout'}},
    ]

    summary = evaluation.summarize('answers', [{'id': 'diagnostics', 'turns': units}])

    assert summary['runtime_errors'] == 1 and summary['units'] == 3
    assert summary['deterministic_checks'] == {'total': 1, 'passed': 0, 'failed': 1}
    classifier = summary['classifier_diagnostics']
    assert classifier['turns_with_intent'] == 2 and classifier['turns_missing_intent'] == 1
    assert classifier['status_counts'] == {'classified': 1, 'failed': 1, 'not_reported': 1}
    assert classifier['request_count'] == 3 and classifier['retried_turns'] == 1
    assert classifier['attempt_count'] == 3
    assert classifier['expected_intent_turns'] == 3
    assert classifier['matched_expected_intent_turns'] == 1
    assert classifier['mismatched_expected_intent_turns'] == 2
    assert classifier['failure_category_counts'] == {'authentication': 1}
    assert classifier['attempt_failure_category_counts'] == {'authentication': 1, 'truncated': 1}
    assert classifier['finish_reason_counts'] == {'length': 1, 'not_reported': 1, 'stop': 1}
    assert classifier['http_status_counts'] == {'403': 1}
    assert classifier['reported_token_usage'] == {
        'prompt_tokens': 15, 'completion_tokens': 25, 'total_tokens': 60, 'reasoning_tokens': 10}
    review = summary['answer_review_diagnostics']
    assert review['turns_with_review'] == 2 and review['turns_missing_review'] == 1
    assert review['status_counts'] == {'completed': 1, 'not_reported': 1, 'unavailable': 1}
    assert review['unavailable_reason_counts'] == {'review_call_failed': 1}
    assert review['changed_turns'] == 1 and review['issue_quote_status_counts'] == {'unmatched': 1}
    focus = summary['memory_focus_diagnostics']
    assert focus['status_counts'] == {'clarify': 1, 'not_reported': 1, 'selected': 1}
    assert focus['reason_counts'] == {'ambiguous_memory': 1, 'not_reported': 1, 'unique_cited_memory': 1}
    assert focus['registration_status_counts'] == {'not_reported': 1, 'not_selected': 1, 'registered': 1}
    assert summary['semantic_review']['score'] is None


def test_missing_or_malformed_optional_diagnostics_cannot_hide_an_answer_failure():
    unit = {'error': {'type': 'DeliberateFailure'}, 'result': {
        'turn_intent': {'classifier': {'attempts': None}},
        'answer_review': {'issues': None}, 'memory_focus': {'registration': None}}}

    summary = evaluation.summarize('answers', [{'turns': [unit]}])

    assert summary['runtime_errors'] == 1
    assert summary['classifier_diagnostics']['attempt_count'] == 0
    assert summary['answer_review_diagnostics']['issue_quote_status_counts'] == {}
    assert summary['memory_focus_diagnostics']['registration_status_counts'] == {'not_reported': 1}


def test_focus_expectation_checks_actual_resolution_and_registration(conn):
    expected = {'memory_focus': {'status': 'clarify', 'reason': 'ambiguous_memory'}}
    # A model answer saying "please clarify" cannot substitute for actual binding state.
    result = {'answer': '请澄清。', 'memory_focus': {'status': 'selected', 'reason': 'unique_cited_memory'}}
    checks = evaluation._answer_checks({}, conn, result, expected, {}, {})
    assert all(not check['passed'] for check in checks if check['name'].startswith('memory_focus_'))
    result['memory_focus'] = {'status': 'clarify', 'reason': 'ambiguous_memory'}
    assert all(check['passed'] for check in evaluation._answer_checks({}, conn, result, expected, {}, {}))
    expected = {'memory_focus': {'status': 'selected', 'registration_status': 'registered'}}
    result['memory_focus'] = {'status': 'selected', 'registration': {'status': 'budget_exceeded'}}
    checks = evaluation._answer_checks({}, conn, result, expected, {}, {})
    assert next(check for check in checks if check['name'] == 'memory_focus_registration_status')['passed'] is False


def test_query_write_guard_checks_database_changes_even_when_classifier_says_query(conn):
    from app.store import repo
    app = {'repo': repo}
    before = evaluation._data_snapshot(conn)
    pid = repo.create_person(conn, '不应新增', status='active')
    repo.add_alias(conn, pid, '别名', source='test')
    repo.add_memory(conn, 'semantic', '模型把问句误当事实', pid, .9, None)
    changes = evaluation._changes(before, evaluation._data_snapshot(conn))
    checks = evaluation._write_checks(app, conn, {'turn_intent': {'intent': 'query'}},
                                     {'intent': 'query', 'no_memory_writes': True}, {}, changes)
    assert checks[0]['passed'] is True
    assert checks[1]['passed'] is False
    assert {'person', 'person_alias', 'memory_item'} <= set(checks[1]['changed_tables'])


def test_audit_evidence_is_allowed_but_mixed_record_field_must_actually_be_saved(conn):
    from app.store import repo
    pid = repo.create_person(conn, '实际落库', status='active')
    before = evaluation._data_snapshot(conn)
    repo.add_evidence(conn, 'text', excerpt='仅查询，留原文')
    changes = evaluation._changes(before, evaluation._data_snapshot(conn))
    assert evaluation._write_checks({'repo': repo}, conn, {}, {'no_memory_writes': True}, {}, changes)[0]['passed']
    expect = {'saved_fields': [{'person': 'p', 'field': 'city', 'value': '苏州'}]}
    # A model claim or a pending item cannot substitute for the exact current field.
    assert not evaluation._write_checks({'repo': repo}, conn, {}, expect, {'p': pid}, {})[0]['passed']
    repo.upsert_field(conn, pid, 'city', '苏州', 1., None, valid_from=1000)
    assert evaluation._write_checks({'repo': repo}, conn, {}, expect, {'p': pid}, {})[0]['passed']
