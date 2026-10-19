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
    case = {'id': 'instrumentation', 'turns': [{'question': '老陈有哪些记录？',
             'expect': {'source_records': ['chen_food'], 'review_points': ['人工审阅仍待完成']}}]}
    result = evaluation._dialogue(app, conn, case, people, records)
    turn = result['turns'][0]
    assert extracted == ['老陈有哪些记录？']
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
