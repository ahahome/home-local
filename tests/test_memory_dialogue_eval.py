"""Frozen dialogue contracts and negative controls, never LLM semantic scoring."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from eval import memory_questions as evaluation


FIXTURE = evaluation.RUNNER_ROOT / 'eval/fixtures/memory_dialogue_v012.json'
FROZEN_HASH = '31156596f125bbf920f1d347ce8b3a374c492d6f3952ca13e6862142922891d5'


def test_v012_questions_and_expectations_are_frozen_before_model_sampling():
    fixture = evaluation.load_fixture(FIXTURE)
    assert evaluation.fixture_hash(fixture) == FROZEN_HASH
    turns = [turn for case in fixture['dialogues'] for turn in case['turns']]
    assert len(fixture['dialogues']) == 8 and len(turns) == 20
    previous = evaluation.load_fixture(evaluation.RUNNER_ROOT / 'eval/fixtures/answer_stability_v010.json')
    questions = {turn['question'] for case in previous['dialogues'] for turn in case['turns']}
    assert not questions & {turn['question'] for turn in turns}
    assert all(turn['expect']['review_points'] for turn in turns)
    assert not any(turn['expect'].get('limited_fact_rules') for turn in turns)
    assert sum(turn['expect']['intent'] == 'query' for turn in turns) == 18
    assert all(turn['expect']['no_memory_writes'] is (turn['expect']['intent'] == 'query') for turn in turns)
    assert {'withdraw', 'restore', 'edit'} <= {action['op'] for turn in turns for action in turn.get('actions', [])}
    assert any(action.get('person') == 'gu' for turn in turns for action in turn.get('actions', []))


@pytest.mark.parametrize('field', ['required_dependency_records', 'required_cited_records'])
@pytest.mark.parametrize('labels', [[], ['xu_phone', 'xu_phone']])
def test_required_evidence_contracts_reject_empty_or_duplicate_denominators(field, labels):
    fixture = json.loads(FIXTURE.read_text())
    fixture['dialogues'][0]['turns'][0]['expect'][field] = labels
    with pytest.raises(ValueError, match='nonempty, unique'):
        evaluation.validate_fixture(fixture)


@pytest.mark.parametrize('status,labels', [
    ('selected', []), ('selected', ['xu_phone', 'xu_phone']),
    ('selected', ['xu_company']), ('clarify', ['xu_phone']),
])
def test_exact_focus_labels_reject_nonmemory_duplicates_and_empty_selection(status, labels):
    fixture = json.loads(FIXTURE.read_text())
    fixture['dialogues'][0]['turns'][1]['expect']['memory_focus'] = {'status': status, 'records': labels}
    with pytest.raises(ValueError, match='memory_focus records'):
        evaluation.validate_fixture(fixture)


def collection(conn):
    from app.agent import memory_sources
    from app.store import repo
    from app.util import sha256_text
    pid = repo.create_person(conn, '集合评测', status='active')
    snapshots, records = [], {}
    for number, text in enumerate(('周三午休方便接电话', '项目事项都写邮件'), 1):
        eid = repo.add_evidence(conn, 'text', excerpt=text)
        mid = repo.add_memory(conn, 'semantic', text, pid, .95, eid, ts=1000 * number)
        records['memory_' + str(number)] = {'type': 'memory', 'id': mid}
        snapshots.append(memory_sources.memory_source(conn, mid))
    conn.commit()
    registered = [dict(source, id='S' + str(index)) for index, source in enumerate(snapshots, 1)]
    cid = repo.create_conversation(conn, title='集合评测')
    anchor_id = repo.add_message(conn, cid, 'assistant', '两条原话。[[S1]][[S2]]', json.dumps({
        'sources': registered, 'source_dependencies': registered}, ensure_ascii=False))
    question = '这些记录有哪些区别？'
    user_id = repo.add_message(conn, cid, 'user', question)
    binding = {'version': 1, 'status': 'selected', 'reason': 'cited_memory_set', 'selection_mode': 'set',
               'conversation_id': cid, 'user_message_id': user_id, 'source_sha256': sha256_text(question),
               'anchor_assistant_message_id': anchor_id, 'reference_snapshots': deepcopy(registered),
               'reference_snapshot': None}
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?',
                 (json.dumps({'memory_focus': binding}, ensure_ascii=False), user_id))
    result = {'conversation_id': cid, 'sources': deepcopy(registered), 'source_dependencies': deepcopy(registered),
              'memory_focus': {'status': 'selected', 'sources': snapshots,
                               'binding': binding,
                               'registration': {'status': 'registered', 'source_ids': ['S1', 'S2']}}}
    result['assistant_message_id'] = repo.add_message(conn, cid, 'assistant', '比较两条。[[S1]][[S2]]',
                                                     json.dumps(result, ensure_ascii=False))
    conn.commit()
    expect = {'allowed_persons': ['p'], 'required_dependency_records': list(records),
              'required_cited_records': list(records),
              'memory_focus': {'status': 'selected', 'registration_status': 'registered', 'records': list(records)}}
    return result, expect, {'p': pid}, records


def checks_by_name(conn, result, expect, people, records):
    return {check['name']: check for check in evaluation._answer_checks({}, conn, result, expect, people, records)}


def test_complete_collection_contract_verifies_both_current_sources_and_registration(conn):
    result, expect, people, records = collection(conn)
    checks = checks_by_name(conn, result, expect, people, records)
    assert all(check['passed'] for check in checks.values()), checks
    assert checks['memory_focus_exact_records']['actual'] == ['memory_1', 'memory_2']
    assert checks['memory_focus_registered_complete_set']['registered_count'] == 2


def test_reading_two_records_cannot_substitute_for_citing_both(conn):
    result, expect, people, records = collection(conn)
    result['sources'] = result['sources'][:1]
    checks = checks_by_name(conn, result, expect, people, records)
    assert checks['required_dependency_record:memory_2']['passed']
    assert not checks['required_cited_record:memory_2']['passed']


def test_final_citations_cannot_substitute_for_actual_dependencies(conn):
    result, expect, people, records = collection(conn)
    result.pop('source_dependencies')
    checks = checks_by_name(conn, result, expect, people, records)
    assert checks['required_cited_record:memory_2']['passed']
    assert not checks['required_dependency_record:memory_2']['passed']
    assert not checks['cited_source_was_read:1']['passed']
    assert not checks['memory_focus_registered_complete_set']['passed']


@pytest.mark.parametrize('mutation', ['omit_focus', 'duplicate_focus', 'omit_anchor', 'omit_registration', 'duplicate_registration'])
def test_collection_negative_controls_detect_partial_or_duplicated_sets(conn, mutation):
    result, expect, people, records = collection(conn)
    focus = result['memory_focus']
    if mutation == 'omit_focus':
        focus['sources'] = focus['sources'][:1]
    elif mutation == 'duplicate_focus':
        focus['sources'][1] = deepcopy(focus['sources'][0])
    elif mutation == 'omit_anchor':
        focus['binding']['reference_snapshots'] = focus['binding']['reference_snapshots'][:1]
    elif mutation == 'omit_registration':
        focus['registration']['source_ids'] = ['S1']
    else:
        focus['registration']['source_ids'] = ['S1', 'S1']
    checks = checks_by_name(conn, result, expect, people, records)
    assert any(not check['passed'] for name, check in checks.items() if name.startswith('memory_focus_'))


def test_edited_collection_requires_new_snapshots_while_retaining_old_anchor(conn):
    from app.agent import memory_sources
    from app.memory import memory_corrections
    result, expect, people, records = collection(conn)
    old_binding = deepcopy(result['memory_focus']['binding'])
    mid = records['memory_1']['id']
    memory_corrections.correct_memory(conn, mid, {'action': 'edit', 'expected_revision': 0,
                                               'request_id': 'eval-edit', 'text': '更正：先发短信'})
    stale = checks_by_name(conn, result, expect, people, records)
    assert not stale['cited_source_snapshot:0']['passed']
    assert not stale['memory_focus_current_snapshot:0']['passed']
    latest = memory_sources.memory_source(conn, mid)
    result['memory_focus']['sources'][0] = latest
    result['source_dependencies'][0] = dict(latest, id='S1')
    result['sources'][0] = dict(latest, id='S1')
    current = checks_by_name(conn, result, expect, people, records)
    assert all(check['passed'] for check in current.values()), current
    assert result['memory_focus']['binding'] == old_binding
    assert latest['revision'] > old_binding['reference_snapshots'][0]['revision']


@pytest.mark.parametrize('mutation', ['text', 'revision', 'missing_anchor', 'cross_conversation', 'later_anchor', 'earlier_anchor'])
def test_anchor_negative_controls_verify_original_message_position_and_exact_snapshots(conn, mutation):
    from app.store import repo
    result, expect, people, records = collection(conn)
    binding = result['memory_focus']['binding']
    if mutation in {'text', 'revision'}:
        binding['reference_snapshots'][0][mutation] = '伪造原话' if mutation == 'text' else 999
    elif mutation == 'missing_anchor':
        binding['anchor_assistant_message_id'] = 'does-not-exist'
    else:
        anchor_metadata = json.dumps({'sources': binding['reference_snapshots']}, ensure_ascii=False)
        if mutation == 'cross_conversation':
            cid = repo.create_conversation(conn, '另一个对话')
            anchor_id = repo.add_message(conn, cid, 'assistant', '其他对话', anchor_metadata)
        elif mutation == 'later_anchor':
            anchor_id = repo.add_message(conn, result['conversation_id'], 'assistant', '后来的回答', anchor_metadata)
        else:
            anchor_id = binding['anchor_assistant_message_id']
            # A newer final answer now stands between the claimed anchor and
            # the current user. Use rowid order, independent of wall timestamps.
            current_user = binding['user_message_id']
            conn.execute("UPDATE message SET role='assistant' WHERE id=?", (current_user,))
            current_user = repo.add_message(conn, result['conversation_id'], 'user', '这些记录有哪些区别？')
            binding['user_message_id'] = current_user
            result['assistant_message_id'] = repo.add_message(conn, result['conversation_id'], 'assistant', '本轮')
        binding['anchor_assistant_message_id'] = anchor_id
    # Persist the forged binding coherently so output-vs-user equality cannot
    # mask a missing exact-anchor or position check.
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?',
                 (json.dumps({'memory_focus': binding}, ensure_ascii=False), binding['user_message_id']))
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?',
                 (json.dumps({'memory_focus': {'binding': binding}}, ensure_ascii=False), result['assistant_message_id']))
    conn.commit()
    checks = checks_by_name(conn, result, expect, people, records)
    if mutation in {'text', 'revision'}:
        assert checks['memory_focus_persisted_user_binding']['passed']
        assert checks['memory_focus_anchor_precedes_user']['passed']
        assert not checks['memory_focus_anchor_exact_snapshots']['passed']
    else:
        assert not checks['memory_focus_anchor_precedes_user']['passed']


def test_binding_must_belong_to_actual_current_user_and_saved_answer(conn):
    result, expect, people, records = collection(conn)
    result['memory_focus']['binding']['source_sha256'] = 'not-the-user-question'
    checks = checks_by_name(conn, result, expect, people, records)
    assert not checks['memory_focus_persisted_user_binding']['passed']


def test_date_context_cannot_forge_edit_or_restore_time(conn):
    result, _, _, _ = collection(conn)
    source = deepcopy(result['sources'][0])
    source['record_context']['last_user_edit_at'] = source['date']
    assert 'memory time context differs from stored lifecycle audit' in evaluation.source_checks(conn, source)
    source['record_context']['last_user_edit_at'] = None
    source['record_context']['last_lifecycle_change'] = {'action': 'restore', 'created_at': source['date']}
    assert 'memory time context differs from stored lifecycle audit' in evaluation.source_checks(conn, source)


def test_missing_required_prior_source_state_is_not_a_vacuous_pass(conn):
    from app.agent import memory_sources
    from app.store import repo
    result, _, people, records = collection(conn)
    def run_turn(conn, cid, question):
        message_id = repo.add_message(conn, cid, 'assistant', '只用于检查评测器。')
        conn.commit()
        return {'assistant_message_id': message_id, 'answer': '只用于检查评测器。',
                'sources': [], 'source_dependencies': []}
    app = {'repo': repo, 'sources': memory_sources, 'harness': SimpleNamespace(run_turn=run_turn)}
    case = {'id': 'missing-prior', 'turns': [
        {'question': '先查两条记录。', 'expect': {'source_records': ['memory_1', 'memory_2']}},
        {'question': '重新核对。', 'expect': {'previous_source_states': {'memory_1': 'current'}}}]}
    report = evaluation._dialogue(app, conn, case, people, records)
    checks = {check['name']: check for check in report['turns'][1]['deterministic_checks']}
    assert not checks['old_source_state_present:memory_1']['passed']


def test_unexercised_contracts_and_semantic_review_are_not_reported_as_success():
    summary = evaluation.summarize('answers', [{'turns': [{'deterministic_checks': []}]}])
    assert summary['evidence_contract_checks']['required_dependencies'] == {
        'total': 0, 'passed': 0, 'failed': 0, 'status': 'not_exercised'}
    assert summary['semantic_review']['score'] is None
    assert '"accuracy":' not in json.dumps(summary)
