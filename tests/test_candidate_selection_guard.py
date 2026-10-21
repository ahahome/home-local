"""Real harness write boundaries after an actual namesake clarification."""
import json

import pytest

from app.agent import harness, turn_intent
from app.memory import extract
from app.store import repo


class Model:
    def __init__(self, classifications, *, mutation_attempt=False):
        self.classifications = classifications
        self.classifier_calls = []
        self.mutation_attempt = mutation_attempt

    def chat(self, messages, **kwargs):
        system = messages[0]['content']
        if '写入意图分类器' in system:
            text = json.loads(messages[-1]['content'])['message']
            self.classifier_calls.append(text)
            value = self.classifications.get(text, {'intent': 'query', 'record_spans': [], 'reason': 'query'})
            return {'content': json.dumps(value, ensure_ascii=False)}
        if '本轮回答的资料编辑' in system:
            data = json.loads(messages[-1]['content'])
            wrote = any(item.get('status') in {'ok', 'pending', 'failed'}
                        for item in data['runtime_context'].get('outcomes', []))
            segments = [{'kind': 'receipt', 'text': '', 'source_ids': []}] if wrote else [
                {'kind': 'source_record', 'text': '', 'source_ids': [source['id']]}
                for source in data['sources'] if source['type'] == 'memory']
            if not segments:
                segments = [{'kind': 'no_answer', 'text': '', 'source_ids': []}]
            return {'content': json.dumps({'answer_type': 'no_answer' if segments[0]['kind'] == 'no_answer' else 'factual',
                                          'segments': segments, 'issues': []}, ensure_ascii=False)}
        if self.mutation_attempt:
            self.mutation_attempt = False
            return {'content': '', 'tool_calls': [{'id': 'bad-selection-note', 'type': 'function',
                'function': {'name': 'note', 'arguments': json.dumps({'text': '研发的那位，叫研发沈岑。'}, ensure_ascii=False)}}]}
        return {'content': '按已有资料回答。'}


def snapshot(conn):
    return {table: [tuple(row) for row in conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
            for table in ('person', 'person_alias', 'person_field', 'memory_item', 'interaction',
                          'interaction_person', 'promise', 'pending_item', 'applied_patch')}


@pytest.fixture
def namesakes(conn):
    first = repo.create_person(conn, '沈岑', status='active')
    second = repo.create_person(conn, '沈岑', status='active')
    repo.add_alias(conn, first, '研发沈岑', 'test')
    repo.add_alias(conn, second, '门店沈岑', 'test')
    for pid, label, memory in [(first, '研发', '研发沈岑说报告先放共享文件夹，只在周五讨论。'),
                                (second, '门店', '门店沈岑说补货的事情发邮件。')]:
        eid = repo.add_evidence(conn, 'text', source_ref='selection-test:' + label, excerpt=memory)
        repo.add_memory(conn, 'semantic', memory, person_id=pid, confidence=1., evidence_id=eid)
    conn.commit()
    return first, second


def clarified_conversation(conn, monkeypatch, model):
    monkeypatch.setattr(harness, 'get_llm', lambda: model)
    cid = repo.create_conversation(conn)
    first = harness.run_turn(conn, cid, '沈岑的报告怎么交？')
    assert first['conversation_scope']['mode'] == 'clarify'
    assert first['answer_review']['reason'] == 'deterministic_clarification'
    return cid, first


@pytest.mark.parametrize('span', ['研发的那位，叫研发沈岑。', '就是研发沈岑。', '选研发沈岑。'])
def test_existing_selection_is_readonly_and_cannot_call_write_tools(conn, monkeypatch, namesakes, span):
    text = span + '把他的报告交付和讨论时间说清楚。'
    model = Model({text: {'intent': 'mixed', 'record_spans': [span], 'reason': 'model thinks this names a person'}},
                  mutation_attempt=True)
    cid, clarification = clarified_conversation(conn, monkeypatch, model)
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('selection must not extract'))
    before = snapshot(conn)
    result = harness.run_turn(conn, cid, text)
    assert snapshot(conn) == before
    decision = result['turn_intent']
    assert decision['intent'] == 'query' and decision['allow_writes'] is False
    assert turn_intent.write_text(decision, text) is None
    guard = decision['write_guard']
    assert guard['policy'] == 'existing_candidate_selection'
    assert guard['person_id'] == namesakes[0]
    assert guard['previous_user_message_id'] == clarification['conversation_scope']['user_message_id']
    assert guard['removed_spans'] == [span] and guard['retained_spans'] == []
    assert guard['original_decision']['intent'] == 'mixed'
    assert guard['original_decision']['record_spans'] == [span]
    assert guard['original_decision']['classifier'] == decision['classifier']
    assert result['conversation_scope']['person_ids'] == [namesakes[0]]
    assert result['saved'] == result['pending'] == []
    assert any(receipt['status'] == 'skipped' for receipt in result['receipts'])
    mid = result['conversation_scope']['user_message_id']
    stored = json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0])
    assert stored['turn_intent'] == decision
    assert conn.execute('SELECT excerpt FROM evidence WHERE id=?', (decision['evidence_id'],)).fetchone()[0] == text
    repeated = harness.run_turn(conn, cid, text, user_message_id=mid)
    assert repeated['turn_intent'] == decision
    assert model.classifier_calls.count(text) == 1


@pytest.mark.parametrize('include_selection', [False, True])
def test_genuine_city_update_after_clarification_stays_mixed_and_is_saved(conn, monkeypatch, namesakes,
                                                                        include_selection):
    selection, city = '研发的那位，叫研发沈岑。', '研发沈岑搬到宁波了。'
    text = (selection if include_selection else '') + city + '他之前的报告怎么交？'
    spans = ([selection] if include_selection else []) + [city]
    model = Model({text: {'intent': 'mixed', 'record_spans': spans, 'reason': 'new city and question'}})
    cid, _ = clarified_conversation(conn, monkeypatch, model)
    extracted = []

    def extract_city(source, *args):
        extracted.append(source)
        return extract.sanitize({'people': [{'name': '研发沈岑', 'canonical_name': '研发沈岑', 'aliases': [],
            'fields': [{'field': 'city', 'value': '宁波', 'confidence': 1., 'evidence': city}],
            'facts': [], 'promises': [], 'ledger': [], 'relations': []}], 'my_notes': []}, source=source)

    monkeypatch.setattr(harness.extract, 'extract_from_text', extract_city)
    result = harness.run_turn(conn, cid, text)
    assert extracted == [city]
    assert result['turn_intent']['intent'] == 'mixed' and result['turn_intent']['allow_writes']
    assert result['turn_intent']['record_spans'] == [city]
    assert repo.current_fields(conn, namesakes[0])['city']['value'] == '宁波'
    assert repo.current_fields(conn, namesakes[1]) == {}
    assert result['saved'] and result['pending'] == []
    if include_selection:
        assert result['turn_intent']['write_guard']['original_decision']['record_spans'] == spans
    else:
        assert 'write_guard' not in result['turn_intent']


def test_explicit_remember_alias_is_not_suppressed_by_selection_guard(conn, monkeypatch, namesakes):
    span = '请记住：研发沈岑也叫小岑。'
    text = span + '他的报告怎么交？'
    model = Model({text: {'intent': 'mixed', 'record_spans': [span], 'reason': 'explicit alias memory'}})
    cid, _ = clarified_conversation(conn, monkeypatch, model)
    extracted = []

    def extract_alias(source, *args):
        extracted.append(source)
        return extract.sanitize({'people': [{'name': '研发沈岑', 'canonical_name': '研发沈岑',
            'aliases': [{'alias': '小岑', 'confidence': 1.}],
            'fields': [], 'facts': [], 'promises': [], 'ledger': [], 'relations': []}], 'my_notes': []}, source=source)

    monkeypatch.setattr(harness.extract, 'extract_from_text', extract_alias)
    result = harness.run_turn(conn, cid, text)
    assert result['turn_intent']['intent'] == 'mixed' and result['turn_intent']['allow_writes']
    assert 'write_guard' not in result['turn_intent']
    assert extracted == [span]
    assert '小岑' in repo.aliases_of(conn, namesakes[0])
    assert '小岑' not in repo.aliases_of(conn, namesakes[1])


@pytest.mark.parametrize('span', ['码头的那位，叫研发沈岑。', '叫研发沈岑，研发沈岑搬到宁波了。'])
def test_unknown_background_or_combined_real_fact_is_not_removed(conn, monkeypatch, namesakes, span):
    text = span + '他的报告怎么交？'
    model = Model({text: {'intent': 'mixed', 'record_spans': [span], 'reason': 'assertion'}})
    cid, _ = clarified_conversation(conn, monkeypatch, model)
    called = []
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda source, *args:
                        called.append(source) or {'people': [], 'my_notes': []})
    result = harness.run_turn(conn, cid, text)
    assert result['turn_intent']['intent'] == 'mixed' and result['turn_intent']['allow_writes']
    assert 'write_guard' not in result['turn_intent']
    assert called == [span]


def test_selection_grammar_alone_without_actual_preceding_clarification_is_not_a_guard(conn, monkeypatch, namesakes):
    span = '叫研发沈岑。'
    text = span + '他的报告怎么交？'
    model = Model({text: {'intent': 'mixed', 'record_spans': [span], 'reason': 'assertion'}})
    monkeypatch.setattr(harness, 'get_llm', lambda: model)
    called = []
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda source, *args:
                        called.append(source) or {'people': [], 'my_notes': []})
    result = harness.run_turn(conn, repo.create_conversation(conn), text)
    assert result['turn_intent']['allow_writes']
    assert 'write_guard' not in result['turn_intent']
    assert called == [span]


@pytest.mark.parametrize('bad_scope', [[], None, {'candidates': [[]]}, {'candidates': [None]},
                                      {'candidates': {'unexpected': 'mapping'}}])
def test_malformed_preceding_clarification_metadata_does_not_crash_or_grant_policy(conn, monkeypatch, namesakes,
                                                                                 bad_scope):
    span = '叫研发沈岑。'
    text = span + '他的报告怎么交？'
    model = Model({text: {'intent': 'mixed', 'record_spans': [span], 'reason': 'assertion'}})
    cid, clarification = clarified_conversation(conn, monkeypatch, model)
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?', (
        json.dumps({'conversation_scope': bad_scope}), clarification['assistant_message_id']))
    conn.commit()
    called = []
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda source, *args:
                        called.append(source) or {'people': [], 'my_notes': []})
    result = harness.run_turn(conn, cid, text)
    assert 'write_guard' not in result['turn_intent']
    assert result['turn_intent']['allow_writes']
    assert called == [span]
