"""Real harness/SQLite boundaries; each model phase has its own explicit script."""
import json

import pytest

from app.agent import harness
from app.memory import extract
from app.store import repo


class PhaseModel:
    def __init__(self, classification, answers, receipt):
        self.classification = classification
        self.answers = list(answers)
        self.receipt = receipt
        self.classifier_calls, self.answer_calls, self.review_calls = [], [], []

    def chat(self, messages, **kwargs):
        system = messages[0]['content']
        if '写入意图分类器' in system:
            self.classifier_calls.append(messages)
            if isinstance(self.classification, Exception):
                raise self.classification
            return {'content': json.dumps(self.classification, ensure_ascii=False)}
        if '本轮回答的资料编辑' in system:
            context = json.loads(messages[-1]['content'])
            self.review_calls.append(context)
            actual = context['runtime_context'].get('outcomes', [])
            has_result = any(item['status'] in {'ok', 'pending', 'failed'} for item in actual)
            return {'content': json.dumps({'answer_type': 'factual' if has_result else 'clarification', 'segments': [
                {'kind': 'receipt' if has_result else 'clarification',
                 'text': '' if has_result else self.receipt, 'source_ids': []}], 'issues': []}, ensure_ascii=False)}
        self.answer_calls.append(messages)
        if not self.answers:
            pytest.fail('answer script exhausted: classifier/reviewer consumed an answer slot')
        return self.answers.pop(0)


def call(name, args, id):
    return {'id': id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}


def persistent_state(conn):
    return {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY rowid')]
            for table in ('person', 'person_alias', 'person_field', 'memory_item', 'interaction',
                          'interaction_person', 'pending_item', 'promise', 'ledger_entry', 'reminder',
                          'research', 'skill', 'applied_patch')}


def test_query_does_not_extract_or_write_even_when_answer_model_calls_mutation_tools(conn, monkeypatch):
    text = '系统现在记录了哪些信息'
    model = PhaseModel({'intent': 'query', 'record_spans': [], 'reason': 'read only'}, [
        {'content': '', 'tool_calls': [
            call('save_person_fact', {'person': '虚构人物', 'field': 'city', 'value': '杭州'}, 'bad-person'),
            call('note', {'text': '我喜欢喝茶'}, 'bad-memory'),
            call('create_reminder', {'text': '发消息', 'fire_date': '2026-11-01'}, 'bad-reminder')]},
        {'content': '本轮未新增记忆。'}], '本轮未新增记忆。')
    monkeypatch.setattr(harness, 'get_llm', lambda: model)
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args:
                        pytest.fail('pure query must not invoke extraction'))
    cid = repo.create_conversation(conn)
    before = persistent_state(conn)
    result = harness.run_turn(conn, cid, text)
    assert persistent_state(conn) == before
    assert result['turn_intent']['intent'] == 'query' and not result['turn_intent']['allow_writes']
    assert result['saved'] == result['pending'] == []
    assert len(result['receipts']) == 3 and all(r['status'] == 'skipped' for r in result['receipts'])
    assert len(model.classifier_calls) == len(model.review_calls) == 1
    assert result['answer_review']['status'] == 'completed'
    assert len(model.answer_calls) == 2 and not model.answers
    tool_messages = [row for row in repo.list_messages(conn, cid) if row['role'] == 'tool']
    assert len(tool_messages) == 3 and all(json.loads(row['content'])['write_blocked'] for row in tool_messages)
    audit = conn.execute('SELECT * FROM evidence').fetchall()
    assert len(audit) == 1 and audit[0]['excerpt'] == text


def test_mixed_message_extracts_only_assertion_while_evidence_keeps_complete_original(conn, monkeypatch):
    pid = repo.create_person(conn, '林舟', status='active')
    text, span = '林舟住杭州。陈宁住上海是真的吗？', '林舟住杭州。'
    model = PhaseModel({'intent': 'mixed', 'record_spans': [span], 'reason': 'new fact plus question'},
                       [{'content': '已记录林舟的居住地杭州。'}], '已记录林舟的居住地杭州。')
    monkeypatch.setattr(harness, 'get_llm', lambda: model)
    extraction_inputs = []
    def fake_extract(source, *args):
        extraction_inputs.append(source)
        return extract.sanitize({'people': [{'name': '林舟', 'canonical_name': '林舟',
            'fields': [{'field': 'city', 'value': '杭州', 'confidence': .95, 'evidence': span}],
            'aliases': [], 'facts': [], 'promises': [], 'ledger': [], 'relations': []}], 'my_notes': []}, source=source)
    monkeypatch.setattr(harness.extract, 'extract_from_text', fake_extract)
    cid = repo.create_conversation(conn)
    result = harness.run_turn(conn, cid, text)
    assert extraction_inputs == [span]
    assert repo.current_fields(conn, pid)['city']['value'] == '杭州'
    assert conn.execute("SELECT COUNT(*) FROM person WHERE display_name='陈宁'").fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM pending_item').fetchone()[0] == 0
    interactions = repo.recent_interactions(conn, pid)
    assert len(interactions) == 1 and interactions[0]['content'] == span and interactions[0]['kind'] == 'note'
    assert conn.execute('SELECT excerpt FROM evidence WHERE id=?',
                        (result['turn_intent']['evidence_id'],)).fetchone()[0] == text
    user = next(row for row in repo.list_messages(conn, cid) if row['role'] == 'user')
    assert json.loads(user['tool_calls'])['turn_intent']['record_spans'] == [span]
    assert len(model.classifier_calls) == len(model.answer_calls) == len(model.review_calls) == 1
    assert result['answer_review']['status'] == 'completed'


def test_classifier_failure_is_persisted_readonly_and_retry_reuses_that_decision(conn, monkeypatch):
    text = '我喜欢喝茶。'
    tool_turn = {'content': '', 'tool_calls': [call('note', {'text': text}, 'denied-note')]}
    model = PhaseModel(RuntimeError('provider unavailable'), [tool_turn, {'content': '不能确认记事意图。'},
                      tool_turn, {'content': '不能确认记事意图。'}], '暂无法确认记事意图，本轮未新增记忆。')
    monkeypatch.setattr(harness, 'get_llm', lambda: model)
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('unknown must not extract'))
    cid = repo.create_conversation(conn)
    before = persistent_state(conn)
    first = harness.run_turn(conn, cid, text)
    mid = first['conversation_scope']['user_message_id']
    second = harness.run_turn(conn, cid, text, user_message_id=mid)
    assert persistent_state(conn) == before
    assert first['turn_intent'] == second['turn_intent']
    assert second['turn_intent']['intent'] == 'unknown' and second['turn_intent']['status'] == 'failed'
    assert second['answer_review']['status'] == 'completed'
    assert second['answer'] == '本轮处理结果请以下方回执为准。'
    assert any(item['status'] == 'skipped' and '未保存新记忆' in item['detail']
               for item in second['outcomes'])
    assert len(model.classifier_calls) == 1 and len(model.review_calls) == 2
    rows = repo.list_messages(conn, cid)
    assert len([row for row in rows if row['role'] == 'user']) == 1
    final = json.loads(rows[-1]['tool_calls'])
    assert final['turn_intent']['status'] == 'failed' and not final['saved'] and not final['pending']
    assert conn.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == 1
