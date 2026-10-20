"""A committed write can restart answer generation once, without replaying it."""
from copy import deepcopy
import json

import pytest

from app.agent import harness, tools
from app.agent.memory_sources import source_state
from app.agent.source_registry import SourceChangedError
from app.memory import memory_corrections
from app.store import repo
from app.util import new_id


QUESTION = '是乐团周汐。帮我记下：乐团周汐现在住在长沙。再查她演出邀约需要先发什么。'
RECORD = '帮我记下：乐团周汐现在住在长沙。'
MUSIC = '乐团周汐说演出邀约先邮件发场地与曲目范围，排练期间不接电话。'
UPDATED = '乐团周汐更新后的邀约记录：先邮件说明场地和曲目范围。'


def tool_call(name, args, suffix):
    return {'id': 'recovery-' + suffix, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}


class Model:
    def __init__(self, conn, person_id, memory_id, *, pending_only=False, second_change=False, query=False):
        self.conn, self.person_id, self.memory_id = conn, person_id, memory_id
        self.pending_only, self.second_change, self.query = pending_only, second_change, query
        self.classifier_calls = []
        self.drafts = []
        self.reviews = 0
        self.sent_initial_tool = False
        self.sent_recovery_tools = False

    def edit_memory(self, text):
        revision = self.conn.execute('SELECT revision FROM memory_item WHERE id=?', (self.memory_id,)).fetchone()[0]
        memory_corrections.correct_memory(self.conn, self.memory_id, {
            'action': 'edit', 'expected_revision': revision, 'request_id': new_id(), 'text': text})

    def chat(self, messages, **kwargs):
        system = messages[0]['content']
        if '写入意图分类器' in system:
            question = json.loads(messages[-1]['content'])['message']
            self.classifier_calls.append(question)
            mixed = question == QUESTION and not self.query
            return {'content': json.dumps({'intent': 'mixed' if mixed else 'query',
                'record_spans': [RECORD] if mixed else [], 'reason': 'controlled test classification'}, ensure_ascii=False)}
        if '本轮回答的资料编辑' in system:
            self.reviews += 1
            payload = json.loads(messages[-1]['content'])
            if self.reviews == 1 and (self.pending_only or self.query):
                self.edit_memory(UPDATED)
            if self.reviews == 2 and self.second_change:
                self.edit_memory('恢复期间再次改变的资料，第二次应拒绝终稿。')
            segments = [{'kind': 'source_record', 'text': '', 'source_ids': [source['id']]}
                        for source in payload['sources'] if source['type'] in {'field', 'memory'}]
            if not self.query:
                segments.append({'kind': 'receipt', 'text': '', 'source_ids': []})
            return {'content': json.dumps({'answer_type': 'factual', 'segments': segments, 'issues': []}, ensure_ascii=False)}

        recovery = any('现在仅使用本次重新读取的资料' in message['content']
                       for message in messages if isinstance(message.get('content'), str))
        self.drafts.append({'recovery': recovery, 'messages': deepcopy(messages),
                            'tools': [tool['function']['name'] for tool in kwargs['tools']]})
        args = {'person': '乐团周汐', 'person_id': self.person_id, 'field': 'city', 'value': '长沙'}
        if not recovery and not self.sent_initial_tool and not (self.pending_only or self.query):
            self.sent_initial_tool = True
            return {'role': 'assistant', 'content': '',
                    'tool_calls': [tool_call('save_person_fact', args, 'first-write')]}
        if recovery and not self.sent_recovery_tools:
            self.sent_recovery_tools = True
            assert not set(self.drafts[-1]['tools']) & tools._WRITE_TOOLS
            # A noncompliant model may still request an unadvertised write.
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                tool_call('save_person_fact', args, 'replayed-write'),
                tool_call('get_person', {'name': '乐团周汐'}, 'fresh-read')]}
        return {'content': ('FRESH_ANSWER_DRAFT' if recovery else 'REJECTED_FIRST_DRAFT') + ' [[S1]]'}


@pytest.fixture
def setup_recovery(conn, monkeypatch):
    pottery = repo.create_person(conn, '周汐', status='active')
    music = repo.create_person(conn, '周汐', status='active')
    repo.add_alias(conn, pottery, '陶艺周汐', 'test')
    repo.add_alias(conn, music, '乐团周汐', 'test')
    eid = repo.add_evidence(conn, 'text', source_ref='recovery-test-music', excerpt=MUSIC)
    mid = repo.add_memory(conn, 'semantic', MUSIC, person_id=music, confidence=.95, evidence_id=eid)
    conn.commit()
    inputs = []

    def extract(source, *args):
        inputs.append(source)
        # Preserve the real failure shape: extraction loses the existing unique
        # alias, leaving the entire patch pending; the later grounded tool writes
        # the city to the user's selected person after the initial card snapshot.
        return {'people': [{'name': '周汐', 'canonical_name': '周汐', 'aliases': [],
            'fields': [{'field': 'city', 'value': '长沙', 'confidence': .9,
                        'evidence': '现在住在长沙', '_source_verified': True}],
            'facts': [], 'relations': [], 'promises': [], 'ledger': []}], 'my_notes': []}

    monkeypatch.setattr(harness.extract, 'extract_from_text', extract)

    def start(**options):
        model = Model(conn, music, mid, **options)
        monkeypatch.setattr(harness, 'get_llm', lambda: model)
        cid = repo.create_conversation(conn, 'Source recovery test')
        first = harness.run_turn(conn, cid, '周汐的邀约应该先怎么沟通？')
        assert first['conversation_scope']['mode'] == 'clarify'
        return cid, model, inputs, music, mid

    return start


def last_user(conn, cid):
    row = conn.execute("SELECT * FROM message WHERE conversation_id=? AND role='user' ORDER BY rowid DESC LIMIT 1",
                       (cid,)).fetchone()
    return row, json.loads(row['tool_calls'])


def final_assistants(conn, cid):
    return [row for row in conn.execute("SELECT * FROM message WHERE conversation_id=? AND role='assistant'", (cid,))
            if not isinstance(json.loads(row['tool_calls'] or '{}'), list)]


def test_committed_city_and_pending_survive_fresh_readonly_answer_without_reextracting(conn, setup_recovery):
    cid, model, inputs, music, mid = setup_recovery()
    events = []

    result = harness.run_turn(conn, cid, QUESTION, emit=lambda kind, data: events.append((kind, data)))

    assert inputs == [RECORD]
    assert model.classifier_calls == ['周汐的邀约应该先怎么沟通？', QUESTION]
    assert model.reviews == 2
    assert conn.execute('SELECT COUNT(*) FROM person_field WHERE person_id=? AND field=?', (music, 'city')).fetchone()[0] == 1
    assert repo.current_fields(conn, music)['city']['value'] == '长沙'
    assert conn.execute('SELECT COUNT(*) FROM applied_patch WHERE person_id=? AND field=?', (music, 'city')).fetchone()[0] == 1
    pending = conn.execute('SELECT * FROM pending_item').fetchall()
    assert len(pending) == 1 and pending[0]['kind'] == 'person_identity' and pending[0]['resolved_at'] is None
    assert result['pending_ids'] == [pending[0]['id']]
    assert any(outcome['status'] == 'pending' for outcome in result['outcomes'])
    assert any(outcome['status'] == 'ok' and outcome['detail'] == 'city=长沙' for outcome in result['outcomes'])
    assert len([receipt for receipt in result['receipts'] if receipt['type'] == 'save_person_fact' and receipt['status'] == 'ok']) == 1
    assert any(receipt['type'] == 'save_person_fact' and receipt['status'] == 'skipped' for receipt in result['receipts'])
    assert result['turn_intent']['intent'] == 'mixed' and result['turn_intent']['allow_writes'] is True
    user, metadata = last_user(conn, cid)
    assert metadata['turn_intent'] == result['turn_intent']
    assert metadata['answer_recovery']['status'] == 'recovered'
    audit = metadata['answer_recovery']['rejected_revisions']
    assert len(audit) == 1 and audit[0]['draft'].startswith('REJECTED_FIRST_DRAFT')
    old_profile = next(source for source in audit[0]['source_dependencies'] if source['type'] == 'profile_context')
    assert 'city=长沙' not in old_profile['text']
    new_profile = next(source for source in result['source_dependencies'] if source['type'] == 'profile_context')
    assert 'city=长沙' in new_profile['text']
    assert all(source_state(conn, source)['status'] == 'current' for source in result['source_dependencies'])
    assert MUSIC in result['answer'] and '长沙' in result['answer']
    assert 'REJECTED_FIRST_DRAFT' not in result['answer']
    assert 'rejected_revisions' not in result['answer_recovery']
    assert len(final_assistants(conn, cid)) == 2
    assert conn.execute("SELECT COUNT(*) FROM message WHERE conversation_id=? AND role='user'", (cid,)).fetchone()[0] == 2
    for draft in (draft for draft in model.drafts if draft['recovery']):
        contents = '\n'.join(message.get('content') or '' for message in draft['messages'])
        assert 'city=长沙' in contents and 'REJECTED_FIRST_DRAFT' not in contents
    assert not any(kind == 'delta' for kind, _ in events)


def test_pending_write_alone_can_recover_after_rereading_changed_memory(conn, setup_recovery):
    cid, model, inputs, music, mid = setup_recovery(pending_only=True)

    result = harness.run_turn(conn, cid, QUESTION)

    assert inputs == [RECORD] and model.reviews == 2
    assert UPDATED in result['answer'] and MUSIC not in result['answer']
    assert repo.current_fields(conn, music) == {}
    assert result['saved'] == [] and len(result['pending_ids']) == 1
    assert result['turn_intent']['intent'] == 'mixed'
    assert result['answer_recovery']['status'] == 'recovered'


def test_second_source_change_still_refuses_both_drafts_and_never_replays_committed_write(conn, setup_recovery):
    cid, model, inputs, music, mid = setup_recovery(second_change=True)
    events = []

    with pytest.raises(SourceChangedError, match='生成期间发生变化'):
        harness.run_turn(conn, cid, QUESTION, emit=lambda kind, data: events.append((kind, data)))

    assert inputs == [RECORD] and model.reviews == 2
    assert repo.current_fields(conn, music)['city']['value'] == '长沙'
    assert conn.execute('SELECT COUNT(*) FROM applied_patch WHERE person_id=? AND field=?', (music, 'city')).fetchone()[0] == 1
    assert conn.execute('SELECT COUNT(*) FROM pending_item').fetchone()[0] == 1
    assert len(final_assistants(conn, cid)) == 1  # only the preceding clarification
    user, metadata = last_user(conn, cid)
    assert metadata['turn_intent']['intent'] == 'mixed'
    assert metadata['answer_recovery']['status'] == 'exhausted'
    assert len(metadata['answer_recovery']['rejected_revisions']) == 2
    assert not any(kind == 'delta' for kind, _ in events)


def test_query_source_change_keeps_original_refusal_without_recovery(conn, setup_recovery):
    cid, model, inputs, music, mid = setup_recovery(query=True)

    with pytest.raises(SourceChangedError, match='生成期间发生变化'):
        harness.run_turn(conn, cid, '查乐团周汐的演出邀约要求。')

    assert inputs == [] and model.reviews == 1
    assert not any(draft['recovery'] for draft in model.drafts)
    assert len(final_assistants(conn, cid)) == 1
    _, metadata = last_user(conn, cid)
    assert metadata['turn_intent']['intent'] == 'query'
    assert 'answer_recovery' not in metadata


def test_successful_action_receipt_without_extraction_outcomes_also_recovers_once(conn, monkeypatch, setup_recovery):
    cid, _, inputs, music, mid = setup_recovery()
    record = '请在2026-10-21提醒我确认乐团周汐的邀约。'
    question = record + '再查她演出邀约需要先发什么。'

    class ReminderModel(Model):
        def chat(self, messages, **kwargs):
            system = messages[0]['content']
            if '写入意图分类器' in system:
                return {'content': json.dumps({'intent': 'mixed', 'record_spans': [record],
                                              'reason': 'explicit reminder and query'}, ensure_ascii=False)}
            if '本轮回答的资料编辑' not in system and not self.sent_initial_tool and not any(
                    '现在仅使用本次重新读取的资料' in message.get('content', '') for message in messages):
                self.sent_initial_tool = True
                return {'role': 'assistant', 'content': '', 'tool_calls': [tool_call('create_reminder',
                    {'person': '乐团周汐', 'text': '确认乐团周汐的邀约', 'fire_date': '2026-10-21'}, 'reminder')]}
            return super().chat(messages, **kwargs)

    model = ReminderModel(conn, music, mid, pending_only=True)
    monkeypatch.setattr(harness, 'get_llm', lambda: model)

    def empty_extraction(source, *args):
        inputs.append(source)
        return {'people': [], 'my_notes': []}

    monkeypatch.setattr(harness.extract, 'extract_from_text', empty_extraction)

    result = harness.run_turn(conn, cid, question)

    assert inputs == [record] and model.reviews == 2
    assert result['saved'] == result['pending'] == []
    assert conn.execute('SELECT COUNT(*) FROM reminder').fetchone()[0] == 1
    assert any(receipt['type'] == 'reminder' and receipt['status'] == 'ok' for receipt in result['receipts'])
    assert result['answer_recovery']['status'] == 'recovered'
    assert UPDATED in result['answer']
