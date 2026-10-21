"""Real SQLite/harness evidence boundaries for deterministic memory follow-ups."""
from copy import deepcopy
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

from app.agent import conversation_scope, harness, memory_focus, turn_intent
from app.agent.memory_sources import memory_source
from app.agent.source_registry import SourceChangedError
from app.config import tz
from app.llm import LLMError
from app.jobs import tasks
from app.memory import memory_corrections
from app.store import repo
from app.util import new_id, sha256_text


DATE_QUESTION = '那9月11日是这条记忆所标的日期，还是我实际修改它的时间？'


def stamp(day):
    return int(datetime.fromisoformat(day + 'T12:00:00').replace(tzinfo=tz()).timestamp() * 1000)


def seed(conn, name='林舟', text='非紧急事项请发短邮件。'):
    pid = repo.create_person(conn, name, status='active')
    mid = repo.add_memory(conn, 'semantic', text, pid, .95, None, ts=stamp('2026-09-11'))
    conn.execute('UPDATE memory_item SET created_at=?,updated_at=? WHERE id=?',
                 (stamp('2026-09-12'), stamp('2026-09-20'), mid))
    conn.commit()
    return pid, mid


def preceding_answer(conn, cid, memory_ids):
    # A previous answer's label cannot be carried into this turn's registry.
    sources = [{**memory_source(conn, mid), 'id': f'S{index + 7}'}
               for index, mid in enumerate(memory_ids)]
    citation_text = ''.join(f'[[{source["id"]}]]' for source in sources)
    aid = repo.add_message(conn, cid, 'assistant', '此前可核查的记忆。 ' + citation_text,
                           json.dumps({'sources': sources, 'source_dependencies': sources}, ensure_ascii=False))
    conn.commit()
    return aid, sources


def domain_state(conn):
    return {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY rowid')]
            for table in ('person', 'person_alias', 'person_field', 'memory_item', 'interaction',
                          'interaction_person', 'pending_item', 'promise', 'ledger_entry', 'reminder',
                          'research', 'skill', 'applied_patch', 'memory_correction')}


def prohibit_model_and_writes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('deterministic memory dates/clarifications must not call model, classifier, extraction, tools or review')

    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=forbidden))
    monkeypatch.setattr(harness.turn_intent, 'classify_turn', forbidden)
    monkeypatch.setattr(harness.extract, 'extract_from_text', forbidden)
    monkeypatch.setattr(harness.tools, 'execute_tool', forbidden)
    monkeypatch.setattr(harness.answer_review, 'review_answer', forbidden)


def user_metadata(conn, cid):
    user = conn.execute("SELECT * FROM message WHERE conversation_id=? AND role='user' ORDER BY rowid DESC LIMIT 1",
                        (cid,)).fetchone()
    return user, json.loads(user['tool_calls'])


@pytest.mark.parametrize('question', [DATE_QUESTION, '这条记忆是什么时候被我修改的？'])
def test_date_followup_persists_query_and_registered_source_without_any_model_phase(conn, monkeypatch, question):
    pid, mid = seed(conn)
    cid = repo.create_conversation(conn, 'Date integration')
    aid, old_sources = preceding_answer(conn, cid, [mid])
    prohibit_model_and_writes(monkeypatch)
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, question)

    assert domain_state(conn) == before
    intent = result['turn_intent']
    assert intent['intent'] == 'query' and intent['status'] == 'ok' and intent['allow_writes'] is False
    assert intent['record_spans'] == [] and intent['classifier']['request_count'] == 0
    assert intent['classifier']['method'] == 'deterministic_readonly'
    assert result['saved'] == result['pending'] == result['failed'] == result['receipts'] == []
    assert result['conversation_scope']['person_ids'] == [pid]
    assert len(result['sources']) == len(result['source_dependencies']) == 1
    assert result['sources'][0]['id'] == 'S1' and result['sources'][0]['entity_id'] == mid
    assert old_sources[0]['id'] == 'S7' and '[[S7]]' not in result['answer']
    assert '[[S1]]' in result['answer']
    assert '所标的日期是 2026-09-11' in result['answer']
    assert '首次录入日期是 2026-09-12' in result['answer']
    assert '没有可核验的人工编辑时间' in result['answer']
    assert '2026-09-20' not in result['answer']
    assert result['answer_review']['status'] == 'not_needed'
    assert result['answer_review']['reason'] == 'memory_date_answer'
    assert result['memory_focus']['binding']['anchor_assistant_message_id'] == aid
    assert result['memory_focus']['registration'] == {'status': 'registered', 'source_id': 'S1', 'source_ids': ['S1']}
    user, metadata = user_metadata(conn, cid)
    assert metadata['turn_intent'] == intent
    assert metadata['memory_focus'] == result['memory_focus']['binding']
    assert metadata['conversation_scope'] == result['conversation_scope']
    audit = conn.execute('SELECT * FROM evidence WHERE id=?', (intent['evidence_id'],)).fetchone()
    assert audit['excerpt'] == question and audit['source_ref'] == user['id']
    saved = conn.execute('SELECT * FROM message WHERE id=?', (result['assistant_message_id'],)).fetchone()
    assert saved['content'] == result['answer'] and json.loads(saved['tool_calls'])['sources'] == result['sources']


@pytest.mark.parametrize('question', ['这条的时间依据可以逐项核对吗？', '那条记录的时间依据可以逐项核对吗？'])
def test_possessive_time_query_uses_current_source_and_normal_model_path_without_a_new_date_template(conn, monkeypatch, question):
    pid, mid = seed(conn)
    cid = repo.create_conversation(conn, 'Possessive reference integration')
    aid, old_sources = preceding_answer(conn, cid, [mid])
    monkeypatch.setattr(memory_corrections, 'now_ms', lambda: stamp('2026-09-17'))
    memory_corrections.correct_memory(conn, mid, {
        'action': 'edit', 'expected_revision': 0, 'request_id': new_id(), 'text': '人工更新后的联系安排。'})
    calls = []

    def chat(messages, **kwargs):
        system = messages[0]['content']
        if '写入意图分类器' in system:
            calls.append('classify')
            return {'content': json.dumps({'intent': 'query', 'record_spans': [], 'reason': '只查询已有记录'})}
        if '本轮回答的资料编辑' in system:
            calls.append('review')
            payload = json.loads(messages[-1]['content'])
            assert payload['runtime_context']['memory_focus']['source_ids'] == ['S1']
            source = next(s for s in payload['sources'] if s['id'] == 'S1')
            assert source['text'] == '人工更新后的联系安排。'
            assert source['时间信息']['最近人工编辑时间'].startswith('2026-09-17')
            return {'content': json.dumps({'answer_type': 'factual', 'segments': [
                {'kind': 'source_record', 'text': '', 'source_ids': ['S1']}], 'issues': []})}
        calls.append('generate')
        assert '人工更新后的联系安排。' in '\n'.join(m['content'] for m in messages)
        return {'content': '这条在2026-09-11由当事人改过。 [[S1]]'}

    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('query must not extract'))
    monkeypatch.setattr(harness.tools, 'execute_tool', lambda *args, **kwargs: pytest.fail('focused source is already read'))
    before = domain_state(conn)
    result = harness.run_turn(conn, cid, question)

    assert not memory_focus.is_date_question(question)
    assert calls == ['classify', 'generate', 'review']
    assert domain_state(conn) == before
    assert result['turn_intent']['intent'] == 'query' and result['turn_intent']['allow_writes'] is False
    assert result['memory_focus']['status'] == 'selected'
    assert result['memory_focus']['binding']['anchor_assistant_message_id'] == aid
    assert result['memory_focus']['binding']['reference_snapshot'] == old_sources[0]
    assert result['conversation_scope']['person_ids'] == [pid]
    current = next(source for source in result['source_dependencies'] if source.get('entity_id') == mid)
    assert current['text'] == '人工更新后的联系安排。' and current['revision'] == 1
    assert result['answer_review']['status'] == 'completed'
    assert result['answer_review']['method'] == 'server_record_revision'
    assert '记忆所标日期（不代表录入或修改日期）：2026-09-11' in result['answer']
    assert '首次录入时间：2026-09-12' in result['answer']
    assert '最近人工编辑时间：2026-09-17' in result['answer']
    assert '由当事人改过' not in result['answer']
    assert [source['entity_id'] for source in result['sources']] == [mid]


def test_date_followup_rereads_manual_edit_and_registers_latest_quotation(conn, monkeypatch):
    _, mid = seed(conn)
    cid = repo.create_conversation(conn, 'Edited memory date')
    _, old_sources = preceding_answer(conn, cid, [mid])
    monkeypatch.setattr(memory_corrections, 'now_ms', lambda: stamp('2026-09-16'))
    memory_corrections.correct_memory(conn, mid, {
        'action': 'edit', 'expected_revision': 0, 'request_id': new_id(),
        'text': '后来明确说：非紧急事项发简短邮件，不打电话。'})
    prohibit_model_and_writes(monkeypatch)
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, DATE_QUESTION)

    assert domain_state(conn) == before
    source = result['sources'][0]
    assert source['text'] == '后来明确说：非紧急事项发简短邮件，不打电话。'
    assert source['revision'] == 1 and source['id'] == 'S1'
    assert source['record_context']['last_user_edit_at'] == stamp('2026-09-16')
    assert result['source_dependencies'] == result['sources']
    assert result['memory_focus']['binding']['reference_snapshot'] == old_sources[0]
    assert '实际人工编辑日期是 2026-09-16' in result['answer']
    assert '所标的日期是 2026-09-11' in result['answer']
    assert result['turn_intent']['intent'] == 'query'


def test_multiple_memories_require_clarification_and_do_not_register_arbitrary_source(conn, monkeypatch):
    _, first = seed(conn)
    _, second = seed(conn, '陈岚', '重要事项请先打电话。')
    cid = repo.create_conversation(conn, 'Ambiguous memory date')
    preceding_answer(conn, cid, [second, first])
    prohibit_model_and_writes(monkeypatch)
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, DATE_QUESTION)

    assert domain_state(conn) == before
    assert result['turn_intent']['intent'] == 'query' and not result['turn_intent']['allow_writes']
    assert result['conversation_scope']['blocked'] and result['conversation_scope']['person_ids'] == []
    assert result['memory_focus']['status'] == 'clarify' and result['memory_focus']['reason'] == 'ambiguous_memory'
    assert result['sources'] == result['source_dependencies'] == []
    assert result['memory_focus']['registration']['status'] == 'not_selected'
    assert '多条记忆' in result['answer'] and '[[' not in result['answer']
    assert result['saved'] == result['pending'] == result['failed'] == []


def same_person_pair(conn):
    pid, first = seed(conn, text='项目事项先发邮件。')
    second = repo.add_memory(conn, 'semantic', '周三午休方便接电话。', pid, .95, None, ts=stamp('2026-09-13'))
    conn.execute('UPDATE memory_item SET created_at=? WHERE id=?', (stamp('2026-09-14'), second))
    conn.commit()
    return pid, first, second


def test_set_followup_carries_reread_sources_through_generation_review_and_persistence(conn, monkeypatch):
    pid, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Set comparison integration')
    aid, old_sources = preceding_answer(conn, cid, [first, second])
    monkeypatch.setattr(memory_corrections, 'now_ms', lambda: stamp('2026-09-18'))
    memory_corrections.correct_memory(conn, second, {
        'action': 'edit', 'expected_revision': 0, 'request_id': new_id(), 'text': '仅月初周三午休方便接电话。'})
    calls = []
    def chat(messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            return {'content': json.dumps({'intent': 'query', 'record_spans': [], 'reason': '只比较已有记录'})}
        if '本轮回答的资料编辑' in messages[0]['content']:
            payload = json.loads(messages[-1]['content'])
            calls.append(payload)
            assert payload['runtime_context']['memory_focus']['source_ids'] == ['S1', 'S2']
            return {'content': json.dumps({'answer_type': 'factual', 'segments': [
                {'kind': 'source_quote', 'text': '', 'source_ids': ['S1']},
                {'kind': 'source_quote', 'text': '', 'source_ids': ['S2']}], 'issues': []})}
        text = '\n'.join(m['content'] for m in messages)
        assert '仅月初周三午休方便接电话。' in text
        assert '本轮追问已重新读取的记忆' in text and '[[S7]]' not in text
        return {'content': '两条记录分别涉及邮件和电话。 [[S1]] [[S2]]'}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('query must not extract'))
    before = domain_state(conn)
    result = harness.run_turn(conn, cid, '这些记录能证明工作发邮件、私事打电话吗？')
    assert domain_state(conn) == before
    assert result['conversation_scope']['person_ids'] == [pid]
    assert result['memory_focus']['binding']['anchor_assistant_message_id'] == aid
    assert result['memory_focus']['binding']['reference_snapshots'] == old_sources
    assert result['memory_focus']['registration']['source_ids'] == ['S1', 'S2']
    assert [s['entity_id'] for s in result['memory_focus']['sources']] == [first, second]
    assert result['memory_focus']['sources'][1]['revision'] == 1
    assert {s['entity_id'] for s in result['sources']} == {first, second}
    assert '仅月初周三午休方便接电话。' in result['answer']
    assert len(calls) == 1 and result['answer_review']['semantic_verified'] is False
    saved = json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?',
                                   (result['assistant_message_id'],)).fetchone()[0])
    assert saved['memory_focus'] == result['memory_focus']


def test_set_date_followup_is_fully_readonly_and_keeps_each_date_role(conn, monkeypatch):
    pid, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Set date integration')
    preceding_answer(conn, cid, [first, second])
    prohibit_model_and_writes(monkeypatch)
    before = domain_state(conn)
    result = harness.run_turn(conn, cid, '这两条是什么时候记下的，后来又是哪天改过？')
    assert domain_state(conn) == before
    assert result['turn_intent']['classifier']['request_count'] == 0
    assert result['conversation_scope']['person_ids'] == [pid]
    assert result['memory_focus']['registration']['source_ids'] == ['S1', 'S2']
    assert len(result['sources']) == len(result['source_dependencies']) == 2
    assert '首次录入日期是 2026-09-12' in result['answer']
    assert '首次录入日期是 2026-09-14' in result['answer']
    assert '所标的日期是 2026-09-11' in result['answer']
    assert '所标的日期是 2026-09-13' in result['answer']
    assert '2026-09-20' not in result['answer']


def comparison_classifier_only(monkeypatch):
    """Keep the real intent classifier; forbid every later model/read/write phase."""
    phases = []
    def forbidden(*args, **kwargs):
        pytest.fail('literal comparison must not generate, review, extract or fetch extra context')
    def chat(messages, **kwargs):
        assert '写入意图分类器' in messages[0]['content']
        phases.append('classifier')
        return {'content': json.dumps({'intent': 'query', 'record_spans': [], 'reason': '只对照已有记录文字'})}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', forbidden)
    monkeypatch.setattr(harness.tools, 'execute_tool', forbidden)
    monkeypatch.setattr(harness.answer_review, 'review_answer', forbidden)
    monkeypatch.setattr(harness, '_context_blocks', forbidden)
    return phases


@pytest.mark.parametrize('question', [
    '这些记录有什么区别？',
    '这些记录说的是同一种联系安排吗？哪些地方对得上，哪些地方还不能确定？',
    '刚才两条现在再对照一下，原文和能支持的结论有没有变化？',
])
def test_literal_set_comparison_classifies_once_rereads_current_text_and_persists_complete_anchor(conn, monkeypatch, question):
    pid, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Literal set comparison')
    aid, old_sources = preceding_answer(conn, cid, [first, second])
    monkeypatch.setattr(memory_corrections, 'now_ms', lambda: stamp('2026-09-18'))
    latest_text = '仅月初周三午休先发短信，确认后才通电话。'
    memory_corrections.correct_memory(conn, second, {
        'action': 'edit', 'expected_revision': 0, 'request_id': new_id(), 'text': latest_text})
    phases = comparison_classifier_only(monkeypatch)
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, question)

    assert phases == ['classifier']
    assert domain_state(conn) == before
    assert result['turn_intent']['intent'] == 'query' and result['turn_intent']['allow_writes'] is False
    assert result['turn_intent']['classifier']['request_count'] == 1
    assert result['conversation_scope']['person_ids'] == [pid]
    assert result['saved'] == result['pending'] == result['failed'] == result['receipts'] == []
    assert result['answer_review'] == {'version': 1, 'status': 'not_needed',
                                       'reason': 'memory_text_comparison', 'semantic_verified': False}
    assert result['memory_focus']['binding']['anchor_assistant_message_id'] == aid
    assert result['memory_focus']['binding']['reference_snapshots'] == old_sources
    assert result['memory_focus']['registration']['source_ids'] == ['S1', 'S2']
    assert result['sources'] == result['source_dependencies']
    assert [source['entity_id'] for source in result['sources']] == [first, second]
    assert result['sources'][1]['revision'] == 1 and result['sources'][1]['text'] == latest_text
    assert latest_text in result['answer'] and old_sources[1]['text'] not in result['answer']
    assert result['answer'].count('保存文字已变化') == result['answer'].count('保存文字未变化') == 1
    assert '所标的日期是 2026-09-13' in result['answer']
    assert '首次录入日期是 2026-09-14' in result['answer']
    assert '实际人工编辑日期是 2026-09-18' in result['answer']
    assert '[[S7]]' not in result['answer'] and '[[S8]]' not in result['answer']
    assert '避免被' not in result['answer'] and '私事' not in result['answer']
    _, user_meta = user_metadata(conn, cid)
    assert user_meta['memory_focus'] == result['memory_focus']['binding']
    saved = conn.execute('SELECT content,tool_calls FROM message WHERE id=?',
                          (result['assistant_message_id'],)).fetchone()
    assert saved['content'] == result['answer']
    assert json.loads(saved['tool_calls'])['memory_focus'] == result['memory_focus']
    assert json.loads(saved['tool_calls'])['sources'] == result['sources']
    anchor = json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?', (aid,)).fetchone()[0])
    assert anchor['sources'] == old_sources


def test_literal_set_comparison_distinguishes_lifecycle_revision_from_text_change(conn, monkeypatch):
    _, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Restored literal comparison')
    _, old_sources = preceding_answer(conn, cid, [first, second])
    monkeypatch.setattr(memory_corrections, 'now_ms', lambda: stamp('2026-09-18'))
    memory_corrections.correct_memory(conn, second, {
        'action': 'withdraw', 'expected_revision': 0, 'request_id': new_id()})
    memory_corrections.correct_memory(conn, second, {
        'action': 'restore', 'expected_revision': 1, 'request_id': new_id()})
    phases = comparison_classifier_only(monkeypatch)
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, '这些记录有什么区别？')

    assert phases == ['classifier'] and domain_state(conn) == before
    assert result['sources'][1]['revision'] == 2
    assert result['memory_focus']['binding']['reference_snapshots'] == old_sources
    assert result['answer'].count('保存文字未变化') == 2 and '保存文字已变化' not in result['answer']
    assert '最近一次恢复操作日期是 2026-09-18' in result['answer']
    assert '没有可核验的人工编辑时间' in result['answer']
    assert result['answer_review']['reason'] == 'memory_text_comparison'


def test_literal_set_comparison_still_refuses_to_save_if_any_source_changes_after_projection(conn, monkeypatch):
    _, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Literal comparison save guard')
    preceding_answer(conn, cid, [first, second])
    phases = comparison_classifier_only(monkeypatch)
    actual_comparison = memory_focus.comparison_answer
    def withdraw_before_save(focus, source_ids, text, intent):
        answer = actual_comparison(focus, source_ids, text, intent)
        assert answer is not None
        memory_corrections.correct_memory(conn, second, {
            'action': 'withdraw', 'expected_revision': 0, 'request_id': new_id()})
        return answer
    monkeypatch.setattr(memory_focus, 'comparison_answer', withdraw_before_save)

    with pytest.raises(SourceChangedError, match='生成期间发生变化'):
        harness.run_turn(conn, cid, '这些记录有什么区别？')

    assert phases == ['classifier']
    assert conn.execute("SELECT COUNT(*) FROM message WHERE conversation_id=? AND role='assistant'",
                        (cid,)).fetchone()[0] == 1
    _, metadata = user_metadata(conn, cid)
    assert metadata['turn_intent']['intent'] == 'query'
    assert metadata['memory_focus']['status'] == 'selected'


@pytest.mark.parametrize('mixed', [False, True])
def test_comparison_with_drafting_keeps_generation_and_explicit_fact_in_normal_flow(conn, monkeypatch, mixed):
    from app.memory import extract
    pid, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Drafting comparison compound request')
    preceding_answer(conn, cid, [first, second])
    span = '林舟现在住杭州。'
    text = '这些记录比较一下，再帮我起草一段发给林舟的消息。' + (span if mixed else '')
    phases, extraction_inputs = [], []
    def chat(messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            phases.append('classifier')
            return {'content': json.dumps({'intent': 'mixed' if mixed else 'query',
                'record_spans': [span] if mixed else [], 'reason': '记事和起草' if mixed else '比较和起草'})}
        phases.append('drafting')
        assert any(message['role'] == 'user' and message['content'] == text for message in messages)
        assert '本轮追问已重新读取的记忆' in '\n'.join(message['content'] for message in messages)
        return {'content': '给林舟的消息草稿：请确认这两条联系安排各自适用的事项。 [[S1]] [[S2]]'}
    def fake_extract(source, *args):
        extraction_inputs.append(source)
        return extract.sanitize({'people': [{'name': '林舟', 'canonical_name': '林舟',
            'fields': [{'field': 'city', 'value': '杭州', 'confidence': .95, 'evidence': span}],
            'aliases': [], 'facts': [], 'promises': [], 'ledger': [], 'relations': []}], 'my_notes': []}, source=source)
    def review(llm, question, draft, sources, **kwargs):
        phases.append('review')
        assert question == text
        return {'answer': draft, 'metadata': {'status': 'test', 'semantic_verified': False}, 'warnings': []}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', fake_extract)
    monkeypatch.setattr(harness.answer_review, 'review_answer', review)
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, text)

    assert phases == ['classifier', 'drafting', 'review']
    assert result['answer'].startswith('给林舟的消息草稿')
    assert result['memory_focus']['status'] == 'selected'
    assert result['answer_review'].get('reason') != 'memory_text_comparison'
    assert {first, second} <= {source['entity_id'] for source in result['sources']}
    if mixed:
        assert extraction_inputs == [span]
        assert result['turn_intent']['intent'] == 'mixed' and result['turn_intent']['allow_writes']
        assert repo.current_fields(conn, pid)['city']['value'] == '杭州' and result['saved']
    else:
        assert extraction_inputs == [] and domain_state(conn) == before
        assert result['turn_intent']['intent'] == 'query' and not result['turn_intent']['allow_writes']


def test_set_member_change_during_generation_refuses_to_save_stale_comparison(conn, monkeypatch):
    _, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Concurrent set integration')
    preceding_answer(conn, cid, [first, second])
    def chat(messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            return {'content': json.dumps({'intent': 'query', 'record_spans': [], 'reason': '比较'})}
        memory_corrections.correct_memory(conn, second, {
            'action': 'withdraw', 'expected_revision': 0, 'request_id': new_id()})
        return {'content': '此前两条记录。 [[S1]] [[S2]]'}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test', 'semantic_verified': False}, 'warnings': []})
    with pytest.raises(SourceChangedError):
        harness.run_turn(conn, cid, '刚才两条具体差在哪里？')
    assistants = conn.execute("SELECT COUNT(*) FROM message WHERE conversation_id=? AND role='assistant'", (cid,)).fetchone()[0]
    assert assistants == 1


def test_plural_reference_in_mixed_message_does_not_suppress_explicit_new_fact(conn, monkeypatch):
    from app.memory import extract
    pid, first, second = same_person_pair(conn)
    cid = repo.create_conversation(conn, 'Mixed set integration')
    preceding_answer(conn, cid, [first, second])
    span = '林舟现在住杭州。'
    text = '这些记录是什么时候录入的？' + span
    extraction_inputs = []
    def chat(messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            return {'content': json.dumps({'intent': 'mixed', 'record_spans': [span], 'reason': '新事实和查询'})}
        return {'content': '请核对本轮处理回执。'}
    def fake_extract(source, *args):
        extraction_inputs.append(source)
        return extract.sanitize({'people': [{'name': '林舟', 'canonical_name': '林舟',
            'fields': [{'field': 'city', 'value': '杭州', 'confidence': .95, 'evidence': span}],
            'aliases': [], 'facts': [], 'promises': [], 'ledger': [], 'relations': []}], 'my_notes': []}, source=source)
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', fake_extract)
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test', 'semantic_verified': False}, 'warnings': []})
    result = harness.run_turn(conn, cid, text)
    assert result['memory_focus']['status'] == 'selected'
    assert result['turn_intent']['intent'] == 'mixed' and result['turn_intent']['allow_writes']
    assert extraction_inputs == [span] and repo.current_fields(conn, pid)['city']['value'] == '杭州'
    assert len(result['saved']) > 0


@pytest.mark.parametrize('text', ['记住：这条是我每天回家的路线。', '请记住：这些记录是我个人的出行安排。',
                                  '记住：这条的内容是我每天喝两杯水。', '请保存：那条的路线从今天改走东门。',
                                  '记住：那条记录是我的饮水提醒。'])
def test_explicit_personal_note_is_not_blocked_by_new_memory_reference_shortcuts(conn, monkeypatch, text):
    calls = []
    def chat(messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            return {'content': json.dumps({'intent': 'record', 'record_spans': [text], 'reason': '明确记录'})}
        return {'content': '请核对处理回执。'}
    def extract(source, *args):
        calls.append(source)
        return {'people': [], 'my_notes': [{'kind': 'semantic', 'text': text, 'confidence': .95, 'evidence': text}]}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', extract)
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test', 'semantic_verified': False}, 'warnings': []})
    cid = repo.create_conversation(conn, 'Literal note integration')
    result = harness.run_turn(conn, cid, text)
    assert result['memory_focus']['status'] == 'not_applicable'
    assert result['turn_intent']['intent'] == 'record' and calls == [text]
    assert result['saved']
    assert conn.execute('SELECT text FROM memory_item WHERE person_id IS NULL').fetchone()[0] == text


def test_date_source_change_before_save_raises_and_does_not_persist_stale_answer(conn, monkeypatch):
    _, mid = seed(conn)
    cid = repo.create_conversation(conn, 'Concurrent date change')
    preceding_answer(conn, cid, [mid])
    prohibit_model_and_writes(monkeypatch)
    actual_date_answer = memory_focus.date_answer
    before_assistants = conn.execute("SELECT COUNT(*) FROM message WHERE conversation_id=? AND role='assistant'",
                                     (cid,)).fetchone()[0]

    def change_before_save(focus, source_id, text):
        answer = actual_date_answer(focus, source_id, text)
        assert answer is not None
        memory_corrections.correct_memory(conn, mid, {
            'action': 'edit', 'expected_revision': 0, 'request_id': new_id(), 'text': '生成期间被人工改成电话。'})
        return answer

    monkeypatch.setattr(memory_focus, 'date_answer', change_before_save)
    with pytest.raises(SourceChangedError, match='生成期间发生变化'):
        harness.run_turn(conn, cid, DATE_QUESTION)

    assert conn.execute("SELECT COUNT(*) FROM message WHERE conversation_id=? AND role='assistant'",
                        (cid,)).fetchone()[0] == before_assistants
    _, metadata = user_metadata(conn, cid)
    assert metadata['turn_intent']['intent'] == 'query'
    assert metadata['memory_focus']['status'] == 'selected'
    assert memory_source(conn, mid)['text'] == '生成期间被人工改成电话。'


def test_real_review_receives_confirmed_aliases_only_for_the_resolved_scope(conn, monkeypatch):
    selected, selected_mid = seed(conn, '唐宁', '非紧急事项请发邮件。')
    outside, _ = seed(conn, '陈岚', '不属于本轮人物的私密资料。')
    repo.add_alias(conn, selected, '糖糖', 'user')
    repo.add_alias(conn, outside, '岚姐', 'user')
    conn.commit()
    phases, review_payloads = [], []

    def chat(messages, **kwargs):
        system = messages[0]['content']
        if '写入意图分类器' in system:
            phases.append('classifier')
            return {'content': json.dumps({'intent': 'query', 'record_spans': [], 'reason': '查询既有偏好'})}
        if '本轮回答的资料编辑' in system:
            phases.append('review')
            payload = json.loads(messages[-1]['content'])
            review_payloads.append(deepcopy(payload))
            source = next(item for item in payload['sources'] if item['type'] == 'memory')
            return {'content': json.dumps({'answer_type': 'factual', 'segments': [
                {'kind': 'fact', 'text': '糖糖的记录写明非紧急事项请发邮件。', 'source_ids': [source['id']]}],
                'issues': []}, ensure_ascii=False)}
        phases.append('answer')
        return {'content': '糖糖的资料记载非紧急事项请发邮件。 [[S1]]'}

    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('query must not extract'))
    cid = repo.create_conversation(conn, 'Alias scope integration')
    result = harness.run_turn(conn, cid, '糖糖的沟通偏好是什么？')

    assert phases == ['classifier', 'answer', 'review']
    assert result['answer_review']['status'] == 'completed'
    assert result['conversation_scope']['person_ids'] == [selected]
    assert result['sources'][0]['entity_id'] == selected_mid
    identity = review_payloads[0]['runtime_context']['confirmed_identity']
    assert identity['status'] == 'single' and len(identity['people']) == 1
    assert identity['people'][0]['id'] == selected and identity['people'][0]['name'] == '唐宁'
    assert {'唐宁', '糖糖'} <= set(identity['people'][0]['aliases'])
    runtime = json.dumps(review_payloads[0]['runtime_context'], ensure_ascii=False)
    assert outside not in runtime and '岚姐' not in runtime and '陈岚' not in runtime
    assert all(source.get('person', {}).get('id') == selected for source in review_payloads[0]['sources'])
    assert '不属于本轮人物的私密资料' not in json.dumps(review_payloads[0], ensure_ascii=False)


def test_model_configuration_failure_persists_user_classification_and_readonly_response(conn, monkeypatch):
    seed(conn)
    calls = []

    def unavailable():
        calls.append('configuration')
        raise LLMError('private configuration details must never be stored', category='config')

    def forbidden(*args, **kwargs):
        pytest.fail('configuration failure must not attempt extraction, tools or review')

    monkeypatch.setattr(harness, 'get_llm', unavailable)
    monkeypatch.setattr(harness.turn_intent, 'get_llm', unavailable)
    monkeypatch.setattr(harness.extract, 'extract_from_text', forbidden)
    monkeypatch.setattr(harness.tools, 'execute_tool', forbidden)
    monkeypatch.setattr(harness.answer_review, 'review_answer', forbidden)
    cid = repo.create_conversation(conn, 'Unavailable configuration')
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, '请记住林舟现在住杭州。')

    assert domain_state(conn) == before
    assert result['turn_intent']['intent'] == 'unknown'
    assert result['turn_intent']['status'] == 'failed' and result['turn_intent']['allow_writes'] is False
    diagnostic = result['turn_intent']['classifier']
    assert diagnostic['failure_category'] == 'config' and diagnostic['request_count'] == 0
    assert diagnostic['retryable'] is False and len(diagnostic['attempts']) == 1
    assert result['saved'] == result['pending'] == result['failed'] == []
    assert result['outcomes'][0]['status'] == 'skipped' and '未保存新记忆' in result['outcomes'][0]['detail']
    assert '模型服务当前不可用' in result['answer'] and 'private configuration' not in json.dumps(result)
    assert result['answer_review']['status'] == 'not_needed' and result['answer_review']['reason'] == 'model_unavailable'
    user, metadata = user_metadata(conn, cid)
    assert metadata['turn_intent'] == result['turn_intent']
    assert conn.execute('SELECT excerpt FROM evidence WHERE id=?',
                        (result['turn_intent']['evidence_id'],)).fetchone()[0] == user['content']
    assert conn.execute('SELECT content FROM message WHERE id=?',
                        (result['assistant_message_id'],)).fetchone()[0] == result['answer']
    assert calls == ['configuration', 'configuration']
    monkeypatch.setattr(harness.turn_intent, 'get_llm', forbidden)
    repeated = harness.run_turn(conn, cid, user['content'], user_message_id=user['id'])
    assert repeated['turn_intent'] == result['turn_intent']
    assert len([row for row in repo.list_messages(conn, cid) if row['role'] == 'user']) == 1
    assert domain_state(conn) == before


def test_failed_note_classification_cannot_be_hidden_by_editor_no_answer(conn, monkeypatch):
    pid, _ = seed(conn, '顾舟')
    cid = repo.create_conversation(conn, 'Failed note stays visible after review')
    text = '记一下：顾舟现在住厦门，后面再搬家我会另外告诉你。'
    phases = []
    editor_response = {'answer_type': 'no_answer', 'segments': [
        {'kind': 'no_answer', 'text': '', 'source_ids': []}], 'issues': []}
    original_draft = '本轮没有保存顾舟住厦门的信息。'
    def chat(messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            phases.append('classifier')
            return {'content': '```json\n{"intent":"record"}\n```'}
        if '本轮回答的资料编辑' in messages[0]['content']:
            phases.append('review')
            return {'content': json.dumps(editor_response, ensure_ascii=False)}
        phases.append('answer')
        return {'content': original_draft}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('invalid intent must not extract'))
    monkeypatch.setattr(harness.tools, 'execute_tool', lambda *args, **kwargs: pytest.fail('invalid intent must not use tools'))
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, text)

    assert phases == ['classifier', 'classifier', 'answer', 'review']
    assert domain_state(conn) == before and repo.current_fields(conn, pid).get('city') is None
    assert result['turn_intent']['status'] == 'invalid' and result['turn_intent']['intent'] == 'unknown'
    assert result['turn_intent']['allow_writes'] is False
    notice = turn_intent.readonly_outcome(result['turn_intent'])['detail']
    assert result['intent_notice'] == notice and result['answer'] == notice
    assert '响应格式未通过校验' in result['answer'] and '未保存新记忆或执行写入' in result['answer']
    assert '当前可核查资料不足以回答这个问题' not in result['answer']
    assert result['saved'] == result['pending'] == result['failed'] == result['receipts'] == []
    # The stable runtime notice must not falsify the editor's audit trail.
    review = result['answer_review']
    assert review['status'] == 'completed' and review['answer_type'] == 'no_answer'
    assert review['original_answer'] == original_draft
    assert review['revised_answer'] == '当前可核查资料不足以回答这个问题。'
    assert json.loads(review['raw_response']) == editor_response
    saved = conn.execute('SELECT content,tool_calls FROM message WHERE id=?',
                          (result['assistant_message_id'],)).fetchone()
    metadata = json.loads(saved['tool_calls'])
    assert saved['content'] == notice and metadata['intent_notice'] == notice
    assert metadata['answer_review'] == review


def test_successful_query_does_not_add_failure_notice_to_editor_no_answer(conn, monkeypatch):
    seed(conn, '顾舟')
    cid = repo.create_conversation(conn, 'Normal query without write failure notice')
    phases = []
    def chat(messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            phases.append('classifier')
            return {'content': json.dumps({'intent': 'query', 'record_spans': [], 'reason': '仅核实血型资料'})}
        if '本轮回答的资料编辑' in messages[0]['content']:
            phases.append('review')
            return {'content': json.dumps({'answer_type': 'no_answer', 'segments': [
                {'kind': 'no_answer', 'text': '', 'source_ids': []}], 'issues': []})}
        phases.append('answer')
        return {'content': '没有查到可核验的血型记录。'}
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('query must not extract'))
    before = domain_state(conn)

    result = harness.run_turn(conn, cid, '顾舟有可核验的血型记录吗？')

    assert phases == ['classifier', 'answer', 'review'] and domain_state(conn) == before
    assert result['turn_intent']['intent'] == 'query' and result['turn_intent']['status'] == 'classified'
    assert result['turn_intent']['allow_writes'] is False
    assert result['intent_notice'] is None
    assert result['answer'] == '当前可核查资料不足以回答这个问题。'
    assert '未保存' not in result['answer'] and '分类' not in result['answer']
    metadata = json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?',
                                      (result['assistant_message_id'],)).fetchone()[0])
    assert metadata['intent_notice'] is None
    assert metadata['answer_review'] == result['answer_review']


@pytest.mark.parametrize('reanalyze', [False, True])
def test_relative_date_extraction_keeps_original_day_for_retry_and_explicit_analysis(conn, monkeypatch, reanalyze):
    seed(conn)
    cid = repo.create_conversation(conn, 'Stable relative date')
    text = '请记住下周三给林舟发报价。'
    original = repo.add_message(conn, cid, 'user', text)
    conn.execute('UPDATE message SET created_at=? WHERE id=?', (stamp('2026-10-12'), original))
    conn.commit()
    conversation_scope.for_user_message(conn, cid, original, text)
    if reanalyze:
        monkeypatch.setattr(turn_intent, 'classify_turn', lambda text, **kwargs:
            turn_intent.validate_intent({'intent': 'unknown', 'record_spans': [], 'reason': 'first failed'}, text))
        turn_intent.for_user_message(conn, cid, original, text)
        uid = repo.add_message(conn, cid, 'user', text, json.dumps({'reanalysis': {
            'source_user_message_id': original, 'source_sha256': sha256_text(text)}}))
        conn.commit()
        context = {'context_message_id': original}
    else:
        uid, context = original, {}
    monkeypatch.setattr(turn_intent, 'classify_turn', lambda text, **kwargs:
        turn_intent.validate_intent({'intent': 'record', 'record_spans': [text], 'reason': 'explicit record'}, text))
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=lambda *args, **kwargs: {'content': '已检查。'}))
    monkeypatch.setattr(harness, 'now_ms', lambda: stamp('2026-10-19'))
    extracted = []

    def extract(text, known_names, today):
        extracted.append((text, today))
        return {'people': [], 'my_notes': []}

    monkeypatch.setattr(harness.extract, 'extract_from_text', extract)
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test', 'semantic_verified': False}, 'warnings': []})
    first = harness.run_turn(conn, cid, text, user_message_id=uid, **context)
    repeated = harness.run_turn(conn, cid, text, user_message_id=uid, **context)
    assert first['turn_intent'] == repeated['turn_intent']
    assert extracted == [(text, '2026-10-12'), (text, '2026-10-12')]


@pytest.mark.parametrize('new_status', ['done', 'partial'])
def test_stale_retry_cannot_requeue_a_task_after_another_attempt_completed(conn, monkeypatch, new_status):
    from app.config import get_settings
    from app.store.db import _connect

    cid = repo.create_conversation(conn, 'Retry race')
    task = tasks.submit(conn, 'chat', 'retry', {'conversation_id': cid, 'text': '查询现有记录'}, dispatch=False)
    conn.execute("UPDATE task SET status='partial' WHERE id=?", (task['id'],))
    conn.commit()
    actual_read, dispatches = tasks.read, []
    previous_seq = conn.execute('SELECT MAX(seq) FROM task_event WHERE task_id=?', (task['id'],)).fetchone()[0]

    def stale_read(connection, tid):
        snapshot = actual_read(connection, tid)
        other = _connect(get_settings().abs_db_path)
        try:
            other.execute("UPDATE task SET status='running' WHERE id=?", (tid,))
            tasks.event(other, tid, 'result', {'answer': 'another attempt already completed'})
            # A fast round can finish with the same status and millisecond.
            # Its event cursor must still invalidate the old retry snapshot.
            other.execute('UPDATE task SET status=?,updated_at=? WHERE id=?',
                          (new_status, snapshot['updated_at'], tid))
            other.commit()
        finally:
            other.close()
        return snapshot

    monkeypatch.setattr(tasks, 'read', stale_read)
    monkeypatch.setattr(tasks, 'start', lambda tid: dispatches.append(tid))
    with pytest.raises(ValueError, match='任务状态已变化'):
        tasks.retry(conn, task['id'])
    assert dispatches == []
    assert actual_read(conn, task['id'])['status'] == new_status
    assert conn.execute('SELECT MAX(seq) FROM task_event WHERE task_id=?',
                        (task['id'],)).fetchone()[0] == previous_seq + 1
