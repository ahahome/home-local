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
    assert result['memory_focus']['registration'] == {'status': 'registered', 'source_id': 'S1'}
    user, metadata = user_metadata(conn, cid)
    assert metadata['turn_intent'] == intent
    assert metadata['memory_focus'] == result['memory_focus']['binding']
    assert metadata['conversation_scope'] == result['conversation_scope']
    audit = conn.execute('SELECT * FROM evidence WHERE id=?', (intent['evidence_id'],)).fetchone()
    assert audit['excerpt'] == question and audit['source_ref'] == user['id']
    saved = conn.execute('SELECT * FROM message WHERE id=?', (result['assistant_message_id'],)).fetchone()
    assert saved['content'] == result['answer'] and json.loads(saved['tool_calls'])['sources'] == result['sources']


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
