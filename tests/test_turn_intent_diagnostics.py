"""Bounded strict classification and durable failure audits; no model service calls."""
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent import tools, turn_intent
from app.llm import LLM, LLMError
from app.store import repo


TEXT = '林舟住杭州。'
RECORD = {'intent': 'record', 'record_spans': [TEXT], 'reason': 'explicit fact'}


class ScriptedClassifier:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.responses:
            pytest.fail('classifier exceeded scripted request budget')
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response(value=RECORD, **metadata):
    return {'content': json.dumps(value, ensure_ascii=False), 'response_metadata': metadata}


@pytest.mark.parametrize(('first', 'category'), [
    (TimeoutError('sk-private-timeout'), 'timeout'),
    (ConnectionError('sk-private-network'), 'request'),
    ({'content': ''}, 'empty_content'),
    ({'content': '```json\n{}\n```'}, 'schema'),
    ({'content': '{"intent":"query"}'}, 'schema'),
    (response(finish_reason='length'), 'truncated'),
])
def test_only_one_recoverable_retry_can_produce_a_valid_grounded_decision(first, category):
    client = ScriptedClassifier(first, response())
    result = turn_intent.classify_turn(TEXT, llm=client)
    assert result['allow_writes'] and result['record_spans'] == [TEXT]
    audit = result['classifier']
    assert audit['tier'] == 'mini' and audit['request_count'] == 2
    assert [a['failure_category'] for a in audit['attempts']] == [category, None]
    assert audit['failure_category'] is None and not audit['retryable']
    assert all(c[1]['max_attempts'] == 1 and c[1]['response_metadata'] for c in client.calls)
    assert [c[1]['max_tokens'] for c in client.calls] == [1800, 3600 if category in {
        'empty_content', 'truncated'} else 1800]
    assert client.calls[0][0] == client.calls[1][0]  # Never replay rejected provider text.
    assert 'sk-private' not in json.dumps(result)


@pytest.mark.parametrize(('raw', 'category'), [
    ('Here is JSON: ```json\n{"intent":"record","record_spans":["林舟住杭州。"],"reason":"ok"}\n```', 'json'),
    ('{"intent":"query","intent":"record","record_spans":["林舟住杭州。"],"reason":"ok"}', 'json'),
    ('{"intent":"record","record_spans":["林舟住杭州。"],"reason":NaN}', 'json'),
    ('{"intent":"record","record_spans":["林舟住杭州。"]', 'json'),
    ('[]', 'schema'),
    ('null', 'schema'),
    ('{"intent":"record","record_spans":["林舟住杭州。"],"reason":"ok","extra":true}', 'schema'),
])
def test_second_invalid_body_stays_readonly_without_prose_or_syntax_salvage(raw, category):
    client = ScriptedClassifier({'content': raw}, {'content': raw})
    result = turn_intent.classify_turn(TEXT, llm=client)
    assert not result['allow_writes'] and result['intent'] == 'unknown'
    assert len(client.calls) == result['classifier']['request_count'] == 2
    assert result['classifier']['failure_category'] == category
    assert result['classifier']['retryable']
    assert all('content' not in a and 'raw_response' not in a for a in result['classifier']['attempts'])


def fenced_response(value=RECORD, **metadata):
    return {'content': '```json\n' + json.dumps(value, ensure_ascii=False) + '\n```',
            'response_metadata': metadata}


def test_complete_json_fence_avoids_resampling_but_still_validates_exact_record_span():
    client = ScriptedClassifier(fenced_response())
    result = turn_intent.classify_turn(TEXT, llm=client)
    assert result['record_spans'] == [TEXT] and result['allow_writes']
    assert len(client.calls) == result['classifier']['request_count'] == 1
    assert result['classifier']['attempts'][0]['response_envelope'] == 'json_fence'
    assert result['classifier']['failure_category'] is None


def test_invalid_plain_json_can_recover_with_complete_fence_with_bounded_audit():
    client = ScriptedClassifier({'content': '{'}, fenced_response())
    result = turn_intent.classify_turn(TEXT, llm=client)
    assert result['allow_writes'] and len(client.calls) == 2
    first, second = result['classifier']['attempts']
    assert first['failure_category'] == 'json' and first['protocol_error'] == 'invalid_json'
    assert second['response_envelope'] == 'json_fence'
    assert client.calls[0][0] == client.calls[1][0]


def test_fenced_semantic_unknown_stays_readonly_on_ordinary_task_retry(conn):
    cid = repo.create_conversation(conn)
    mid = repo.add_message(conn, cid, 'user', TEXT)
    client = ScriptedClassifier(fenced_response({'intent': 'unknown', 'record_spans': [], 'reason': 'uncertain'}))
    first = turn_intent.for_user_message(conn, cid, mid, TEXT, llm=client)
    retry = ScriptedClassifier(fenced_response())
    second = turn_intent.for_user_message(conn, cid, mid, TEXT, llm=retry)
    assert first == second and not second['allow_writes'] and not retry.calls
    assert first['intent'] == 'unknown' and len(client.calls) == 1
    assert first['classifier']['failure_category'] is None


def test_fenced_unsafe_span_does_not_gain_a_format_retry():
    text, span = '不要记录：顾念住青岛，只是假设。', '顾念住青岛'
    value = {'intent': 'record', 'record_spans': [span], 'reason': 'unsafe cut'}
    client = ScriptedClassifier(fenced_response(value), fenced_response())
    result = turn_intent.classify_turn(text, llm=client)
    assert not result['allow_writes'] and len(client.calls) == 1
    assert result['classifier']['failure_category'] == 'span'
    assert not result['classifier']['retryable']
    assert result['classifier']['attempts'][0]['response_envelope'] == 'json_fence'


def test_fenced_content_with_provider_length_finish_still_fails_readonly():
    client = ScriptedClassifier(fenced_response(finish_reason='length'), fenced_response(finish_reason='length'))
    result = turn_intent.classify_turn(TEXT, llm=client)
    assert not result['allow_writes'] and len(client.calls) == 2
    assert result['classifier']['failure_category'] == 'truncated'
    assert all('response_envelope' not in attempt for attempt in result['classifier']['attempts'])


def test_successful_fenced_classification_persists_only_its_grounded_write(conn):
    text = '请记住：顾念现在住青岛。'
    cid = repo.create_conversation(conn)
    mid = repo.add_message(conn, cid, 'user', text)
    value = {'intent': 'record', 'record_spans': [text], 'reason': 'full instruction'}
    first = turn_intent.for_user_message(conn, cid, mid, text,
                                        llm=ScriptedClassifier(fenced_response(value)))
    result = tools.execute_tool(conn, 'save_person_fact',
                                {'person': '顾念', 'field': 'city', 'value': '青岛'},
                                source_msg_id=mid, source_text=text, turn_intent=first)
    assert result['ok']
    assert conn.execute('SELECT COUNT(*) FROM person').fetchone()[0] == 1
    raw = conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0]
    saved = json.loads(raw)['turn_intent']
    assert saved['record_spans'] == [text]
    assert saved['classifier']['attempts'][0]['response_envelope'] == 'json_fence'
    assert 'raw_response' not in raw


def test_length_limited_empty_output_has_reasoning_token_diagnostics():
    empty = {'content': '', 'response_metadata': {'finish_reason': 'length',
             'usage': {'completion_tokens': 1800, 'reasoning_tokens': 1800}, 'model': 'qwen-test'}}
    client = ScriptedClassifier(empty, {'content': ''})
    result = turn_intent.classify_turn(TEXT, llm=client)
    attempt = result['classifier']['attempts'][0]
    assert attempt['failure_category'] == 'truncated' and attempt['content_chars'] == 0
    assert attempt['usage'] == {'completion_tokens': 1800, 'reasoning_tokens': 1800}
    assert attempt['finish_reason'] == 'length' and attempt['model'] == 'qwen-test'
    assert result['classifier']['failure_category'] == 'empty_content'


def test_response_metadata_does_not_copy_arbitrary_provider_data():
    provider = {'finish_reason': {'sk-private': True}, 'headers': {'Authorization': 'sk-private'},
                'request_id': 'sk-private', 'model': 'sk-private',
                'usage': {'prompt_tokens': True, 'total_tokens': -1, 'completion_tokens': 5,
                          'reasoning_tokens': 'sk-private', 'secret': 'sk-private'}}
    result = turn_intent.classify_turn(TEXT, llm=ScriptedClassifier(response(**provider)))
    assert result['allow_writes']
    assert result['classifier']['attempts'][0]['usage'] == {'completion_tokens': 5}
    assert 'sk-private' not in json.dumps(result)


@pytest.mark.parametrize(('text', 'span'), [
    ('假设：林舟住杭州，那如何联系？', '林舟住杭州'),
    ('不要记住林舟住杭州，只是个示例。', '林舟住杭州'),
    ('引用旧回答：“林舟住杭州”。是否属实？', '林舟住杭州'),
    ('如果林舟住杭州，就替他记住这个地址。', '就替他记住这个地址。'),
    ('请记住林舟住杭州。', '林舟住杭州。'),
    ('请记住：林舟住杭州。', '林舟住杭州。'),
    ('林舟告诉我：他住杭州。', '他住杭州。'),
    ('请记住：如果林舟来杭州就提醒我。', '如果林舟来杭州就提醒我。'),
    ('林舟说“我不吃香菜”。帮我想菜单。', '我不吃香菜'),
])
def test_unsafe_span_is_not_a_recoverable_format_failure(text, span):
    value = {'intent': 'record', 'record_spans': [span], 'reason': 'unsafe cut'}
    client = ScriptedClassifier(response(value), response(RECORD))
    result = turn_intent.classify_turn(text, llm=client)
    assert not result['allow_writes'] and len(client.calls) == 1
    assert result['classifier']['failure_category'] == 'span'
    assert not result['classifier']['retryable']


@pytest.mark.parametrize('text', [
    '请记住林舟现在不住杭州，改住北京了。',
    '请记住：如果林舟来杭州就提醒我。',
    '林舟告诉我如果不紧急不要打电话，发邮件就行。',
])
def test_complete_explicit_attributed_conditional_or_negative_span_is_preserved(text):
    value = {'intent': 'record', 'record_spans': [text], 'reason': 'full literal span'}
    result = turn_intent.classify_turn(text, llm=ScriptedClassifier(response(value)))
    assert turn_intent.write_text(result, text) == text


def test_semantic_unknown_is_a_successful_readonly_decision_without_sampling_again():
    client = ScriptedClassifier(response({'intent': 'unknown', 'record_spans': [], 'reason': 'unconfirmed'}))
    result = turn_intent.classify_turn(TEXT, llm=client)
    assert result['status'] == 'classified' and not result['allow_writes']
    assert result['classifier']['failure_category'] is None and len(client.calls) == 1
    assert '无法可靠判断' in turn_intent.readonly_outcome(result)['detail']


@pytest.mark.parametrize('error', [
    LLMError('sk-private', category='config'),
    RuntimeError('sk-private'),
    TypeError('sk-private unexpected keyword'),
])
def test_hard_configuration_or_unrecognized_error_never_retries(error):
    client = ScriptedClassifier(error, response())
    result = turn_intent.classify_turn(TEXT, llm=client)
    assert not result['allow_writes'] and len(client.calls) == 1
    assert not result['classifier']['retryable'] and 'sk-private' not in json.dumps(result)


def test_config_initialization_is_diagnosed_without_a_request(monkeypatch):
    def unavailable():
        raise LLMError('sk-private configuration details', category='config')
    monkeypatch.setattr(turn_intent, 'get_llm', unavailable)
    result = turn_intent.classify_turn(TEXT)
    assert result['classifier']['request_count'] == 0
    assert result['classifier']['failure_category'] == 'config'
    outcome = turn_intent.readonly_outcome(result)
    assert '配置不可用' in outcome['detail'] and not outcome['retryable']
    assert 'sk-private' not in json.dumps(result)


def test_real_llm_request_budget_does_not_multiply_internal_retries(monkeypatch):
    llm = LLM.__new__(LLM)
    llm.s = SimpleNamespace(model_mini='test-mini', model_mid='mid', model_think='think',
                            model_pro='pro', model_vision='vision')
    create = Mock(side_effect=TimeoutError('sk-private timeout'))
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr('app.llm.time.sleep', lambda _: pytest.fail('hidden retry/backoff'))
    result = turn_intent.classify_turn(TEXT, llm=llm)
    assert create.call_count == 2 and result['classifier']['request_count'] == 2
    assert result['classifier']['failure_category'] == 'timeout' and result['classifier']['retryable']
    assert 'sk-private' not in json.dumps(result)


def test_failed_attempts_persist_across_sqlite_connections_and_cannot_start_writing(conn):
    cid = repo.create_conversation(conn)
    mid = repo.add_message(conn, cid, 'user', TEXT)
    client = ScriptedClassifier(TimeoutError('sk-private'), {'content': 'before ```json\n{}\n```'})
    first = turn_intent.for_user_message(conn, cid, mid, TEXT, llm=client)
    db_path = conn.execute('PRAGMA database_list').fetchone()['file']
    with sqlite3.connect(db_path) as other:
        other.row_factory = sqlite3.Row
        retry = ScriptedClassifier(response())
        second = turn_intent.for_user_message(other, cid, mid, TEXT, llm=retry)
        assert first == second and not retry.calls
        result = tools.execute_tool(other, 'save_person_fact',
                                    {'person': '林舟', 'field': 'city', 'value': '杭州'},
                                    source_text=TEXT, turn_intent=second)
        assert result['write_blocked']
        assert other.execute('SELECT COUNT(*) FROM person').fetchone()[0] == 0
        assert other.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 0
        assert other.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == 1
        raw = other.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0]
        assert 'sk-private' not in raw
        assert json.loads(raw)['turn_intent']['classifier']['attempts'] == first['classifier']['attempts']


def test_one_format_retry_then_success_persists_exact_decision_and_real_field_write(conn):
    cid = repo.create_conversation(conn)
    mid = repo.add_message(conn, cid, 'user', TEXT)
    first = turn_intent.for_user_message(conn, cid, mid, TEXT,
                                        llm=ScriptedClassifier({'content': ''}, response()))
    result = tools.execute_tool(conn, 'save_person_fact',
                                {'person': '林舟', 'field': 'city', 'value': '杭州'},
                                source_msg_id=mid, source_text=TEXT, turn_intent=first)
    assert result['ok']
    person = conn.execute("SELECT id FROM person WHERE display_name='林舟'").fetchone()[0]
    assert repo.current_fields(conn, person)['city']['value'] == '杭州'
    retry = ScriptedClassifier(RuntimeError('must not request'))
    second = turn_intent.for_user_message(conn, cid, mid, TEXT, llm=retry)
    assert first == second and not retry.calls


@pytest.mark.parametrize('saved', [None, {}, {'version': 0},
                                   {'version': 1, 'source_sha256': 'different'}])
def test_invalid_saved_decision_cannot_get_a_new_writable_sample(conn, saved):
    cid = repo.create_conversation(conn)
    mid = repo.add_message(conn, cid, 'user', TEXT, json.dumps({'turn_intent': saved}))
    client = ScriptedClassifier(response())
    result = turn_intent.for_user_message(conn, cid, mid, TEXT, llm=client)
    assert not result['allow_writes'] and not client.calls
    assert result['classifier']['failure_category'] == 'schema'


@pytest.mark.parametrize('raw', ['{', 'null', '[]', 'NaN',
                                '{"turn_intent":{},"turn_intent":{}}'])
def test_corrupt_user_audit_cannot_get_a_new_writable_sample(conn, raw):
    cid = repo.create_conversation(conn)
    mid = repo.add_message(conn, cid, 'user', TEXT, raw)
    client = ScriptedClassifier(response())
    first = turn_intent.for_user_message(conn, cid, mid, TEXT, llm=client)
    second = turn_intent.for_user_message(conn, cid, mid, TEXT, llm=client)
    assert first == second and not first['allow_writes'] and not client.calls
    assert first['classifier']['failure_category'] == 'schema'


def test_readonly_date_policy_persists_query_with_zero_requests_and_cannot_override_saved_write(conn):
    cid = repo.create_conversation(conn)
    text = '这条记忆是什么时候改的？'
    mid = repo.add_message(conn, cid, 'user', text)
    client = ScriptedClassifier(response())
    first = turn_intent.for_user_message(conn, cid, mid, text, llm=client,
                                        readonly_reason='memory_date_question')
    second = turn_intent.for_user_message(conn, cid, mid, text, llm=client)
    assert first == second and first['intent'] == 'query' and first['status'] == 'ok'
    assert first['reason'] == 'memory_date_question' and not first['allow_writes']
    assert first['classifier']['request_count'] == 0 and not client.calls
    written_mid = repo.add_message(conn, cid, 'user', TEXT)
    written = turn_intent.for_user_message(conn, cid, written_mid, TEXT, llm=ScriptedClassifier(response()))
    reused = turn_intent.for_user_message(conn, cid, written_mid, TEXT,
                                         readonly_reason='memory_date_question')
    assert reused == written and reused['allow_writes']


@pytest.mark.parametrize(('category', 'label'), [('request', '服务暂时不可用'),
                                               ('timeout', '服务请求超时'), ('json', '格式未通过校验'),
                                               ('empty_content', '格式未通过校验'), ('span', '原文片段')])
def test_readonly_receipt_distinguishes_failure_kind_and_reanalysis_availability(category, label):
    result = turn_intent._diagnosed(turn_intent._readonly(TEXT, 'test', 'failed'),
                                    [{'request_started': True}], category, category != 'span')
    receipt = turn_intent.readonly_outcome(result)
    assert label in receipt['detail'] and '未保存新记忆' in receipt['detail']
    assert receipt['retryable'] is (category != 'span')
    assert ('可重新分析' in receipt['detail']) is receipt['retryable']
