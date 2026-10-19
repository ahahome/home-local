"""Write boundaries use real SQLite; semantic classification is reviewed separately."""
import json

import pytest

from app.agent import tools, turn_intent
from app.memory import apply as apply_mod, extract
from app.store import repo


class Classifier:
    def __init__(self, result=None, raw=None, error=None):
        self.result, self.raw, self.error = result, raw, error
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return {'content': self.raw if self.raw is not None else json.dumps(self.result, ensure_ascii=False)}


def decision(text, intent='query', spans=()):
    return turn_intent.validate_intent({'intent': intent, 'record_spans': list(spans), 'reason': 'test'}, text)


def state(conn):
    tables = ('person', 'person_alias', 'person_field', 'memory_item', 'interaction', 'interaction_person',
              'pending_item', 'promise', 'ledger_entry', 'reminder', 'research', 'skill', 'applied_patch',
              'evidence', 'relation_edge')
    return {name: [tuple(row) for row in conn.execute(f'SELECT * FROM {name} ORDER BY rowid')]
            for name in tables}


@pytest.mark.parametrize('text', ['林舟怎么联系', '林舟的血型是什么？', '假设林舟在上海，该怎么安排？',
                                      '你上次说“林舟喜欢电话”，可靠吗？', '把“林舟住上海”改成英文例句'])
def test_queries_are_not_classified_by_question_mark(text):
    client = Classifier({'intent': 'query', 'record_spans': [], 'reason': '只查询或讨论'})
    result = turn_intent.classify_turn(text, llm=client)
    assert result['intent'] == 'query' and not result['allow_writes']
    assert result['extraction_text'] == ''
    assert client.calls[0][1]['temperature'] == 0
    assert json.loads(client.calls[0][0][-1]['content'])['message'] == text


@pytest.mark.parametrize('raw', ['```json\n{}\n```', 'null', '[]',
                                '{"intent":"query","record_spans":[]}',
                                '{"intent":"record","record_spans":["林舟住上海"],"reason":"ok","allow_writes":true}',
                                '{"intent":[],"record_spans":[],"reason":"bad"}'])
def test_invalid_json_or_schema_stays_readonly(raw):
    result = turn_intent.classify_turn('林舟住上海', llm=Classifier(raw=raw))
    assert result['intent'] == 'unknown' and not result['allow_writes']


def test_failure_does_not_expose_provider_error_or_authorize_write():
    result = turn_intent.classify_turn('帮我记住林舟住上海', llm=Classifier(error=RuntimeError('private provider detail')))
    assert result['status'] == 'failed'
    assert result['intent'] == 'unknown' and not result['allow_writes']
    assert 'private' not in str(result)


def test_duplicate_json_keys_or_classifier_tool_calls_are_not_authorization():
    raw = '{"intent":"query","intent":"record","record_spans":["林舟住上海。"],"reason":"test"}'
    result = turn_intent.classify_turn('林舟住上海。', llm=Classifier(raw=raw))
    assert not result['allow_writes']
    class ToolClassifier:
        def chat(self, *args, **kwargs):
            return {'content': '{"intent":"record","record_spans":["林舟住上海。"],"reason":"test"}',
                    'tool_calls': []}
    assert not turn_intent.classify_turn('林舟住上海。', llm=ToolClassifier())['allow_writes']


@pytest.mark.parametrize(('text', 'span'), [
    ('如果林舟住上海，该怎么安排？', '林舟住上海'),
    ('如果林舟住上海，该怎么安排？', '如果林舟住上海'),
    ('假设：林舟住上海，那怎么联系？', '林舟住上海'),
    ('不要记录：林舟住上海，只是个示例。', '林舟住上海'),
    ('引用旧回答：“林舟住上海”。靠谱吗？', '林舟住上海'),
    ('> 林舟住上海\n这是真的吗？', '> 林舟住上海'),
    ('林舟住上海吗？', '林舟住上海'),
    ('林舟住上海，林舟住上海是否属实？', '林舟住上海'),
    ('林舟住上海。', '林舟现在住上海。'),
])
def test_model_cannot_cut_away_hypothesis_quotation_or_question_suffix(text, span):
    result = decision(text, 'mixed', [span])
    assert not result['allow_writes'] and result['status'] == 'invalid'


@pytest.mark.parametrize(('text', 'spans'), [
    ('林舟搬到杭州了，他以前住哪里？', ['林舟搬到杭州了']),
    ('林舟现在不住杭州，改住北京了。之前呢？', ['林舟现在不住杭州，改住北京了。']),
    ('林舟说“我不吃香菜”，帮我想个菜单。', ['林舟说“我不吃香菜”']),
    ('请记住林舟住上海', ['请记住林舟住上海']),
])
def test_true_assertions_and_mixed_messages_keep_grounded_spans(text, spans):
    result = decision(text, 'mixed', spans)
    assert result['allow_writes']
    assert result['extraction_text'] == '\n'.join(spans)
    assert turn_intent.write_text(result, text) == result['extraction_text']


@pytest.mark.parametrize(('text', 'span'), [
    ('唐宁说如果是急事就发短信，平时发邮件。', '唐宁说如果是急事就发短信，平时发邮件。'),
    ('唐宁说如果是急事就发短信，平时发邮件。他以前怎么联系？', '唐宁说如果是急事就发短信，平时发邮件。'),
    ('请记住：如果他来杭州就提醒我。', '请记住：如果他来杭州就提醒我。'),
    ('请记住：如果他来杭州就提醒我。之前去过哪里？', '请记住：如果他来杭州就提醒我。'),
    ('唐宁告诉我如果不紧急不要打电话，发邮件就行。', '唐宁告诉我如果不紧急不要打电话，发邮件就行。'),
])
def test_true_conditional_preferences_and_explicit_rules_keep_full_context(text, span):
    result = decision(text, 'mixed', [span])
    assert result['allow_writes'] and result['extraction_text'] == span


@pytest.mark.parametrize(('text', 'span'), [
    ('如果唐宁说急事发短信，那该怎么回复？', '如果唐宁说急事发短信'),
    ('如果唐宁住杭州，记住他可能不想打电话。', '记住他可能不想打电话。'),
    ('唐宁说如果是急事就发短信，平时发邮件。', '平时发邮件。'),
    ('请记住：如果他来杭州就提醒我。', '如果他来杭州就提醒我。'),
    ('你上次说如果是急事就发短信，这是真的吗？', '你上次说如果是急事就发短信'),
    ('不要记录：如果他来杭州就提醒我。', '如果他来杭州就提醒我。'),
    ('如果陈宁来杭州，唐宁说如果急事就发短信，那你怎么回复？', '唐宁说如果急事就发短信'),
])
def test_conditional_context_cannot_be_removed_or_borrowed_from_old_answer(text, span):
    assert not decision(text, 'record', [span])['allow_writes']


def test_span_metadata_cannot_authorize_different_source():
    text = '林舟住上海。'
    result = decision(text, 'record', [text])
    assert turn_intent.write_text(result, '假设林舟住上海。') is None
    result['extraction_text'] = '林舟住北京。'
    assert turn_intent.write_text(result, text) is None


def test_user_decision_audit_is_idempotent_and_preserves_scope(conn):
    cid = repo.create_conversation(conn)
    text = '林舟怎么联系'
    mid = repo.add_message(conn, cid, 'user', text, json.dumps({'conversation_scope': {'mode': 'single'}}))
    before = state(conn)
    client = Classifier({'intent': 'query', 'record_spans': [], 'reason': '查询'})
    first = turn_intent.for_user_message(conn, cid, mid, text, llm=client)
    second = turn_intent.for_user_message(conn, cid, mid, text, llm=Classifier(error=AssertionError('must reuse')))
    assert first == second and len(client.calls) == 1
    after = state(conn)
    assert len(after['evidence']) == len(before['evidence']) + 1
    assert {k: v for k, v in before.items() if k != 'evidence'} == {k: v for k, v in after.items() if k != 'evidence'}
    metadata = json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0])
    assert metadata['conversation_scope']['mode'] == 'single'
    assert metadata['turn_intent']['intent'] == 'query'
    assert conn.execute('SELECT excerpt FROM evidence WHERE id=?', (first['evidence_id'],)).fetchone()[0] == text


def test_unknown_failure_cannot_become_a_write_on_retry(conn):
    cid = repo.create_conversation(conn)
    text = '林舟住上海。'
    mid = repo.add_message(conn, cid, 'user', text)
    first = turn_intent.for_user_message(conn, cid, mid, text, llm=Classifier(error=RuntimeError('offline')))
    retry = Classifier({'intent': 'record', 'record_spans': [text], 'reason': '后来样本'})
    second = turn_intent.for_user_message(conn, cid, mid, text, llm=retry)
    assert first == second and not second['allow_writes'] and not retry.calls


def test_audit_cannot_attach_to_another_conversation_or_changed_message(conn):
    cid, other = repo.create_conversation(conn), repo.create_conversation(conn)
    mid = repo.add_message(conn, cid, 'user', '林舟住上海。')
    with pytest.raises(ValueError):
        turn_intent.for_user_message(conn, other, mid, '林舟住上海。')
    with pytest.raises(ValueError):
        turn_intent.for_user_message(conn, cid, mid, '林舟住北京。')
    assert conn.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == 0


@pytest.mark.parametrize('name', sorted(tools._WRITE_TOOLS))
@pytest.mark.parametrize('intent_name', ['query', 'unknown'])
def test_all_write_tools_are_blocked_before_any_sqlite_changes(conn, name, intent_name):
    text = '林舟的沟通偏好是什么？'
    before = state(conn)
    result = tools.execute_tool(conn, name, {}, source_text=text, turn_intent=decision(text, intent_name))
    assert result['ok'] is False and result['write_blocked'] is True
    assert result['receipt']['status'] == 'skipped'
    assert '未保存' in result['receipt']['detail']
    assert state(conn) == before


def test_query_read_tools_still_work_with_boundary(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    text = '林舟的资料是什么？'
    before = state(conn)
    result = tools.execute_tool(conn, 'get_person', {'person_id': pid}, source_text=text, turn_intent=decision(text))
    assert result['ok'] and result['person_id'] == pid
    assert state(conn) == before


def test_mixed_turn_can_save_fact_but_not_question_content_or_other_person(conn):
    text = '林舟住杭州，陈宁是不是住上海？'
    boundary = decision(text, 'mixed', ['林舟住杭州'])
    before = state(conn)
    for args in [{'person': '林舟', 'field': 'city', 'value': '上海'},
                 {'person': '陈宁', 'field': 'city', 'value': '杭州'}]:
        result = tools.execute_tool(conn, 'save_person_fact', args, source_text=text, turn_intent=boundary)
        assert result['write_blocked'] and state(conn) == before
    result = tools.execute_tool(conn, 'save_person_fact', {'person': '林舟', 'field': 'city', 'value': '杭州'},
                                source_text=text, source_msg_id='mixed-source', turn_intent=boundary)
    assert result['ok']
    pid = conn.execute("SELECT id FROM person WHERE display_name='林舟'").fetchone()[0]
    assert repo.current_fields(conn, pid)['city']['value'] == '杭州'
    assert conn.execute("SELECT COUNT(*) FROM person WHERE display_name='陈宁'").fetchone()[0] == 0
    assert conn.execute('SELECT excerpt FROM evidence WHERE source_ref=?', ('mixed-source',)).fetchone()[0] == '林舟住杭州'


def test_mixed_transfer_or_action_from_query_cannot_create_pending_or_reminder(conn):
    text = '林舟住杭州，我给陈宁转过300元吗，需要提醒我吗？'
    boundary = decision(text, 'mixed', ['林舟住杭州'])
    before = state(conn)
    for name, args in [('record_transfer', {'person': '陈宁', 'amount_yuan': 300, 'direction': 'out'}),
                       ('create_reminder', {'text': '联系林舟', 'fire_date': '2026-11-01'}),
                       ('save_skill', {'name': '联系', 'body': '林舟住杭州'}),
                       ('deep_research', {'topic': '林舟'}),
                       ('complete_promise', {'promise_id': 'unknown'})]:
        result = tools.execute_tool(conn, name, args, source_text=text, turn_intent=boundary)
        assert result['write_blocked'] and state(conn) == before


def test_explicit_research_request_can_create_proposal(conn, monkeypatch):
    from app.agent import research
    monkeypatch.setattr(research, 'websearch', lambda *args, **kwargs: [])
    monkeypatch.setattr(research, 'direct_sources', lambda *args, **kwargs: [])
    monkeypatch.setattr(research, '_proposal', lambda *args, **kwargs:
                        {'questions': ['了解产品'], 'subtasks': [{'scope': '产品事实', 'queries': []}]})
    text = '请调研星河科技的产品。'
    result = tools.execute_tool(conn, 'deep_research', {'topic': '星河科技的产品'}, source_text=text,
                                turn_intent=decision(text, 'record', [text]))
    assert result['ok']
    assert repo.get_research(conn, result['receipt']['id']) is not None


def test_explicit_reminder_request_can_create_reminder(conn):
    text = '请在2026-11-01提醒我发送合同。'
    result = tools.execute_tool(conn, 'create_reminder', {'text': '发送合同', 'fire_date': '2026-11-01'},
                                source_text=text, turn_intent=decision(text, 'record', [text]))
    assert result['ok']
    assert conn.execute('SELECT text FROM reminder WHERE id=?', (result['receipt']['id'],)).fetchone()[0] == '发送合同'


@pytest.mark.parametrize('args', [
    {'text': '林舟住上海', 'fire_date': '2026-11-01'},
    {'text': '发合同', 'fire_date': '2027-12-31'},
    {'text': '林舟住上海', 'fire_date': '2027-12-31'},
])
def test_reminder_arguments_cannot_come_from_excluded_question_or_invention(conn, args):
    span = '请在2026-11-01提醒我发合同。'
    text = span + '林舟住上海吗，2027-12-31是什么日子？'
    before = state(conn)
    result = tools.execute_tool(conn, 'create_reminder', args, source_text=text,
                                turn_intent=decision(text, 'mixed', [span]))
    assert result['write_blocked'] and result['receipt']['status'] == 'skipped'
    assert state(conn) == before


@pytest.mark.parametrize('span', ['明天提醒我发合同。', '11月1日提醒我发合同。',
                                '请在2026-11-01或2026-11-02提醒我发合同。'])
def test_reminder_relative_missing_year_or_ambiguous_date_requires_explicit_day(conn, span):
    before = state(conn)
    result = tools.execute_tool(conn, 'create_reminder', {'text': '发合同', 'fire_date': '2026-11-01'},
                                source_text=span, turn_intent=decision(span, 'record', [span]))
    assert result['write_blocked'] and '日历日期' in result['error']
    assert state(conn) == before


def test_reminder_cannot_borrow_date_from_another_approved_action(conn):
    first, second = '请在2026-11-01提醒我发合同。', '请在2027-12-31提醒我打电话。'
    text = first + second
    before = state(conn)
    result = tools.execute_tool(conn, 'create_reminder', {'text': '发合同', 'fire_date': '2027-12-31'},
                                source_text=text, turn_intent=decision(text, 'record', [first, second]))
    assert result['write_blocked'] and state(conn) == before


def test_reminder_accepts_verifiable_chinese_calendar_date(conn):
    text = '请在2026年11月1日提醒我发合同。'
    result = tools.execute_tool(conn, 'create_reminder', {'text': '发合同', 'fire_date': '2026-11-01'},
                                source_text=text, turn_intent=decision(text, 'record', [text]))
    assert result['ok'] and conn.execute('SELECT COUNT(*) FROM reminder').fetchone()[0] == 1


@pytest.mark.parametrize('args', [
    {'name': '联络', 'body': '林舟住上海'},
    {'name': '陌生技能', 'body': '先发邮件'},
    {'name': '联络', 'body': '先发邮件', 'style_notes': '林舟住上海'},
    {'name': '联络', 'body': '先发邮件', 'trigger_hint': '林舟住上海'},
])
def test_skill_all_text_arguments_must_be_in_approved_instruction(conn, args):
    span = '保存技能，名称联络，正文先发邮件。'
    text = span + '林舟住上海吗？'
    before = state(conn)
    result = tools.execute_tool(conn, 'save_skill', args, source_text=text,
                                turn_intent=decision(text, 'mixed', [span]))
    assert result['write_blocked'] and state(conn) == before


def test_skill_explicit_name_body_and_metadata_are_saved(conn):
    text = '保存技能，名称联络，正文先发邮件，风格简短，触发词联系客户。'
    result = tools.execute_tool(conn, 'save_skill', {'name': '联络', 'body': '先发邮件',
        'style_notes': '简短', 'trigger_hint': '联系客户'}, source_text=text,
        turn_intent=decision(text, 'record', [text]))
    assert result['ok']
    row = repo.get_skill(conn, '联络')
    assert row['body'] == '先发邮件' and row['style_notes'] == '简短'


def test_skill_cannot_borrow_body_from_other_approved_fact(conn):
    first, second = '保存技能，名称联络，正文先发邮件。', '林舟住杭州。'
    text = first + second
    before = state(conn)
    result = tools.execute_tool(conn, 'save_skill', {'name': '联络', 'body': '林舟住杭州'},
                                source_text=text, turn_intent=decision(text, 'record', [first, second]))
    assert result['write_blocked'] and state(conn) == before


@pytest.mark.parametrize('topic', ['陈宁的资产', '从未提及的主题'])
def test_research_topic_cannot_come_from_excluded_question_or_invention(conn, topic):
    span = '请调研星河科技的产品。'
    text = span + '陈宁的资产有多少？'
    before = state(conn)
    result = tools.execute_tool(conn, 'deep_research', {'topic': topic}, source_text=text,
                                turn_intent=decision(text, 'mixed', [span]))
    assert result['write_blocked'] and state(conn) == before


def test_research_cannot_borrow_target_from_another_approved_fact(conn):
    first, second = '请调研星河科技的产品。', '林舟住杭州。'
    text = first + second
    before = state(conn)
    result = tools.execute_tool(conn, 'deep_research', {'topic': '林舟'}, source_text=text,
                                turn_intent=decision(text, 'record', [first, second]))
    assert result['write_blocked'] and state(conn) == before


def test_complete_promise_does_not_accept_arbitrary_id_because_message_says_completed(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    selected = repo.add_promise(conn, 'me', '发送合同', pid)
    unrelated = repo.add_promise(conn, 'me', '支付尾款', pid)
    span = '林舟的发送合同已完成。'
    text = span + '支付尾款完成了吗？'
    before = state(conn)
    result = tools.execute_tool(conn, 'complete_promise', {'promise_id': unrelated}, source_text=text,
                                turn_intent=decision(text, 'mixed', [span]))
    assert result['write_blocked'] and state(conn) == before
    valid = tools.execute_tool(conn, 'complete_promise', {'promise_id': selected}, source_text=text,
                               turn_intent=decision(text, 'mixed', [span]))
    assert valid['ok']
    assert conn.execute('SELECT status FROM promise WHERE id=?', (selected,)).fetchone()[0] == 'done'
    assert conn.execute('SELECT status FROM promise WHERE id=?', (unrelated,)).fetchone()[0] == 'open'


def test_complete_promise_requires_unambiguous_owner_and_content_or_exact_id(conn):
    first, second = repo.create_person(conn, '林舟', status='active'), repo.create_person(conn, '陈宁', status='active')
    selected = repo.add_promise(conn, 'me', '发送合同', first)
    other = repo.add_promise(conn, 'me', '发送合同', second)
    text = '发送合同已完成。'
    before = state(conn)
    boundary = decision(text, 'record', [text])
    denied = tools.execute_tool(conn, 'complete_promise', {'promise_id': other}, source_text=text,
                                turn_intent=boundary)
    assert denied['write_blocked'] and state(conn) == before
    scope = {'mode': 'single', 'person_ids': [first], 'blocked': False}
    denied = tools.execute_tool(conn, 'complete_promise', {'promise_id': other}, source_text=text,
                                turn_intent=boundary, read_scope=scope)
    assert denied['write_blocked'] and state(conn) == before
    accepted = tools.execute_tool(conn, 'complete_promise', {'promise_id': selected}, source_text=text,
                                  turn_intent=boundary, read_scope=scope)
    assert accepted['ok']


def test_complete_promise_duplicate_wording_needs_exact_promise_id(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    first, second = repo.add_promise(conn, 'me', '发送合同', pid), repo.add_promise(conn, 'me', '发送合同', pid)
    text = '林舟的发送合同已完成。'
    before = state(conn)
    denied = tools.execute_tool(conn, 'complete_promise', {'promise_id': second}, source_text=text,
                                turn_intent=decision(text, 'record', [text]))
    assert denied['write_blocked'] and state(conn) == before
    precise = f'请把承诺 {first} 标记为完成。'
    accepted = tools.execute_tool(conn, 'complete_promise', {'promise_id': first}, source_text=precise,
                                  turn_intent=decision(precise, 'record', [precise]))
    assert accepted['ok']
    assert conn.execute('SELECT status FROM promise WHERE id=?', (second,)).fetchone()[0] == 'open'


def test_complete_promise_exact_id_still_obeys_selected_person_scope(conn):
    first, second = repo.create_person(conn, '林舟', status='active'), repo.create_person(conn, '陈宁', status='active')
    target = repo.add_promise(conn, 'me', '发送合同', second)
    text = f'请把承诺 {target} 标记为完成。'
    before = state(conn)
    result = tools.execute_tool(conn, 'complete_promise', {'promise_id': target}, source_text=text,
                                turn_intent=decision(text, 'record', [text]),
                                read_scope={'mode': 'single', 'person_ids': [first], 'blocked': False})
    assert result['write_blocked'] and state(conn) == before


@pytest.mark.parametrize(('name', 'args', 'span'), [
    ('save_person_fact', {'person': '林舟', 'field': 'city', 'value': '杭州', 'valid_from': '2027-12-31'},
     '林舟从2026-11-01起住杭州。'),
    ('create_promise', {'person': '林舟', 'who': 'me', 'content': '发送合同', 'due_date': '2027-12-31'},
     '我答应林舟在2026-11-01前发送合同。'),
    ('record_transfer', {'person': '林舟', 'direction': 'out', 'amount_yuan': 100, 'occurred_date': '2027-12-31'},
     '我在2026-11-01给林舟转了100元。'),
    ('create_promise', {'who': 'me', 'content': '发送合同', 'due_date': '2027-12-31'},
     '我答应在2026-11-01前发送合同。'),
])
def test_shared_write_date_cannot_be_invented_or_borrowed_from_question(conn, name, args, span):
    text = span + '2027-12-31是什么日子？'
    before = state(conn)
    result = tools.execute_tool(conn, name, args, source_text=text, turn_intent=decision(text, 'mixed', [span]))
    assert result['write_blocked'] and state(conn) == before


@pytest.mark.parametrize(('name', 'args', 'text'), [
    ('save_person_fact', {'person': '林舟', 'field': 'city', 'value': '杭州', 'valid_from': '2026-11-01'},
     '林舟现在住杭州。'),
    ('create_promise', {'who': 'me', 'content': '发送合同', 'due_date': '2026-11-01'}, '我答应明天发送合同。'),
    ('record_transfer', {'person': '林舟', 'direction': 'out', 'amount_yuan': 100, 'occurred_date': '2026-11-01'},
     '我昨天给林舟转了100元。'),
])
def test_shared_write_unknown_or_relative_date_cannot_authorize_model_calendar_guess(conn, name, args, text):
    before = state(conn)
    result = tools.execute_tool(conn, name, args, source_text=text, turn_intent=decision(text, 'record', [text]))
    assert result['write_blocked'] and state(conn) == before


def test_shared_write_cannot_borrow_date_from_other_approved_fact(conn):
    first, second = '林舟住杭州。', '陈宁从2026-11-01开始休假。'
    text = first + second
    before = state(conn)
    result = tools.execute_tool(conn, 'save_person_fact', {'person': '林舟', 'field': 'city',
        'value': '杭州', 'valid_from': '2026-11-01'}, source_text=text,
        turn_intent=decision(text, 'record', [first, second]))
    assert result['write_blocked'] and state(conn) == before


@pytest.mark.parametrize(('name', 'args', 'text'), [
    ('save_person_fact', {'person': '林舟', 'field': 'city', 'value': '杭州', 'valid_from': '2026-11-01'},
     '林舟从2026年11月1日起住杭州。'),
    ('create_promise', {'who': 'me', 'content': '发送合同', 'due_date': '2026-11-01'},
     '我答应在2026-11-01前发送合同。'),
    ('record_transfer', {'person': '林舟', 'direction': 'out', 'amount_yuan': 100, 'occurred_date': '2026-11-01'},
     '我在2026-11-01给林舟转了100元。'),
    ('save_person_fact', {'person': '林舟', 'field': 'city', 'value': '杭州'}, '林舟住杭州。'),
    ('create_promise', {'who': 'me', 'content': '发送合同'}, '我答应发送合同。'),
    ('record_transfer', {'person': '林舟', 'direction': 'out', 'amount_yuan': 100}, '我给林舟转了100元。'),
])
def test_shared_write_explicit_matching_date_or_omitted_default_is_preserved(conn, name, args, text):
    result = tools.execute_tool(conn, name, args, source_text=text,
                                turn_intent=decision(text, 'record', [text]))
    assert result['ok'] and not result.get('write_blocked')


@pytest.mark.parametrize(('name', 'args'), [
    ('save_person_fact', {'person': '林舟', 'field': 'city', 'value': '上海'}),
    ('create_promise', {'person': '林舟', 'who': 'me', 'content': '整理报价'}),
    ('record_transfer', {'person': '林舟', 'direction': 'in', 'amount_yuan': 200}),
    ('record_transfer', {'person': '林舟', 'direction': 'in', 'amount_yuan': 100}),
    ('record_transfer', {'person': '林舟', 'direction': 'out', 'amount_yuan': 100, 'note': '整理报价'}),
])
def test_shared_writes_cannot_mix_people_values_promises_or_transfer_arguments_across_spans(conn, name, args):
    first, second = repo.create_person(conn, '林舟', status='active'), repo.create_person(conn, '陈宁', status='active')
    spans = ['林舟住杭州。', '陈宁住上海。', '我答应林舟发送合同。', '我答应陈宁整理报价。',
             '我给林舟转了100元。', '陈宁给我转了200元。']
    text = ''.join(spans) + '他们之前住哪里？'
    before = state(conn)

    result = tools.execute_tool(conn, name, args, source_text=text,
                                turn_intent=decision(text, 'mixed', spans))

    assert result['write_blocked'] and state(conn) == before
    assert not repo.current_fields(conn, first) and not repo.current_fields(conn, second)


def test_shared_field_values_stay_with_their_own_person_in_actual_sqlite(conn):
    first, second = repo.create_person(conn, '林舟', status='active'), repo.create_person(conn, '陈宁', status='active')
    spans = ['林舟住杭州。', '陈宁住上海。']
    text = ''.join(spans) + '他们之前住哪里？'
    boundary = decision(text, 'mixed', spans)

    for pid, value in [(first, '杭州'), (second, '上海')]:
        result = tools.execute_tool(conn, 'save_person_fact', {'person_id': pid, 'field': 'city', 'value': value},
                                    source_text=text, turn_intent=boundary)
        assert result['ok'] and not result.get('write_blocked')

    assert repo.current_fields(conn, first)['city']['value'] == '杭州'
    assert repo.current_fields(conn, second)['city']['value'] == '上海'


def test_shared_promise_counterparty_and_body_use_one_approved_span(conn):
    first, second = repo.create_person(conn, '林舟', status='active'), repo.create_person(conn, '陈宁', status='active')
    spans = ['我答应林舟发送合同。', '陈宁答应整理报价。']
    text = ''.join(spans)
    boundary = decision(text, 'record', spans)

    for pid, who, content in [(first, 'me', '发送合同'), (second, 'them', '整理报价')]:
        result = tools.execute_tool(conn, 'create_promise', {'person_id': pid, 'who': who, 'content': content},
                                    source_text=text, turn_intent=boundary)
        assert result['ok'] and not result.get('write_blocked')

    assert [tuple(row) for row in conn.execute('SELECT person_id,who,content FROM promise ORDER BY rowid')] == [
        (first, 'me', '发送合同'), (second, 'them', '整理报价')]


def test_shared_personal_note_cannot_join_different_approved_spans(conn):
    spans = ['我喜欢喝茶。', '我最近喜欢散步。']
    text = ''.join(spans)
    boundary = decision(text, 'record', spans)
    before = state(conn)

    denied = tools.execute_tool(conn, 'note', {'text': '\n'.join(spans)}, source_text=text, turn_intent=boundary)

    assert denied['write_blocked'] and state(conn) == before
    accepted = tools.execute_tool(conn, 'note', {'text': spans[1]}, source_text=text, turn_intent=boundary)
    assert accepted['ok']
    assert [row[0] for row in conn.execute('SELECT text FROM memory_item')] == [spans[1]]


def test_shared_transfer_direction_check_cannot_consume_another_approved_transfer(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    spans = ['我在2026-11-01给林舟转了100元。', '林舟给我转了100元。']
    text = ''.join(spans)

    result = tools.execute_tool(conn, 'record_transfer', {'person_id': pid, 'direction': 'out',
        'amount_yuan': 100, 'occurred_date': '2026-11-01'}, source_text=text,
        turn_intent=decision(text, 'record', spans))

    assert result['ok'] and not result.get('write_blocked')
    assert [tuple(row) for row in conn.execute('SELECT person_id,direction,amount_cents FROM ledger_entry')] == [
        (pid, 'out', 10000)]


def test_shared_span_binding_preserves_single_person_pronouns(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    spans = ['他住杭州。', '我给他转了100元。']
    text = ''.join(spans)
    boundary = decision(text, 'record', spans)
    scope = {'mode': 'single', 'person_ids': [pid], 'blocked': False}

    field = tools.execute_tool(conn, 'save_person_fact', {'person_id': pid, 'field': 'city', 'value': '杭州'},
                               source_text=text, turn_intent=boundary, read_scope=scope)
    transfer = tools.execute_tool(conn, 'record_transfer', {'person_id': pid, 'direction': 'out', 'amount_yuan': 100},
                                  source_text=text, turn_intent=boundary, read_scope=scope)

    assert field['ok'] and transfer['ok']
    assert repo.current_fields(conn, pid)['city']['value'] == '杭州'
    assert conn.execute('SELECT person_id,direction,amount_cents FROM ledger_entry').fetchone()[:] == (pid, 'out', 10000)


def test_mixed_extraction_uses_only_new_assertion_and_keeps_original_audit(conn):
    cid = repo.create_conversation(conn)
    text = '林舟住杭州，陈宁住上海是真的吗？'
    mid = repo.add_message(conn, cid, 'user', text)
    result = turn_intent.for_user_message(conn, cid, mid, text, llm=Classifier(
        {'intent': 'mixed', 'record_spans': ['林舟住杭州'], 'reason': '前半句是新事实'}))
    payload = extract.sanitize({'people': [{'name': '林舟', 'canonical_name': '林舟', 'aliases': [],
        'fields': [{'field': 'city', 'value': '杭州', 'confidence': .95, 'evidence': '林舟住杭州'}],
        'facts': [], 'promises': [], 'ledger': [], 'relations': []}], 'my_notes': []},
        source=result['extraction_text'])
    applied = apply_mod.apply_extraction(conn, payload, mid, result['extraction_text'], evidence_id=result['evidence_id'])
    assert applied['person_ids']
    assert conn.execute("SELECT COUNT(*) FROM person WHERE display_name='陈宁'").fetchone()[0] == 0
    assert conn.execute('SELECT excerpt FROM evidence WHERE id=?', (result['evidence_id'],)).fetchone()[0] == text


def test_legacy_execute_tool_without_turn_intent_keeps_existing_contract(conn):
    result = tools.execute_tool(conn, 'note', {'text': '我喜欢清淡饮食'})
    assert result['ok']
    assert conn.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 1


@pytest.mark.parametrize('boundary', [{}, {'intent': 'record', 'allow_writes': True}, 'record'])
def test_malformed_boundary_is_not_legacy_permission(conn, boundary):
    before = state(conn)
    result = tools.execute_tool(conn, 'note', {'text': '我喜欢清淡饮食'}, source_text='我喜欢清淡饮食',
                                turn_intent=boundary)
    assert result['write_blocked'] and state(conn) == before
