"""Deterministic conversation identity and read boundaries, with no model accuracy claims."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.agent import conversation_scope, harness
from app.store import repo


def person(conn, name, **fields):
    pid = repo.create_person(conn, name, status='active')
    for index, (field, value) in enumerate(fields.items()):
        repo.upsert_field(conn, pid, field, value, 1., None, valid_from=1000 + index)
    conn.commit()
    return pid


def turn(conn, cid, text):
    mid = repo.add_message(conn, cid, 'user', text)
    conn.commit()
    return conversation_scope.for_user_message(conn, cid, mid, text)


def conversation(conn):
    cid = repo.create_conversation(conn, 'Scope test')
    conn.commit()
    return cid


def test_explicit_person_pronoun_switch_and_conversation_isolation(conn):
    first, second = person(conn, '林舟'), person(conn, '陈岚')
    cid = conversation(conn)
    initial = turn(conn, cid, '林舟喜欢什么沟通方式？')
    inherited = turn(conn, cid, '她的近况呢？')
    assert inherited['mode'] == 'single' and inherited['person_ids'] == [first]
    assert inherited['origin_message_id'] == initial['user_message_id']
    assert inherited['user_message_id'] != initial['user_message_id']
    switched = turn(conn, cid, '换成陈岚，她的偏好是什么？')
    assert switched['person_ids'] == [second]
    assert turn(conn, cid, '他最近怎么样？')['person_ids'] == [second]
    assert turn(conn, conversation(conn), '她的近况呢？')['blocked'] is True


def test_unrelated_user_topic_clears_focus_and_global_is_explicit(conn):
    person(conn, '林舟')
    cid = conversation(conn)
    turn(conn, cid, '查询林舟')
    assert turn(conn, cid, '解释一下归并排序')['mode'] == 'none'
    assert turn(conn, cid, '她的偏好呢？')['blocked'] is True
    global_scope = turn(conn, cid, '比较所有人的沟通偏好')
    assert global_scope['mode'] == 'global' and not global_scope['blocked']
    assert global_scope['person_ids'] == []
    assert turn(conn, cid, '她的偏好呢？')['blocked'] is True
    assert turn(conn, cid, '其他排序算法有哪些？')['mode'] == 'none'
    assert turn(conn, cid, '第一季度计划怎么安排？')['mode'] == 'none'


@pytest.mark.parametrize('reference', ['她呢？', '他喜欢什么？', '他们的偏好呢？', '两人谁更喜欢邮件？'])
def test_multiple_people_never_choose_by_pronoun_or_read_order(conn, reference):
    first, second = person(conn, '林舟'), person(conn, '陈岚')
    cid = conversation(conn)
    scope = turn(conn, cid, '比较陈岚与林舟的沟通偏好')
    assert scope['mode'] == 'multiple' and scope['person_ids'] == [second, first]
    repo.add_message(conn, cid, 'tool', json.dumps({'scoped_person': {'id': first}}))
    repo.add_message(conn, cid, 'assistant', '最后谈到林舟', json.dumps({'conversation_scope': scope}))
    conn.commit()
    unresolved = turn(conn, cid, reference)
    assert unresolved['blocked'] and unresolved['person_ids'] == []
    assert {candidate['person_id'] for candidate in unresolved['candidates']} == {first, second}


def test_namesake_candidates_background_selection_and_id_precedence(conn):
    first = person(conn, '同名林舟', city='杭州', company='甲公司')
    second = person(conn, '同名林舟', city='上海', company='乙公司')
    cid = conversation(conn)
    initial = turn(conn, cid, '同名林舟的沟通偏好是什么？')
    assert initial['mode'] == 'clarify' and initial['person_ids'] == []
    candidates = {candidate['person_id']: candidate for candidate in initial['candidates']}
    assert set(candidates) == {first, second}
    assert candidates[first]['display_name'] == '同名林舟'
    assert candidates[first]['summary'] == '杭州 · 甲公司'
    ordinal = turn(conn, cid, '第二个的偏好')
    assert ordinal['blocked'] and ordinal['reason'] == 'ordinal_selection'
    selected = turn(conn, cid, '杭州那个')
    assert selected['mode'] == 'single' and selected['person_ids'] == [first]
    assert turn(conn, cid, '她呢？')['person_ids'] == [first]
    assert turn(conn, cid, f'人物 ID: {second}。\n同名林舟的沟通偏好是什么？')['person_ids'] == [second]
    assert turn(conn, conversation(conn), '上海的同名林舟喜欢什么？')['person_ids'] == [second]


def test_ambiguous_background_or_nonselection_does_not_choose_candidate(conn):
    person(conn, '同名林舟', city='杭州', company='甲公司')
    person(conn, '同名林舟', city='杭州', company='乙公司')
    cid = conversation(conn)
    turn(conn, cid, '同名林舟的偏好？')
    assert turn(conn, cid, '杭州那个')['mode'] != 'single'
    turn(conn, cid, '同名林舟的偏好？')
    assert turn(conn, cid, '杭州天气很好')['mode'] == 'none'
    assert turn(conn, cid, '第二个')['blocked']


@pytest.mark.parametrize('selection', ['2', '二', '选2', '选择 2 号'])
@pytest.mark.parametrize('shared_name', [False, True])
def test_numeric_replies_do_not_guess_candidate_order_or_clear_pending_scope(conn, selection, shared_name):
    first = person(conn, '林舟')
    second = person(conn, '林舟' if shared_name else '陈岚')
    cid = conversation(conn)
    turn(conn, cid, '林舟的偏好？' if shared_name else '比较林舟与陈岚的偏好')
    scope = turn(conn, cid, selection)
    assert scope['blocked'] and scope['reason'] == 'ordinal_selection'
    assert {candidate['person_id'] for candidate in scope['candidates']} == {first, second}
    assert scope['person_ids'] == []


@pytest.mark.parametrize('text', ['换成不存在的王琪，她的沟通偏好？', '换个人，她的偏好呢？',
                                  '@王琪，她的偏好呢？'])
def test_unknown_explicit_subject_does_not_inherit_previous_person(conn, text):
    person(conn, '林舟')
    cid = conversation(conn)
    turn(conn, cid, '林舟的沟通偏好？')
    scope = turn(conn, cid, text)
    assert scope['blocked'] and scope['reason'] == 'unresolved_subject'
    assert scope['person_ids'] == []


def test_switch_marker_with_pronoun_can_keep_unambiguous_single_subject(conn):
    pid = person(conn, '林舟')
    cid = conversation(conn)
    turn(conn, cid, '林舟的偏好？')
    assert turn(conn, cid, '换成她的沟通偏好呢？')['person_ids'] == [pid]


@pytest.mark.parametrize('invalid_action', ['archive', 'merge'])
def test_inactive_previous_person_and_explicit_invalid_id_never_fall_back_to_all(conn, invalid_action):
    pid = person(conn, '林舟')
    cid = conversation(conn)
    turn(conn, cid, '林舟的偏好？')
    if invalid_action == 'archive':
        repo.archive_person(conn, pid)
    else:
        repo.set_person_status(conn, pid, 'merged')
    conn.commit()
    scope = turn(conn, cid, '她的偏好？')
    assert scope['mode'] == 'clarify' and scope['blocked']
    assert turn(conn, cid, f'人物 ID: {pid}')['blocked']
    assert turn(conn, cid, '人物 ID: does-not-exist')['blocked']


def test_failed_or_cancelled_user_turn_rebuilds_focus_without_assistant(conn):
    first, second = person(conn, '林舟'), person(conn, '陈岚')
    cid = conversation(conn)
    turn(conn, cid, '林舟的偏好？')
    failed = repo.add_message(conn, cid, 'user', '陈岚的偏好？')
    conn.commit()
    latest = turn(conn, cid, '她呢？')
    assert latest['person_ids'] == [second] and latest['person_ids'] != [first]
    assert latest['origin_message_id'] == failed


def test_scope_is_saved_on_user_before_generation_and_retries_ignore_later_turns(conn):
    first, second = person(conn, '林舟'), person(conn, '陈岚')
    cid = conversation(conn)
    initial = turn(conn, cid, '林舟的偏好？')
    row = conn.execute('SELECT tool_calls FROM message WHERE id=?', (initial['user_message_id'],)).fetchone()
    assert json.loads(row['tool_calls'])['conversation_scope'] == initial
    turn(conn, cid, '陈岚的偏好？')
    retry = conversation_scope.for_user_message(conn, cid, initial['user_message_id'], '林舟的偏好？')
    assert retry['person_ids'] == [first]
    assert turn(conn, cid, '她呢？')['person_ids'] == [second]
    with pytest.raises(ValueError):
        conversation_scope.for_user_message(conn, conversation(conn), initial['user_message_id'], '林舟的偏好？')


def test_all_explicit_people_are_retained_and_order_uses_text_not_creation(conn):
    ids = {name: person(conn, name) for name in ['林舟', '陈岚', '周敏', '赵宁']}
    scope = turn(conn, conversation(conn), '比较赵宁、周敏、陈岚与林舟')
    assert scope['person_ids'] == [ids[name] for name in ['赵宁', '周敏', '陈岚', '林舟']]


def model_stub(monkeypatch, chat):
    # Scope tests script answer/history calls only. Keep the real classification
    # persistence, tools and source validation while isolating the two new model
    # stages, which have their own unit/integration tests.
    def readonly_classifier(text, **kwargs):
        return harness.turn_intent.validate_intent({
            'intent': 'query', 'record_spans': [], 'reason': 'Scope integration reads existing records.'}, text)

    def unchanged_review(llm, question, draft_answer, sources, *, context=None):
        return {'answer': draft_answer, 'metadata': {
            'version': 1, 'status': 'test_double', 'semantic_verified': False,
            'original_answer': draft_answer, 'changed': False}, 'warnings': []}

    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: {'people': [], 'my_notes': []})
    monkeypatch.setattr(harness.turn_intent, 'classify_turn', readonly_classifier)
    monkeypatch.setattr(harness.answer_review, 'review_answer', unchanged_review)
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))


def test_harness_unresolved_reference_returns_deterministic_clarification_without_answer_model(conn, monkeypatch):
    pid = person(conn, '林舟')
    repo.add_memory(conn, 'episodic', 'unrelated private background', pid, .95, None)
    conn.commit()
    model_stub(monkeypatch, lambda *args, **kwargs: pytest.fail('unresolved reference must not ask model to guess'))
    result = harness.run_turn(conn, conversation(conn), '她的偏好呢？')
    assert result['conversation_scope']['blocked']
    assert result['sources'] == result['source_dependencies'] == []
    assert '哪一位' in result['answer']
    message = conn.execute('SELECT tool_calls FROM message WHERE id=?', (result['assistant_message_id'],)).fetchone()
    assert json.loads(message['tool_calls'])['conversation_scope'] == result['conversation_scope']


def test_harness_same_scope_reaches_every_read_and_outside_recent_person_is_not_injected(conn, monkeypatch):
    first, second, unrelated = [person(conn, name) for name in ['林舟', '陈岚', '赵宁']]
    repo.add_memory(conn, 'episodic', 'must not inject third-person private detail', unrelated, .95, None)
    conn.commit()
    calls, scopes = [], []

    def chat(messages, **kwargs):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': str(index), 'type': 'function', 'function': {'name': 'query_memory',
                 'arguments': json.dumps({'query': '偏好', 'person_id': pid})}}
                for index, pid in enumerate([second, first])]}
        return {'role': 'assistant', 'content': '暂无可核查的偏好。'}

    def execute(conn, name, args, **kwargs):
        scopes.append(deepcopy(kwargs['read_scope']))
        return {'ok': True, 'results': []}

    model_stub(monkeypatch, chat)
    monkeypatch.setattr(harness.tools, 'execute_tool', execute)
    cid = conversation(conn)
    result = harness.run_turn(conn, cid, '比较林舟和陈岚的沟通偏好')
    assert len(scopes) == 2 and scopes[0] == scopes[1] == result['conversation_scope']
    assert result['conversation_scope']['mode'] == 'multiple'
    assert result['conversation_scope']['person_ids'] == [first, second]
    assert all('must not inject' not in message['content'] for message in calls[0])
    assert turn(conn, cid, '她呢？')['blocked']


def test_harness_generation_failure_keeps_new_user_scope(conn, monkeypatch):
    person(conn, '林舟')
    second = person(conn, '陈岚')
    cid = conversation(conn)
    turn(conn, cid, '林舟的偏好？')
    model_stub(monkeypatch, lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('generation failed')))
    with pytest.raises(RuntimeError, match='generation failed'):
        harness.run_turn(conn, cid, '陈岚的偏好？')
    assert turn(conn, cid, '她呢？')['person_ids'] == [second]


@pytest.mark.parametrize('compressed_history', [False, True])
def test_harness_retry_uses_original_question_and_excludes_later_history_and_summary(conn, monkeypatch,
                                                                                   compressed_history):
    first = person(conn, '林舟')
    person(conn, '陈岚')
    cid = conversation(conn)
    if compressed_history:
        for index in range(15):
            repo.add_message(conn, cid, 'user' if index % 2 == 0 else 'assistant',
                             f'早期话题 {index}：' + '已有的会话内容' * 20)
        monkeypatch.setattr(harness, 'get_settings', lambda: SimpleNamespace(ctx_budget_chars=100))
    question = '林舟的沟通偏好？'
    original = turn(conn, cid, question)
    turn(conn, cid, '陈岚的新问题，later-user-marker')
    repo.add_message(conn, cid, 'assistant', 'later-assistant-marker，陈岚的后续回答。')
    repo.set_kv(conn, f'summ_text_{cid}', 'future-summary-marker，陈岚的后续摘要。')
    repo.set_kv(conn, f'summ_count_{cid}', '999')
    conn.commit()
    calls = []

    def chat(messages, **kwargs):
        calls.append(deepcopy(messages))
        if kwargs['tier'] == 'mini':
            return {'role': 'assistant', 'content': '仅来自重试问题以前的会话摘要。'}
        assert messages[-1] == {'role': 'user', 'content': question}
        return {'role': 'assistant', 'content': '没有可核查的偏好记录。'}

    model_stub(monkeypatch, chat)
    result = harness.run_turn(conn, cid, question, user_message_id=original['user_message_id'])
    assert result['conversation_scope']['person_ids'] == [first]
    assert result['conversation_scope']['user_message_id'] == original['user_message_id']
    assert calls
    assert all(marker not in str(calls) for marker in
               ['later-user-marker', 'later-assistant-marker', 'future-summary-marker'])
    assert repo.get_kv(conn, f'summ_text_{cid}') == 'future-summary-marker，陈岚的后续摘要。'
    if compressed_history:
        assert len(calls) == 2
        assert calls[0][0]['content'].startswith('把此前摘要')
