"""A follow-up selects persisted quotations, never a tool's last returned row."""
from copy import deepcopy
from datetime import datetime
import json

import pytest

from app.agent import conversation_scope, memory_focus
from app.agent.memory_sources import memory_source
from app.agent.source_registry import SourceChangedError, SourceRegistry
from app.config import tz
from app.memory import memory_corrections
from app.store import repo
from app.util import new_id


QUESTION = '那9月11日是这条记忆所标的日期，还是我实际修改它的时间？'


def stamp(day):
    return int(datetime.fromisoformat(day + 'T12:00:00').replace(tzinfo=tz()).timestamp() * 1000)


def memory(conn, name='林舟', text='偏好短邮件', person_id=False):
    pid = repo.create_person(conn, name, status='active') if person_id is False else person_id
    mid = repo.add_memory(conn, 'semantic', text, pid, .9, None, ts=stamp('2026-09-11'))
    conn.execute('UPDATE memory_item SET created_at=?,updated_at=? WHERE id=?',
                 (stamp('2026-09-12'), stamp('2026-09-20'), mid))
    conn.commit()
    return pid, mid


def conversation(conn):
    cid = repo.create_conversation(conn, 'Memory focus test')
    conn.commit()
    return cid


def quote(conn, mid, sid='S1'):
    return {**memory_source(conn, mid), 'id': sid}


def answer(conn, cid, sources=(), dependencies=None):
    metadata = {'sources': list(sources),
                'source_dependencies': list(sources if dependencies is None else dependencies)}
    aid = repo.add_message(conn, cid, 'assistant', '可核查的回答', json.dumps(metadata, ensure_ascii=False))
    conn.commit()
    return aid


def followup(conn, cid, text=QUESTION, uid=None):
    if uid is None:
        uid = repo.add_message(conn, cid, 'user', text)
        conn.commit()
    scope = conversation_scope.for_user_message(conn, cid, uid, text)
    return uid, memory_focus.for_user_message(conn, cid, uid, text, scope)


def metadata(conn, uid):
    return json.loads(conn.execute('SELECT tool_calls FROM message WHERE id=?', (uid,)).fetchone()[0])


def change(conn, mid, monkeypatch, day='2026-09-15', action='edit', **edits):
    monkeypatch.setattr(memory_corrections, 'now_ms', lambda: stamp(day))
    row = conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()
    conn.commit()
    return memory_corrections.correct_memory(conn, mid, {
        'action': action, 'expected_revision': row['revision'], 'request_id': new_id(), **edits})


def test_cited_memory_wins_over_uncited_dependencies_and_scope_uses_its_owner(conn):
    pid, mid = memory(conn)
    _, other = memory(conn, '陈岚', '偏好电话')
    cid = conversation(conn)
    selected, ignored = quote(conn, mid, 'S2'), quote(conn, other)
    aid = answer(conn, cid, [selected], [ignored, selected])
    uid, focus = followup(conn, cid)
    assert focus['status'] == 'selected' and focus['source']['entity_id'] == mid
    assert focus['source']['person']['id'] == pid
    assert focus['binding']['anchor_assistant_message_id'] == aid
    assert focus['read_scope']['person_ids'] == [pid]
    assert metadata(conn, uid)['memory_focus'] == focus['binding']
    assert metadata(conn, uid)['conversation_scope'] == focus['read_scope']


def test_only_unique_memory_dependency_can_supply_an_uncited_anchor(conn):
    _, mid = memory(conn)
    cid = conversation(conn)
    source = quote(conn, mid)
    answer(conn, cid, [], [{'type': 'profile_context', 'entity_id': 'other', 'text': 'card'}, source])
    _, focus = followup(conn, cid, '这个偏好是什么时候编辑的？')
    assert focus['status'] == 'selected' and focus['source']['entity_id'] == mid
    assert focus['reason'] == 'unique_memory_dependency'


@pytest.mark.parametrize('reverse', [False, True])
def test_multiple_memory_sources_clarify_independently_of_tool_order(conn, reverse):
    _, first = memory(conn)
    _, second = memory(conn, '陈岚')
    cid = conversation(conn)
    sources = [quote(conn, first), quote(conn, second, 'S2')]
    answer(conn, cid, [], list(reversed(sources)) if reverse else sources)
    _, focus = followup(conn, cid)
    assert focus['status'] == 'clarify' and focus['reason'] == 'ambiguous_memory'
    assert focus['source'] is None and focus['read_scope']['blocked']


def test_duplicate_labels_of_same_memory_deduplicate_but_conflicting_owners_do_not(conn):
    pid, mid = memory(conn)
    other = repo.create_person(conn, '陈岚', status='active')
    cid = conversation(conn)
    source = quote(conn, mid)
    duplicate = {**deepcopy(source), 'id': 'S2'}
    answer(conn, cid, [duplicate, source])
    _, focus = followup(conn, cid)
    assert focus['status'] == 'selected' and focus['source']['person']['id'] == pid
    conflict = {**deepcopy(source), 'person': {'id': other, 'display_name': '陈岚'}}
    answer(conn, cid, [duplicate, conflict])
    _, focus = followup(conn, cid)
    assert focus['status'] == 'clarify' and focus['reason'] == 'ambiguous_memory'


def test_latest_final_answer_without_source_does_not_fall_back_to_earlier_quote(conn):
    _, mid = memory(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    last = answer(conn, cid)
    _, focus = followup(conn, cid)
    assert focus['status'] == 'clarify' and focus['reason'] == 'missing_memory'
    assert focus['binding']['anchor_assistant_message_id'] == last


def test_intermediate_tool_call_assistant_is_skipped_but_later_turns_do_not_steer_retry(conn):
    _, first = memory(conn)
    _, second = memory(conn, '陈岚')
    cid = conversation(conn)
    anchor = answer(conn, cid, [quote(conn, first)])
    repo.add_message(conn, cid, 'assistant', '', json.dumps([{'id': 'call1'}]))
    conn.commit()
    uid, focus = followup(conn, cid)
    assert focus['binding']['anchor_assistant_message_id'] == anchor
    repo.add_message(conn, cid, 'user', '陈岚的情况呢？')
    answer(conn, cid, [quote(conn, second)])
    _, retried = followup(conn, cid, uid=uid)
    assert retried['binding'] == focus['binding']
    assert retried['source']['entity_id'] == first


def test_other_conversations_and_future_answers_never_supply_an_anchor(conn):
    _, mid = memory(conn)
    cid, other_cid = conversation(conn), conversation(conn)
    answer(conn, other_cid, [quote(conn, mid)])
    uid = repo.add_message(conn, cid, 'user', QUESTION)
    answer(conn, cid, [quote(conn, mid)])
    _, focus = followup(conn, cid, uid=uid)
    assert focus['status'] == 'clarify' and focus['reason'] == 'missing_memory'
    assert focus['binding']['anchor_assistant_message_id'] is None


@pytest.mark.parametrize('tamper', ['conversation', 'user', 'hash', 'anchor', 'snapshot', 'status', 'reason'])
def test_corrupt_saved_binding_cannot_reselect_a_memory(conn, tamper):
    _, mid = memory(conn)
    cid, other_cid = conversation(conn), conversation(conn)
    foreign = answer(conn, other_cid, [quote(conn, mid)])
    answer(conn, cid, [quote(conn, mid)])
    uid, _ = followup(conn, cid)
    saved = metadata(conn, uid)
    focus = saved['memory_focus']
    if tamper == 'conversation':
        focus['conversation_id'] = other_cid
    elif tamper == 'user':
        focus['user_message_id'] = 'another-user'
    elif tamper == 'hash':
        focus['source_sha256'] = 'wrong'
    elif tamper == 'anchor':
        focus['anchor_assistant_message_id'] = foreign
    elif tamper == 'status':
        focus.pop('status')
    elif tamper == 'reason':
        focus['reason'] = None
    else:
        focus['reference_snapshot']['text'] = 'fabricated quotation'
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?', (json.dumps(saved), uid))
    conn.commit()
    _, retried = followup(conn, cid, uid=uid)
    assert retried['status'] == 'clarify' and retried['reason'] == 'invalid_binding'
    assert retried['source'] is None


def test_binding_to_older_same_conversation_answer_is_rejected(conn):
    _, first = memory(conn)
    _, second = memory(conn, '陈岚')
    cid = conversation(conn)
    older_source = quote(conn, first)
    older = answer(conn, cid, [older_source])
    answer(conn, cid, [quote(conn, second)])
    uid, _ = followup(conn, cid)
    saved = metadata(conn, uid)
    saved['memory_focus'].update(anchor_assistant_message_id=older, reference_snapshot=older_source)
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?', (json.dumps(saved), uid))
    conn.commit()
    _, retried = followup(conn, cid, uid=uid)
    assert retried['status'] == 'clarify' and retried['reason'] == 'invalid_binding'


def test_message_id_conversation_and_content_are_validated(conn):
    cid = conversation(conn)
    uid, _ = followup(conn, cid)
    for supplied_cid, supplied_text in [(conversation(conn), QUESTION), (cid, '修改后的问题')]:
        with pytest.raises(ValueError):
            memory_focus.for_user_message(conn, supplied_cid, uid, supplied_text, {})


def test_retry_reloads_latest_text_and_true_manual_edit_audit_without_rebinding(conn, monkeypatch):
    _, mid = memory(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    uid, initial = followup(conn, cid)
    change(conn, mid, monkeypatch, text='偏好简短邮件，不要电话')
    _, retried = followup(conn, cid, uid=uid)
    assert retried['binding'] == initial['binding']
    assert retried['source']['text'] == '偏好简短邮件，不要电话'
    assert retried['source']['record_context']['last_user_edit_at'] == stamp('2026-09-15')


@pytest.mark.parametrize('action', ['withdraw', 'archive', 'purge', 'move', 'delete_memory'])
def test_unavailable_or_moved_memory_never_falls_back_or_exposes_new_owner(conn, monkeypatch, action):
    pid, mid = memory(conn)
    _, alternative = memory(conn, '陈岚', '另一条可用记忆')
    new_owner = repo.create_person(conn, '不可泄露的新人物', status='active')
    cid = conversation(conn)
    original, other = quote(conn, mid), quote(conn, alternative, 'S2')
    answer(conn, cid, [original], [other, original])
    uid, _ = followup(conn, cid)
    if action == 'withdraw':
        change(conn, mid, monkeypatch, action='withdraw')
    elif action == 'archive':
        repo.archive_person(conn, pid)
    elif action == 'purge':
        repo.purge_person(conn, pid)
    elif action == 'move':
        change(conn, mid, monkeypatch, person_id=new_owner)
    else:
        conn.execute('DELETE FROM memory_item WHERE id=?', (mid,))
    conn.commit()
    _, focus = followup(conn, cid, uid=uid)
    assert focus['status'] == 'clarify' and focus['source'] is None
    assert focus['reason'] == ('moved_memory' if action == 'move' else 'unavailable_memory')
    assert focus['binding']['reference_snapshot'] == original
    assert '不可泄露的新人物' not in json.dumps(focus, ensure_ascii=False)


def test_retry_after_restore_checks_availability_again_and_preserves_real_edit_time(conn, monkeypatch):
    _, mid = memory(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    uid, _ = followup(conn, cid)
    change(conn, mid, monkeypatch, text='改成邮件')
    change(conn, mid, monkeypatch, day='2026-09-16', action='withdraw')
    _, blocked = followup(conn, cid, uid=uid)
    assert blocked['status'] == 'clarify'
    change(conn, mid, monkeypatch, day='2026-09-17', action='restore')
    _, restored = followup(conn, cid, uid=uid)
    assert restored['status'] == 'selected'
    assert restored['source']['record_context']['last_user_edit_at'] == stamp('2026-09-15')
    assert restored['source']['record_context']['last_lifecycle_change']['created_at'] == stamp('2026-09-17')


def test_same_name_uses_quoted_identity_and_personal_memory_has_no_person_scope(conn):
    first, mid = memory(conn, '同名林舟')
    memory(conn, '同名林舟')
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    _, focus = followup(conn, cid)
    assert focus['status'] == 'selected' and focus['read_scope']['person_ids'] == [first]
    _, personal = memory(conn, person_id=None)
    answer(conn, cid, [quote(conn, personal)])
    _, focus = followup(conn, cid)
    assert focus['status'] == 'selected' and focus['read_scope']['mode'] == 'none'


@pytest.mark.parametrize('multiple', [False, True])
@pytest.mark.parametrize('selection', ['换成陈岚，这条记忆什么时候编辑的？', '改为全库比较林舟的这个偏好'])
def test_new_person_and_global_scope_take_precedence_even_with_ambiguous_memories(conn, multiple, selection):
    _, mid = memory(conn)
    new_pid, other = memory(conn, '陈岚')
    cid = conversation(conn)
    sources = [quote(conn, mid)] + ([quote(conn, other, 'S2')] if multiple else [])
    answer(conn, cid, sources)
    _, focus = followup(conn, cid, selection)
    assert focus['status'] == 'not_applicable' and focus['reason'] == 'explicit_scope_precedence'
    if '全库' in selection:
        assert focus['read_scope']['mode'] == 'global'
    else:
        assert focus['read_scope']['person_ids'] == [new_pid]


def test_clarification_retry_persists_blocked_scope_again(conn):
    cid = conversation(conn)
    uid, initial = followup(conn, cid)
    assert initial['status'] == 'clarify'
    _, retried = followup(conn, cid, uid=uid)
    assert retried['read_scope']['blocked']
    assert metadata(conn, uid)['conversation_scope'] == retried['read_scope']


@pytest.mark.parametrize('text', ['今天天气如何？', '她最近怎么样？', '9月11日是什么节日？'])
def test_ordinary_questions_do_not_use_memory_reference_resolution(conn, text):
    cid = conversation(conn)
    _, focus = followup(conn, cid, text)
    assert focus['status'] == 'not_applicable'


@pytest.mark.parametrize('text,reason', [
    ('这条记忆 S1 什么时候编辑的？', 'unanchored_source_id'),
    ('上一条回答的 S2 什么时候编辑的？', 'unknown_source_id'),
    ('上一条回答的 S1 和 S2 是什么？', 'ambiguous_memory'),
])
def test_source_ids_require_one_cited_label_in_explicit_previous_answer(conn, text, reason):
    _, mid = memory(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)], [quote(conn, mid), quote(conn, mid, 'S2')])
    _, focus = followup(conn, cid, text)
    assert focus['status'] == 'clarify' and focus['reason'] == reason


def test_explicit_previous_answer_label_selects_one_of_multiple_memories(conn):
    _, first = memory(conn)
    _, second = memory(conn, '陈岚')
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, first), quote(conn, second, 'S2')])
    _, focus = followup(conn, cid, '上一条回答中的 s2 什么时候编辑的？')
    assert focus['status'] == 'selected' and focus['source']['entity_id'] == second


def test_budget_is_checked_before_registration_and_final_save_rechecks_changes(conn, monkeypatch):
    _, mid = memory(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    _, focus = followup(conn, cid)
    registry = SourceRegistry(conn)
    insufficient = memory_focus.register_source(focus, registry, 1)
    assert insufficient['status'] == 'budget_exceeded'
    assert registry.sources == {} and registry.dependencies == []
    registered = memory_focus.register_source(focus, registry, 12000)
    assert registered['source_id'] == 'S1' and '所标日期' in registered['context']
    assert registry.dependencies[0]['entity_id'] == mid
    change(conn, mid, monkeypatch, text='生成期间发生修改')
    with pytest.raises(SourceChangedError):
        registry.save_answer(lambda: pytest.fail('changed source must not save'))


def test_date_answer_uses_labeled_recorded_edit_and_restore_dates_separately(conn, monkeypatch):
    _, mid = memory(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    uid, _ = followup(conn, cid)
    change(conn, mid, monkeypatch, text='编辑后的邮件偏好')
    change(conn, mid, monkeypatch, day='2026-09-16', action='withdraw')
    change(conn, mid, monkeypatch, day='2026-09-17', action='restore')
    _, focus = followup(conn, cid, uid=uid)
    registry = SourceRegistry(conn)
    registered = memory_focus.register_source(focus, registry, 12000)
    result = memory_focus.date_answer(focus, registered['source_id'], QUESTION)
    assert '所标的日期是 2026-09-11' in result
    assert '首次录入日期是 2026-09-12' in result
    assert '实际人工编辑日期是 2026-09-15' in result
    assert '恢复操作日期是 2026-09-17' in result
    _, sources, warnings = registry.citations(result, require_citations=True)
    assert len(sources) == 1 and not any(w['code'] == 'uncited' for w in warnings)


def test_unknown_edit_date_is_not_inferred_from_updated_at_or_revision(conn):
    _, mid = memory(conn)
    conn.execute('UPDATE memory_item SET revision=8 WHERE id=?', (mid,))
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    _, focus = followup(conn, cid)
    result = memory_focus.date_answer(focus, 'S1', QUESTION)
    assert '没有可核验的人工编辑时间' in result
    assert '2026-09-20' not in result


@pytest.mark.parametrize('text', [QUESTION, '这条记忆的人工编辑时间？', '刚才那条记录什么时候修改的？',
                                  '这条记忆是什么时候被我修改的？', '这条记忆是在什么时候被我修改的？',
                                  '这个偏好的实际人工编辑时间是在什么时候？'])
def test_date_readonly_detection_accepts_only_complete_questions(text):
    assert memory_focus.is_date_question(text)
    for extra in ['；再记录她喜欢电话', '然后修改成9月12日', '，并告诉我她现在住哪里']:
        assert not memory_focus.is_date_question(text + extra)
    assert not memory_focus.is_date_question('这个偏好现在还适用吗？')
    assert memory_focus.date_answer({'status': 'selected', 'source': {}}, 'S1', '这个偏好现在还适用吗？') is None
