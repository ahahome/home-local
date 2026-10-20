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
    ('上一条回答的 S1 和 S2 是什么？', 'unknown_source_id'),
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


def memory_pair(conn):
    pid, first = memory(conn, text='项目事项写邮件。')
    _, second = memory(conn, text='周三午休方便接电话。', person_id=pid)
    return pid, first, second


@pytest.mark.parametrize('text', ['这些记录能说明工作发邮件、私事打电话吗？',
                                  '刚才两条有什么区别？', '这两条记录分别适用于什么情况？',
                                  '上一条回答中的 S2 和 S7 能支持这个结论吗？'])
def test_plural_focus_reloads_entire_cited_same_person_set_and_excludes_uncited_memory(conn, text):
    pid, first, second = memory_pair(conn)
    _, unused = memory(conn, '陈岚', '未引用的其他人物资料')
    cid = conversation(conn)
    old = [quote(conn, first, 'S7'), quote(conn, second, 'S2')]
    anchor = answer(conn, cid, old, [*old, quote(conn, unused, 'S8')])
    uid, focus = followup(conn, cid, text)
    assert focus['status'] == 'selected' and focus['reason'] == 'cited_memory_set'
    assert focus['source'] is None and focus['read_scope']['person_ids'] == [pid]
    assert [s['entity_id'] for s in focus['sources']] == [second, first]
    assert focus['binding']['reference_snapshot'] is None
    assert focus['binding']['reference_snapshots'] == list(reversed(old))
    assert focus['binding']['anchor_assistant_message_id'] == anchor
    assert metadata(conn, uid)['memory_focus'] == focus['binding']


@pytest.mark.parametrize('case,reason', [
    ('two_of_three', 'ambiguous_memory_set'), ('only_one', 'ambiguous_memory_set'),
    ('different_people', 'ambiguous_memory_owner'), ('mixed_field', 'ambiguous_memory_set'),
    ('uncited_only', 'ambiguous_memory_set'), ('unknown_id', 'unknown_source_id')])
def test_plural_focus_never_infers_unseen_subset_or_cross_person_set(conn, case, reason):
    pid, first, second = memory_pair(conn)
    _, other = memory(conn, '陈岚')
    _, third = memory(conn, person_id=pid)
    sources = [quote(conn, first), quote(conn, second, 'S2')]
    text = '这两条记录有什么差异？'
    dependencies = None
    if case == 'two_of_three':
        sources.append(quote(conn, third, 'S3'))
    elif case == 'only_one':
        sources = sources[:1]
    elif case == 'different_people':
        sources[1] = quote(conn, other, 'S2')
    elif case == 'mixed_field':
        sources.append({'id': 'S3', 'type': 'person_field', 'entity_id': 'field', 'text': '公司档案'})
    elif case == 'uncited_only':
        sources, dependencies = [], sources
    else:
        text = '上一条回答的 S1 和 S9 能对照吗？'
    cid = conversation(conn)
    answer(conn, cid, sources, dependencies)
    _, focus = followup(conn, cid, text)
    assert focus['status'] == 'clarify' and focus['reason'] == reason
    assert focus['sources'] == [] and focus['source'] is None and focus['read_scope']['blocked']


def test_explicit_ids_choose_complete_subset_and_generic_plural_accepts_three_records(conn):
    pid, first, second = memory_pair(conn)
    _, third = memory(conn, person_id=pid)
    cid = conversation(conn)
    sources = [quote(conn, first), quote(conn, second, 'S2'), quote(conn, third, 'S3')]
    answer(conn, cid, sources)
    _, pair = followup(conn, cid, '上一条回答的 S3 与 S1 有什么不同？')
    assert [s['entity_id'] for s in pair['sources']] == [first, third]
    _, entire = followup(conn, cid, '这些记忆分别讲了什么？')
    assert [s['entity_id'] for s in entire['sources']] == [first, second, third]


@pytest.mark.parametrize('action', ['withdraw', 'archive', 'move', 'delete_memory'])
def test_one_unavailable_member_blocks_whole_set_without_revealing_new_owner(conn, monkeypatch, action):
    pid, first, second = memory_pair(conn)
    new_owner = repo.create_person(conn, '不能沿用的新人物', status='active')
    cid = conversation(conn)
    old = [quote(conn, first), quote(conn, second, 'S2')]
    answer(conn, cid, old)
    uid, initial = followup(conn, cid, '这些记录支持什么结论？')
    if action == 'withdraw':
        change(conn, second, monkeypatch, action='withdraw')
    elif action == 'move':
        change(conn, second, monkeypatch, person_id=new_owner)
    elif action == 'archive':
        repo.archive_person(conn, pid)
    else:
        conn.execute('DELETE FROM memory_item WHERE id=?', (second,))
    conn.commit()
    _, blocked = followup(conn, cid, '这些记录支持什么结论？', uid=uid)
    assert blocked['status'] == 'clarify'
    assert blocked['reason'] == ('moved_memory' if action == 'move' else 'unavailable_memory')
    assert blocked['sources'] == [] and blocked['source'] is None
    assert blocked['binding'] == initial['binding']
    assert '不能沿用的新人物' not in json.dumps(blocked, ensure_ascii=False)


def test_retry_keeps_set_anchor_but_reads_edits_and_restore_again(conn, monkeypatch):
    _, first, second = memory_pair(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, first), quote(conn, second, 'S2')])
    text = '这些记录分别怎么说？'
    uid, initial = followup(conn, cid, text)
    change(conn, second, monkeypatch, text='仅每月首个周三午休接电话。')
    change(conn, first, monkeypatch, action='withdraw')
    _, blocked = followup(conn, cid, text, uid=uid)
    assert blocked['status'] == 'clarify'
    change(conn, first, monkeypatch, action='restore')
    _, restored = followup(conn, cid, text, uid=uid)
    assert restored['binding'] == initial['binding']
    assert restored['sources'][1]['text'] == '仅每月首个周三午休接电话。'
    assert restored['sources'][1]['revision'] == 1
    assert restored['sources'][0]['record_context']['last_lifecycle_change']['action'] == 'restore'


@pytest.mark.parametrize('tamper', ['drop', 'replace', 'mode', 'singular'])
def test_corrupted_set_binding_cannot_silently_change_members(conn, tamper):
    pid, first, second = memory_pair(conn)
    _, third = memory(conn, person_id=pid)
    cid = conversation(conn)
    sources = [quote(conn, first), quote(conn, second, 'S2'), quote(conn, third, 'S3')]
    answer(conn, cid, sources)
    text = '这些记录有什么区别？'
    uid, _ = followup(conn, cid, text)
    saved = metadata(conn, uid)
    binding = saved['memory_focus']
    if tamper == 'drop':
        binding['reference_snapshots'] = binding['reference_snapshots'][:2]
    elif tamper == 'replace':
        binding['reference_snapshots'][1] = deepcopy(sources[0])
    elif tamper == 'mode':
        binding['selection_mode'] = 'single'
    else:
        binding['reference_snapshot'] = deepcopy(sources[0])
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?', (json.dumps(saved), uid))
    conn.commit()
    _, focus = followup(conn, cid, text, uid=uid)
    assert focus['status'] == 'clarify' and focus['reason'] == 'invalid_binding'
    assert focus['sources'] == []


def test_legacy_single_binding_keeps_its_anchor_without_reclassification(conn):
    _, mid = memory(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, mid)])
    uid, _ = followup(conn, cid)
    saved = metadata(conn, uid)
    saved['memory_focus'].pop('reference_snapshots')
    saved['memory_focus'].pop('selection_mode')
    conn.execute('UPDATE message SET tool_calls=? WHERE id=?', (json.dumps(saved), uid))
    conn.commit()
    _, retried = followup(conn, cid, uid=uid)
    assert retried['status'] == 'selected' and retried['sources'][0]['entity_id'] == mid
    assert retried['binding'] == saved['memory_focus']


@pytest.mark.parametrize('text', ['记住：这条是我每天回家的路线。', '请记住：这些记录是我个人的出行安排。',
                                  '这条是我每天回家的路线。', '那条公路是什么时候修好的？'])
def test_new_shorthand_does_not_turn_an_unrelated_note_into_a_blocking_memory_reference(conn, text):
    cid = conversation(conn)
    _, focus = followup(conn, cid, text)
    assert focus['status'] == 'not_applicable' and not focus['read_scope']['blocked']


def test_ordinary_singular_shorthand_after_two_records_requires_selection(conn):
    _, first, second = memory_pair(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, first), quote(conn, second, 'S2')])
    _, focus = followup(conn, cid, '这条后来是什么时候改的？')
    assert focus['status'] == 'clarify' and focus['reason'] == 'ambiguous_memory'
    assert focus['sources'] == []


def test_set_registration_checks_combined_budget_before_any_registry_mutation(conn, monkeypatch):
    _, first, second = memory_pair(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, first, 'S7'), quote(conn, second, 'S8')])
    _, focus = followup(conn, cid, '这些记录有什么不同？')
    registry = SourceRegistry(conn)
    _, extra = memory(conn, '陈岚')
    registry.register(memory_source(conn, extra))
    before = deepcopy(registry.dependencies)
    full = memory_focus.register_source(focus, SourceRegistry(conn), 12000)
    insufficient = memory_focus.register_source(focus, registry, len(full['context']) - 1)
    assert insufficient['status'] == 'budget_exceeded' and insufficient['source_ids'] == []
    assert registry.dependencies == before
    registered = memory_focus.register_source(focus, registry, 12000)
    assert registered['source_id'] is None and registered['source_ids'] == ['S2', 'S3']
    assert [json.loads(line)['id'] for line in registered['context'].split('\n\n')] == ['S2', 'S3']
    assert len(registry.dependencies) == 3
    change(conn, second, monkeypatch, text='生成期间修改第二条')
    with pytest.raises(SourceChangedError):
        registry.save_answer(lambda: pytest.fail('entire set must be rechecked before save'))


@pytest.mark.parametrize('text', ['这条记录最初是什么时候录入的？', '这些记录分别是什么时候修改的？',
                                  '这两条记录的录入时间和修改时间分别是哪天？',
                                  '这条是什么时候记下的？', '这条后来改过吗，恢复又是在什么时候？',
                                  '这两条是什么时候记下的，后来又是哪天改过？'])
def test_explicit_date_roles_support_set_and_recorded_date_without_swallowing_writes(text):
    assert memory_focus.is_date_question(text)
    for suffix in ['，再记住陈岚住北京。', '；把这两条改成今天', '然后恢复这条记录']:
        assert not memory_focus.is_date_question(text + suffix)


def test_set_date_template_keeps_each_sources_date_roles_and_citations_separate(conn, monkeypatch):
    _, first, second = memory_pair(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, first), quote(conn, second, 'S2')])
    change(conn, second, monkeypatch, day='2026-09-18', text='第二条修改后的文字')
    text = '这两条记录的录入时间和修改时间分别是哪天？'
    _, focus = followup(conn, cid, text)
    registry = SourceRegistry(conn)
    registered = memory_focus.register_source(focus, registry, 12000)
    result = memory_focus.date_answer(focus, registered['source_ids'], text)
    assert '第 1 条记忆' in result and '第 2 条记忆' in result
    assert '没有可核验的人工编辑时间' in result and '实际人工编辑日期是 2026-09-18' in result
    assert len(registry.citations(result)[1]) == 2
    assert '2026-09-20' not in result


@pytest.mark.parametrize('text', [
    '这些记录有什么区别？', '刚才两条有什么不同？',
    '这些记录说的是同一种联系安排吗？哪些地方对得上，哪些地方还不能确定？',
    '刚才两条现在再对照一下，原文和能支持的结论有没有变化？',
])
def test_complete_readonly_collection_comparisons_are_recognized(text):
    assert memory_focus.is_comparison_question(text)


@pytest.mark.parametrize('text', [
    '根据这两条记录帮我写一封邮件。',
    '对照这些记录，帮我给他起草一条短信。',
    '这些记录有什么区别？顺便记住林舟住杭州。',
    '刚才两条有什么不同？然后恢复第一条。',
    '这些记录说的是同一种联系安排吗？再把电话那条改成今天。',
    '这些记录有什么区别，给我推荐明天怎么联系他。',
    '这些记录里的饭店电话是多少？',
    '这条记录有什么不同？',
    '', None,
])
def test_collection_comparison_recognition_never_captures_drafting_writes_or_unknown_requests(text):
    assert not memory_focus.is_comparison_question(text)


def comparison_focus(conn):
    pid, first, second = memory_pair(conn)
    cid = conversation(conn)
    answer(conn, cid, [quote(conn, first, 'S7'), quote(conn, second, 'S8')])
    text = '这些记录有什么区别？'
    uid, focus = followup(conn, cid, text)
    return pid, first, second, cid, uid, focus, text


def render_comparison(conn, focus, text, intent=None):
    registry = SourceRegistry(conn)
    registered = memory_focus.register_source(focus, registry, 12000)
    result = memory_focus.comparison_answer(focus, registered['source_ids'], text,
                                          {'intent': 'query', 'allow_writes': False} if intent is None else intent)
    return result, registry


def test_collection_comparison_outputs_current_text_date_roles_and_only_current_citations(conn):
    _, first, second, _, _, focus, text = comparison_focus(conn)
    result, registry = render_comparison(conn, focus, text)
    assert isinstance(result, str) and result
    assert all(source['text'] in result for source in focus['sources'])
    assert result.count('保存文字未变化') == 2 and '保存文字已变化' not in result
    assert '2026-09-11' in result and '2026-09-12' in result
    assert '所标' in result and '录入' in result and '人工编辑' in result
    _, citations, warnings = registry.citations(result)
    assert [source['id'] for source in citations] == ['S1', 'S2']
    assert {source['entity_id'] for source in citations} == {first, second}
    assert not any(warning['code'] == 'invalid_citation' for warning in warnings)
    assert 'S7' not in result and 'S8' not in result
    assert '工作用邮件、私事打电话' not in result


def test_collection_comparison_compares_current_text_to_each_old_identity_without_rebinding(conn, monkeypatch):
    _, first, second, cid, uid, old_focus, text = comparison_focus(conn)
    old_binding = deepcopy(old_focus['binding'])
    change(conn, second, monkeypatch, text='仅月初周三午休接电话，先发短信。')
    _, current = followup(conn, cid, text, uid=uid)
    # The formatter must align entities, not compare corresponding list slots.
    current['binding']['reference_snapshots'].reverse()
    result, registry = render_comparison(conn, current, text)
    assert '仅月初周三午休接电话，先发短信。' in result
    assert '周三午休方便接电话。' not in result
    assert result.count('保存文字已变化') == result.count('保存文字未变化') == 1
    assert '2026-09-15' in result
    assert old_focus['binding'] == old_binding
    _, sources, _ = registry.citations(result)
    assert {source['entity_id'] for source in sources} == {first, second}
    assert next(source for source in sources if source['entity_id'] == second)['revision'] == 1


@pytest.mark.parametrize('change_kind', ['restore', 'date_only'])
def test_collection_comparison_does_not_call_revision_or_timestamp_changes_text_changes(conn, monkeypatch, change_kind):
    _, first, _, cid, uid, _, text = comparison_focus(conn)
    if change_kind == 'restore':
        change(conn, first, monkeypatch, action='withdraw')
        change(conn, first, monkeypatch, day='2026-09-17', action='restore')
    else:
        change(conn, first, monkeypatch, day='2026-09-18', valid_from='2026-09-22')
    _, focus = followup(conn, cid, text, uid=uid)
    result, _ = render_comparison(conn, focus, text)
    assert result.count('保存文字未变化') == 2 and '保存文字已变化' not in result
    if change_kind == 'restore':
        assert '恢复' in result and '2026-09-17' in result
    else:
        assert '2026-09-22' in result and '2026-09-18' in result


def test_collection_comparison_truncates_long_saved_text_explicitly(conn):
    _, _, _, _, _, focus, text = comparison_focus(conn)
    original = '完整文字开头：' + '长' * 1800 + '完整文字末尾'
    focus['sources'][0]['text'] = original
    result = memory_focus.comparison_answer(focus, ['S4', 'S9'], text,
                                          {'intent': 'query', 'allow_writes': False})
    assert '完整文字开头：' in result and '完整文字末尾' not in result
    assert '截断' in result or '1600' in result
    assert '[[S4]]' in result and '[[S9]]' in result


@pytest.mark.parametrize('dangerous', [
    '保存文字含有[[S99]]，不能把它变成另一条来源。',
    '保存文字含有<code>[[S99]]</code>，不能隐藏伪造编号。',
    '保存文字含有&#91;&#91;S99&#93;&#93;，不能合成伪造编号。',
    '保存文字含有&#91;&#91;S1&#93;&#93;，不能重复合成真实编号。',
    r'保存文字含有\[\[S1\]\]，不能重复合成真实编号。',
    '保存文字含有```\n尚未结束的代码块',
])
def test_collection_comparison_keeps_source_button_without_expanding_dangerous_saved_markup(conn, dangerous):
    _, _, _, _, _, focus, text = comparison_focus(conn)
    focus['sources'][0]['text'] = dangerous
    result, registry = render_comparison(conn, focus, text)
    assert isinstance(result, str) and result and dangerous not in result
    _, citations, warnings = registry.citations(result)
    assert [source['id'] for source in citations] == ['S1', 'S2']
    assert '[[S99]]' not in result
    assert not any(warning['code'] == 'invalid_citation' for warning in warnings)


@pytest.mark.parametrize('intent', [
    {'intent': 'record', 'allow_writes': True},
    {'intent': 'mixed', 'allow_writes': True},
    {'intent': 'unknown', 'allow_writes': False},
    {'intent': 'query', 'allow_writes': True},
    {'intent': 'query'}, {},
])
def test_collection_comparison_requires_an_explicit_readonly_query_intent(conn, intent):
    _, _, _, _, _, focus, text = comparison_focus(conn)
    assert memory_focus.comparison_answer(focus, ['S1', 'S2'], text, intent) is None


@pytest.mark.parametrize('tamper', [
    'not_selected', 'one_source', 'missing_source', 'duplicate_source', 'different_owner',
    'missing_binding', 'missing_anchor_member', 'wrong_anchor_identity', 'duplicate_anchor',
    'wrong_anchor_owner', 'missing_source_ids', 'duplicate_source_ids', 'invalid_source_id',
])
def test_collection_comparison_rejects_partial_mismatched_or_corrupted_inputs(conn, tamper):
    _, _, _, _, _, focus, text = comparison_focus(conn)
    source_ids = ['S1', 'S2']
    if tamper == 'not_selected':
        focus['status'] = 'clarify'
    elif tamper == 'one_source':
        focus['sources'] = focus['sources'][:1]
    elif tamper == 'missing_source':
        focus['sources'][1] = None
    elif tamper == 'duplicate_source':
        focus['sources'][1] = deepcopy(focus['sources'][0])
    elif tamper == 'different_owner':
        focus['sources'][1]['person'] = {'id': 'new-owner', 'display_name': '不应展示的人物'}
    elif tamper == 'missing_binding':
        focus['binding'] = None
    elif tamper == 'missing_anchor_member':
        focus['binding']['reference_snapshots'] = focus['binding']['reference_snapshots'][:1]
    elif tamper == 'wrong_anchor_identity':
        focus['binding']['reference_snapshots'][1]['entity_id'] = 'different-memory'
    elif tamper == 'duplicate_anchor':
        focus['binding']['reference_snapshots'][1] = deepcopy(focus['binding']['reference_snapshots'][0])
    elif tamper == 'wrong_anchor_owner':
        focus['binding']['reference_snapshots'][1]['person']['id'] = 'old-owner'
    elif tamper == 'missing_source_ids':
        source_ids = ['S1']
    elif tamper == 'duplicate_source_ids':
        source_ids = ['S1', 'S1']
    else:
        source_ids = ['S1', 'S99 invalid']
    assert memory_focus.comparison_answer(focus, source_ids, text,
                                         {'intent': 'query', 'allow_writes': False}) is None


@pytest.mark.parametrize('text', [
    '根据这两条记录给他起草邮件。',
    '这些记录有什么区别？记住林舟住杭州。',
    '这些记录的饭店电话是多少？',
])
def test_collection_comparison_formatter_does_not_handle_unrecognized_query_even_with_valid_sources(conn, text):
    _, _, _, _, _, focus, _ = comparison_focus(conn)
    assert memory_focus.comparison_answer(focus, ['S1', 'S2'], text,
                                         {'intent': 'query', 'allow_writes': False}) is None
