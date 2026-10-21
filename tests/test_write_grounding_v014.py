"""Writes bind each actual mention and keep identity labels out of facts."""
import json

import pytest

from app.agent import harness, tools, turn_intent
from app.memory import apply, extract
from app.memory.write_grounding import freeze_write_aliases
from app.store import repo


def person(name='周汐', **items):
    return {'name': name, 'canonical_name': name, 'fields': [], 'aliases': [],
            'facts': [], 'relations': [], 'promises': [], 'ledger': [], 'value_read': None, **items}


def field(key, value, proof):
    return {'field': key, 'value': value, 'confidence': .96, 'evidence': proof}


def scope(*ids):
    return {'version': 1, 'mode': 'single' if len(ids) == 1 else 'multiple',
            'person_ids': list(ids), 'blocked': False}


@pytest.fixture
def namesakes(conn):
    a, b = (repo.create_person(conn, '周汐', status='active') for _ in range(2))
    repo.add_alias(conn, a, '陶艺周汐', 'test')
    repo.add_alias(conn, b, '乐团周汐', 'test')
    conn.commit()
    return a, b


def write(conn, source, people, *, read_scope=None, source_id='test-v014', forced=None):
    patch = extract.sanitize({'people': people, 'my_notes': []}, source=source)
    return apply.apply_extraction(conn, patch, source_id, source, read_scope=read_scope, forced=forced)


def test_shared_canonical_keeps_the_original_selected_alias_and_does_not_repeat_identity_pending(conn, namesakes):
    a, b = namesakes
    source = '帮我记下：乐团周汐现在住在长沙。'
    p = person('周汐', fields=[field('city', '长沙', source)])
    result = write(conn, source, [p], read_scope=scope(b))
    assert repo.current_fields(conn, b)['city']['value'] == '长沙'
    assert repo.current_fields(conn, a) == {}
    assert not result['pending_ids']
    assert result['write_audit'][0]['person_id'] == b
    repeated = write(conn, source, [p], read_scope=scope(b))
    assert any(o['status'] == 'duplicate' for o in repeated['outcomes'])
    assert conn.execute('SELECT COUNT(*) FROM person_field').fetchone()[0] == 1


def test_source_alias_overrules_model_invented_canonical(conn, namesakes):
    a, b = namesakes
    source = '乐团周汐现在住在长沙。'
    result = write(conn, source, [person('乐团周汐', canonical_name='虚构规范名',
        fields=[field('city', '长沙', source)])], read_scope=scope(b))
    assert result['person_ids'] == [b]
    assert len(repo.all_active_persons(conn)) == 2
    assert '虚构规范名' not in repo.aliases_of(conn, b)
    assert repo.current_fields(conn, a) == {}


def test_two_namesakes_in_one_batch_bind_each_patch_without_canonical_dictionary_collision(conn, namesakes):
    a, b = namesakes
    source = '陶艺周汐搬到苏州，乐团周汐搬到长沙。'
    people = [person('陶艺周汐', canonical_name='周汐', fields=[field('city', '苏州', '陶艺周汐搬到苏州')]),
              person('乐团周汐', canonical_name='周汐', fields=[field('city', '长沙', '乐团周汐搬到长沙')])]
    write(conn, source, people, read_scope=scope(a, b))
    assert repo.current_fields(conn, a)['city']['value'] == '苏州'
    assert repo.current_fields(conn, b)['city']['value'] == '长沙'


def test_unknown_person_is_not_forced_into_the_selected_known_person(conn, namesakes):
    a, b = namesakes
    source = '陶艺周汐搬到苏州，何晴搬到南京。'
    write(conn, source, [person('陶艺周汐', fields=[field('city', '苏州', '陶艺周汐搬到苏州')]),
                         person('何晴', fields=[field('city', '南京', '何晴搬到南京')])], read_scope=scope(a))
    other = repo.get_person_by_name(conn, '何晴')
    assert other and other['id'] not in namesakes
    assert repo.current_fields(conn, other['id'])['city']['value'] == '南京'
    assert repo.current_fields(conn, a)['city']['value'] == '苏州'
    assert repo.current_fields(conn, b) == {}


def test_pronoun_is_only_a_locator_and_never_a_registered_alias(conn, namesakes):
    a, b = namesakes
    source = '帮我记下她现在住在长沙。'
    write(conn, source, [person('她', fields=[field('city', '长沙', source)])], read_scope=scope(b))
    assert repo.current_fields(conn, b)['city']['value'] == '长沙'
    assert '她' not in repo.aliases_of(conn, b)
    assert len(repo.all_active_persons(conn)) == 2


@pytest.mark.parametrize('state', ['archived', 'merged'])
def test_scoped_and_forced_writes_recheck_person_availability(conn, namesakes, state):
    _, b = namesakes
    if state == 'archived':
        conn.execute('UPDATE person SET deleted_at=1 WHERE id=?', (b,))
    else:
        repo.set_person_status(conn, b, 'merged')
    conn.commit()
    source = '乐团周汐搬到长沙。'
    p = person('乐团周汐', fields=[field('city', '长沙', source)])
    with pytest.raises(ValueError):
        write(conn, source, [p], read_scope=scope(b))
    with pytest.raises(ValueError):
        write(conn, source, [p], forced={'乐团周汐': b})
    assert not repo.current_fields(conn, b)


def test_alias_qualifier_is_not_company_fact_or_value_judgement(conn, namesakes):
    _, b = namesakes
    source = '乐团周汐现在住在长沙。'
    result = write(conn, source, [person('周汐', fields=[field('city', '长沙', source), field('company', '乐团', source)],
        facts=[{'text': '周汐在乐团工作', 'confidence': .99, 'evidence': source}],
        value_read={'text': '周汐有乐团工作背景', 'confidence': .99, 'evidence': source})], read_scope=scope(b))
    assert set(repo.current_fields(conn, b)) == {'city'}
    assert not conn.execute('SELECT 1 FROM memory_item WHERE person_id=?', (b,)).fetchone()
    assert len([a for a in result['write_audit'] if a['reason'] == 'identity_label_is_not_fact']) == 3
    assert not result['pending_ids']


def test_explicit_job_and_workplace_stay_valid_even_when_they_repeat_alias_words(conn, namesakes):
    _, b = namesakes
    source = '乐团周汐在乐团工作，担任首席。'
    write(conn, source, [person('乐团周汐', fields=[field('company', '乐团', source), field('title', '首席', source)],
        facts=[{'text': '周汐在乐团工作', 'confidence': .99, 'evidence': source}])], read_scope=scope(b))
    assert {k: f['value'] for k, f in repo.current_fields(conn, b).items()} == {'company': '乐团', 'title': '首席'}
    assert conn.execute('SELECT 1 FROM memory_item WHERE person_id=?', (b,)).fetchone()


def test_incomplete_company_requires_confirmation_without_inventing_a_title(conn):
    source = '祝遥现在在云岸策展工作。'
    result = write(conn, source, [person('祝遥', fields=[field('company', '云岸', source), field('title', '策展', source)])])
    pid = repo.get_person_by_name(conn, '祝遥')['id']
    assert repo.current_fields(conn, pid) == {}
    assert len(result['pending_ids']) == 1
    pending = next(a for a in result['write_audit'] if a['action'] == 'pending')
    assert pending['candidate']['value'] == '云岸' and pending['source_phrases'] == ['云岸策展']


@pytest.mark.parametrize('suffix', ['远程', '兼职', '长期', '和青屿轮流'])
def test_workplace_suffix_is_never_automatically_added_to_company(conn, suffix):
    source = f'祝遥在云岸{suffix}工作。'
    result = write(conn, source, [person('祝遥', fields=[field('company', '云岸', source)])])
    pid = repo.get_person_by_name(conn, '祝遥')['id']
    assert repo.current_fields(conn, pid) == {}
    assert len(result['pending_ids']) == 1
    assert repo.get_pending(conn, result['pending_ids'][0])['kind'] == 'field'


def test_complete_company_value_is_saved_from_actual_source(conn):
    source = '祝遥现在在云岸策展工作。'
    write(conn, source, [person('祝遥', fields=[field('company', '云岸策展', source)])])
    pid = repo.get_person_by_name(conn, '祝遥')['id']
    assert repo.current_fields(conn, pid)['company']['value'] == '云岸策展'


@pytest.mark.parametrize('predicate', ['做', '担任', '从事', '任'])
def test_explicit_title_predicate_keeps_both_correct_fields(conn, predicate):
    source = f'祝遥现在在云岸{predicate}策展工作。'
    write(conn, source, [person('祝遥', fields=[field('company', '云岸', source), field('title', '策展', source)])])
    pid = repo.get_person_by_name(conn, '祝遥')['id']
    assert {k: f['value'] for k, f in repo.current_fields(conn, pid).items()} == {'company': '云岸', 'title': '策展'}


@pytest.mark.parametrize('prefix', ['不在', '可能在', '如果在', '以前在'])
def test_uncertain_or_historical_clause_is_not_rewritten_as_a_new_complete_workplace(conn, prefix):
    source = f'祝遥{prefix}云岸策展工作。'
    result = write(conn, source, [person('祝遥', fields=[field('company', '云岸', source)])])
    assert not any(a['action'] == 'normalized' for a in result['write_audit'])


@pytest.mark.parametrize('with_id', [False, True])
def test_shared_tool_write_also_rejects_alias_inference(conn, namesakes, with_id):
    _, b = namesakes
    source = '乐团周汐现在住在长沙。'
    args = {'person': '周汐', 'field': 'company', 'value': '乐团'}
    if with_id: args['person_id'] = b
    decision = turn_intent.validate_intent({'intent': 'record', 'record_spans': [source], 'reason': 'test'}, source)
    result = tools.execute_tool(conn, 'save_person_fact', args, source_text=source,
                                source_msg_id='tool-inferred', read_scope=scope(b), turn_intent=decision)
    assert not repo.current_fields(conn, b)
    assert not result.get('pending_ids')
    assert result['ok'] is True
    assert any(a['reason'] == 'identity_label_is_not_fact' for a in result['write_audit'])


def test_untrusted_canonical_does_not_erase_a_real_city(conn, namesakes):
    _, b = namesakes
    source = '乐团周汐现在住在长沙。'
    write(conn, source, [person('乐团周汐', canonical_name='长沙', fields=[field('city', '长沙', source)])],
          read_scope=scope(b))
    assert repo.current_fields(conn, b)['city']['value'] == '长沙'


def test_new_named_subject_before_pronoun_does_not_inherit_previous_person(conn, namesakes):
    _, b = namesakes
    source = '何晴说她搬到南京了。'
    result = write(conn, source, [person('她', fields=[field('city', '南京', source)])], read_scope=scope(b))
    assert repo.current_fields(conn, b) == {}
    assert len(repo.all_active_persons(conn)) == 2
    assert not result['pending_ids']
    decision = turn_intent.validate_intent({'intent': 'record', 'record_spans': [source], 'reason': 'test'}, source)
    denied = tools.execute_tool(conn, 'save_person_fact', {'person': '周汐', 'person_id': b,
        'field': 'city', 'value': '南京'}, source_text=source, source_msg_id='pronoun-other',
        read_scope=scope(b), turn_intent=decision)
    assert denied['ok'] is False and not repo.current_fields(conn, b)


def test_model_canonical_cannot_silently_redirect_a_new_name_to_existing_person(conn, namesakes):
    a, b = namesakes
    source = '何晴搬到南京。'
    result = write(conn, source, [person('何晴', canonical_name='周汐', fields=[field('city', '南京', source)])],
                   read_scope=scope(b))
    assert repo.current_fields(conn, a) == repo.current_fields(conn, b) == {}
    assert '何晴' not in repo.aliases_of(conn, b)
    assert len(result['pending_ids']) == 1
    assert repo.get_pending(conn, result['pending_ids'][0])['kind'] == 'person_identity'


@pytest.mark.parametrize('wrong_name', ['陶艺周汐', '不存在的别名'])
def test_tool_name_and_id_must_agree(conn, namesakes, wrong_name):
    a, b = namesakes
    source = f'{wrong_name}搬到长沙。'
    before = conn.execute('SELECT COUNT(*) FROM person_field').fetchone()[0]
    result = tools.execute_tool(conn, 'save_person_fact', {'person': wrong_name, 'person_id': b,
        'field': 'city', 'value': '长沙'}, source_text=source)
    assert result['ok'] is False
    assert conn.execute('SELECT COUNT(*) FROM person_field').fetchone()[0] == before


def test_tool_cannot_write_another_namesake_through_the_shared_short_name(conn, namesakes):
    a, b = namesakes
    source = '乐团周汐搬到长沙。'
    result = tools.execute_tool(conn, 'save_person_fact', {'person': '周汐', 'person_id': a,
        'field': 'city', 'value': '长沙'}, source_text=source, source_msg_id='wrong-owner', read_scope=scope(b),
        turn_intent=turn_intent.validate_intent({'intent': 'record', 'record_spans': [source], 'reason': 'test'}, source))
    assert result['ok'] is False
    assert repo.current_fields(conn, a) == repo.current_fields(conn, b) == {}


@pytest.fixture
def record_model(monkeypatch):
    source = '帮我记下：乐团周汐现在住在长沙。'
    hooks = {}

    class Model:
        def chat(self, messages, **kwargs):
            if '写入意图分类器' in messages[0]['content']:
                if hooks.get('classify'):
                    hooks['classify']()
                text = json.loads(messages[-1]['content'])['message']
                return {'content': json.dumps({'intent': 'record' if text == source else 'query',
                    'record_spans': [source] if text == source else [], 'reason': 'test'}, ensure_ascii=False)}
            if '本轮回答的资料编辑' in messages[0]['content']:
                return {'content': json.dumps({'answer_type': 'no_answer', 'issues': [], 'segments': [
                    {'kind': 'no_answer', 'text': '', 'source_ids': []}]})}
            return {'content': '按实际处理回执核对。'}

    monkeypatch.setattr(harness, 'get_llm', lambda: Model())
    return source, hooks


@pytest.fixture
def recorded_turn(conn, monkeypatch, namesakes, record_model):
    source, _ = record_model
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda text, *args:
        extract.sanitize({'people': [person('周汐', fields=[field('city', '长沙', text)])]}, source=text))
    cid = repo.create_conversation(conn)
    result = harness.run_turn(conn, cid, source)
    return cid, source, result['conversation_scope']['user_message_id']


@pytest.mark.parametrize('phase', ['classify', 'extract'])
@pytest.mark.parametrize('change', ['delete', 'transfer', 'share'])
def test_write_rejects_alias_changes_while_waiting_on_a_model(
        conn, monkeypatch, namesakes, record_model, phase, change):
    a, b = namesakes
    source, hooks = record_model

    def mutate():
        if change != 'share':
            conn.execute('DELETE FROM person_alias WHERE person_id=? AND alias=?', (b, '乐团周汐'))
        if change != 'delete':
            repo.add_alias(conn, a, '乐团周汐', 'test-during-model')
        conn.commit()

    def extraction(text, *args):
        if phase == 'extract':
            mutate()
        return extract.sanitize({'people': [person('乐团周汐',
            fields=[field('city', '长沙', text)])]}, source=text)

    if phase == 'classify':
        hooks['classify'] = mutate
    monkeypatch.setattr(harness.extract, 'extract_from_text', extraction)
    result = harness.run_turn(conn, repo.create_conversation(conn), source)
    assert result['conversation_scope']['person_ids'] == [b]
    assert result['failed'] == [{'kind': 'extract', 'status': 'failed',
                                'detail': '记事原文中的人物称呼已变化，请重新指定人物'}]
    assert not result['saved'] and not result['pending_ids']
    assert repo.current_fields(conn, a) == repo.current_fields(conn, b) == {}
    assert len(repo.all_active_persons(conn)) == 2


@pytest.mark.parametrize('change', ['delete', 'transfer', 'share'])
@pytest.mark.parametrize('tool_name', ['save_person_fact', 'create_reminder'])
def test_id_only_write_tools_recheck_frozen_alias_ownership(conn, namesakes, change, tool_name):
    a, b = namesakes
    source = ('乐团周汐搬到长沙。' if tool_name == 'save_person_fact'
              else '提醒我2026-11-01联系乐团周汐。')
    write_scope = freeze_write_aliases(conn, scope(b), source)
    if change != 'share':
        conn.execute('DELETE FROM person_alias WHERE person_id=? AND alias=?', (b, '乐团周汐'))
    if change != 'delete':
        repo.add_alias(conn, a, '乐团周汐', 'test-during-tool-model')
    conn.commit()
    args = {'person_id': a if change == 'transfer' else b}
    if tool_name == 'save_person_fact':
        args.update(field='city', value='长沙')
    else:
        args.update(text='联系乐团周汐', fire_date='2026-11-01')
    decision = turn_intent.validate_intent({'intent': 'record', 'record_spans': [source],
                                          'reason': 'test'}, source)
    result = tools.execute_tool(conn, tool_name, args, source_text=source,
                                source_msg_id='id-only-alias-change', read_scope=write_scope,
                                turn_intent=decision)
    assert result == {'ok': False, 'error': '记事原文中的人物称呼已变化，请重新指定人物'}
    assert repo.current_fields(conn, a) == repo.current_fields(conn, b) == {}
    assert conn.execute('SELECT COUNT(*) FROM reminder').fetchone()[0] == 0


def test_retry_after_later_person_selection_keeps_original_write_person(conn, namesakes, recorded_turn):
    a, b = namesakes
    cid, source, mid = recorded_turn
    harness.run_turn(conn, cid, '查陶艺周汐的资料。')
    repeated = harness.run_turn(conn, cid, source, user_message_id=mid)
    assert repeated['conversation_scope']['person_ids'] == [b]
    assert repo.current_fields(conn, a) == {}
    assert repo.current_fields(conn, b)['city']['value'] == '长沙'
    assert conn.execute('SELECT COUNT(*) FROM person_field').fetchone()[0] == 1


def test_old_write_cannot_follow_an_alias_transferred_to_another_person(conn, namesakes, recorded_turn):
    a, b = namesakes
    cid, source, mid = recorded_turn
    metadata = conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0]
    conn.execute('DELETE FROM person_alias WHERE person_id=? AND alias=?', (b, '乐团周汐'))
    repo.add_alias(conn, a, '乐团周汐', 'test-transfer')
    conn.commit()
    with pytest.raises(ValueError, match='原记事选择的人物已变化'):
        harness.run_turn(conn, cid, source, user_message_id=mid)
    assert conn.execute('SELECT tool_calls FROM message WHERE id=?', (mid,)).fetchone()[0] == metadata
    assert repo.current_fields(conn, a) == {}
    assert repo.current_fields(conn, b)['city']['value'] == '长沙'
