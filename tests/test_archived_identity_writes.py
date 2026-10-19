"""Archived aliases cannot silently become new people through automatic extraction."""
from types import SimpleNamespace

import pytest

from app.agent import conversation_scope, harness
from app.memory import apply as apply_mod, disambiguate
from app.memory.writes import AmbiguousPerson, resolve_person
from app.store import repo


def extraction(canonical='老何', mention=None):
    return {'people': [{
        'canonical_name': canonical, 'name': mention or canonical,
        'fields': [{'field': 'city', 'value': '上海', 'confidence': .95, 'evidence': '住上海'}],
        'aliases': [], 'facts': [], 'promises': [], 'ledger': [], 'relations': [], 'value_read': None,
    }], 'my_notes': []}


def archived_person(conn, name='何川', alias='老何'):
    pid = repo.create_person(conn, name, status='active')
    repo.add_alias(conn, pid, alias, 'manual', 1.)
    repo.upsert_field(conn, pid, 'city', '深圳', 1., None)
    repo.add_memory(conn, 'semantic', 'archived private preference', pid, .95, None)
    repo.archive_person(conn, pid)
    conn.commit()
    return pid


def scope(conn, text):
    cid = repo.create_conversation(conn, 'Archived identity test')
    mid = repo.add_message(conn, cid, 'user', text)
    conn.commit()
    return conversation_scope.for_user_message(conn, cid, mid, text)


@pytest.mark.parametrize('canonical,mention', [('老何', '老何'), ('何川', '老何'),
                                             ('何川', '新称呼'), ('新称呼', '老何')])
def test_extraction_skips_if_canonical_or_mention_is_an_archived_only_alias(conn, canonical, mention):
    pid = archived_person(conn)
    before = {table: conn.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
              for table in ('person', 'person_alias', 'person_field', 'memory_item')}
    result = apply_mod.apply_extraction(conn, extraction(canonical, mention), 'archived-question',
                                       f'{mention}现在住上海吗？')
    assert result['person_ids'] == result['pending_ids'] == []
    assert len(result['outcomes']) == 1
    assert result['outcomes'][0]['kind'] == 'person'
    assert result['outcomes'][0]['status'] == 'skipped'
    assert '已归档' in result['outcomes'][0]['detail']
    assert repo.get_person(conn, pid)['deleted_at'] is not None
    assert repo.current_fields(conn, pid)['city']['value'] == '深圳'
    assert all(conn.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] == count
               for table, count in before.items())
    evidence = conn.execute('SELECT * FROM evidence WHERE source_ref=?', ('archived-question',)).fetchone()
    assert evidence is not None and mention in evidence['excerpt']


@pytest.mark.parametrize('name', ['何川', '老何'])
def test_resolve_person_create_rejects_archived_alias_without_changing_disambiguation_contract(conn, name):
    pid = archived_person(conn)
    assert disambiguate.resolve(conn, name).kind == 'new'
    with pytest.raises(ValueError, match='已归档'):
        resolve_person(conn, name, create=True)
    assert conn.execute('SELECT COUNT(*) FROM person').fetchone()[0] == 1
    assert repo.get_person(conn, pid)['deleted_at'] is not None


def test_active_shared_exact_alias_keeps_normal_resolution_and_ambiguity(conn):
    archived = archived_person(conn)
    active = repo.create_person(conn, '何舟', status='active')
    repo.add_alias(conn, active, '老何', 'manual', 1.)
    conn.commit()
    assert resolve_person(conn, '老何', create=True) == active
    result = apply_mod.apply_extraction(conn, extraction(), 'active-question', '老何住上海')
    assert result['person_ids'] == [active]
    assert repo.current_fields(conn, active)['city']['value'] == '上海'
    assert repo.current_fields(conn, archived)['city']['value'] == '深圳'
    other = repo.create_person(conn, '何宁', status='active')
    repo.add_alias(conn, other, '老何', 'manual', 1.)
    conn.commit()
    with pytest.raises(AmbiguousPerson):
        resolve_person(conn, '老何', create=True)
    ambiguous = apply_mod.apply_extraction(conn, extraction(), 'ambiguous-question', '老何住上海')
    assert ambiguous['person_ids'] == [] and ambiguous['pending_ids']


def test_forced_active_user_choice_and_explicit_manual_new_person_are_preserved(conn):
    archived = archived_person(conn)
    chosen = repo.create_person(conn, '何舟', status='active')
    conn.commit()
    result = apply_mod.apply_extraction(conn, extraction(), 'confirmed-question', '老何住上海',
                                       forced={'老何': chosen})
    assert result['person_ids'] == [chosen]
    assert repo.current_fields(conn, chosen)['city']['value'] == '上海'
    assert repo.current_fields(conn, archived)['city']['value'] == '深圳'
    manual = repo.create_person(conn, '何川', status='active')
    conn.commit()
    assert manual != archived and resolve_person(conn, '何川', create=True) == manual


def test_purge_removes_alias_reservation_and_allows_automatic_new_person(conn):
    archived = archived_person(conn)
    repo.purge_person(conn, archived)
    conn.commit()
    result = apply_mod.apply_extraction(conn, extraction(), 'new-after-purge', '老何住上海')
    assert len(result['person_ids']) == 1
    assert result['person_ids'][0] != archived
    assert repo.get_person(conn, result['person_ids'][0])['display_name'] == '老何'
    assert resolve_person(conn, '何川', create=True) != archived


@pytest.mark.parametrize('question', ['老何怎么联系？', '何川怎么联系？', '比较陈岚和老何的偏好'])
def test_explicit_archived_alias_in_fresh_scope_requires_unavailable_clarification(conn, question):
    archived = archived_person(conn)
    repo.create_person(conn, '陈岚', status='active')
    conn.commit()
    result = scope(conn, question)
    assert result['blocked'] and result['reason'] == 'unavailable_person'
    assert result['person_ids'] == result['candidates'] == []
    assert archived not in str(result)


@pytest.mark.parametrize('short,long', [('何川', '何川川'), ('Ann', 'Ann Lee')])
def test_active_longer_alias_covers_archived_short_name_only_at_same_occurrence(conn, short, long):
    archived_person(conn, short, short)
    active = repo.create_person(conn, long, status='active')
    conn.commit()
    result = scope(conn, f'{long}的沟通偏好？')
    assert not result['blocked'] and result['person_ids'] == [active]
    mixed = scope(conn, f'比较{long}与{short}的偏好')
    assert mixed['blocked'] and mixed['reason'] == 'unavailable_person'


def test_active_shared_alias_and_explicit_id_take_precedence_for_scope(conn):
    archived_person(conn)
    active = repo.create_person(conn, '何舟', status='active')
    repo.add_alias(conn, active, '老何', 'manual', 1.)
    conn.commit()
    assert scope(conn, '老何怎么联系？')['person_ids'] == [active]
    assert scope(conn, f'人物 ID: {active}。何川怎么联系？')['person_ids'] == [active]


def test_harness_archived_alias_query_keeps_evidence_without_extracting_recreating_or_reading_person(conn, monkeypatch):
    archived = archived_person(conn)
    repo.create_person(conn, '陈言', status='active')
    cid = repo.create_conversation(conn, 'Fresh archived query')
    conn.commit()
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: pytest.fail('archived query is read-only'))
    monkeypatch.setattr(harness.turn_intent, 'classify_turn', lambda text, **kwargs:
        harness.turn_intent.validate_intent({'intent': 'query', 'record_spans': [], 'reason': 'query fixture'}, text))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda *args, **kwargs:
        pytest.fail('deterministic unavailable clarification needs no model review'))
    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(
        chat=lambda *args, **kwargs: pytest.fail('archived query must not ask answer model to guess')))
    result = harness.run_turn(conn, cid, '老何怎么联系？')
    assert result['conversation_scope']['reason'] == 'unavailable_person'
    assert result['conversation_scope']['candidates'] == []
    assert result['sources'] == result['source_dependencies'] == []
    assert '不可用' in result['answer'] and '陈言' not in result['answer']
    assert result['turn_intent']['intent'] == 'query'
    assert any(outcome['kind'] == 'intent' and outcome['status'] == 'skipped' for outcome in result['outcomes'])
    assert conn.execute("SELECT COUNT(*) FROM person WHERE display_name='老何'").fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM interaction').fetchone()[0] == 0
    assert repo.current_fields(conn, archived)['city']['value'] == '深圳'
    source_id = result['conversation_scope']['user_message_id']
    assert conn.execute('SELECT 1 FROM evidence WHERE source_ref=?', (source_id,)).fetchone()
