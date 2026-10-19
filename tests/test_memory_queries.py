"""Natural lexical questions and enforced read scopes on real SQLite records."""
from types import SimpleNamespace

import pytest

from app.agent import tools
from app.memory import fts, retrieve, vector
from app.store import repo
from app.util import new_id, now_ms


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from app import llm
    monkeypatch.setattr(llm, 'get_llm', lambda: (_ for _ in ()).throw(AssertionError('No model expected')))


@pytest.fixture
def people(conn):
    first = repo.create_person(conn, '陈宁', status='active')
    second = repo.create_person(conn, '李安', status='active')
    third = repo.create_person(conn, '周敏', status='active')
    repo.add_alias(conn, first, '宁宁', 'manual')
    return SimpleNamespace(first=first, second=second, third=third)


def memory(conn, pid, text, *, ts=None):
    eid = repo.add_evidence(conn, 'text', source_ref='query-test:' + new_id(), excerpt=text)
    return repo.add_memory(conn, 'semantic', text, pid, .9, eid, ts=ts)


def read_scope(*ids):
    return {'mode': 'single' if len(ids) == 1 else 'multiple', 'person_ids': list(ids), 'blocked': False}


def query(conn, text, **args):
    return tools.execute_tool(conn, 'query_memory', {'query': text, 'retrieval_mode': 'fts', **args})


@pytest.mark.parametrize('text', ['邮件？', '请问邮件呢？', '邮件!!!', '邮件，邮件。'])
def test_punctuation_and_function_words_do_not_require_empty_index_tokens(conn, people, text):
    mid = memory(conn, people.first, '习惯用邮件沟通，白天不接电话。')
    result = query(conn, text, person_id=people.first)
    assert result['ok'] and [row['id'] for row in result['results']] == [mid]
    assert result['results'][0]['match_type'] == 'fts_exact'


def test_query_cleanup_preserves_negation_numbers_and_temporal_terms():
    terms = fts.query_terms('请问不喜欢电话，2026年之前呢？')
    assert '不' in terms and '2026' in terms and '之前' in terms
    assert '?' not in terms and '？' not in terms and '请问' not in terms
    assert fts.tokenize('不接电话。').endswith('。')  # Existing index tokenization remains intact.


@pytest.mark.parametrize('text', ['宁宁平时喜欢怎么沟通', '陈宁的沟通偏好是什么？'])
def test_natural_question_partial_match_stays_with_resolved_person(conn, people, text):
    desired = memory(conn, people.first, '习惯用邮件沟通，白天不接电话。')
    memory(conn, people.second, '平时喜欢用电话沟通，沟通偏好是电话。')
    result = query(conn, text)
    assert result['ok'] and result['scope_inferred']
    assert [row['id'] for row in result['results']] == [desired]
    source = result['results'][0]['source']
    assert source['person']['id'] == people.first and source['text'] == '习惯用邮件沟通，白天不接电话。'
    assert result['results'][0]['match_type'] in {'fts_partial', 'text_partial'}


def test_compound_substring_filters_visible_person_set_before_candidate_limit(conn, people):
    first = memory(conn, people.first, '白天不接电话。')
    second = memory(conn, people.second, '下班后方便接电话。')
    for _ in range(30):
        memory(conn, people.third, '电话')
    hidden = repo.create_person(conn, 'Archived', status='active')
    for _ in range(20):
        memory(conn, hidden, '接电话')
    conn.execute('UPDATE person SET deleted_at=1 WHERE id=?', (hidden,))
    withdrawn = memory(conn, people.first, '电话')
    conn.execute("UPDATE memory_item SET status='withdrawn' WHERE id=?", (withdrawn,))
    results = retrieve.hybrid_search(conn, '电话', 2, person_ids=[people.first, people.second], retrieval_mode='fts')
    assert {row['id'] for row in results} == {first, second}
    assert all(row['source']['person']['id'] in {people.first, people.second} for row in results)


def test_exact_match_precedes_partial_candidate_even_when_partial_is_newer(conn, people):
    exact = memory(conn, people.first, '邮件 电话', ts=1000)
    partial = memory(conn, people.first, '邮件', ts=now_ms())
    results = retrieve.hybrid_search(conn, '邮件 电话', person_id=people.first, retrieval_mode='fts')
    assert [row['id'] for row in results] == [exact, partial]
    assert results[0]['match_type'] == 'fts_exact' and results[1]['match_type'].endswith('_partial')


@pytest.mark.parametrize('text', ['量子芯片', '请问是什么呢？', '!!!', '陈宁一般怎样联系比较合适'])
def test_no_answer_and_filler_only_questions_do_not_fall_back_to_recent_records(conn, people, text):
    memory(conn, people.first, '习惯用邮件沟通，白天不接电话。')
    result = query(conn, text, person_id=people.first)
    assert result['ok'] and result['results'] == []


def test_unscoped_search_does_not_relax_all_keywords_across_every_person(conn, people):
    memory(conn, people.first, 'quartz background')
    assert query(conn, 'quartz zephyr')['results'] == []
    assert query(conn, 'quartz zephyr', person_id=people.first)['results']


def test_multi_person_memory_and_interaction_scopes_exclude_third_person(conn, people):
    expected = {memory(conn, people.first, 'quartz first'), memory(conn, people.second, 'quartz second')}
    interaction, _ = repo.insert_interaction(conn, now_ms(), 'chat', 'quartz contact', new_id(),
                                             [(people.second, 'participant')], kind='contact')
    expected.add(interaction)
    for _ in range(20):
        memory(conn, people.third, 'quartz')
    result = query(conn, '陈宁 李安 quartz', person_ids=[people.first, people.second], top_k=3)
    assert result['ok'] and {row['id'] for row in result['results']} == expected
    assert set(result['scoped_person_ids']) == {people.first, people.second}


def test_personal_scope_relaxes_only_unowned_visible_records(conn, people):
    personal = memory(conn, None, '习惯用邮件沟通')
    memory(conn, people.first, '沟通偏好')
    result = query(conn, '沟通偏好是什么？', scope='personal')
    assert result['ok'] and [row['id'] for row in result['results']] == [personal]
    assert result['results'][0]['source']['person'] is None


def test_fts_mode_never_enters_embedding_channel(conn, people, monkeypatch):
    from app import llm
    calls = []
    monkeypatch.setattr(llm, 'get_llm', lambda: calls.append('called'))
    memory(conn, people.first, 'quartz')
    assert retrieve.hybrid_search(conn, 'quartz', retrieval_mode='fts')
    assert calls == []


def test_vector_allowed_ids_are_a_person_set_before_similarity_limit(conn, people, monkeypatch):
    from app import llm
    class FakeLLM:
        def embed(self, texts):
            return [[1., 0.]]
    monkeypatch.setattr(llm, 'get_llm', FakeLLM)
    index = vector.VectorIndex()
    monkeypatch.setattr(vector, 'get_index', lambda: index)
    for _ in range(20):
        mid = memory(conn, people.third, 'foreign vector')
        index.add(conn, 'memory', mid, 'foreign vector', [1., 0.])
    desired = set()
    for pid in (people.first, people.second):
        mid = memory(conn, pid, 'scoped vector')
        desired.add(mid)
        index.add(conn, 'memory', mid, 'scoped vector', [.8, .2])
    results = retrieve.hybrid_search(conn, 'unmatchedtoken', 2, person_ids=[people.first, people.second])
    assert {row['id'] for row in results} == desired


@pytest.mark.parametrize('changes', [
    {'person_id': 'one', 'person_ids': []}, {'person_ids': [], 'scope': 'personal'},
    {'person_ids': 'one'}, {'retrieval_mode': 'unknown'},
])
def test_invalid_scope_combinations_do_not_expand_to_global(conn, changes):
    with pytest.raises(ValueError):
        retrieve.hybrid_search(conn, 'quartz', **changes)
    assert retrieve.hybrid_search(conn, 'quartz', person_ids=[]) == []


@pytest.mark.parametrize('tool,args', [('query_memory', {'query': 'quartz'}), ('get_person', {'name': '陈宁'})])
def test_blocked_scope_does_not_read_profile_or_memory(conn, people, monkeypatch, tool, args):
    monkeypatch.setattr(repo, 'get_person', lambda *a, **k: pytest.fail('blocked read queried a person'))
    result = tools.execute_tool(conn, tool, args, read_scope={
        'mode': 'clarify', 'blocked': True, 'reason': '请选择人物', 'person_ids': [],
        'candidates': [{'person_id': people.first, 'display_name': '陈宁'}]})
    assert not result['ok'] and result['needs_confirmation'] and result['error'] == '请选择人物'
    assert result['candidates'][0]['person_id'] == people.first


@pytest.mark.parametrize('tool,args', [
    ('query_memory', {'query': 'quartz'}), ('get_person', {}),
])
def test_single_scope_supplies_missing_person_id(conn, people, tool, args):
    mid = memory(conn, people.first, 'quartz first')
    memory(conn, people.second, 'quartz second')
    result = tools.execute_tool(conn, tool, {**args, 'retrieval_mode': 'fts'}, read_scope=read_scope(people.first))
    assert result['ok']
    if tool == 'query_memory':
        assert [row['id'] for row in result['results']] == [mid]
    else:
        assert result['person_id'] == people.first and '陈宁' in result['card']


@pytest.mark.parametrize('tool,args', [
    ('query_memory', {'query': 'quartz', 'person_id': 'third'}),
    ('query_memory', {'query': 'quartz', 'person_ids': ['first', 'third']}),
    ('query_memory', {'query': 'quartz', 'person': '周敏'}),
    ('get_person', {'person_id': 'third'}), ('get_person', {'name': '周敏'}),
    ('get_person', {'person_ids': ['first'], 'name': '周敏'}),
])
def test_model_cannot_read_a_person_outside_the_turn_scope(conn, people, tool, args):
    args = dict(args)
    if args.get('person_id'):
        args['person_id'] = getattr(people, args['person_id'])
    if args.get('person_ids'):
        args['person_ids'] = [getattr(people, key) for key in args['person_ids']]
    result = tools.execute_tool(conn, tool, args, read_scope=read_scope(people.first, people.second))
    assert not result['ok'] and result['needs_confirmation']
    assert 'card' not in result and 'results' not in result


def test_multiple_scope_is_a_set_for_memory_but_requires_a_person_for_profile(conn, people):
    desired = {memory(conn, people.first, 'quartz first'), memory(conn, people.second, 'quartz second')}
    memory(conn, people.third, 'quartz third')
    scope = read_scope(people.first, people.second)
    result = tools.execute_tool(conn, 'query_memory', {'query': 'quartz', 'retrieval_mode': 'fts'}, read_scope=scope)
    assert result['ok'] and {row['id'] for row in result['results']} == desired
    result = tools.execute_tool(conn, 'get_person', {}, read_scope=scope)
    assert not result['ok'] and result['needs_confirmation']
    selected = tools.execute_tool(conn, 'get_person', {'name': '陈宁'}, read_scope=scope)
    assert selected['ok'] and selected['person_id'] == people.first


def test_previously_resolved_same_name_is_limited_to_the_chosen_id(conn):
    first = repo.create_person(conn, '张伟', status='active')
    second = repo.create_person(conn, '张伟', status='active')
    expected = memory(conn, first, 'quartz first')
    memory(conn, second, 'quartz second')
    result = tools.execute_tool(conn, 'query_memory', {'query': 'quartz', 'person': '张伟', 'retrieval_mode': 'fts'},
                                read_scope=read_scope(first))
    assert result['ok'] and [row['id'] for row in result['results']] == [expected]
    ambiguous = tools.execute_tool(conn, 'query_memory', {'query': '张伟 quartz', 'retrieval_mode': 'fts'})
    assert ambiguous['needs_confirmation']


def test_empty_or_stale_restricted_scope_never_reads_globally(conn, people):
    memory(conn, people.third, 'quartz foreign')
    empty = tools.execute_tool(conn, 'query_memory', {'query': 'quartz'}, read_scope=read_scope())
    assert empty['needs_confirmation']
    conn.execute('UPDATE person SET deleted_at=1 WHERE id=?', (people.first,))
    stale = tools.execute_tool(conn, 'query_memory', {'query': 'quartz'}, read_scope=read_scope(people.first))
    assert stale['needs_confirmation'] and not stale.get('results')


def test_legacy_and_explicit_global_scope_remain_available_and_writes_are_not_restricted(conn, people):
    expected = memory(conn, people.third, 'quartz foreign')
    for scope in (None, {'mode': 'none', 'person_ids': []}, {'mode': 'global', 'person_ids': []}):
        result = tools.execute_tool(conn, 'query_memory', {'query': 'quartz', 'retrieval_mode': 'fts'}, read_scope=scope)
        assert result['ok'] and [row['id'] for row in result['results']] == [expected]
    result = tools.execute_tool(conn, 'note', {'text': 'a user note'}, read_scope={'blocked': True})
    assert result['ok'] and result.get('receipt')


def test_removing_names_keeps_ascii_words_and_short_cjk_substrings_intact(conn, people):
    repo.add_alias(conn, people.first, 'Ann', 'manual')
    repo.add_alias(conn, people.first, '安', 'manual')
    assert tools._scoped_query(conn, '陈宁 Ann Annual 安全 安', people.first) == 'Annual 安全'
