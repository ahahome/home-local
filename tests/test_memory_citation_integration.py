"""Real retrieval/citation API contracts with a scripted model, not an LLM accuracy score."""
from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import harness
from app.agent.source_registry import SourceChangedError
from app.memory import memory_corrections
from app.store import db, repo
from app.util import new_id
from eval.memory_retrieval import evaluate, load_fixture


@pytest.fixture
def api(conn, monkeypatch):
    from app import llm
    from app.api import main, memories, message_sources

    for module in (main, memories, message_sources):
        monkeypatch.setattr(module, 'get_conn', lambda: conn)
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *args: {'people': [], 'my_notes': []})
    monkeypatch.setattr(llm, 'get_llm', lambda: (_ for _ in ()).throw(RuntimeError('offline FTS test')))
    monkeypatch.setattr(socket.socket, 'connect', lambda *args: pytest.fail('citation tests must remain offline'))
    client = TestClient(main.app)
    response = client.post('/api/conversations', json={'title': 'Citation integration'})
    assert response.status_code == 200, response.text
    yield SimpleNamespace(client=client, cid=response.json()['id'], monkeypatch=monkeypatch)
    client.close()


def seed_person(conn, name='Lin'):
    pid = repo.create_person(conn, name, status='active')
    conn.commit()
    return pid


def seed_memory(conn, pid, text='communication preference: phone'):
    eid = repo.add_evidence(conn, 'text', source_ref='citation:' + new_id(), excerpt=text)
    mid = repo.add_memory(conn, 'semantic', text, pid, .95, eid)
    conn.commit()
    return mid, eid


def change(conn, mid, action='edit', **changes):
    revision = conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0]
    return memory_corrections.correct_memory(conn, mid, {
        'action': action, 'expected_revision': revision, 'request_id': new_id(), **changes})


class ReadingModel:
    """Only script model choices; production tools, registration and persistence stay real."""
    def __init__(self, tool, args, answer=None, before_answer=None):
        self.tool, self.args = tool, args
        self.answer, self.before_answer = answer, before_answer
        self.calls, self.read = [], None

    def chat(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        if len(self.calls) == 1:
            return {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'read-' + new_id(), 'type': 'function', 'function': {
                    'name': self.tool, 'arguments': json.dumps(self.args)}}]}
        assert len(self.calls) == 2
        self.read = json.loads(next(message['content'] for message in reversed(messages) if message['role'] == 'tool'))
        if self.before_answer:
            self.before_answer(self.read)
        if self.answer:
            content = self.answer(self.read)
        elif self.read.get('results'):
            source = self.read['results'][0]['source']
            content = source['text'] + ' [[' + source['id'] + ']]'
        elif self.read.get('needs_confirmation'):
            content = 'Please choose the correct person before continuing.'
        else:
            content = 'No active matching memory is available.'
        return {'role': 'assistant', 'content': content}


def ask(api, tool='query_memory', args=None, answer=None, before_answer=None, text='Check communication preference'):
    model = ReadingModel(tool, args or {}, answer, before_answer)
    api.monkeypatch.setattr(harness, 'get_llm', lambda: model)
    response = api.client.post(f'/api/conversations/{api.cid}/messages', json={'text': text})
    assert response.status_code == 200, response.text
    return response.json(), model


def source_url(api, result, source=None):
    source = source or result['sources'][0]
    return f"/api/conversations/{api.cid}/messages/{result['assistant_message_id']}/sources/{source['id']}"


def source_detail(api, result, source=None):
    response = api.client.get(source_url(api, result, source))
    assert response.status_code == 200, response.text
    return response.json()


def final_row(conn, result):
    return dict(conn.execute('SELECT * FROM message WHERE id=?', (result['assistant_message_id'],)).fetchone())


def test_cited_answer_has_real_record_evidence_and_clickable_registered_backend_route(api, conn):
    pid = seed_person(conn)
    mid, eid = seed_memory(conn, pid)

    result, model = ask(api, args={'query': 'communication preference', 'person_id': pid})

    assert result['answer'] == 'communication preference: phone [[S1]]'
    assert len(result['sources']) == 1
    source = result['sources'][0]
    assert source['entity_id'] == mid and source['type'] == 'memory'
    assert source['person']['id'] == pid
    assert source['text'] == 'communication preference: phone'
    assert source['evidence']['id'] == eid and source['evidence']['excerpt'] == source['text']
    assert source['temporal_status'] == 'historical_background'
    assert source['revision'] == 0
    assert model.read['results'][0]['source_id'] == source['id']
    clicked = source_detail(api, result)
    assert clicked['source'] == source and clicked['state']['status'] == 'current'
    messages = api.client.get(f'/api/conversations/{api.cid}/messages').json()['messages']
    rendered = next(message for message in messages if message['id'] == result['assistant_message_id'])
    assert rendered['metadata']['sources'] == result['sources']
    assert 'historical_context' in {warning['code'] for warning in result['source_warnings']}


def test_phone_edit_email_withdraw_restore_retrieval_and_old_answer_remain_consistent(api, conn):
    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)
    args = {'query': 'communication preference', 'person_id': pid}
    initial, _ = ask(api, args=args)
    initial_row = final_row(conn, initial)

    change(conn, mid, text='communication preference: email')

    clicked = source_detail(api, initial)
    assert clicked['source']['text'] == 'communication preference: phone'
    assert clicked['state']['status'] == 'updated'
    assert clicked['state']['current']['text'] == 'communication preference: email'
    edited, model = ask(api, args=args)
    assert edited['answer'] == 'communication preference: email [[S1]]'
    assert edited['sources'][0]['entity_id'] == mid
    assert model.read['results'][0]['text'] == 'communication preference: email'
    assert all('communication preference: phone' not in message['content'] for message in model.calls[0])
    assert final_row(conn, initial) == initial_row
    edited_row = final_row(conn, edited)

    change(conn, mid, action='withdraw')

    assert source_detail(api, initial)['state']['status'] == 'withdrawn'
    assert source_detail(api, edited)['state']['status'] == 'withdrawn'
    withdrawn, model = ask(api, args=args)
    assert model.read['results'] == [] and withdrawn['sources'] == []
    assert 'phone' not in withdrawn['answer'] and 'email' not in withdrawn['answer']
    assert final_row(conn, initial) == initial_row and final_row(conn, edited) == edited_row

    change(conn, mid, action='restore')

    restored, model = ask(api, args=args)
    assert model.read['results'][0]['text'] == 'communication preference: email'
    assert restored['sources'][0]['revision'] == 3
    assert source_detail(api, restored)['state']['status'] == 'current'
    assert source_detail(api, edited)['state']['status'] == 'updated'
    assert final_row(conn, initial) == initial_row and final_row(conn, edited) == edited_row


def test_same_name_requires_confirmation_and_explicit_person_id_isolates_citations(api, conn):
    first, second = seed_person(conn, '同名林舟'), seed_person(conn, '同名林舟')
    first_mid, _ = seed_memory(conn, first, 'communication preference: phone')
    second_mid, _ = seed_memory(conn, second, 'communication preference: email')

    ambiguous, model = ask(api, args={'query': 'communication preference', 'person': '同名林舟'})

    assert model.read['ok'] is False and model.read['needs_confirmation'] is True
    assert {candidate['person_id'] for candidate in model.read['candidates']} == {first, second}
    assert not model.read.get('results') and ambiguous['sources'] == []
    selected, _ = ask(api, args={'query': 'communication preference', 'person_id': second})
    assert selected['sources'][0]['entity_id'] == second_mid
    assert selected['sources'][0]['person']['id'] == second
    assert 'email' in selected['answer'] and 'phone' not in selected['answer']
    assert all(source['entity_id'] != first_mid for source in selected['sources'])
    implicit, model = ask(api, args={'query': '同名林舟 communication preference'})
    assert model.read['needs_confirmation'] is True and implicit['sources'] == []


def test_current_and_historical_city_sources_keep_temporal_status_and_only_used_sources(api, conn):
    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)
    for value, timestamp in [('Hangzhou', 1000), ('Shanghai', 2000)]:
        eid = repo.add_evidence(conn, 'text', source_ref='city:' + value, excerpt=value)
        repo.upsert_field(conn, pid, 'city', value, 1., eid, valid_from=timestamp)
    conn.commit()

    def answer(read):
        current = next(source for source in read['sources'] if source['type'] == 'field'
                       and source['temporal_status'] == 'current_field')
        old = next(source for source in read['sources'] if source['type'] == 'field'
                   and source['temporal_status'] == 'historical_field')
        return f"Current city: {current['text']} [[{current['id']}]]. Previous city: {old['text']} [[{old['id']}]]."

    result, model = ask(api, tool='get_person', args={'person_id': pid}, answer=answer)

    assert len(result['sources']) == 2
    current, old = result['sources']
    assert current['text'] == 'Shanghai' and current['temporal_status'] == 'current_field'
    assert old['text'] == 'Hangzhou' and old['temporal_status'] == 'historical_field'
    assert old['valid_to'] == current['valid_from'] == 2000
    assert source_detail(api, result, current)['state']['status'] == 'current'
    assert source_detail(api, result, old)['state']['status'] == 'current'
    assert all(source['entity_id'] != mid for source in result['sources'])
    assert any(source['entity_id'] == mid for source in model.read['sources'])
    assert 'city=Shanghai' in model.read['card'] and 'city=Hangzhou' not in model.read['card']


def test_archive_restore_move_and_purge_update_source_state_without_rewriting_old_answer(api, conn):
    pid, target = seed_person(conn), seed_person(conn, 'Chen')
    mid, _ = seed_memory(conn, pid)
    result, _ = ask(api, args={'query': 'communication preference', 'person_id': pid})
    original = final_row(conn, result)

    assert api.client.delete(f'/api/persons/{pid}').status_code == 200
    assert source_detail(api, result)['state']['status'] == 'unavailable'
    assert final_row(conn, result) == original
    assert api.client.post(f'/api/persons/{pid}/restore').status_code == 200
    assert source_detail(api, result)['state']['status'] == 'updated'
    assert final_row(conn, result) == original

    change(conn, mid, person_id=target)

    moved = source_detail(api, result)
    assert moved['source']['person']['id'] == pid
    assert moved['state']['status'] == 'moved' and moved['state']['current']['person']['id'] == target
    assert final_row(conn, result) == original
    old_owner, old_model = ask(api, args={'query': 'communication preference', 'person_id': pid})
    assert old_model.read['results'] == [] and old_owner['sources'] == []
    new_owner, _ = ask(api, args={'query': 'communication preference', 'person_id': target})
    assert new_owner['sources'][0]['person']['id'] == target
    assert api.client.post(f'/api/persons/{target}/purge').status_code == 200
    assert source_detail(api, result)['state']['status'] == 'unavailable'
    assert source_detail(api, new_owner)['state']['status'] == 'unavailable'
    assert final_row(conn, result) == original


@pytest.mark.parametrize('answer,expected_codes', [
    ('No citation attached.', {'uncited'}),
    ('Fabricated citation [[S999]].', {'invalid_citation', 'uncited'}),
])
def test_uncited_or_fabricated_markers_do_not_become_trusted_sources(api, conn, answer, expected_codes):
    pid = seed_person(conn)
    seed_memory(conn, pid)

    result, _ = ask(api, args={'query': 'communication preference', 'person_id': pid}, answer=lambda read: answer)

    assert result['sources'] == []
    assert expected_codes <= {warning['code'] for warning in result['source_warnings']}
    assert '[[S999]]' not in result['answer']
    response = api.client.get(source_url(api, result, {'id': 'S999'}))
    assert response.status_code == 404


def test_uncited_answer_tracks_read_dependencies_and_hides_obsolete_history_without_db_rewrite(api, conn):
    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)
    result, _ = ask(api, args={'query': 'communication preference', 'person_id': pid},
                    answer=lambda read: read['results'][0]['text'])
    immutable = final_row(conn, result)
    metadata = json.loads(immutable['tool_calls'])
    assert result['answer'] == 'communication preference: phone' and result['sources'] == []
    assert 'uncited' in {warning['code'] for warning in result['source_warnings']}
    assert len(metadata['source_dependencies']) == 1
    assert metadata['source_dependencies'][0]['entity_id'] == mid
    assert metadata['source_dependencies'][0]['text'] == 'communication preference: phone'
    assert any(message['content'] == result['answer'] for message in harness._history_messages(conn, api.cid))

    change(conn, mid, text='communication preference: email')

    history = harness._history_messages(conn, api.cid)
    assert all('communication preference: phone' not in message['content'] for message in history)
    assert final_row(conn, result) == immutable
    assert api.client.get(source_url(api, result, {'id': 'S1'})).status_code == 404


@pytest.mark.parametrize('markdown', [
    '`[[S1]]`',
    '```text\n[[S1]]\n```',
    '    [[S1]]',
    '[See [[S1]]](https://example.invalid/source)',
    '[Source](https://example.invalid/[[S1]])',
    '<https://example.invalid/[[S1]]>',
    'https://example.com/[[S1]]',
    '<a href="https://example.com">[[S1]]</a>',
    '<code>[[S1]]</code>',
    '<pre>[[S1]]</pre>',
])
def test_code_or_link_markers_do_not_count_as_a_visible_answer_citation(api, conn, markdown):
    pid = seed_person(conn)
    seed_memory(conn, pid)

    result, _ = ask(api, args={'query': 'communication preference', 'person_id': pid},
                    answer=lambda read: markdown)

    assert result['sources'] == []
    assert 'uncited' in {warning['code'] for warning in result['source_warnings']}
    assert api.client.get(source_url(api, result, {'id': 'S1'})).status_code == 404


def test_visible_citation_survives_code_example_containing_an_unknown_marker(api, conn):
    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)

    def answer(read):
        sid = read['results'][0]['source_id']
        return f"{read['results'][0]['text']} [[{sid}]]\n\nExample: `[[S999]]`"

    result, _ = ask(api, args={'query': 'communication preference', 'person_id': pid}, answer=answer)

    assert len(result['sources']) == 1 and result['sources'][0]['entity_id'] == mid
    assert not {'uncited', 'invalid_citation'} & {warning['code'] for warning in result['source_warnings']}
    assert source_detail(api, result)['state']['status'] == 'current'


@pytest.mark.parametrize('context_budget', [200, 1000, 2400])
def test_initial_profile_cards_and_sources_share_the_context_budget(api, conn, context_budget):
    from app.agent.source_registry import SourceRegistry

    pid = seed_person(conn)
    repo.update_value_read(conn, pid, 'Long profile assessment ' * 80, None)
    for index in range(4):
        eid = repo.add_evidence(conn, 'text', excerpt='field evidence ' * 20)
        repo.upsert_field(conn, pid, 'trait.' + str(index), 'long field value ' * 15, 1., eid,
                          valid_from=1000 + index)
    seed_memory(conn, pid, 'communication preference ' * 50)
    settings = harness.get_settings().model_copy(update={'ctx_budget_chars': context_budget})
    api.monkeypatch.setattr(harness, 'get_settings', lambda: settings)
    registry = SourceRegistry(conn)

    content = harness._context_blocks(conn, [pid], registry)

    assert len(content) <= context_budget // 2
    blocks = [json.loads(line) for line in content.split('\n\n') if line]
    exposed = {block['id'] for block in blocks if 'id' in block}
    assert set(registry.sources) == exposed
    profiles = [dependency for dependency in registry.dependencies if dependency['type'] == 'profile_context']
    assert len(profiles) == sum('profile_context' in block for block in blocks)
    for dependency in profiles:
        assert any(block.get('profile_context') == dependency['text'] for block in blocks)


def test_card_only_read_tracks_profile_change_for_history_without_rewriting_answer(api, conn):
    pid = seed_person(conn)
    repo.update_value_read(conn, pid, 'original profile assessment', None)
    conn.commit()
    result, model = ask(api, tool='get_person', args={'person_id': pid}, answer=lambda read: read['card'])
    immutable = final_row(conn, result)
    assert model.read['sources'] == [] and result['sources'] == []
    dependencies = json.loads(immutable['tool_calls'])['source_dependencies']
    assert dependencies == [{'type': 'profile_context', 'entity_id': pid, 'text': model.read['card']}]

    repo.update_value_read(conn, pid, 'corrected profile assessment', None)
    conn.commit()

    assert all('original profile assessment' not in message['content']
               for message in harness._history_messages(conn, api.cid))
    assert final_row(conn, result) == immutable


def test_card_only_change_during_generation_prevents_final_answer(api, conn):
    pid = seed_person(conn)
    repo.update_value_read(conn, pid, 'original profile assessment', None)
    conn.commit()

    def mutate(read):
        assert read['sources'] == []
        repo.update_value_read(conn, pid, 'corrected profile assessment', None)
        conn.commit()

    with pytest.raises(SourceChangedError):
        ask(api, tool='get_person', args={'person_id': pid}, answer=lambda read: read['card'], before_answer=mutate)

    assert not conn.execute("SELECT 1 FROM message WHERE conversation_id=? AND role='assistant' "
                            "AND json_valid(tool_calls) AND json_type(tool_calls)='object'", (api.cid,)).fetchone()


def test_rereading_changed_profile_does_not_discard_the_earlier_dependency(api, conn):
    from app.agent.source_registry import SourceRegistry
    from app.memory.compile_card import build_card

    pid = seed_person(conn)
    registry = SourceRegistry(conn)
    registry.register_profile(pid, build_card(conn, pid))
    repo.update_value_read(conn, pid, 'changed during tool rounds', None)
    conn.commit()
    registry.register_profile(pid, build_card(conn, pid))

    assert len(registry.dependencies) == 2
    with pytest.raises(SourceChangedError):
        registry.save_answer(lambda: pytest.fail('stale dependency must prevent saving'))


def test_latin_person_name_does_not_scope_annual_or_expose_its_profile(api, conn):
    ann, bob = seed_person(conn, 'Ann'), seed_person(conn, 'Bob')
    ann_mid, _ = seed_memory(conn, ann)
    bob_mid, _ = seed_memory(conn, bob, 'Annual planning communication schedule')
    repo.update_value_read(conn, ann, 'private Ann profile assessment', None)
    conn.commit()

    result, model = ask(api, args={'query': 'Annual planning'}, text='Annual planning')

    assert model.read['scope_inferred'] is False and model.read['scoped_person'] is None
    assert result['sources'][0]['entity_id'] == bob_mid
    assert all(dependency['entity_id'] != ann for dependency in result['source_dependencies'])
    assert harness._detect_mentioned_person_ids(conn, 'Annual planning') == []
    assert harness._detect_mentioned_person_ids(conn, 'Ask Ann about communication') == [ann]
    explicit, model = ask(api, args={'query': 'Ann communication preference'})
    assert model.read['scope_inferred'] is True and model.read['scoped_person']['id'] == ann
    assert explicit['sources'][0]['entity_id'] == ann_mid


@pytest.mark.parametrize('short,long', [('Ann', 'Ann Lee'), ('林舟', '林舟舟')])
def test_separate_mentions_of_nested_names_keep_both_people_in_scope(api, conn, short, long):
    first, second = seed_person(conn, short), seed_person(conn, long)
    query = short + ' 与 ' + long + ' 的沟通偏好'

    result = harness.tools.execute_tool(conn, 'query_memory', {'query': query})

    assert result['ok'] and result['scoped_person'] is None and not result['scope_inferred']
    assert set(harness._detect_mentioned_person_ids(conn, query)) == {first, second}
    single = harness.tools.execute_tool(conn, 'query_memory', {'query': long})
    assert single['scoped_person']['id'] == second


def test_multiple_read_tools_consume_one_cumulative_source_budget(api, conn):
    pid = seed_person(conn)
    for index in range(10):
        seed_memory(conn, pid, f'communication preference {index}: ' + 'long background ' * 100)
    settings = harness.get_settings().model_copy(update={'ctx_budget_chars': 8000})
    api.monkeypatch.setattr(harness, 'get_settings', lambda: settings)
    calls, reads = [], []

    def chat(messages, **kwargs):
        calls.append(deepcopy(messages))
        if len(calls) == 1:
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'read-' + str(index), 'type': 'function', 'function': {
                    'name': 'query_memory', 'arguments': json.dumps({
                        'query': 'communication preference', 'person_id': pid, 'top_k': 8})}}
                for index in range(3)]}
        reads.extend(json.loads(message['content']) for message in messages if message['role'] == 'tool')
        source = reads[0]['results'][0]['source']
        return {'role': 'assistant', 'content': source['text'] + ' [[' + source['id'] + ']]'}

    api.monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    response = api.client.post(f'/api/conversations/{api.cid}/messages', json={'text': 'Check communication preference'})

    assert response.status_code == 200, response.text
    result = response.json()
    assert len(reads) == 3 and 0 < len(reads[0]['results']) < 8
    assert reads[0]['truncated'] is True
    assert all(read.get('truncated') and not read.get('results') for read in reads[1:])
    exposed = {hit['id'] for read in reads for hit in read.get('results', [])}
    assert {dependency['entity_id'] for dependency in result['source_dependencies']} == exposed
    assert len(result['sources']) == 1 and result['sources'][0]['entity_id'] in exposed
    assert sum(len(message['content']) for message in calls[1] if message['role'] == 'tool') <= 8000


def test_source_ids_are_bound_to_the_saved_assistant_and_conversation(api, conn):
    pid = seed_person(conn)
    seed_memory(conn, pid)
    result, _ = ask(api, args={'query': 'communication preference', 'person_id': pid})
    path = source_url(api, result)
    other_cid = repo.create_conversation(conn, 'Other')
    conn.commit()
    assert api.client.get(path.replace(api.cid, other_cid)).status_code == 404
    assert api.client.get(source_url(api, result, {'id': 'S98765'})).status_code == 404
    user_id = conn.execute("SELECT id FROM message WHERE conversation_id=? AND role='user'", (api.cid,)).fetchone()[0]
    assert api.client.get(path.replace(result['assistant_message_id'], user_id)).status_code == 404
    tool_assistant_id = conn.execute("SELECT id FROM message WHERE conversation_id=? AND role='assistant' "
        "AND id!=?", (api.cid, result['assistant_message_id'])).fetchone()[0]
    assert api.client.get(path.replace(result['assistant_message_id'], tool_assistant_id)).status_code == 404


def test_model_cannot_register_a_fabricated_source_without_reading_storage(api, conn):
    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)
    api.monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=lambda *args, **kwargs: {
        'role': 'assistant', 'content': 'Fabricated source for ' + mid + ' [[S1]]'}))

    response = api.client.post(f'/api/conversations/{api.cid}/messages', json={'text': 'Answer a general question'})

    assert response.status_code == 200, response.text
    result = response.json()
    assert result['sources'] == [] and '[[S1]]' not in result['answer']
    assert 'invalid_citation' in {warning['code'] for warning in result['source_warnings']}
    assert api.client.get(source_url(api, result, {'id': 'S1'})).status_code == 404


def test_repeated_real_citation_is_deduplicated_and_unknown_marker_is_removed(api, conn):
    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)

    def answer(read):
        sid = read['results'][0]['source_id']
        return f'First citation [[{sid}]], repeated [[{sid}]], unknown [[S999]].'

    result, _ = ask(api, args={'query': 'communication preference', 'person_id': pid}, answer=answer)

    assert len(result['sources']) == 1 and result['sources'][0]['entity_id'] == mid
    assert result['answer'].count('[[S1]]') == 2 and '[[S999]]' not in result['answer']
    assert 'invalid_citation' in {warning['code'] for warning in result['source_warnings']}
    assert 'uncited' not in {warning['code'] for warning in result['source_warnings']}


@pytest.mark.parametrize('operation', ['edit', 'withdraw', 'move', 'archive'])
def test_source_changes_on_independent_connection_during_answer_prevent_final_persistence(api, conn, operation):
    pid, target = seed_person(conn), seed_person(conn, 'Chen')
    mid, _ = seed_memory(conn, pid)
    database = conn.execute('PRAGMA database_list').fetchone()['file']

    def mutate(read):
        assert read['results'][0]['source']['text'] == 'communication preference: phone'
        other = db._connect(database)
        try:
            if operation == 'archive':
                repo.archive_person(other, pid)
                other.commit()
            elif operation == 'move':
                change(other, mid, person_id=target)
            elif operation == 'edit':
                change(other, mid, text='communication preference: email')
            else:
                change(other, mid, action='withdraw')
        finally:
            other.close()

    with pytest.raises(SourceChangedError, match='引用资料在生成期间发生变化'):
        ask(api, args={'query': 'communication preference', 'person_id': pid}, before_answer=mutate)

    final_messages = [row for row in repo.list_messages(conn, api.cid) if row['role'] == 'assistant'
                      and row['tool_calls'] and isinstance(json.loads(row['tool_calls']), dict)]
    assert final_messages == []
    assert conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0] == 1


def test_change_to_an_exposed_but_uncited_source_also_prevents_stale_answer(api, conn):
    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)
    eid = repo.add_evidence(conn, 'text', excerpt='Shanghai')
    repo.upsert_field(conn, pid, 'city', 'Shanghai', 1., eid, valid_from=2000)
    conn.commit()

    def mutate(read):
        assert any(source['entity_id'] == mid for source in read['sources'])
        change(conn, mid, text='communication preference: email')

    def answer(read):
        city = next(source for source in read['sources'] if source['type'] == 'field')
        return 'City: ' + city['text'] + ' [[' + city['id'] + ']]'

    with pytest.raises(SourceChangedError):
        ask(api, tool='get_person', args={'person_id': pid}, answer=answer, before_answer=mutate)

    assert not conn.execute("SELECT 1 FROM message WHERE conversation_id=? AND role='assistant' "
                            "AND json_valid(tool_calls) AND json_type(tool_calls)='object'", (api.cid,)).fetchone()


def test_long_history_summary_invalidates_on_source_change_and_then_reuses_cache(api, conn):
    from app.agent.memory_sources import memory_source

    pid = seed_person(conn)
    mid, _ = seed_memory(conn, pid)
    initial, _ = ask(api, args={'query': 'communication preference', 'person_id': pid})
    immutable = final_row(conn, initial)
    for index in range(20):
        repo.add_message(conn, api.cid, 'user' if index % 2 == 0 else 'assistant',
                         f'Long history item {index}: ' + 'context ' * 10)
    conn.commit()
    settings = harness.get_settings().model_copy(update={'ctx_budget_chars': 80})
    api.monkeypatch.setattr(harness, 'get_settings', lambda: settings)
    summary_calls = []

    def summarize(messages, **kwargs):
        summary_calls.append(deepcopy(messages))
        return {'content': f'Summary generation {len(summary_calls)}'}

    api.monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=summarize))
    before = harness._history_messages(conn, api.cid)
    assert len(summary_calls) == 1
    assert 'Summary generation 1' in before[0]['content']
    assert 'communication preference: phone' in summary_calls[0][-1]['content']
    signature_before = repo.get_kv(conn, f'summ_sources_{api.cid}')
    assert harness._history_messages(conn, api.cid) == before
    assert len(summary_calls) == 1

    change(conn, mid, text='communication preference: email')

    after = harness._history_messages(conn, api.cid)
    assert len(summary_calls) == 2
    assert 'Summary generation 2' in after[0]['content']
    assert 'communication preference: phone' not in summary_calls[1][-1]['content']
    assert repo.get_kv(conn, f'summ_sources_{api.cid}') != signature_before
    assert final_row(conn, initial) == immutable
    assert harness._history_messages(conn, api.cid) == after
    assert len(summary_calls) == 2

    current = memory_source(conn, mid) | {'id': 'S1'}
    repo.add_message(conn, api.cid, 'assistant', 'communication preference: email [[S1]]',
                     json.dumps({'sources': [current], 'source_warnings': []}))
    conn.commit()
    harness._history_messages(conn, api.cid)
    assert len(summary_calls) == 2
    assert final_row(conn, initial) == immutable


def test_legacy_summary_without_source_signature_is_not_regenerated(api, conn):
    for index in range(20):
        repo.add_message(conn, api.cid, 'user' if index % 2 == 0 else 'assistant',
                         f'Legacy history {index}: ' + 'context ' * 10)
    repo.set_kv(conn, f'summ_count_{api.cid}', '8')
    repo.set_kv(conn, f'summ_text_{api.cid}', 'Existing legacy summary')
    conn.commit()
    settings = harness.get_settings().model_copy(update={'ctx_budget_chars': 80})
    api.monkeypatch.setattr(harness, 'get_settings', lambda: settings)
    api.monkeypatch.setattr(harness, 'get_llm', lambda: pytest.fail('valid legacy cache should be reused'))

    messages = harness._history_messages(conn, api.cid)

    assert 'Existing legacy summary' in messages[0]['content']
    assert repo.get_kv(conn, f'summ_text_{api.cid}') == 'Existing legacy summary'


def test_offline_retrieval_evaluation_reports_cases_and_detects_wrong_semantic_expectations():
    fixture = load_fixture()
    report = evaluate(fixture)
    assert report['summary']['cases'] == len(fixture['cases'])
    assert report['summary']['pass_rate'] == report['summary']['recall_at_k'] == 1.
    assert report['summary']['mrr'] == report['summary']['source_snapshot_accuracy'] == 1.
    assert all(case['passed'] and not case['failures'] for case in report['details'])
    assert report['limitations']

    wrong = deepcopy(fixture)
    wrong['cases'] = [next(case for case in wrong['cases'] if case['id'] == 'edited-email-preference')]
    wrong['cases'][0]['expect']['contains'] = ['电话']
    wrong['cases'][0]['expect']['ids'] = ['namesake']
    failed = evaluate(wrong)
    assert failed['summary']['pass_rate'] == failed['summary']['recall_at_k'] == 0.
    assert failed['summary']['mrr'] == 0.
    assert failed['details'][0]['failures']


def test_retrieval_evaluator_rejects_real_evidence_id_with_forged_excerpt(monkeypatch):
    from eval import memory_retrieval

    fixture = load_fixture()
    fixture['cases'] = [next(case for case in fixture['cases'] if case['id'] == 'initial-phone-preference')]
    execute = memory_retrieval.tools.execute_tool

    def corrupt_source(*args, **kwargs):
        result = execute(*args, **kwargs)
        result['results'][0]['source']['evidence']['excerpt'] = 'forged evidence text'
        return result

    monkeypatch.setattr(memory_retrieval.tools, 'execute_tool', corrupt_source)
    report = evaluate(fixture)

    assert report['summary']['recall_at_k'] == 1.
    assert report['summary']['pass_rate'] == report['summary']['source_snapshot_accuracy'] == 0.
    assert 'source evidence content differs from the stored evidence' in report['details'][0]['failures']
