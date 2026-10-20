"""Server presentation boundaries, not a model semantic accuracy estimate."""
import json

import pytest

from app.agent.answer_contract import presentation_contract
from app.agent.answer_review import review_answer
from app.agent.source_registry import _prose_references
from app.agent import harness, turn_intent
from app.agent.source_registry import SourceChangedError
from app.memory import memory_corrections
from app.store import repo
from app.util import new_id
SOURCES = [{
    'id': 'S1', 'type': 'memory', 'entity_id': 'm-lin',
    'person': {'id': 'p-lin', 'display_name': '林舟'}, 'text': '项目事项先发邮件。',
    'ts': 1791511200000, 'temporal_status': 'historical_background',
    'record_context': {'status': 'active', 'recorded_at': 1791511200000, 'last_user_edit_at': None},
}]


class Reviewer:
    def __init__(self, response):
        self.response = response

    def chat(self, *args, **kwargs):
        return self.response


def segment(kind='fact', text='项目事项先发邮件。', ids=None):
    return {'kind': kind, 'text': text, 'source_ids': ['S1'] if ids is None else ids}


def proposed(segments, response_type):
    return {'answer_type': response_type, 'segments': segments, 'issues': []}


def review(segments, *, question='林舟记录里怎么写的？', response_type='factual', raw=None):
    model = Reviewer({'content': raw or json.dumps(proposed(segments, response_type), ensure_ascii=False)})
    result = review_answer(model, question, '她已经改成私事打电话，并已保存。', SOURCES,
                           context={'presentation': presentation_contract(question)})
    return result


def test_fabricated_factual_rewrite_is_replaced_by_registered_record_and_dates():
    result = review([segment(text='林舟今年47岁，历史从未编辑。')])
    assert result['metadata']['status'] == 'completed'
    assert '47岁' not in result['answer']
    assert SOURCES[0]['text'] in result['answer']
    assert '人工编辑' in result['answer']
    assert _prose_references(result['answer']) == ['S1']
    assert result['metadata']['rendered_segments'] == [segment('source_record', '', ['S1'])]
    assert result['metadata']['semantic_verified'] is False


def test_query_cannot_keep_fabricated_facts_inside_advice_or_clarification():
    result = review([segment('advice', '她已换公司，所以改用电话。', []),
                     segment('clarification', '她在10月1日已改过，是否还要再改？', []),
                     segment('source_record', '', ['S1'])], response_type='mixed')
    assert result['metadata']['status'] == 'completed'
    assert '换公司' not in result['answer'] and '10月1日' not in result['answer']
    assert '她' not in result['answer']
    assert result['metadata']['suppressed_segment_kinds'] == ['advice']


def test_query_does_not_fall_back_to_unreviewed_draft_or_dump_all_background():
    result = review([], raw='bad json')
    assert result['metadata']['status'] == 'unavailable'
    assert '请重试回答' in result['answer']
    assert '私事' not in result['answer'] and '已保存' not in result['answer']
    assert _prose_references(result['answer']) == []


def test_missing_answer_does_not_show_unselected_person_background():
    result = review([segment('no_answer', '', [])], question='林舟什么血型？', response_type='no_answer')
    assert result['answer'] == '当前可核查资料不足以回答这个问题。'
    assert _prose_references(result['answer']) == []


def test_explicit_draft_retains_requested_text_beside_server_rendered_sources():
    result = review([segment('source_record', '', ['S1']),
                     segment('draft', '你好，想先确认方便联系的渠道。', [])],
                    question='帮我起草一段给林舟的沟通消息。', response_type='mixed')
    assert result['metadata']['status'] == 'completed'
    assert '你好，想先确认方便联系的渠道。' in result['answer']
    assert '生成内容' in result['answer']
    assert SOURCES[0]['text'] in result['answer']


@pytest.mark.parametrize('question,expected', [
    ('林舟之前怎么说的？', False), ('对照这些记录的适用范围。', False),
    ('她什么时候改过？', False), ('帮我起草一条消息，再记下沟通结果。', True),
    ('给我建议如何准备下一次沟通。', True), ('怎么跟进比较好？', True),
    ('不要建议，只查林舟的记录。', False), ('Draft a short message for Lin.', True),
    ('我想倾诉，能陪我聊聊吗？', True), ('你好！', True), ('你好，敏姐的血型？', False),
    ('太长了，精简到三句话。', True), ('把上一段压缩为一条短信。', True),
    ('再短一点，并列出发送前要做的步骤。', True), ('请翻译成英文。', True),
    ('Make it shorter and list what to check before sending.', True),
    ('林舟的记录是否被缩短过？', False), ('这条记录有没有被改写过？', False),
    ('改写后的记录是何时放进系统的？', False),
    ('只查他的公司，再给我一个沟通建议。', True),
    ('不用给我建议，只查已有记录。', False), ('请不要起草消息，只查原文。', False),
    ('Please do not rewrite this, show the record.', False),
    ('请将上面的回答润色一下。', True), ('给个建议，别把未问到的安排当定了。', True),
    ('翻译稿是哪天保存的？', False), ('缩短这个操作是何时做的？', False),
    ('把之前翻译过的内容查出来。', False), ('把这条改写过的时间查出来。', False),
    ('他当时写给我的内容是什么？', False),
    ('不要起草但请给我一个沟通建议。', True), ('不必写长文但给我一点建议。', True),
    ('太长了，缩成两句，再提醒我发送前应该补哪一步。', True),
    ('保留原意，压到三十字以内，语气随和一点；然后提醒我发之前还有什么没补。', True),
    ('压成一条短信。', True), ('缩到40个字以内。', True),
    ('语气正式一些。', True), ('把口吻改得随和一点。', True),
    ('提醒我发送前还需要补什么。', True), ('列出发送之前的准备步骤。', True),
    ('记录中的“压到三十字”是什么时候写的？', False),
    ('他上次提醒我什么？', False), ('语气正式的原文是哪天保存的？', False),
])
def test_generation_permission_does_not_follow_write_intent(question, expected):
    assert presentation_contract(question)['generation_requested'] is expected


def test_whole_json_fence_and_type_mismatch_do_not_discard_valid_segments():
    value = proposed([segment('source_record', '', ['S1'])], 'mixed')
    result = review([], raw='```json\n' + json.dumps(value) + '\n```')
    meta = result['metadata']
    assert meta['status'] == 'completed'
    assert meta['response_envelope'] == 'json_fence'
    assert meta['declared_answer_type'] == 'mixed'
    assert meta['answer_type'] == 'factual'
    assert meta['answer_type_normalized'] is True


def test_freeform_write_claim_cannot_be_promoted_to_a_receipt():
    result = review([segment('receipt', '已保存', [])])
    assert result['metadata']['status'] == 'unavailable'
    assert '已保存' not in result['answer']


def test_generated_html_cannot_hide_the_following_server_record():
    from markdown_it import MarkdownIt
    result = review([segment('draft', '<div hidden class="hide">未闭合生成正文', []),
                     segment('source_record', '', ['S1'])], question='起草一段给林舟的话。', response_type='mixed')
    assert result['metadata']['status'] == 'completed'
    html = MarkdownIt('gfm-like').render(result['answer'])
    assert '<div hidden' not in html
    assert '&lt;div hidden' in html
    assert _prose_references(result['answer']) == ['S1']


def setup_people(conn, monkeypatch):
    people = []
    for name, text in [('林舟', '项目事项先发邮件。'), ('陈岚', '仅周三午休方便接电话。')]:
        pid = repo.create_person(conn, name, status='active')
        mid = repo.add_memory(conn, 'semantic', text, pid, .95, None)
        people.append((pid, mid, text))
    cid = repo.create_conversation(conn, 'Evidence integration')
    conn.commit()
    monkeypatch.setattr(turn_intent, 'classify_turn', lambda text, **kw: turn_intent.validate_intent({
        'intent': 'query', 'record_spans': [], 'reason': 'Read existing records.'}, text))
    return cid, people


class StreamingModel:
    def __init__(self, selected_text, before_review=None):
        self.selected_text, self.before_review = selected_text, before_review

    def chat_stream(self, *args, on_delta, **kwargs):
        on_delta('她现在已改用电话，已保存。')
        return {'role': 'assistant', 'content': '她现在已改用电话，已保存。'}

    def chat(self, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        source = next(s for s in payload['sources'] if s['text'] == self.selected_text)
        if self.before_review:
            self.before_review()
        return {'content': json.dumps(proposed([segment('source_record', '', [source['id']])], 'factual'))}


def test_first_query_and_explicit_person_switch_render_only_selected_person_without_draft_delta(conn, monkeypatch):
    cid, people = setup_people(conn, monkeypatch)
    events = []
    for (pid, _, text), question in zip(people, ['林舟的沟通记录是什么？', '换成陈岚，查沟通记录。']):
        monkeypatch.setattr(harness, 'get_llm', lambda: StreamingModel(text))
        result = harness.run_turn(conn, cid, question, emit=lambda kind, data: events.append((kind, data)))
        assert result['conversation_scope']['person_ids'] == [pid]
        assert result['answer_review']['status'] == 'completed'
        assert result['answer_review']['semantic_verified'] is False
        assert len(result['sources']) == 1 and result['sources'][0]['person']['id'] == pid
        assert text in result['answer']
        assert '已保存' not in result['answer']
        stored = conn.execute('SELECT content FROM message WHERE id=?', (result['assistant_message_id'],)).fetchone()[0]
        assert stored == result['answer']
    assert not any(kind == 'delta' for kind, _ in events)


def test_sources_changed_during_review_cannot_be_saved_even_when_card_was_valid(conn, monkeypatch):
    cid, people = setup_people(conn, monkeypatch)
    _, mid, text = people[0]

    def withdraw():
        memory_corrections.correct_memory(conn, mid, {
            'action': 'withdraw', 'expected_revision': 0, 'request_id': new_id()})

    monkeypatch.setattr(harness, 'get_llm', lambda: StreamingModel(text, withdraw))
    with pytest.raises(SourceChangedError):
        harness.run_turn(conn, cid, '林舟的沟通记录是什么？')
    assert conn.execute("SELECT COUNT(*) FROM message WHERE conversation_id=? AND role='assistant'", (cid,)).fetchone()[0] == 0
