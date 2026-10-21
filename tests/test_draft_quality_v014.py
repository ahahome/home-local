"""Draft task shape and side-effect boundaries, never model accuracy claims."""
import json
from types import SimpleNamespace

import pytest

from app.agent import followup, harness, turn_intent
from app.agent.answer_contract import is_draft_edit_followup, is_text_edit_request, presentation_contract
from app.agent.answer_review import review_answer
from app.agent.source_registry import _prose_references
from app.jobs import tasks
from app.store import repo


SOURCE = {
    'id': 'S1', 'type': 'memory', 'entity_id': 'memory-contact-order',
    'person': {'id': 'person-example', 'display_name': '许岚'},
    'text': '讨论定制时先邮件给数量和交期，确认清单后再约电话。',
    'record_context': {'status': 'active'},
}


def segment(kind, text='', ids=None):
    return {'kind': kind, 'text': text, 'source_ids': ids or []}


def review(question, segments, *, draft='给我准备一封邮件。'):
    payload = {'answer_type': 'mixed', 'segments': segments, 'issues': []}
    llm = SimpleNamespace(chat=lambda *a, **kw: {'content': json.dumps(payload, ensure_ascii=False)})
    return review_answer(llm, question, draft, [SOURCE], context={
        'presentation': presentation_contract(question),
        'confirmed_identity': {'status': 'single', 'people': [
            {'id': 'person-example', 'name': '许岚', 'aliases': []}]},
    })


@pytest.mark.parametrize('question', [
    '拟一封询问定制价格的短邮件，数量和交期我还没整理好。',
    '帮我写给她，提醒我发送之前还要补什么。',
    '太长了，精简成两句，再列出发送前要补的信息。',
    '保留原意，压到三十字以内，语气随和一点。',
    'Please write a short email to her.',
    'Translate this message into English.',
    '帮我起封邮件。',
    '帮我给她起封邮件。',
    '照她的沟通要求帮我起封邮件。',
    '请拟封信。',
    '起条消息。',
    '改成两句话就够了，别太正式，末尾另列我点发送之前要补齐的项目。',
    '改为三行。',
    '将上面的草稿改到五十字。',
])
def test_explicit_drafting_and_revision_require_actual_draft_shape(question):
    contract = presentation_contract(question)
    assert contract['generation_requested'] is True
    assert contract['draft_requested'] is True
    assert contract['semantic_verified'] is False


@pytest.mark.parametrize('question', [
    '许岚以前写给我的内容是什么？',
    '这条翻译稿是什么时候保存的？',
    '请不要起草，只查已保存的记录。',
    '给我一个准备沟通的建议。',
    '你好！',
])
def test_readback_advice_and_greeting_do_not_require_draft(question):
    assert presentation_contract(question)['draft_requested'] is False


@pytest.mark.parametrize('question', [
    'When did she write an email?',
    'Did she write a letter?',
    '查她 write an email 这段记录的日期。',
    'When did I write a short email to her?',
    'write an email 这段记录是哪天保存的？',
    '那封邮件是谁帮我起的？',
    '之前帮我起封邮件的记录是哪天保存的？',
    '她以前帮我起封邮件。',
    '什么时候改成两句话的？',
    '查一下改成三行的记录。',
    '不要起封邮件，只查记录。',
    '不用改成两句话，只给我看原文。',
])
def test_english_writing_readback_does_not_open_generation(question):
    contract = presentation_contract(question)
    assert contract['generation_requested'] is False
    assert contract['draft_requested'] is False


@pytest.mark.parametrize('question', [
    '改成两句', '翻译成英文', '语气随和一点', '把这封邮件缩短',
    '把上面的草稿翻译成英文。', '请将上一段精简一点。',
    '改成两句话就够了，别太正式，末尾另列我点发送之前要补齐的项目。',
    '保留原意，压到三十字以内，语气随和一点；然后提醒我发之前还有什么没补。',
    '太长了，改成两句。', '不够自然，语气随和一点。',
])
def test_closed_draft_edits_can_refer_only_to_immediate_previous_output(question):
    assert is_draft_edit_followup(question) is True


@pytest.mark.parametrize('question', [
    '帮我起封邮件。', '起草一条新消息。', '给我一个沟通建议。',
    '把王琪的邮件缩短。', '把新的邮件缩短。', '把我的邮件翻译成英文。',
    '把这封邮件缩短：王琪你好。', '改成两句。下面是新文稿：王琪你好。',
    '改成两句，王琪的近况是什么？', '换个人，改成两句。',
    '改成两句的记录何时保存？', '查询现有记录。', '保留原意。',
    '发送前列出还要补齐的项目。', '不要改成两句。',
    '太长了。', '不够自然。', '太长了，给王琪起草邮件。',
    '太长了，改成两句，新文稿：王琪你好。',
])
def test_new_objects_pasted_text_and_nonediting_queries_never_borrow_draft_recipient(question):
    assert is_draft_edit_followup(question) is False


@pytest.mark.parametrize('question', [
    '把王琪的邮件缩短。', '把这封邮件缩短：王琪说她会过来。',
    '改成两句，王琪说她会过来。', '把给她的邮件翻译成英文。',
    '太长了，把这封邮件缩短：王琪说她会过来。',
    '把这封有日期的邮件缩短：王琪说她会过来。',
    '把这封有时间的邮件缩短：王琪说她会过来。',
])
def test_wide_edit_detector_routes_new_object_and_pasted_pronoun_through_strict_boundary(question):
    assert is_text_edit_request(question) is True
    assert is_draft_edit_followup(question) is False


@pytest.mark.parametrize('question', [
    '她的偏好呢？', '帮我给她起封邮件。', '帮我给她起草一封邮件。',
    '给我一个建议。', '她的翻译稿是什么时候保存的？',
    '请不要改成两句，只查记录。',
])
def test_wide_edit_detector_does_not_include_new_drafts_advice_or_readback(question):
    assert is_text_edit_request(question) is False


def test_wide_scope_gate_does_not_grant_generation_even_for_readback_shaped_like_edit_command():
    question = '改成两句的记录是什么时候保存的？'
    assert is_text_edit_request(question) is True
    assert is_draft_edit_followup(question) is False
    contract = presentation_contract(question)
    assert contract['generation_requested'] is False and contract['draft_requested'] is False


@pytest.mark.parametrize('segments', [
    [segment('source_record', ids=['S1'])],
    [segment('advice', '先把所需资料整理好。')],
    [segment('no_answer')],
])
def test_records_or_advice_alone_do_not_count_as_requested_draft(segments):
    result = review('起草一封邮件，数量和交期还没定。', segments)
    assert result['metadata']['status'] == 'unavailable'
    assert result['metadata']['unavailable_reason'] == 'requested_draft_missing'
    assert '没有生成所请求的草稿' in result['answer']
    assert '请重试回答' in result['answer']
    assert _prose_references(result['answer']) == []


def test_placeholder_draft_preserves_sources_and_separate_sending_checklist():
    text = ('主题：定制报价咨询\n许岚，你好：\n'
            '想咨询【待补：数量】份定制，期望交期为【待补：交期】。'
            '请先确认清单，确认后我们再约电话。\n'
            '【待补：规格清单内容或附件】')
    checklist = '发送前：补齐数量、交期和规格清单；核对附件后再发送。'
    result = review('按记录拟一封短邮件，数量和交期还没整理好。', [
        segment('source_record', ids=['S1']), segment('draft', text), segment('advice', checklist)])
    meta = result['metadata']
    assert meta['status'] == 'completed'
    assert meta['generation_task'] == {
        'draft_requested': True, 'draft_present': True, 'clarification_present': False,
        'status': 'draft_provided', 'method': 'segment_shape_only', 'semantic_verified': False,
    }
    assert text in result['answer'] and checklist in result['answer']
    assert SOURCE['text'] in result['answer']
    assert _prose_references(result['answer']) == ['S1']
    assert meta['semantic_verified'] is False


def test_genuinely_missing_original_text_keeps_specific_question_and_audits_incomplete_task():
    question = '请把那封邮件翻译成英文。'
    needed = '请贴出要翻译的邮件原文。'
    result = review(question, [segment('clarification', needed)])
    assert result['metadata']['status'] == 'completed'
    assert needed in result['answer']
    assert '请补充要查询的人物' not in result['answer']
    assert result['metadata']['generation_task']['status'] == 'needs_input'
    assert result['metadata']['generation_task']['draft_present'] is False


def test_generation_clarification_does_not_silently_replace_empty_question_with_generic_prompt():
    result = review('起草一封邮件。', [segment('clarification')])
    assert result['metadata']['unavailable_reason'] == 'empty_generation_clarification'


def test_specific_generated_clarification_is_literal_and_cannot_hide_source_record():
    from markdown_it import MarkdownIt
    text = '请贴出原文。<div hidden>补充内容'
    result = review('请翻译成英文。', [
        segment('clarification', text), segment('source_record', ids=['S1'])])
    assert result['metadata']['status'] == 'completed'
    html = MarkdownIt('gfm-like').render(result['answer'])
    assert '<div hidden>' not in html and '&lt;div hidden&gt;' in html
    assert _prose_references(result['answer']) == ['S1']


def test_clarification_cannot_create_fake_source_or_keep_facts_in_readonly_query():
    result = review('请起草邮件。', [segment('clarification', '请确认[[S999]]里的材料。')])
    assert result['metadata']['unavailable_reason'] == 'inline_reference_not_allowed'
    readonly = review('许岚记录原文是什么？', [
        segment('clarification', '她已经整理好所有材料，是否立即发送？'),
        segment('source_record', ids=['S1'])])
    assert readonly['metadata']['status'] == 'completed'
    assert '已经整理好' not in readonly['answer']
    assert '请补充要查询的人物' in readonly['answer']


def test_prepared_email_with_material_placeholders_preserves_open_promise_and_never_counts_contact(
        conn, monkeypatch):
    pid = repo.create_person(conn, '许岚', status='active')
    eid = repo.add_evidence(conn, 'text', excerpt=SOURCE['text'])
    mid = repo.add_memory(conn, 'semantic', SOURCE['text'], pid, 1, eid)
    promise_id = repo.add_promise(conn, 'me', '发送规格清单', pid, evidence_id=eid)
    conn.commit()
    monkeypatch.setattr(tasks, 'start', lambda tid: None)
    task = followup.create_followup(conn, pid,
        '写一封定制报价邮件，数量、交期和清单还没有准备好。', 'draft-quality-1')
    fid = task['payload']['followup_id']
    sid = 'memory:' + mid
    draft = ('主题：定制报价咨询\n许岚，你好：\n'
             '数量：【待补：数量】；交期：【待补：交期】。\n'
             '【待补：规格清单内容或附件】\n确认清单后我们再约电话。\n\n'
             '发送前：补齐上述信息，核对实际附件后再发送。')
    calls = []

    def chat_json(messages, **kwargs):
        data = json.loads(messages[-1]['content'])
        calls.append(data)
        assert [p['id'] for p in data['brief']['open_promises']] == [promise_id]
        return {'draft': draft, 'source_ids': [sid]}

    monkeypatch.setattr(followup, 'get_llm', lambda: SimpleNamespace(chat_json=chat_json))
    before = {table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
              for table in ('interaction', 'memory_item', 'person_field', 'promise', 'pending_item')}
    record = followup.prepare_followup(conn, fid)
    assert record['status'] == 'ready' and record['draft'] == draft
    assert record['draft_source_ids'] == [sid]
    assert repo.open_promises(conn, pid)[0]['id'] == promise_id
    assert {table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            for table in before} == before
    # Retrying a ready preparation reuses its stored editable draft, rather
    # than treating the placeholder or checklist as a request to act.
    assert followup.prepare_followup(conn, fid)['draft'] == draft
    assert len(calls) == 1


def test_closed_chat_revision_rereads_recipient_sources_and_remains_readonly(conn, monkeypatch):
    pid = repo.create_person(conn, '许岚', status='active')
    other = repo.create_person(conn, '陈岚', status='active')
    eid = repo.add_evidence(conn, 'text', excerpt=SOURCE['text'])
    mid = repo.add_memory(conn, 'semantic', SOURCE['text'], pid, .95, eid)
    repo.add_memory(conn, 'episodic', '不应借编辑请求读取的其他人物私密记录。', other, .95, eid)
    cid = repo.create_conversation(conn, 'Closed revision integration')
    conn.commit()
    observed = []

    def classify(text, **kwargs):
        return turn_intent.validate_intent({
            'intent': 'query', 'record_spans': [], 'reason': '只起草或编辑，没有记事请求。'}, text)

    def chat(messages, **kwargs):
        if '本轮回答的资料编辑' not in messages[0]['content']:
            return {'content': '邮件正文：数量【待补】，交期【待补】。确认清单后再约电话。'}
        data = json.loads(messages[-1]['content'])
        scope = data['runtime_context']['read_scope']
        assert scope['person_ids'] == [pid]
        assert data['runtime_context']['presentation']['draft_requested'] is True
        source = next(s for s in data['sources'] if s.get('entity_id') == mid)
        assert source['text'] == SOURCE['text']
        assert '不应借编辑请求读取' not in json.dumps(data, ensure_ascii=False)
        observed.append(scope)
        return {'content': json.dumps({'answer_type': 'mixed', 'segments': [
            segment('source_record', ids=[source['id']]),
            segment('draft', '数量【待补】，交期【待补】。确认清单后再约电话。'),
            segment('advice', '发送前补齐数量和交期，并核对清单。')], 'issues': []}, ensure_ascii=False)}

    monkeypatch.setattr(harness, 'get_llm', lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(turn_intent, 'classify_turn', classify)
    monkeypatch.setattr(harness.extract, 'extract_from_text', lambda *a, **kw: pytest.fail('draft edits are readonly'))
    first = harness.run_turn(conn, cid, '许岚那边帮我起封邮件，数量和交期还没定。')
    assert first['answer_review']['generation_task']['draft_present'] is True
    second = harness.run_turn(conn, cid,
        '太长了，改成两句话就够了，别太正式，末尾另列我点发送之前要补齐的项目。')
    assert second['conversation_scope']['reason'] == 'inherited_draft_edit'
    assert second['conversation_scope']['draft_anchor_assistant_message_id'] == first['assistant_message_id']
    assert any(s.get('entity_id') == mid for s in second['source_dependencies'])
    assert all(s.get('person', {}).get('id') != other for s in second['source_dependencies'])
    assert second['answer_review']['generation_task']['draft_present'] is True
    assert first['saved'] == first['pending'] == second['saved'] == second['pending'] == []
    assert conn.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 2
    assert conn.execute('SELECT COUNT(*) FROM interaction').fetchone()[0] == 0
    assert len(observed) == 2
