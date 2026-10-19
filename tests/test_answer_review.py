import hashlib
import json

import pytest

from app.agent.answer_review import review_answer
from app.agent.source_registry import compact_source


SOURCES = [
    {'id': 'S1', 'type': 'memory', 'entity_id': 'm1', 'person': {'display_name': '林舟'},
     'text': '林舟说非紧急事项请发邮件。', 'temporal_status': 'historical_background'},
    {'id': 'S2', 'type': 'memory', 'entity_id': 'm2', 'person': {'display_name': '林舟'},
     'text': '林舟说沟通前先电话联系。', 'temporal_status': 'historical_background'},
]
DRAFT = '工作发邮件，私事打电话。'


class Reviewer:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def segment(kind='fact', text='已记录非紧急事项请发邮件。', ids=None):
    return {'kind': kind, 'text': text, 'source_ids': ['S1'] if ids is None else ids}


def proposed(segments=None, answer_type='factual', issues=None):
    return {'answer_type': answer_type, 'segments': [segment()] if segments is None else segments,
            'issues': [] if issues is None else issues}


def run(value, *, sources=None, draft=DRAFT, context=None):
    llm = Reviewer({'content': json.dumps(value, ensure_ascii=False)})
    result = review_answer(llm, '林舟怎么联系？', draft, SOURCES if sources is None else sources,
                           context=context)
    return result, llm


def codes(result):
    return {warning['code'] for warning in result['warnings']}


def assert_fallback(result, llm, draft=DRAFT):
    assert result['answer'] == draft
    assert result['metadata']['status'] == 'unavailable'
    assert result['metadata']['semantic_verified'] is False
    assert result['metadata']['revised_answer'] is None
    assert 'review_unavailable' in codes(result)
    assert len(llm.calls) == 1


def test_conflicting_records_can_be_cited_without_inventing_a_reconciliation():
    value = proposed([
        segment(text='一条记录写明非紧急事项发邮件。'),
        segment(text='另一条记录写明沟通前先电话联系。', ids=['S2']),
        segment('advice', '先确认这两条偏好各自适用的范围。', []),
    ], answer_type='mixed', issues=[{
        'code': 'conflict_resolution', 'quote': '工作发邮件，私事打电话。',
        'detail': '资料未说明工作与私事的区别，不能据此分配渠道。', 'source_ids': ['S1', 'S2'],
    }])

    result, llm = run(value)

    assert result['answer'] == ('一条记录写明非紧急事项发邮件。 [[S1]]\n\n'
                                '另一条记录写明沟通前先电话联系。 [[S2]]\n\n'
                                '建议：先确认这两条偏好各自适用的范围。')
    review = result['metadata']
    assert review['status'] == 'completed'
    assert review['semantic_verified'] is False
    assert review['original_answer'] == DRAFT
    assert review['revised_answer'] == result['answer']
    assert json.loads(review['raw_response']) == value
    assert review['issues'] == value['issues']
    payload = json.loads(llm.calls[0][0][1]['content'])
    assert payload['sources'] == [compact_source(source) for source in SOURCES]
    assert 'temporal_status' not in payload['sources'][0]
    assert '记录状态' in payload['sources'][0]
    assert review['input_sha256'] == hashlib.sha256(llm.calls[0][0][1]['content'].encode()).hexdigest()
    assert len(review['source_snapshot_sha256']) == 64
    assert len(llm.calls) == 1
    assert 'tools' not in llm.calls[0][1]


def test_no_answer_is_fixed_prose_without_irrelevant_citations():
    result, _ = run(proposed([segment('no_answer', '', [])], 'no_answer'))

    assert result['answer'] == '当前可核查资料不足以回答这个问题。'
    assert result['metadata']['answer_type'] == 'no_answer'
    assert result['metadata']['status'] == 'completed'
    assert result['warnings'] == []


def test_absent_sources_do_not_force_a_no_answer_to_cite_something():
    result, _ = run(proposed([segment('no_answer', '', [])], 'no_answer'), sources=[])

    assert result['metadata']['status'] == 'completed'
    assert '[[' not in result['answer']


def test_no_answer_label_cannot_smuggle_a_personal_fact_into_uncited_suppression():
    result, llm = run(proposed([segment('no_answer', '林舟的血型是O型。', [])], 'no_answer'))

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'no_answer_contains_content'


def test_factual_segment_cannot_hide_behind_a_no_answer_classification():
    result, llm = run(proposed([segment()], 'no_answer'))

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'inconsistent_answer_type'


def test_pure_advice_gets_an_explicit_prefix_without_fabricating_a_source():
    result, _ = run(proposed([segment('advice', '可以先发一条简短消息询问方便的渠道。', [])], 'advice'),
                    sources=[])

    assert result['answer'] == '建议：可以先发一条简短消息询问方便的渠道。'
    assert result['metadata']['status'] == 'completed'
    assert result['metadata']['semantic_verified'] is False

@pytest.mark.parametrize('value', [
    proposed([]),
    proposed([segment(text='   ')]),
    proposed([segment(ids=[])]),
    proposed([segment(ids=['S999'])]),
    proposed([segment(ids=['S1', 'S1'])]),
    proposed([segment(ids=['profile_context'])]),
    proposed([segment('advice', '建议邮件。', ['S1'])], 'advice'),
    proposed([segment('receipt', '已记录偏好。', [])]),
    proposed([segment()], issues=[{'code': 'citation_missing', 'quote': '并不存在的原文',
                                  'detail': '不应伪造原稿引文。', 'source_ids': []}]),
    proposed([segment()], issues=[{'code': 'other', 'quote': '', 'detail': '引用未登记来源',
                                  'source_ids': ['S999']}]),
    {'answer_type': 'factual', 'segments': [segment()], 'issues': [], 'confidence': 1.0},
])
def test_untrusted_review_contract_failure_preserves_original_and_warns(value):
    result, llm = run(value)

    assert_fallback(result, llm)
    assert 'uncited' in codes(result)


@pytest.mark.parametrize('text', [
    '看这条 [[S999]]。', '`[[S1]]`', '```text\n[[S1]]\n```',
    '[链接 [[S1]]](https://example.invalid)', '<script>[[S1]]</script>',
    '&#91;&#91;S1&#93;&#93;', r'\[\[S1\]\]',
    '```text\n这段不闭合', '<code>这段不闭合',
])
def test_model_references_or_markdown_that_hides_server_references_is_rejected(text):
    result, llm = run(proposed([segment(text=text)]))

    assert_fallback(result, llm)


def test_an_unclosed_fence_in_an_earlier_segment_cannot_hide_later_source_buttons():
    result, llm = run(proposed([
        segment('advice', '请确认渠道\n\n```text\n尚未闭合', []), segment(),
    ], 'mixed'))

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'references_not_visible'


@pytest.mark.parametrize('raw', ['', 'not JSON', '```json\n{}\n```', '[]',
                                '{"answer_type":"no_answer","answer_type":"factual"}'])
def test_bad_json_is_not_repaired_with_a_second_model_call(raw):
    llm = Reviewer({'content': raw})

    result = review_answer(llm, '林舟的血型？', DRAFT, SOURCES)

    assert_fallback(result, llm)


def test_model_tool_action_is_rejected_without_executing_any_tool():
    llm = Reviewer({'content': '', 'tool_calls': [{'function': {'name': 'add_memory'}}]})

    result = review_answer(llm, '林舟怎么联系？', DRAFT, SOURCES)

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'unexpected_model_action'


def test_provider_failure_does_not_expose_exception_body_or_claim_review_success():
    llm = Reviewer(RuntimeError('provider echoed an API credential: very-secret-value'))

    result = review_answer(llm, '林舟怎么联系？', DRAFT, SOURCES)

    assert_fallback(result, llm)
    assert 'very-secret-value' not in json.dumps(result)
    assert result['metadata']['unavailable_reason'] == 'review_call_failed'


def test_fallback_keeps_original_invalid_reference_for_harness_warning_merge():
    draft = '错误引用 [[profile_context]]，真实引用 [[S1]]。'
    llm = Reviewer({'content': 'bad json'})

    result = review_answer(llm, '林舟怎么联系？', draft, SOURCES)

    assert_fallback(result, llm, draft)
    assert result['metadata']['invalid_original_references'] == ['profile_context']
    assert 'uncited' not in codes(result)


def test_oversize_response_keeps_a_bounded_audit_record_and_its_full_hash():
    raw = 'x' * 100000
    llm = Reviewer({'content': raw})

    result = review_answer(llm, '林舟怎么联系？', DRAFT, SOURCES)

    assert_fallback(result, llm)
    assert result['metadata']['raw_response_truncated'] is True
    assert len(result['metadata']['raw_response']) == 48000
    assert result['metadata']['raw_response_sha256'] == hashlib.sha256(raw.encode()).hexdigest()


def test_oversize_input_does_not_send_a_partial_request_to_the_model():
    llm = Reviewer({'content': 'never called'})

    result = review_answer(llm, '林舟怎么联系？', DRAFT, SOURCES,
                           context={'outcomes': [{'detail': 'x' * 100000}]})

    assert llm.calls == []
    assert result['answer'] == DRAFT
    assert result['metadata']['unavailable_reason'] == 'input_budget_exceeded'


def test_duplicate_source_snapshots_are_rejected_before_any_model_call():
    llm = Reviewer({'content': 'never called'})

    result = review_answer(llm, '林舟怎么联系？', DRAFT, [SOURCES[0], SOURCES[0]])

    assert llm.calls == []
    assert result['metadata']['unavailable_reason'] == 'duplicate_source_snapshot'


def test_runtime_receipt_can_be_preserved_but_is_not_a_new_memory_source():
    context = {'outcomes': [{'kind': 'memory', 'status': 'pending', 'detail': '待用户确认'}]}
    result, _ = run(proposed([segment('receipt', '', [])]), context=context)

    assert result['answer'] == '本轮处理结果请以下方回执为准。'
    assert result['metadata']['status'] == 'completed'
    assert result['metadata']['source_ids'] == ['S1', 'S2']
    assert result['metadata']['semantic_verified'] is False


@pytest.mark.parametrize('status', ['ok', 'pending', 'failed'])
def test_receipt_pointer_uses_only_a_runtime_result_status(status):
    context = {'receipts': [{'type': 'reminder', 'status': status}]}

    result, _ = run(proposed([segment('receipt', '', [])]), context=context)

    assert result['metadata']['status'] == 'completed'
    assert result['answer'] == '本轮处理结果请以下方回执为准。'
    assert '成功' not in result['answer']


def test_nested_tool_receipt_outcomes_can_enable_the_fixed_pointer():
    context = {'receipts': [{'type': 'save_person_fact', 'outcomes': [{'status': 'ok'}]}]}

    result, _ = run(proposed([segment('receipt', '', [])]), context=context)

    assert result['metadata']['status'] == 'completed'


def test_query_audit_receipt_cannot_claim_it_saved_a_fictitious_blood_type():
    context = {'intent': 'query', 'outcomes': [{'kind': 'intent', 'status': 'skipped',
                                               'detail': '仅保留审计原文，未保存记忆。'}]}

    result, llm = run(proposed([segment('receipt', '已保存林舟的血型为O型。', [])]), context=context)

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'receipt_contains_content'
    assert 'O型' not in result['answer']


def test_skipped_only_query_cannot_add_even_an_empty_receipt_segment():
    context = {'intent': 'query', 'outcomes': [{'kind': 'intent', 'status': 'skipped'}]}

    result, llm = run(proposed([segment('receipt', '', [])]), context=context)

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'receipt_without_runtime_results'


def test_even_a_real_success_cannot_authorize_freeform_model_receipt_claims():
    context = {'outcomes': [{'kind': 'memory', 'status': 'ok'}]}

    result, llm = run(proposed([segment('receipt', '已记录用户并未告知的事实。', [])]), context=context)

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'receipt_contains_content'


def test_a_cited_claim_is_not_labeled_as_semantically_verified():
    # The contract can reject invented identifiers, but cannot prove that S1
    # entails this incorrect age. Keep that practical limit explicit.
    result, _ = run(proposed([segment(text='林舟今年47岁。')]))

    assert result['metadata']['status'] == 'completed'
    assert result['metadata']['semantic_verified'] is False
