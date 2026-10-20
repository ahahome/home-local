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


def run(value, *, sources=None, draft=DRAFT, context=None, question='林舟怎么联系？'):
    llm = Reviewer({'content': json.dumps(value, ensure_ascii=False)})
    result = review_answer(llm, question, draft, SOURCES if sources is None else sources,
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
    assert review['raw_issues'] == value['issues']
    assert review['issues'] == [{**value['issues'][0], 'quote_status': 'matched'}]
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


def test_blood_type_no_answer_can_drop_available_but_irrelevant_background():
    sources = [
        {**SOURCES[0], 'person': {'id': 'p-min', 'display_name': '敏姐'},
         'text': '敏姐住杭州，喜欢羽毛球。'},
        {**SOURCES[1], 'person': {'id': 'p-min', 'display_name': '敏姐'},
         'text': '敏姐参加过邮件培训。'},
    ]
    draft = '没有查到敏姐的血型。她住杭州、喜欢羽毛球，还参加过邮件培训。 [[S1]][[S2]]'

    result, llm = run(proposed([segment('no_answer', '', [])], 'no_answer'), sources=sources,
                      draft=draft, question='敏姐是什么血型？')

    assert result['answer'] == '当前可核查资料不足以回答这个问题。'
    assert result['metadata']['original_answer'] == draft
    assert result['metadata']['source_ids'] == ['S1', 'S2']
    payload = json.loads(llm.calls[0][0][1]['content'])
    assert payload['question'] == '敏姐是什么血型？'
    assert '即使这些背景有真实来源' in llm.calls[0][0][0]['content']


def test_editor_receives_confirmed_alias_identity_without_registering_a_new_fact_source():
    identity = {'status': 'confirmed', 'people': [
        {'id': 'p-tang', 'name': '唐宁', 'aliases': ['糖糖']},
    ]}
    source = {**SOURCES[0], 'person': {'id': 'p-tang', 'display_name': '唐宁'},
              'text': '唐宁说非紧急事项请发邮件。'}

    result, llm = run(proposed([segment(text='糖糖有一条记录是“唐宁说非紧急事项请发邮件。”')]),
                      sources=[source], question='糖糖怎么联系？',
                      draft='没有糖糖的档案，建议新建人物。',
                      context={'confirmed_identity': identity})

    payload = json.loads(llm.calls[0][0][1]['content'])
    assert payload['runtime_context']['confirmed_identity'] == identity
    assert payload['sources'][0]['person'] == source['person']
    assert result['metadata']['source_ids'] == ['S1']
    assert result['metadata']['status'] == 'completed'
    assert result['metadata']['semantic_verified'] is False
    assert '不能因资料只写正式姓名而否认别名' in llm.calls[0][0][0]['content']


def test_confirmed_identity_cannot_be_used_as_an_unregistered_citation():
    context = {'confirmed_identity': {'status': 'confirmed', 'people': [
        {'id': 'p-tang', 'name': '唐宁', 'aliases': ['糖糖']},
    ]}}

    result, llm = run(proposed([segment(text='糖糖的血型是O型。', ids=['p-tang'])]),
                      context=context)

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'invalid_source_id'


@pytest.mark.parametrize('edited_at,expected_edit_date', [(1792375200000, '2026-10-19'), (None, '未知')])
def test_edit_date_revision_uses_labelled_source_times_without_exposing_raw_timestamps(
        edited_at, expected_edit_date):
    source = {**SOURCES[0], 'ts': 1791511200000, 'revision': 3,
              'record_context': {'status': 'active', 'recorded_at': 1791511200000,
                                 'last_user_edit_at': edited_at}}
    draft = '林舟在2026年10月9日纠正过之前的说法。 [[S1]]'
    corrected = ('最近人工编辑时间是2026年10月19日；10月9日是记忆所标日期。'
                 if edited_at else '最近人工编辑时间未知，10月9日只是记忆所标日期。')
    issue = {'code': 'date_confusion', 'quote': '林舟在2026年10月9日纠正过之前的说法。',
             'detail': '记忆所标日期不能替代最近人工编辑时间。', 'source_ids': ['S1']}

    result, llm = run(proposed([segment(text=corrected)], issues=[issue]), sources=[source],
                      draft=draft, question='我什么时候改过这条记忆？')

    projection = json.loads(llm.calls[0][0][1]['content'])['sources'][0]
    assert '2026-10-09' in projection['时间信息']['记忆所标日期（不代表录入或修改日期）']
    assert expected_edit_date in projection['时间信息']['最近人工编辑时间']
    assert not {'ts', 'revision', 'record_context', 'temporal_status'} & projection.keys()
    assert result['answer'] == corrected + ' [[S1]]'
    assert result['metadata']['raw_issues'] == [issue]
    assert '最近人工编辑时间；未知就说明不能确认' in llm.calls[0][0][0]['content']


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


@pytest.mark.parametrize('text', [
    '建议：先确认方便联系的渠道。',
    '建议：建议：先确认方便联系的渠道。',
    '  建议 : \n 建议： 建议:先确认方便联系的渠道。 ',
])
def test_advice_prefix_is_normalized_once_when_the_model_echoes_it(text):
    result, llm = run(proposed([segment('advice', text, [])], 'advice'))

    assert result['answer'] == '建议：先确认方便联系的渠道。'
    assert result['metadata']['status'] == 'completed'
    assert json.loads(result['metadata']['raw_response'])['segments'][0]['text'] == text
    assert len(llm.calls) == 1


def test_advice_prefix_normalization_preserves_body_labels_and_old_draft():
    draft = '建议：建议：如果您知道她的血型，可以补充。'
    text = '建议：可以问她“你对建议：先发邮件怎么看”。'
    result, _ = run(proposed([segment('advice', text, [])], 'advice'), draft=draft)

    assert result['answer'] == '建议：可以问她“你对建议：先发邮件怎么看”。'
    assert result['metadata']['original_answer'] == draft
    assert result['metadata']['segments'][0]['text'] == text


@pytest.mark.parametrize('quote', ['工作发邮件私事打电话', '工作发邮件 [[S1]]，私事打电话。',
                                  '并不存在的原文'])
def test_unmatched_issue_quote_is_audited_without_discarding_a_valid_revision(quote):
    issue = {'code': 'conflict_resolution', 'quote': quote,
             'detail': '草稿对渠道作了来源未说明的划分。', 'source_ids': ['S1', 'S2']}
    value = proposed([
        segment(text='一条记录是“林舟说非紧急事项请发邮件。”'),
        segment(text='另一条记录是“林舟说沟通前先电话联系。”', ids=['S2']),
        segment('advice', '请向林舟核实两条记录各自适用的范围。', []),
    ], 'mixed', issues=[issue])

    result, llm = run(value)

    assert result['metadata']['status'] == 'completed'
    assert result['metadata']['semantic_verified'] is False
    assert result['metadata']['raw_issues'] == [issue]
    assert result['metadata']['issues'] == [{**issue, 'quote': '', 'quote_status': 'unmatched'}]
    assert json.loads(result['metadata']['raw_response']) == value
    assert result['warnings'] == []
    assert '工作发邮件' not in result['answer']
    assert '[[S1]]' in result['answer'] and '[[S2]]' in result['answer']
    assert len(llm.calls) == 1


def test_an_omitted_issue_quote_is_explicit_in_normalized_audit():
    issue = {'code': 'other', 'quote': '', 'detail': '资料日期含义需区分。', 'source_ids': []}

    result, _ = run(proposed(issues=[issue]))

    assert result['metadata']['raw_issues'] == [issue]
    assert result['metadata']['issues'] == [{**issue, 'quote_status': 'omitted'}]


@pytest.mark.parametrize('value', [
    proposed([]),
    proposed([segment(text='   ')]),
    proposed([segment(ids=[])]),
    proposed([segment(ids=['S999'])]),
    proposed([segment(ids=['S1', 'S1'])]),
    proposed([segment(ids=['profile_context'])]),
    proposed([segment('advice', '建议邮件。', ['S1'])], 'advice'),
    proposed([segment('receipt', '已记录偏好。', [])]),
    proposed([segment('advice', '建议： 建议: ', [])], 'advice'),
    proposed([segment()], issues=[{'code': 'citation_missing', 'quote': 123,
                                  'detail': '引文类型不合法。', 'source_ids': []}]),
    proposed([segment()], issues=[{'code': 'citation_missing', 'quote': 'x' * 2001,
                                  'detail': '引文超过预算。', 'source_ids': []}]),
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


@pytest.mark.parametrize('bad_segment', [
    segment(ids=['S999']),
    segment(ids=['S1', 'S1']),
    segment(ids=['profile_context']),
    segment(text='明确偏好。 [[S1]]'),
    segment(text='&#91;&#91;S1&#93;&#93;'),
    segment(text='```text\n引用被遮住'),
])
def test_unmatched_issue_quote_never_relaxes_body_or_source_validation(bad_segment):
    issue = {'code': 'other', 'quote': '这是模型改写的草稿',
             'detail': '未逐字匹配只影响审计标签。', 'source_ids': []}

    result, llm = run(proposed([bad_segment], issues=[issue]))

    assert_fallback(result, llm)


def test_unmatched_issue_quote_never_authorizes_a_fabricated_audit_source():
    issue = {'code': 'other', 'quote': '这是模型改写的草稿',
             'detail': '问题标签也只能引用本轮资料。', 'source_ids': ['S999']}

    result, llm = run(proposed(issues=[issue]))

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == 'invalid_source_id'


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


def test_source_quote_comparison_copies_each_snapshot_without_model_rewording():
    sources = [
        {**SOURCES[0], 'text': '林舟说项目事项请发邮件。', 'kind': 'semantic'},
        {**SOURCES[1], 'text': '林舟说周三午休方便接电话，其他时候先留言。', 'kind': 'semantic'},
    ]
    value = proposed([
        segment('source_quote', '', ['S1']),
        segment('source_quote', '', ['S2']),
        segment('advice', '请向林舟确认这两条记录各自适用的范围。', []),
    ], 'mixed')

    result, llm = run(value, sources=sources,
                      question='两条记录能说明工作邮件、私事电话吗？')

    assert result['answer'] == ('记录文字：林舟说项目事项请发邮件。 [[S1]]\n\n'
                                '记录文字：林舟说周三午休方便接电话，其他时候先留言。 [[S2]]\n\n'
                                '建议：请向林舟确认这两条记录各自适用的范围。')
    assert result['metadata']['status'] == 'completed'
    assert result['metadata']['semantic_verified'] is False
    assert result['metadata']['segments'] == value['segments']
    assert 'kind' not in json.loads(llm.calls[0][0][1]['content'])['sources'][0]
    assert sources[0]['kind'] == 'semantic'


def test_source_quote_only_is_factual_and_marks_actual_text_truncation():
    source = {**SOURCES[0], 'text': '记录内容' * 500}

    result, _ = run(proposed([segment('source_quote', '', ['S1'])]), sources=[source])

    assert result['metadata']['status'] == 'completed'
    assert result['metadata']['answer_type'] == 'factual'
    assert result['answer'] == '记录文字：' + source['text'][:1600] + '\n（以上仅显示记录前1600字符。） [[S1]]'


@pytest.mark.parametrize('quote_segment,reason', [
    (segment('source_quote', '林舟说私事打电话。', ['S1']), 'source_quote_requires_single_source'),
    (segment('source_quote', '', []), 'source_quote_requires_single_source'),
    (segment('source_quote', '', ['S1', 'S2']), 'source_quote_requires_single_source'),
    (segment('source_quote', '', ['S999']), 'invalid_source_id'),
])
def test_source_quote_cannot_accept_model_text_or_ambiguous_sources(quote_segment, reason):
    result, llm = run(proposed([quote_segment]))

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == reason


@pytest.mark.parametrize('text,reason', [
    ('', 'source_quote_missing_text'),
    ('嵌入引用 [[S2]]', 'source_quote_not_renderable'),
    ('记录首行\n\n```text\n未闭合代码块', 'references_not_visible'),
    ('<code>未闭合代码标签', 'references_not_visible'),
])
def test_source_quote_preserves_existing_reference_visibility_boundary(text, reason):
    source = {**SOURCES[0], 'text': text}

    result, llm = run(proposed([segment('source_quote', '', ['S1'])]), sources=[source])

    assert_fallback(result, llm)
    assert result['metadata']['unavailable_reason'] == reason


def test_editor_gets_edit_audit_words_and_real_time_labels_without_raw_input_keys():
    source = {**SOURCES[0], 'text': '林舟说项目事项先发邮件。', 'kind': 'semantic',
              'ts': 1791511200000,
              'record_context': {'status': 'active', 'last_user_edit_at': 1792375200000},
              'evidence': {'kind': 'text', 'source_ref': 'memory-correction:1',
                           'excerpt': json.dumps({'user_correction': {
                               'text': '林舟说项目事项先发邮件。', 'valid_from': '2026-10-09'},
                               'edited_by_user': True})}}

    result, llm = run(proposed([segment('source_quote', '', ['S1'])]), sources=[source],
                      question='现在这条记录是怎么写的？')

    payload = json.loads(llm.calls[0][0][1]['content'])
    assert payload['sources'][0]['文字依据']['用户修订文字'] == source['text']
    assert '2026-10-19' in payload['sources'][0]['时间信息']['最近人工编辑时间']
    assert '2026-10-09' in payload['sources'][0]['时间信息']['记忆所标日期（不代表录入或修改日期）']
    assert 'valid_from' not in llm.calls[0][0][1]['content']
    assert 'semantic' not in llm.calls[0][0][1]['content']
    assert result['answer'] == '记录文字：' + source['text'] + ' [[S1]]'
    assert result['metadata']['source_snapshot_sha256'] == hashlib.sha256(
        json.dumps([source], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
