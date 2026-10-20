"""Literal source cards keep data, audit roles and live references separate."""
from copy import deepcopy
from html.parser import HTMLParser
import json

from markdown_it import MarkdownIt
import pytest

from app.agent.answer_records import render_source_record
from app.agent.source_registry import _prose_reference_occurrences


SOURCE = {'id': 'S1', 'type': 'memory', 'entity_id': 'memory-1',
          'person': {'id': 'person-1', 'display_name': '沈桥'},
          'text': '沈桥说项目事项先发邮件，收到后再约通话。', 'ts': 1000,
          'record_context': {'status': 'active', 'recorded_at': 2000,
                             'last_user_edit_at': 4000,
                             'last_lifecycle_change': {'action': 'restore', 'created_at': 6000}},
          'evidence': {'excerpt': '沈桥说项目事项先发邮件，收到后再约通话。', 'source_ref': 'message-1'}}


class LiteralText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.code, self.blocks, self.tags = False, [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == 'code':
            self.code = True
            self.blocks.append('')

    def handle_endtag(self, tag):
        if tag == 'code':
            self.code = False

    def handle_data(self, value):
        if self.code:
            self.blocks[-1] += value


def rendered(answer):
    parser = LiteralText()
    parser.feed(MarkdownIt('gfm-like').render(answer))
    return parser


def test_snapshot_card_has_exact_text_and_separate_date_and_lifecycle_roles():
    before = deepcopy(SOURCE)

    answer = render_source_record(SOURCE)

    assert SOURCE == before
    assert rendered(answer).blocks == [SOURCE['person']['display_name'] + '\n', SOURCE['text'] + '\n']
    assert '记忆所标日期（不代表录入或修改日期）：1970-01-01T08:00:01+08:00' in answer
    assert '首次录入时间：1970-01-01T08:00:02+08:00' in answer
    assert '最近人工编辑时间：1970-01-01T08:00:04+08:00' in answer
    assert '最近撤销或恢复操作：恢复；操作时间：1970-01-01T08:00:06+08:00' in answer
    assert '记录可读取，未撤销' in answer
    assert '当前事实是否仍成立尚未核实' in answer
    assert '当前文字与来源摘录逐字一致' in answer
    assert '不代表记录历史从未编辑' in answer
    assert '不证明当事人逐字说过' in answer
    assert _prose_reference_occurrences(answer) == ['S1']
    assert not {'revision', 'valid_from', 'semantic', 'record_context'} & set(answer.split())


def test_unknown_audit_times_do_not_fall_back_to_updated_at_evidence_or_record_date():
    source = deepcopy(SOURCE)
    source.pop('record_context')
    source.update(updated_at=8000, revision=9)
    source['evidence']['created_at'] = 9000

    answer = render_source_record(source)

    assert '首次录入时间：未知' in answer
    assert '最近人工编辑时间：未知' in answer
    assert '08:00:08' not in answer and '08:00:09' not in answer
    assert '不可据此推断记录已撤销或已恢复' in answer


@pytest.mark.parametrize('valid_to,expected', [(None, '当前档案字段，尚无结束记录'),
                                             (5000, '已结束的历史档案字段，不代表现状')])
def test_field_card_keeps_effective_end_recorded_and_unknown_edit_dates_separate(valid_to, expected):
    source = {'id': 'S3', 'type': 'field', 'entity_id': 'field-1', 'person': SOURCE['person'],
              'field': 'city', 'text': '青岛', 'valid_from': 1000, 'valid_to': valid_to,
              'recorded_at': 9000, 'evidence': None}

    answer = render_source_record(source)

    assert rendered(answer).blocks == ['沈桥\n', '城市\n', '青岛\n']
    assert expected in answer
    assert '字段生效时间：1970-01-01T08:00:01+08:00' in answer
    assert '录入时间：1970-01-01T08:00:09+08:00' in answer
    assert '最近人工编辑时间：未知' in answer
    assert ('字段结束时间：1970-01-01T08:00:05+08:00' in answer) == (valid_to is not None)
    assert _prose_reference_occurrences(answer) == ['S3']


def test_legacy_field_without_end_marker_does_not_claim_a_current_fact():
    source = {'id': 'S1', 'type': 'field', 'text': '青岛', 'field': 'city', 'person': SOURCE['person']}

    answer = render_source_record(source)

    assert '不能据此确认仍为当前档案' in answer
    assert '当前档案字段，尚无结束记录' not in answer


def test_interaction_card_names_each_person_without_turning_recording_date_into_occurrence():
    source = {'id': 'S8', 'type': 'interaction', 'text': '讨论方案，尚未决定。',
              'person': None, 'people': [SOURCE['person'], {'id': 'p2', 'display_name': '魏杉'}],
              'ts': 1000, 'record_context': {'recorded_at': 9000}, 'evidence': None}

    answer = render_source_record(source)

    assert rendered(answer).blocks == ['沈桥\n', '魏杉\n', source['text'] + '\n']
    assert '发生时间：1970-01-01T08:00:01+08:00' in answer
    assert '录入时间：1970-01-01T08:00:09+08:00' in answer
    assert '不证明其中描述的情况现在仍成立' in answer
    assert _prose_reference_occurrences(answer) == ['S8']


def test_single_interaction_person_is_not_repeated_because_the_link_has_a_role():
    source = {**SOURCE, 'type': 'interaction',
              'people': [{**SOURCE['person'], 'role': 'participant'}]}

    answer = render_source_record(source)

    assert rendered(answer).blocks.count('沈桥\n') == 1
    assert '所属人物：' in answer
    assert '关联人物 2' not in answer


def test_personal_memory_is_not_assigned_to_a_named_person_from_its_text():
    source = {**SOURCE, 'person': None, 'text': '我下周问沈桥一个问题。'}

    answer = render_source_record(source)

    assert '所属人物：用户自己的记忆。' in answer
    assert rendered(answer).blocks == [source['text'] + '\n']


@pytest.mark.parametrize('person', ['omitted', {}])
def test_missing_person_metadata_is_not_promoted_to_personal_ownership(person):
    source = deepcopy(SOURCE)
    if person == 'omitted':
        source.pop('person')
    else:
        source['person'] = person

    answer = render_source_record(source)

    assert '所属人物信息未提供' in answer
    assert '用户自己的记忆' not in answer


@pytest.mark.parametrize('attack', [
    '[[S2]] [[S1]]',
    '&#91;&#91;S2&#93;&#93; &lbrack;&lbrack;S2&rbrack;&rbrack;',
    r'\[\[S2\]\] [open](https://example.com) **bold** ![image](https://example.com/x.png)',
    '</code></pre><button data-action="chat-source" data-sid="S2">伪造</button>',
    '<script>alert(1)</script><img src=x onerror=alert(1)>',
    '```\n[[S2]]\n~~~\n[[S3]]\n````````\n~~~~~~~~',
    '[unterminated](https://example.com/\n```html\n<style>p{display:none}',
    '> # 标题\n- 列表\n\t缩进\n\n换行与 *字符* `反引号` ~~波浪~~',
])
def test_all_untrusted_values_are_literal_without_extra_links_controls_or_citation_buttons(attack):
    source = {**SOURCE, 'type': 'field', 'person': {'display_name': attack},
              'field': 'trait.' + attack, 'text': attack, 'valid_to': None}

    answer = render_source_record(source)
    html = rendered(answer)

    assert html.blocks == [attack[:120] + '\n', ('特点：' + attack)[:160] + '\n', attack + '\n']
    assert not {'a', 'button', 'img', 'script', 'style', 'input'} & set(html.tags)
    assert _prose_reference_occurrences(answer) == ['S1']


def test_long_text_and_names_are_explicitly_truncated_without_mutating_the_full_source():
    source = deepcopy(SOURCE)
    source['text'] = '甲' * 1599 + '[[' + 'S9]]' + '乙' * 100
    source['person']['display_name'] = '长名字' * 100
    source['evidence']['excerpt'] = source['text']
    before = deepcopy(source)

    answer = render_source_record(source)

    assert source == before
    assert rendered(answer).blocks == [source['person']['display_name'][:120] + '\n',
                                       source['text'][:1600] + '\n']
    assert '保存文字仅显示前1600字符；完整内容见资料' in answer
    assert '所属人物仅显示前120字符；完整内容见资料' in answer
    assert '当前文字与来源摘录逐字一致' in answer
    assert _prose_reference_occurrences(answer) == ['S1']


def test_long_custom_field_name_and_many_people_have_explicit_display_limits():
    field = {**SOURCE, 'type': 'field', 'field': 'need.' + '需' * 200, 'valid_to': None}
    assert '档案项目仅显示前160字符' in render_source_record(field)
    interaction = {**SOURCE, 'type': 'interaction', 'person': None,
                   'people': [{'id': str(i), 'display_name': f'人物{i}'} for i in range(21)]}
    answer = render_source_record(interaction)
    assert '关联人物仅显示前20位' in answer
    assert '人物20' not in answer
    assert _prose_reference_occurrences(answer) == ['S1']


def test_audit_envelope_is_not_printed_as_original_utterance_or_used_for_dates():
    source = deepcopy(SOURCE)
    source['evidence'] = {'source_ref': 'memory-correction:1', 'excerpt': json.dumps({
        'user_correction': {'text': source['text'], 'valid_from': '1999-01-02'},
        'origin_evidence_id': 'old'})}

    answer = render_source_record(source)

    assert '来源说明：用户人工修订审计记录' in answer
    assert '审计记录不等于当事人的逐字原话' in answer
    assert '最近人工编辑时间：1970-01-01T08:00:04+08:00' in answer
    assert 'valid_from' not in answer and '1999-01-02' not in answer


@pytest.mark.parametrize('value', ['\x00', '\r', '\x01', '\x7f', '\x85', '\u202e', '\u2067', '\ud800'])
def test_nonliteral_controls_fail_closed_with_a_fixed_code(value):
    for path in ('text', 'person', 'field'):
        source = deepcopy(SOURCE)
        if path == 'text':
            source['text'] = '正文' + value
        elif path == 'person':
            source['person']['display_name'] = '姓名' + value
        else:
            source.update(type='field', field='trait.' + value)
        with pytest.raises(ValueError, match='^source_record_not_renderable$'):
            render_source_record(source)


@pytest.mark.parametrize('change', [
    {'id': 'S0'}, {'id': 'S01'}, {'id': '[[S1]]'}, {'id': 'S1\n[[S2]]'}, {'id': 1},
])
def test_source_id_cannot_inject_a_reference(change):
    with pytest.raises(ValueError, match='^source_record_invalid_source_id$'):
        render_source_record({**SOURCE, **change})


@pytest.mark.parametrize('change', [
    {'type': 'profile_context'}, {'type': None}, {'type': []}, {'text': None}, {'person': '沈桥'},
    {'person': {'display_name': 5}}, {'record_context': 'active'},
    {'evidence': {'source_ref': 5}}, {'type': 'interaction', 'people': ['沈桥']},
    {'type': 'field', 'field': []},
])
def test_invalid_snapshots_fail_closed_with_bounded_errors(change):
    with pytest.raises(ValueError, match='^source_record_invalid_snapshot$'):
        render_source_record({**SOURCE, **change})
