"""Server-owned, literal projections of one turn-local source snapshot.

Only the final reference is active prose. Stored strings live in fenced literal
blocks, which both citation renderers exclude. Ordinary Markdown escaping alone
would turn a stored ``[[S2]]`` back into a live reference after rendering.
"""
from __future__ import annotations

import re
import unicodedata

from app.agent.memory_sources import model_evidence_context, model_source_context
from app.agent.source_registry import _prose_reference_occurrences

_SOURCE_ID = re.compile(r'S[1-9]\d*\Z')
_TEXT_LIMIT = 1600
_NAME_LIMIT = 120
_FIELD_LIMIT = 160
_PEOPLE_LIMIT = 20
_KINDS = {'memory': '记忆', 'field': '档案信息', 'interaction': '沟通记录'}
_FIELDS = {'company': '公司', 'title': '职位', 'city': '城市', 'birthday': '生日',
           'phone': '电话', 'email': '电子邮箱', 'family': '家庭'}
_BIDI_CONTROLS = set('\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
_TEXT_BOUNDARY = ('文字核对只比较系统内当前保存的字符串；逐字一致不代表记录历史从未编辑，'
                  '也不证明当事人逐字说过这些话或原始资料完整。')


def _literal(value):
    """Preserve data as code text, never as Markdown, HTML or citation prose."""
    if not isinstance(value, str):
        raise ValueError('source_record_invalid_snapshot')
    # Markdown normalizes CR and NUL; bidi overrides can disguise neighbouring
    # labels. Do not silently rewrite data that cannot be displayed literally.
    if any((unicodedata.category(char) in {'Cc', 'Cs'} and char not in '\n\t')
           or char in _BIDI_CONTROLS for char in value):
        raise ValueError('source_record_not_renderable')
    choices = []
    for char in ('`', '~'):
        longest = max((len(run) for run in re.findall(re.escape(char) + '+', value)), default=0)
        choices.append((max(3, longest + 1), char))
    length, char = min(choices)
    fence = char * length
    return f'{fence}text\n{value}\n{fence}'


def literal_generated_text(value):
    """Isolate generated prose so it cannot hide or restyle adjacent records."""
    return _literal(value)


def _bounded_value(label, value, limit):
    block = label + '：\n\n' + _literal(value[:limit])
    if len(value) > limit:
        block += f'\n\n（{label}仅显示前{limit}字符；完整内容见资料。）'
    return block


def _people(source):
    kind, person = source['type'], source.get('person')
    if person is not None and not isinstance(person, dict):
        raise ValueError('source_record_invalid_snapshot')
    people = source.get('people', []) if kind == 'interaction' else []
    if not isinstance(people, list) or any(not isinstance(item, dict) for item in people):
        raise ValueError('source_record_invalid_snapshot')
    values = [person] if person else []
    for item in people:
        same_person = any(isinstance(item.get('id'), str) and item['id']
                          and item['id'] == known.get('id')
                          and item.get('display_name') == known.get('display_name') for known in values)
        if item not in values and not same_person:
            values.append(item)
    if not values:
        if kind == 'memory' and 'person' in source and person is None:
            return ['所属人物：用户自己的记忆。']
        return ['所属人物：快照未提供人物名称。']
    blocks = []
    for index, item in enumerate(values[:_PEOPLE_LIMIT], 1):
        name = item.get('display_name')
        label = '所属人物' if len(values) == 1 else f'关联人物 {index}'
        if name is None:
            blocks.append(label + '：快照未提供人物名称。')
        elif not isinstance(name, str):
            raise ValueError('source_record_invalid_snapshot')
        else:
            blocks.append(_bounded_value(label, name, _NAME_LIMIT))
    if len(values) > _PEOPLE_LIMIT:
        blocks.append(f'（关联人物仅显示前{_PEOPLE_LIMIT}位；完整名单见资料。）')
    return blocks


def _field_label(source):
    field = source.get('field')
    if field is None:
        return '未提供档案项目名称'
    if not isinstance(field, str):
        raise ValueError('source_record_invalid_snapshot')
    if field.startswith(('trait.', 'need.')):
        prefix, name = field.split('.', 1)
        return ('特点：' if prefix == 'trait' else '需求：') + name
    return _FIELDS.get(field, field)


def render_source_record(source):
    """Render one registered ``memory``, ``field`` or ``interaction`` snapshot.

    This function has no storage reads or writes. It describes exactly the
    supplied snapshot, so callers must retain the registry's source-state guard
    when saving the answer. It does not certify source truth or relevance.
    """
    if not isinstance(source, dict) or not isinstance(source.get('type'), str) \
            or source['type'] not in _KINDS \
            or not isinstance(source.get('text'), str):
        raise ValueError('source_record_invalid_snapshot')
    sid = source.get('id')
    if not isinstance(sid, str) or not _SOURCE_ID.fullmatch(sid):
        raise ValueError('source_record_invalid_source_id')
    if source.get('record_context') is not None and not isinstance(source['record_context'], dict):
        raise ValueError('source_record_invalid_snapshot')
    blocks = ['资料类型：' + _KINDS[source['type']], *_people(source)]
    if source['type'] == 'field':
        blocks.append(_bounded_value('档案项目', _field_label(source), _FIELD_LIMIT))
    blocks.append(_bounded_value('保存文字', source['text'], _TEXT_LIMIT))
    try:
        context = model_source_context(source)
        evidence = model_evidence_context(source)
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise ValueError('source_record_invalid_snapshot') from None
    # A legacy field lacking its end marker must not be promoted to current
    # merely because dict.get() defaults to None.
    if source['type'] == 'field' and 'valid_to' not in source:
        context['事实时效'] = '未提供字段结束信息，不能据此确认仍为当前档案。'
    if source['type'] == 'memory' and ('person' not in source or source['person'] == {}):
        context['记录状态'] = context['记录状态'].split('；')[0] + '；所属人物信息未提供。'
    for key in ('记录状态', '事实时效'):
        if key in context:
            blocks.append(key + '：' + context[key])
    for label, value in context.get('时间信息', {}).items():
        if isinstance(value, dict):
            blocks.append(label + '：' + value['操作'] + '；操作时间：' + value['操作时间'])
        else:
            blocks.append(label + '：' + value)
    if context.get('时间表述边界'):
        blocks.append(context['时间表述边界'])
    if evidence.get('来源说明'):
        blocks.append('来源说明：' + evidence['来源说明'])
    blocks.append('文字核对：' + evidence['文字核对'])
    blocks.append(_TEXT_BOUNDARY)
    blocks.append(f'[[{sid}]]')
    answer = '\n\n'.join(blocks)
    if _prose_reference_occurrences(answer) != [sid]:
        raise ValueError('source_record_not_renderable')
    return answer
