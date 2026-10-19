import pytest

from app.agent.source_registry import SourceRegistry, _prose_references


def registry():
    result = SourceRegistry(None)
    result.register({'type': 'memory', 'entity_id': 'memory-1', 'text': 'A saved fact',
                     'temporal_status': 'current'})
    return result


def codes(warnings):
    return {warning['code'] for warning in warnings}


@pytest.mark.parametrize('identifier', [
    'profile_context', 'unknown_id', 'Sx', 'S-1', 'S999', 'tool.result', 'foo:bar', '123',
])
def test_unknown_ascii_identifiers_in_prose_are_removed_and_warned(identifier):
    answer = f'First [[{identifier}]], repeated [[{identifier}]], real [[S1]] and [[S1]].'

    cleaned, sources, warnings = registry().citations(answer, require_citations=True)

    assert cleaned == 'First , repeated , real [[S1]] and [[S1]].'
    assert [source['id'] for source in sources] == ['S1']
    assert codes(warnings) == {'invalid_citation'}
    assert '已移除' not in warnings[0]['message']


PROTECTED_CONTEXTS = [
    '`[[{id}]]`',
    '``literal ` [[{id}]]``',
    '```text\n[[{id}]]\n```',
    '~~~text\n[[{id}]]\n~~~',
    '```text\n[[{id}]]',
    '    [[{id}]]',
    '[See [[{id}]]](https://example.invalid/source)',
    '[Source](https://example.invalid/[[{id}]])',
    '<https://example.invalid/[[{id}]]>',
    'https://example.com/[[{id}]]',
    '<a href="https://example.com">[[{id}]]</a>',
    '<code>[[{id}]]</code>',
    '<pre>[[{id}]]</pre>',
    '<script>[[{id}]]</script>',
    '<style>[[{id}]]</style>',
    '<iframe>[[{id}]]</iframe>',
    '<textarea>[[{id}]]</textarea>',
    '<title>[[{id}]]</title>',
    '<template>[[{id}]]</template>',
    '![See [[{id}]]](https://example.invalid/image.png)',
    '<img src="https://example.invalid/image.png" alt="[[{id}]]">',
]


@pytest.mark.parametrize('context', PROTECTED_CONTEXTS)
@pytest.mark.parametrize('identifier', ['profile_context', 'S999'])
def test_protected_unknown_identifiers_alone_are_preserved_without_invalid_warning(context, identifier):
    answer = context.format(id=identifier)

    cleaned, sources, warnings = registry().citations(answer, require_citations=True)

    assert cleaned == answer
    assert sources == []
    assert codes(warnings) == {'uncited'}


@pytest.mark.parametrize('context', PROTECTED_CONTEXTS)
@pytest.mark.parametrize('identifier', ['profile_context', 'S999'])
def test_same_unknown_identifier_in_prose_and_protected_context_keeps_literals(context, identifier):
    answer = f'Unknown [[{identifier}]], repeated [[{identifier}]], real [[S1]].\n\n' + context.format(id=identifier)

    cleaned, sources, warnings = registry().citations(answer, require_citations=True)

    assert cleaned == answer
    assert [source['id'] for source in sources] == ['S1']
    assert codes(warnings) == {'invalid_citation'}
    assert '已移除' not in warnings[0]['message']


@pytest.mark.parametrize('prose', [
    '&#91;&#91;S999&#93;&#93;',
    r'\[\[S999\]\]',
])
def test_entity_or_escape_cannot_balance_counts_and_delete_a_protected_literal(prose):
    answer = '`[[S999]]` and ' + prose

    cleaned, sources, warnings = registry().citations(answer, require_citations=True)

    assert cleaned == answer
    assert sources == []
    assert codes(warnings) == {'invalid_citation', 'uncited'}


def test_only_unambiguous_identifiers_are_removed_when_another_is_protected():
    answer = 'Unknown [[profile_context]], [[S999]], real [[S1]]. Example: `[[S999]]`'

    cleaned, sources, warnings = registry().citations(answer, require_citations=True)

    assert cleaned == 'Unknown , [[S999]], real [[S1]]. Example: `[[S999]]`'
    assert [source['id'] for source in sources] == ['S1']
    assert codes(warnings) == {'invalid_citation'}


def test_prose_reference_helper_keeps_deduplication_and_exclusions():
    assert _prose_references('[[S1]] [[profile_context]] [[S1]] `[[S999]]`') == ['S1', 'profile_context']


@pytest.mark.parametrize('answer', ['[[ordinary words]]', '[[中文条目]]'])
def test_non_identifier_brackets_are_preserved(answer):
    cleaned, sources, warnings = registry().citations(answer)

    assert cleaned == answer
    assert sources == []
    assert warnings == []
