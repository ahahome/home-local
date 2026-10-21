"""No provider calls: constrained envelopes never relax strict JSON parsing."""
import json
from pathlib import Path

import pytest

from app.agent.json_protocol import JSONProtocolError, parse_json_object


CASES = json.loads((Path(__file__).parent / 'fixtures' / 'json_protocol.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_independent_transport_fixture(case):
    if 'error' in case:
        with pytest.raises(JSONProtocolError) as failure:
            parse_json_object(case['raw'])
        assert failure.value.code == case['error']
        assert str(failure.value) == case['error']
        assert failure.value.category == ('schema' if case['error'] == 'json_object_required' else 'json')
    else:
        result = parse_json_object(case['raw'])
        assert result.envelope == case['envelope'] and isinstance(result.value, dict)


def test_parser_does_not_modify_literal_span_inside_transport():
    span = '请记住：如果顾念今晚到站，就提醒我。'
    value = {'intent': 'record', 'record_spans': [span], 'reason': 'explicit instruction'}
    raw = '```json\n' + json.dumps(value, ensure_ascii=False) + '\n```'
    assert parse_json_object(raw).value == value


def test_budget_applies_to_full_transport_and_does_not_echo_rejected_text():
    raw = '```json\n{"private":"sk-private"}\n```'
    with pytest.raises(JSONProtocolError) as failure:
        parse_json_object(raw, max_chars=len(raw) - 1)
    assert failure.value.code == 'response_budget_exceeded'
    assert 'sk-private' not in str(failure.value)
    assert parse_json_object(raw, max_chars=len(raw)).value == {'private': 'sk-private'}


@pytest.mark.parametrize('value', [None, 1, {}, []])
def test_nontext_transport_is_schema_failure(value):
    with pytest.raises(JSONProtocolError) as failure:
        parse_json_object(value)
    assert failure.value.category == 'schema' and failure.value.code == 'invalid_response_type'


def test_callers_may_disable_envelope_acceptance():
    with pytest.raises(JSONProtocolError):
        parse_json_object('```json\n{}\n```', allow_fence=False)
    assert parse_json_object('{}', allow_fence=False).envelope == 'plain'


def test_excessive_nesting_fails_with_bounded_diagnostic():
    with pytest.raises(JSONProtocolError) as failure:
        parse_json_object('{"nested":' + '[' * 2000 + '0' + ']' * 2000 + '}')
    assert str(failure.value) == 'invalid_json'
