import pytest
from app.util import parse_json_loose


@pytest.mark.parametrize('raw', [
    '{"scope":"分析“关系记忆”：来源包含,}和字面引号"}',
    '```json\n{"scope":"分析“关系记忆”：来源包含,}和字面引号"}\n```',
    '说明文字\n{"scope":"分析“关系记忆”：来源包含,}和字面引号"}\n结束',
])
def test_valid_json_string_content_is_never_rewritten(raw):
    assert parse_json_loose(raw)['scope'] == '分析“关系记忆”：来源包含,}和字面引号'


def test_repairs_structural_full_width_punctuation_and_trailing_comma_only():
    result = parse_json_loose('{“scope”： “关系记忆”, "literal": "保留：,}",}')
    assert result == {'scope': '关系记忆', 'literal': '保留：,}'}


def test_repairs_truncated_outer_brace_without_corrupting_string():
    assert parse_json_loose('{"scope":"“来源”：测试"')['scope'] == '“来源”：测试'
