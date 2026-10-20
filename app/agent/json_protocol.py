"""Strict object transport for model responses, without extracting prose.

A single complete ``json`` code fence is a transport envelope only. Removing
that envelope never relaxes JSON syntax or the caller's domain validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any


_JSON_FENCE = re.compile(r'```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```', re.I)


class JSONProtocolError(ValueError):
    """A bounded public code; rejected model text never enters this error."""

    def __init__(self, code: str, *, category: str = 'json'):
        super().__init__(code)
        self.code = code
        self.category = category


@dataclass(frozen=True)
class ParsedJSON:
    value: dict[str, Any]
    envelope: str


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def _constant(_value):
    raise ValueError('non_json_constant')


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('non_finite_number')
    return number


def parse_json_object(raw: str, *, max_chars: int = 48000, allow_fence: bool = True) -> ParsedJSON:
    """Accept an entire JSON object or one entire labelled JSON fence.

    Never find a JSON substring inside explanatory text, remove comments,
    repair syntax, merge multiple objects, or accept duplicate keys/non-finite
    numbers. Callers still must validate every field, span and source ID.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError('max_chars must be a positive integer')
    if not isinstance(raw, str):
        raise JSONProtocolError('invalid_response_type', category='schema')
    if len(raw) > max_chars:
        raise JSONProtocolError('response_budget_exceeded')
    body, envelope = raw.strip(), 'plain'
    if allow_fence:
        fence = _JSON_FENCE.fullmatch(body)
        if fence:
            body, envelope = fence.group('body'), 'json_fence'
    try:
        value = json.loads(body, object_pairs_hook=_object, parse_constant=_constant,
                           parse_float=_finite_float)
    except (ValueError, TypeError, RecursionError):
        raise JSONProtocolError('invalid_json') from None
    if not isinstance(value, dict):
        raise JSONProtocolError('json_object_required', category='schema')
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > 64:
            raise JSONProtocolError('invalid_json')
        children = item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
        stack.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))
    return ParsedJSON(value=value, envelope=envelope)
