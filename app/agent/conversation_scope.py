"""Conservative person references, persisted on the user message that established them."""
from __future__ import annotations

import json
import re

from app.agent.tools import _alias_pattern
from app.memory.writes import archived_only_aliases
from app.store import repo

_PRONOUN = re.compile(r'(?<!其)[他她](?!人)|对方|那个人|这个人|刚才那位|前面那位|上面那位|两人|两位|双方|'
                      r'\b(?:he|she|him|her|his|hers|they|them|their|ta)\b', re.I)
_PLURAL = re.compile(r'[他她]们|两人|两位|双方|\b(?:they|them|their)\b', re.I)
_ORDINAL = re.compile(r'第[一二三四五六七八九十百\d]+(?:个|位|名)(?:候选人|人物|人)?(?=$|[\s，。！？,.!?的吧呢])|'
                      r'^\s*第[一二三四五六七八九十百\d]+[\s，。！？,.!?]*$|'
                      r'\b(?:first|second|third|fourth)\s+(?:one|person|candidate)\b', re.I)
_GLOBAL = re.compile(r'所有人|所有人物|全部人物|所有联系人|大家|全库|跨人物|不限定人物|全体|'
                     r'\b(?:everyone|all people|all contacts|across people)\b', re.I)
_ID = re.compile(r'(?<![A-Za-z0-9_-])[0-9a-f]{16}(?![A-Za-z0-9_-])', re.I)
_LABELED_ID = re.compile(r'(?:人物\s*(?:ID|编号)|person[_ -]?id|\bID)\s*[:：=]\s*[`"\']?'
                         r'([A-Za-z0-9_-]{4,80})', re.I)
_NUMERIC_SELECTION = re.compile(r'^\s*(?:(?:选|选择|就选|选中)\s*)?'
                                r'(?:[1-9]\d?|[一二两三四五六七八九十]{1,3})'
                                r'(?:\s*(?:个|位|号))?[\s，。！？,.!?]*$')
_SWITCH_SUBJECT = re.compile(r'(?:换成|改成|换到|改为)\s*([^，。！？,:：;\s]+)|'
                             r'换(?:一个人|个人|一位|个人物)|(?<![\w.])@([^@\s，。！？,:：;]+)')


def _metadata(raw):
    try:
        value = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def mentioned_people(conn, text):
    """Return every explicitly named person, without silently choosing a namesake."""
    aliases = {}
    for row in repo.alias_rows(conn):
        if len(row['alias']) >= 2:
            aliases.setdefault(row['alias'], set()).add(row['person_id'])
    matches = [(match.start(), match.end(), alias, ids) for alias, ids in aliases.items()
               for match in re.finditer(_alias_pattern(alias), text)]
    selected = [item for item in matches if not any(
        other[0] <= item[0] and other[1] >= item[1] and other[1] - other[0] > item[1] - item[0]
        for other in matches)]
    people, ambiguous = [], []
    for _, _, alias, ids in sorted(selected, key=lambda item: (item[0], -item[1])):
        if len(ids) != 1:
            ambiguous.append((alias, sorted(ids)))
        elif next(iter(ids)) not in people:
            people.append(next(iter(ids)))
    return people, list(dict.fromkeys(alias for alias, _ in ambiguous)), ambiguous


def _mentions_archived_person(conn, text):
    aliases = archived_only_aliases(conn)
    if not aliases:
        return False
    active_matches = [match.span() for row in repo.alias_rows(conn) if len(row['alias']) >= 2
                      for match in re.finditer(_alias_pattern(row['alias']), text)]
    for alias in aliases:
        if len(alias) < 2:
            continue
        for match in re.finditer(_alias_pattern(alias), text):
            if not any(start <= match.start() and end >= match.end() and end - start > len(alias)
                       for start, end in active_matches):
                return True
    return False


def _candidates(conn, ids):
    values = []
    for pid in dict.fromkeys(ids):
        person = repo.get_person(conn, pid)
        if not person or person['deleted_at'] is not None or person['status'] == 'merged':
            continue
        fields = repo.current_fields(conn, pid)
        summary = ' · '.join(dict.fromkeys(str(fields[field]['value']) for field in ('city', 'company', 'title')
                                           if field in fields and fields[field]['value']))
        value = {'person_id': pid, 'display_name': person['display_name']}
        if summary:
            value['summary'] = summary
        values.append(value)
    return values


def _background_choice(conn, text, ids, selection_only=False):
    matches = []
    for candidate in _candidates(conn, ids):
        fields = repo.current_fields(conn, candidate['person_id'])
        values = [fields[field]['value'].strip() for field in ('city', 'company', 'title')
                  if isinstance(fields.get(field, {}).get('value'), str)]
        if any(len(value) >= 2 and value in text and (not selection_only or text.strip() == value
               or re.search(r'那个|那位|这位|指的是|说的是|选择|\b(?:the one|from)\b', text, re.I))
               for value in values):
            matches.append(candidate['person_id'])
    return matches[0] if len(matches) == 1 else None


def _scope(conn, mode, ids=(), reason='', origin=None):
    ids = list(dict.fromkeys(ids))
    return {'version': 1, 'mode': mode, 'person_ids': ids if mode in {'single', 'multiple'} else [],
            'blocked': mode == 'clarify', 'reason': reason,
            'candidates': _candidates(conn, ids), 'origin_message_id': origin}


def _valid_previous(conn, scope):
    ids = scope.get('person_ids', [])
    if scope.get('mode') in {'single', 'multiple'} and (
            not ids or len(_candidates(conn, ids)) != len(ids)):
        return _scope(conn, 'clarify', reason='unavailable_person', origin=scope.get('origin_message_id'))
    return scope


def _resolve(conn, text, previous):
    active = {person['id'] for person in repo.all_active_persons(conn)}
    explicit = list(dict.fromkeys([*_LABELED_ID.findall(text), *_ID.findall(text)]))
    if explicit:
        if any(pid not in active for pid in explicit):
            return _scope(conn, 'clarify', reason='unavailable_person')
        return _scope(conn, 'single' if len(explicit) == 1 else 'multiple', explicit, 'explicit_id')

    if _mentions_archived_person(conn, text):
        return _scope(conn, 'clarify', reason='unavailable_person')
    people, _, ambiguous = mentioned_people(conn, text)
    unresolved = []
    for _, ids in ambiguous:
        selected = _background_choice(conn, text, ids)
        if selected:
            people.append(selected)
        else:
            unresolved.extend(ids)
    if unresolved:
        return _scope(conn, 'clarify', unresolved, 'ambiguous_name')
    if people:
        people = list(dict.fromkeys(people))
        return _scope(conn, 'single' if len(people) == 1 else 'multiple', people, 'explicit_name')
    if _ORDINAL.search(text) or (previous.get('candidates') and _NUMERIC_SELECTION.fullmatch(text)):
        return _scope(conn, 'clarify', [item['person_id'] for item in previous.get('candidates', [])],
                      'ordinal_selection')
    if _GLOBAL.search(text):
        return _scope(conn, 'global', reason='explicit_global')
    if previous.get('mode') == 'clarify' and previous.get('candidates'):
        selected = _background_choice(conn, text, [item['person_id'] for item in previous['candidates']],
                                      selection_only=True)
        if selected:
            return _scope(conn, 'single', [selected], 'unique_background')
    if _PRONOUN.search(text):
        for subject in _SWITCH_SUBJECT.finditer(text):
            target = subject.group(1) or subject.group(2)
            if target is None or not _PRONOUN.match(target):
                return _scope(conn, 'clarify', reason='unresolved_subject')
        if previous.get('mode') == 'single' and not _PLURAL.search(text):
            return _scope(conn, 'single', previous['person_ids'], 'inherited_single',
                          previous.get('origin_message_id'))
        return _scope(conn, 'clarify', [item['person_id'] for item in previous.get('candidates', [])],
                      'ambiguous_reference' if previous.get('mode') in {'multiple', 'clarify'} else 'missing_reference')
    return _scope(conn, 'none', reason='no_person_reference')


def for_user_message(conn, conversation_id, user_message_id, text):
    """Rebuild only preceding user choices; failed answers and later messages cannot steer retries."""
    row = conn.execute("SELECT rowid ordering,* FROM message WHERE id=? AND conversation_id=? AND role='user'",
                       (user_message_id, conversation_id)).fetchone()
    if row is None or row['content'] != text:
        raise ValueError('用户消息不属于此对话或内容不一致')
    previous = _scope(conn, 'none')
    rows = conn.execute("SELECT * FROM message WHERE conversation_id=? AND role='user' AND rowid<? ORDER BY rowid",
                        (conversation_id, row['ordering'])).fetchall()
    for item in rows:
        saved = _metadata(item['tool_calls']).get('conversation_scope')
        if isinstance(saved, dict) and saved.get('user_message_id') == item['id'] \
                and saved.get('conversation_id') == conversation_id and saved.get('version') == 1:
            previous = _valid_previous(conn, saved)
        else:
            previous = _resolve(conn, item['content'] or '', previous)
            previous['origin_message_id'] = previous.get('origin_message_id') or item['id']
    scope = _resolve(conn, text, previous)
    scope.update(conversation_id=conversation_id, user_message_id=user_message_id)
    scope['origin_message_id'] = scope.get('origin_message_id') or user_message_id
    metadata = _metadata(row['tool_calls'])
    metadata['conversation_scope'] = scope
    conn.execute("UPDATE message SET tool_calls=? WHERE id=? AND conversation_id=? AND role='user'",
                 (json.dumps(metadata, ensure_ascii=False), user_message_id, conversation_id))
    conn.commit()
    return scope


def clarification(scope):
    if scope['reason'] == 'unavailable_person':
        return '这位人物目前不可用，请重新指定人物。'
    if scope['reason'] == 'ambiguous_name':
        return '这个姓名对应多份档案，请选择具体的人物，或提供城市、公司等可区分的信息。'
    if scope['reason'] == 'ordinal_selection':
        return '请明确选择具体的人物，或提供城市、公司等可区分的信息。'
    return '你指的是哪一位？请说明姓名，或选择对应的人物档案。'
