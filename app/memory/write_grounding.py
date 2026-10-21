"""Narrow, auditable write constraints; these do not verify general semantics."""
from __future__ import annotations

from copy import deepcopy
import re

from app.store import repo

_PRONOUN = re.compile(r'^(?:他|她|对方|这个人|那个人|he|she|him|her)$', re.I)
_RECORD_REQUEST = r'(?:顺便)?(?:帮我|替我)(?:记下|记一下|存下|存一下)'
_SUBJECT = re.compile(r'^\s*(?:(?:' + _RECORD_REQUEST +
                      r'|帮我记下|请记住|记住|记下|记录|补充|更新|保存|请|帮我|记一下)\s*[：:，,]?\s*)*'
                      r'(他|她|对方|这个人|那个人)(?!们|人)')
_TRANSFER_PRONOUN = re.compile(r'^\s*(?:(?:' + _RECORD_REQUEST +
    r'|帮我记下|请记住|记住|记下|记录)\s*[：:，,]?\s*)*'
    r'我\s*(?:在?\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日|号)?\s*)?'
    r'(?:给|向|还|借给|付给|转给)(他|她|对方)(?!们|人)')
_WORKPLACE = re.compile(r'(?<!现)在\s*([^，。；！？;!?\n]{1,60}?)\s*(?:工作|任职|就职)(?=了|的|[，。；！？;!?\n]|$)')
_ROLE_MARKER = re.compile(r'担任|从事|负责|职位|职务|岗位|做|任(?!职)|(?<![A-Za-z])as\s', re.I)
_UNCERTAIN = re.compile(r'不在|没在|未在|不是|并非|如果|假如|要是|可能|也许|是否|没有|不再|不曾|曾经|以前')


def _pattern(name):
    escaped = re.escape(name)
    return (r'(?<![A-Za-z0-9_])' + escaped + r'(?![A-Za-z0-9_])') if re.search(r'[A-Za-z0-9]', name) else escaped


def mentions(conn, source):
    """Longest exact alias occurrences; a shared short name stays ambiguous."""
    aliases = {}
    for row in repo.alias_rows(conn):
        if len(row['alias']) >= 2:
            aliases.setdefault(row['alias'], set()).add(row['person_id'])
    matches = [(m.start(), m.end(), alias, ids) for alias, ids in aliases.items()
               for m in re.finditer(_pattern(alias), source)]
    return [item for item in matches if not any(
        other[0] <= item[0] and other[1] >= item[1] and other[1] - other[0] > item[1] - item[0]
        for other in matches)]


def validate_write_scope(conn, source, scope):
    """Recheck the selected people and alias ownership at every write entry."""
    if not isinstance(scope, dict) or scope.get('blocked') is not False:
        return
    allowed = scope.get('person_ids', [])
    if scope.get('mode') not in {'single', 'multiple'} or not isinstance(allowed, list):
        return
    live = {p['id']: p for p in repo.all_active_persons(conn)}
    if not allowed or any(pid not in live for pid in allowed):
        raise ValueError('原问题选择的人物已不可用，请重新指定人物')
    for alias, expected in scope.get('write_alias_bindings', {}).items():
        current = sorted({row['person_id'] for row in repo.alias_rows(conn) if row['alias'] == alias})
        if re.search(_pattern(alias), source) and current != expected:
            raise ValueError('记事原文中的人物称呼已变化，请重新指定人物')


def scoped_identity(conn, person, source, scope):
    """Bind this mention, never all patches, to a currently available scoped ID."""
    validate_write_scope(conn, source, scope)
    if not isinstance(scope, dict) or scope.get('blocked') is not False:
        return None
    allowed = scope.get('person_ids', [])
    if scope.get('mode') not in {'single', 'multiple'} or not isinstance(allowed, list):
        return None
    name = person.get('name', '')
    found = mentions(conn, source)
    if _PRONOUN.fullmatch(name) or (scoped_pronoun_reference(source) and len(allowed) == 1
                                  and name in repo.aliases_of(conn, allowed[0])):
        # Pronouns are locators only: never register them as aliases. Any named
        # foreign subject or ambiguous name prevents inheriting the selection.
        if len(allowed) != 1 or not scoped_pronoun_reference(source):
            return None
        if any(ids != {allowed[0]} for _, _, _, ids in found):
            return None
        return allowed[0]
    # Original mention is authoritative; a model's canonical spelling cannot
    # redirect an exact selected alias to a different or newly created person.
    if any(len(ids) == 1 and ids.isdisjoint(allowed) and (name in alias or alias in name)
           for _, _, alias, ids in found if name):
        raise ValueError('记事称呼指向了原选择之外的人物，请重新指定人物')
    candidates = [pid for pid in allowed if name in repo.aliases_of(conn, pid)]
    supported = [pid for pid in candidates if any(ids == {pid} and
        (name in alias or alias in name) for _, _, alias, ids in found)
        or pid in source]
    return supported[0] if len(supported) == 1 else None


def freeze_write_aliases(conn, scope, source):
    """Capture exact ownership before waiting on a model, including removals."""
    return {**scope, 'write_alias_bindings': {alias: sorted(ids)
            for _, _, alias, ids in mentions(conn, source)}}


def pronoun_subject(source):
    """Only a leading subject after a closed record prefix may inherit scope."""
    match = _SUBJECT.match(source)
    return match.group(1) if match else None


def is_pronoun(name):
    return bool(_PRONOUN.fullmatch(name))


def scoped_pronoun_reference(source):
    # Preserve an explicit user-to-recipient transfer as well as a leading
    # subject; neither permits an arbitrary new named speaker before "她".
    return pronoun_subject(source) or bool(_TRANSFER_PRONOUN.match(source))


def _mask(source, names):
    for name in sorted({n for n in names if n and not _PRONOUN.fullmatch(n)}, key=len, reverse=True):
        source = re.sub(_pattern(name), lambda m: ' ' * len(m.group()), source)
    return source


def constrain_person(conn, person, source, person_id=None):
    """Preserve explicit predicates; reject facts supported only by a nickname.

    An unquoted workplace phrase does not prove where an institution name ends;
    a partial model value requires confirmation rather than automatic expansion.
    The returned audit retains each candidate changed or skipped.
    """
    p, audit = deepcopy(person), []
    names = [p.get('name', '')]
    if person_id:
        names += repo.aliases_of(conn, person_id)
    # Include longest aliases actually used for this original mention even when
    # the model reduced a unique alias to a shared canonical name.
    names += [alias for _, _, alias, _ in mentions(conn, source)
              if any(name and name in alias for name in names)]
    display = repo.get_person(conn, person_id)['display_name'] if person_id else p.get('name', '')
    qualifiers = {name.replace(display, '').strip() for name in names if display and display in name}
    qualifiers.discard('')

    def supported(kind, item):
        proof = item.get('evidence') or source
        body = _mask(proof, names)
        value = item.get('value', '') if kind == 'field' else item.get('text', '')
        identity_only = (kind == 'field' and value and any(value in name for name in names) and value not in body)
        inferred_qualifier = kind != 'field' and any(q in _mask(value, names) and q not in body for q in qualifiers)
        if identity_only or inferred_qualifier:
            audit.append({'kind': kind, 'action': 'skipped', 'reason': 'identity_label_is_not_fact',
                          'candidate': deepcopy(item)})
            return False
        return True

    p['fields'] = [f for f in p.get('fields', []) if supported('field', f)]
    for key in ('facts',):
        p[key] = [item for item in p.get(key, []) if supported('fact', item)]
    if p.get('value_read') and not supported('value_read', p['value_read']):
        p['value_read'] = None

    for company in [f for f in p['fields'] if f['field'] == 'company']:
        proof = _mask(company.get('evidence') or source, names)
        groups = {m.group(1).strip().strip('「」“”\"《》') for m in _WORKPLACE.finditer(proof)
                  if not _UNCERTAIN.search(proof[max(0, m.start()-12):m.end()])
                  and not _ROLE_MARKER.search(m.group(1))}
        matching = [group for group in groups if group.startswith(company['value'])]
        if any(group != company['value'] for group in matching):
            reason = 'workplace_value_incomplete_or_ambiguous'
            audit.append({'kind': 'field', 'action': 'pending', 'reason': reason,
                          'candidate': deepcopy(company), 'source_phrases': sorted(groups)})
            company['_write_constraint_reason'] = reason

    fields = []
    for field in p['fields']:
        if field['field'] == 'title':
            proof = _mask(field.get('evidence') or source, names)
            # A suffix within an undivided workplace phrase is not independent
            # evidence for a title. Explicit occupation predicates remain valid.
            groups = [m.group(1).strip() for m in _WORKPLACE.finditer(proof)]
            outside = _WORKPLACE.sub('', proof)
            if any(field['value'] in group and not _ROLE_MARKER.search(group) for group in groups) \
                    and field['value'] not in outside:
                audit.append({'kind': 'field', 'action': 'skipped', 'reason': 'title_inside_workplace_phrase',
                              'candidate': deepcopy(field)})
                continue
        fields.append(field)
    p['fields'] = fields
    return p, audit
