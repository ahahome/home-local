"""Immutable quotations of stored memory and their current availability."""
from __future__ import annotations

import json

from app.store import repo
from app.util import now_ms

EVIDENCE_KEYS = ('id', 'kind', 'excerpt', 'source_ref', 'blob_hash', 'created_at')


def _person(conn, person_id):
    if person_id is None:
        return None
    row = repo.get_person(conn, person_id)
    if not row or row['deleted_at'] is not None or row['status'] == 'merged':
        return None
    return {'id': row['id'], 'display_name': row['display_name']}


def _evidence(conn, evidence_id):
    row = conn.execute('SELECT * FROM evidence WHERE id=?', (evidence_id,)).fetchone()
    return {key: row[key] for key in EVIDENCE_KEYS} if row else None


def _original_evidence(conn, initial):
    source, seen = initial, set()
    while source and source['id'] not in seen:
        seen.add(source['id'])
        if not (source['source_ref'] or '').startswith(('outcome-memory-review:', 'memory-correction:')):
            break
        try:
            record = json.loads(source['excerpt'])
        except (json.JSONDecodeError, TypeError):
            break
        if not isinstance(record, dict):
            break
        reference = record.get('source_evidence_id') or record.get('origin_evidence_id')
        if not isinstance(reference, str) or reference in seen:
            break
        original = _evidence(conn, reference)
        if original is None:
            break
        source = original
    return source


def _memory_snapshot(conn, row):
    person = _person(conn, row['person_id'])
    if row['deleted_at'] is not None or row['person_id'] is not None and person is None:
        return None
    source = {
        'type': 'memory', 'entity_id': row['id'], 'person': person, 'text': row['text'],
        'date': row['ts'], 'temporal_status': 'historical_background', 'revision': row['revision'],
        'ts': row['ts'], 'kind': row['kind'], 'evidence': _evidence(conn, row['evidence_id']),
        'captured_at': now_ms(),
    }
    initial = _evidence(conn, row['origin_evidence_id'])
    original = _original_evidence(conn, initial)
    if original:
        source['origin_source'] = original
    if initial and original and initial['id'] != original['id']:
        source['initial_source'] = initial
    return source


def memory_source(conn, memory_id):
    row = conn.execute(f'SELECT m.* FROM memory_item m WHERE m.id=? AND {repo.ACTIVE_MEMORY_WHERE}',
                       (memory_id,)).fetchone()
    return _memory_snapshot(conn, row) if row else None


def _field_snapshot(conn, row):
    person = _person(conn, row['person_id'])
    if person is None:
        return None
    source = {
        'type': 'field',
        'entity_id': json.dumps([row['person_id'], row['field'], row['valid_from']],
                                ensure_ascii=False, separators=(',', ':')),
        'person': person, 'text': row['value'] if row['value'] is not None else '',
        'field': row['field'], 'date': row['valid_from'],
        'valid_from': row['valid_from'], 'valid_to': row['valid_to'], 'recorded_at': row['recorded_at'],
        'temporal_status': 'current_field' if row['valid_to'] is None else 'historical_field',
        'evidence': _evidence(conn, row['evidence_id']), 'captured_at': now_ms(),
    }
    original = _original_evidence(conn, source['evidence'])
    if original:
        source['origin_source'] = original
    return source


def field_sources(conn, person_id, limit=10):
    """Keep every current field and at most ``limit`` recent historical versions."""
    if _person(conn, person_id) is None:
        return []
    current = conn.execute('SELECT * FROM person_field WHERE person_id=? AND valid_to IS NULL '
                           'ORDER BY field,valid_from DESC', (person_id,)).fetchall()
    historical = conn.execute('SELECT * FROM person_field WHERE person_id=? AND valid_to IS NOT NULL '
                              'ORDER BY valid_from DESC,field LIMIT ?', (person_id, max(0, limit))).fetchall()
    return [_field_snapshot(conn, row) for row in [*current, *historical]]


def interaction_source(conn, interaction_id):
    row = conn.execute('SELECT * FROM interaction WHERE id=?', (interaction_id,)).fetchone()
    if row is None:
        return None
    links = conn.execute('SELECT ip.person_id,ip.role FROM interaction_person ip '
                         'WHERE ip.interaction_id=? ORDER BY ip.person_id', (interaction_id,)).fetchall()
    people = []
    for link in links:
        person = _person(conn, link['person_id'])
        if person is None:
            return None
        people.append(person | {'role': link['role']})
    evidence = _evidence(conn, row['source_ref'])
    if evidence is None and row['source_ref']:
        candidates = conn.execute('SELECT * FROM evidence WHERE source_ref=? ORDER BY id LIMIT 2',
                                  (row['source_ref'],)).fetchall()
        if len(candidates) == 1:
            candidate = {key: candidates[0][key] for key in EVIDENCE_KEYS}
            excerpt = candidate['excerpt']
            if isinstance(excerpt, str) and excerpt and (excerpt in row['content'] or row['content'] in excerpt):
                evidence = candidate
    source = {
        'type': 'interaction', 'entity_id': row['id'],
        'person': {key: people[0][key] for key in ('id', 'display_name')} if len(people) == 1 else None,
        'people': people, 'text': row['content'], 'date': row['ts'], 'ts': row['ts'],
        'temporal_status': row['kind'], 'kind': row['kind'], 'channel': row['channel'],
        'source_ref': row['source_ref'], 'evidence': evidence, 'captured_at': now_ms(),
    }
    original = _original_evidence(conn, evidence)
    if original:
        source['origin_source'] = original
    return source


def _unchanged(snapshot, current):
    # Registry labels and capture time describe the quotation, not its stored content.
    return {key: value for key, value in snapshot.items() if key not in {'id', 'captured_at'}} == \
        {key: value for key, value in current.items() if key not in {'id', 'captured_at'}}


def source_state(conn, snapshot):
    if not isinstance(snapshot, dict):
        return {'status': 'unavailable', 'current': None}
    kind, entity_id = snapshot.get('type'), snapshot.get('entity_id')
    current = None
    if kind == 'profile_context' and isinstance(entity_id, str):
        from app.memory.compile_card import build_card
        if _person(conn, entity_id):
            current = {'type': kind, 'entity_id': entity_id, 'text': build_card(conn, entity_id)}
    elif kind == 'memory' and isinstance(entity_id, str):
        row = conn.execute('SELECT * FROM memory_item WHERE id=?', (entity_id,)).fetchone()
        current = _memory_snapshot(conn, row) if row else None
        if current is not None and row['status'] == 'withdrawn':
            return {'status': 'withdrawn', 'current': current}
        quoted_person = snapshot.get('person')
        quoted_person_id = quoted_person.get('id') if isinstance(quoted_person, dict) else None
        if current is not None and (current.get('person') or {}).get('id') != quoted_person_id:
            return {'status': 'moved', 'current': current}
    elif kind == 'field' and isinstance(entity_id, str):
        try:
            identity = json.loads(entity_id)
        except (json.JSONDecodeError, TypeError):
            identity = None
        if isinstance(identity, list) and len(identity) == 3 and isinstance(identity[0], str) \
                and isinstance(identity[1], str) and type(identity[2]) is int:
            row = conn.execute('SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from=?',
                               identity).fetchone()
            current = _field_snapshot(conn, row) if row else None
    elif kind == 'interaction' and isinstance(entity_id, str):
        current = interaction_source(conn, entity_id)
    if current is None:
        return {'status': 'unavailable', 'current': None}
    return {'status': 'current' if _unchanged(snapshot, current) else 'updated', 'current': current}
