"""Fictional records with natural questions: retrieval and real-harness evaluation.

This runner never reports semantic answer accuracy. It separates fixed-query
ranking, deterministic provenance checks, limited string rules and human review.
Application imports happen only inside isolated workers, after --app-root routing.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

RUNNER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = RUNNER_ROOT / 'eval/fixtures/memory_questions_v08.json'
PROVENANCE = 'fictional_records_with_natural_questions'
RANK_CUTOFFS = (1, 3, 5)


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def fixture_hash(fixture):
    return hashlib.sha256(_encoded(fixture).encode()).hexdigest()


def app_source_hash(app_root):
    """Hash implementation files, excluding local settings and runtime data."""
    root = Path(app_root).resolve()
    files = sorted(path for directory in ('app', 'migrations')
                   for path in (root / directory).rglob('*')
                   if path.is_file() and path.suffix in {'.py', '.sql'})
    if not files or not (root / 'app/agent/harness.py').is_file():
        raise ValueError('Application root must contain app/agent/harness.py')
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return digest.hexdigest()


def _git_info(root):
    def read(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    top = read('rev-parse', '--show-toplevel')
    if not top or Path(top).resolve() != Path(root).resolve():
        return {'sha': None, 'dirty': None}
    sha = read('rev-parse', 'HEAD')
    return {'sha': sha, 'dirty': bool(read('status', '--porcelain')) if sha else None}


def load_fixture(path=DEFAULT_FIXTURE):
    fixture = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_fixture(fixture)
    return fixture


def validate_fixture(fixture):
    if fixture.get('provenance') != PROVENANCE:
        raise ValueError('The corpus must explicitly disclose fictional records and natural questions')
    people = [item['key'] for item in fixture.get('persons', [])]
    records = [item['key'] for group in ('fields', 'memories', 'interactions')
               for item in fixture.get(group, [])]
    if len(people) != len(set(people)) or len(records) != len(set(records)):
        raise ValueError('Person and record keys must be unique')
    people, records = set(people), set(records)
    for group in ('fields', 'memories', 'interactions'):
        for item in fixture.get(group, []):
            if item.get('person') is not None and item['person'] not in people:
                raise ValueError('Unknown record owner: ' + str(item['person']))
    for group in ('retrieval_cases', 'dialogues'):
        cases = fixture.get(group, [])
        ids = [case['id'] for case in cases]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Each track requires nonempty, unique case IDs')
        for case in cases:
            turns = case.get('turns', []) if group == 'dialogues' else [case]
            if not turns:
                raise ValueError('A dialogue needs user turns')
            for turn in turns:
                if not turn.get('question', '').strip() or not turn.get('expect'):
                    raise ValueError('Every question requires explicit expectations')
                args = turn.get('args', {})
                if any(key in args for key in ('person_id', 'person_ids')):
                    raise ValueError('Fixed questions cannot receive hidden person IDs')
                expect = turn['expect']
                if expect.get('intent') not in {None, 'query', 'record', 'mixed', 'unknown'}:
                    raise ValueError('Unknown expected turn intent')
                if 'no_memory_writes' in expect and not isinstance(expect['no_memory_writes'], bool):
                    raise ValueError('no_memory_writes must be boolean')
                for field in expect.get('saved_fields', []):
                    if field.get('person') not in people or not field.get('field') or 'value' not in field:
                        raise ValueError('saved_fields needs an explicit person, field and value')
                for key in ('relevant', 'excluded', 'source_records', 'forbidden_source_records'):
                    if not set(expect.get(key, [])) <= records:
                        raise ValueError('Unknown relevance/source record in ' + case['id'])
                for key in ('ambiguity', 'allowed_persons', 'excluded_persons'):
                    if not set(expect.get(key, [])) <= people:
                        raise ValueError('Unknown person expectation in ' + case['id'])
                for rule in expect.get('limited_fact_rules', []):
                    if rule.get('kind') not in {'contains_any', 'not_contains_any'} or not rule.get('values'):
                        raise ValueError('Limited fact rules require supported kinds and explicit values')
                for action in turn.get('actions', []):
                    if action['op'] not in {'edit', 'withdraw', 'restore', 'archive_person', 'restore_person'}:
                        raise ValueError('Unsupported lifecycle action')
                    if 'memory' in action and action['memory'] not in records:
                        raise ValueError('Unknown action record')
    return fixture


def ranking_metrics(ranked, relevant):
    """Unanswerable and ambiguity cases have no ranking denominator."""
    relevant = set(relevant)
    if not relevant:
        return {'recall_at_' + str(k): None for k in RANK_CUTOFFS} | {'mrr_at_5': None}
    result = {'recall_at_' + str(k): len(set(ranked[:k]) & relevant) / len(relevant)
              for k in RANK_CUTOFFS}
    result['mrr_at_5'] = next((1 / (i + 1) for i, key in enumerate(ranked[:5]) if key in relevant), 0.)
    return result


def limited_fact_checks(answer, rules):
    """String probes are intentionally not semantic correctness judgments."""
    text = answer if isinstance(answer, str) else ''
    usable = bool(text.strip()) and not text.lstrip().startswith('（这一轮没有')
    checks = []
    for rule in rules:
        found = [value for value in rule['values'] if value in text]
        passed = usable and (bool(found) if rule['kind'] == 'contains_any' else not found)
        checks.append({**rule, 'passed': passed, 'matched_values': found,
                       'limitation': 'String presence/absence only; human semantic review is still pending'})
    return checks


def _check(name, passed, **details):
    return {'name': name, 'passed': bool(passed), **details}


def _canonical_entity(source):
    entity = source.get('entity_id')
    if source.get('type') == 'field' and isinstance(entity, str):
        try:
            return _encoded(json.loads(entity))
        except (ValueError, TypeError):
            pass
    return entity


def record_key(source, records):
    if not isinstance(source, dict):
        return None
    entity = _canonical_entity(source)
    return next((key for key, record in records.items()
                 if source.get('type') == record['type'] and entity == record['id']), None)


def source_checks(conn, source):
    """Compare quotations with stored rows, not model assertions about quotations."""
    errors = []
    if not isinstance(source, dict) or source.get('type') not in {'memory', 'field', 'interaction'}:
        return ['unsupported source snapshot']
    if type(source.get('date')) is not int or type(source.get('captured_at')) is not int:
        errors.append('missing date or capture timestamp')
    evidence = source.get('evidence')
    stored = conn.execute('SELECT * FROM evidence WHERE id=?',
                          (evidence.get('id') if isinstance(evidence, dict) else None,)).fetchone()
    if stored is None:
        errors.append('source evidence does not exist')
    elif any(evidence.get(key) != stored[key] for key in
             ('id', 'kind', 'excerpt', 'source_ref', 'blob_hash', 'created_at')):
        errors.append('source evidence differs from stored evidence')
    kind = source['type']
    owner = (source.get('person') or {}).get('id')
    if kind == 'memory':
        row = conn.execute('SELECT * FROM memory_item WHERE id=?', (source.get('entity_id'),)).fetchone()
        if row is None:
            errors.append('source memory does not exist')
        else:
            if any(source.get(key) != row[column] for key, column in
                   (('text', 'text'), ('date', 'ts'), ('revision', 'revision'), ('kind', 'kind'))):
                errors.append('memory quotation differs from current revision')
            if owner != row['person_id'] or (evidence or {}).get('id') != row['evidence_id']:
                errors.append('memory owner or evidence mismatch')
            if row['status'] != 'active' or row['deleted_at'] is not None:
                errors.append('withdrawn or deleted memory exposed')
            if source.get('temporal_status') != 'historical_background':
                errors.append('background memory presented with wrong temporal status')
    elif kind == 'field':
        try:
            identity = json.loads(source['entity_id'])
            if len(identity) != 3:
                raise ValueError('bad field identity')
            row = conn.execute('SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from=?',
                               identity).fetchone()
        except (ValueError, TypeError, KeyError):
            row = None
        if row is None:
            errors.append('field version does not exist')
        else:
            temporal = 'current_field' if row['valid_to'] is None else 'historical_field'
            if source.get('text') != row['value'] or source.get('valid_to') != row['valid_to'] \
                    or source.get('temporal_status') != temporal or owner != row['person_id'] \
                    or (evidence or {}).get('id') != row['evidence_id']:
                errors.append('field value, owner, validity or evidence mismatch')
    else:
        row = conn.execute('SELECT * FROM interaction WHERE id=?', (source.get('entity_id'),)).fetchone()
        if row is None or any(source.get(key) != row[column] for key, column in
                              (('text', 'content'), ('date', 'ts'), ('kind', 'kind'), ('source_ref', 'source_ref'))):
            errors.append('interaction quotation differs from stored interaction')
        linked = {row['person_id'] for row in conn.execute(
            'SELECT person_id FROM interaction_person WHERE interaction_id=?', (source.get('entity_id'),))}
        if {person['id'] for person in source.get('people', [])} != linked:
            errors.append('interaction participants mismatch')
    owners = [owner] if owner else [person['id'] for person in source.get('people', [])]
    for pid in owners:
        person = conn.execute('SELECT * FROM person WHERE id=?', (pid,)).fetchone()
        if person is None or person['deleted_at'] is not None or person['status'] == 'merged':
            errors.append('unavailable person exposed')
    return errors


def _seed(app, conn, fixture):
    repo = app['repo']
    people, records = {}, {}
    for person in fixture['persons']:
        pid = repo.create_person(conn, person['name'], status='active')
        people[person['key']] = pid
        for alias in person.get('aliases', []):
            repo.add_alias(conn, pid, alias, source='eval-fictional')
    for item in fixture.get('fields', []):
        pid = people[item['person']]
        eid = repo.add_evidence(conn, 'text', source_ref='eval-v08:' + item['key'], excerpt=item['value'])
        repo.upsert_field(conn, pid, item['field'], item['value'], 1., eid, valid_from=item['valid_from'])
        records[item['key']] = {'type': 'field', 'id': _encoded([pid, item['field'], item['valid_from']])}
    for item in fixture.get('memories', []):
        eid = repo.add_evidence(conn, 'text', source_ref='eval-v08:' + item['key'], excerpt=item['text'])
        mid = repo.add_memory(conn, 'semantic', item['text'], people.get(item.get('person')), .95, eid,
                              ts=item['ts'])
        records[item['key']] = {'type': 'memory', 'id': mid}
    for item in fixture.get('interactions', []):
        reference = 'eval-v08:' + item['key']
        repo.add_evidence(conn, 'text', source_ref=reference, excerpt=item['text'])
        iid, _ = repo.insert_interaction(conn, item['ts'], item['channel'], item['text'], reference,
                                         [(people[item['person']], 'participant')], kind=item['kind'])
        records[item['key']] = {'type': 'interaction', 'id': iid}
    conn.commit()
    return people, records


def _actions(app, conn, actions, people, records):
    for action in actions:
        operation = action['op']
        if operation in {'archive_person', 'restore_person'}:
            getattr(app['repo'], operation)(conn, people[action['person']])
            conn.commit()
            continue
        mid = records[action['memory']]['id']
        revision = conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0]
        body = {'action': operation, 'expected_revision': revision, 'request_id': app['util'].new_id()}
        if 'text' in action:
            body['text'] = action['text']
        if 'person' in action:
            body['person_id'] = people.get(action['person'])
        app['corrections'].correct_memory(conn, mid, body)


def _sources(result):
    return [hit.get('source') for hit in result.get('results', []) if isinstance(hit, dict)] \
        if 'results' in result else result.get('sources', [])


def _retrieval_case(app, conn, case, people, records):
    _actions(app, conn, case.get('actions', []), people, records)
    tool, args = case.get('tool', 'query_memory'), dict(case.get('args', {}))
    if tool == 'query_memory':
        args.update(query=case['question'], top_k=5)
    result = app['tools'].execute_tool(conn, tool, args)
    sources = _sources(result)
    ranked = [record_key({'type': hit.get('type'), 'entity_id': hit.get('id')}, records)
              or 'unlabelled:' + str(hit.get('id'))
              for hit in result.get('results', [])]
    expect, checks = case['expect'], []
    if expect.get('ambiguity'):
        candidates = {item['person_id'] for item in result.get('candidates', [])}
        checks.append(_check('exact_ambiguity_candidates', result.get('needs_confirmation') is True
                             and result.get('ok') is False
                             and candidates == {people[key] for key in expect['ambiguity']} and not ranked))
    elif expect.get('error'):
        checks.append(_check('unavailable_person_rejected', result.get('ok') is False and not ranked))
    else:
        checks.append(_check('tool_completed', result.get('ok') is True))
    if expect.get('empty'):
        checks.append(_check('empty_result', not ranked))
    checks.append(_check('excluded_records_absent', not set(ranked) & set(expect.get('excluded', []))))
    for index, source in enumerate(sources):
        failures = source_checks(conn, source)
        if tool == 'query_memory':
            hit = result['results'][index]
            if not isinstance(source, dict) or any(source.get(key) != hit.get(column) for key, column in
                                                   (('type', 'type'), ('entity_id', 'id'), ('text', 'text'))):
                failures.append('source differs from retrieved hit')
        checks.append(_check('source_snapshot:' + str(index), not failures, failures=failures))
        owners = {(source.get('person') or {}).get('id')} if isinstance(source, dict) else set()
        if isinstance(source, dict):
            owners.update(person['id'] for person in source.get('people', []))
        checks.append(_check('person_isolation:' + str(index),
                             not owners & {people[key] for key in expect.get('excluded_persons', [])}))
        if expect.get('personal_only'):
            checks.append(_check('personal_scope:' + str(index), not owners - {None}))
    for expected in expect.get('fields', []):
        checks.append(_check('field_temporal_state:' + expected['record'], any(
            record_key(source, records) == expected['record']
            and source.get('temporal_status') == expected['temporal_status']
            and source.get('text') == expected['text'] for source in sources if isinstance(source, dict))))
    for needle in expect.get('snapshot_contains', []):
        checks.append(_check('snapshot_content:' + needle,
                             any(needle in str(source.get('text', '')) for source in sources if source)))
    metrics = ranking_metrics(ranked, expect.get('relevant', [])) if tool == 'query_memory' else {}
    return {'id': case['id'], 'tags': case.get('tags', []), 'question': case['question'],
            'tool': tool, 'actual_arguments': args, 'expected': expect, 'result': result,
            'ranked_records': ranked, 'metrics': metrics, 'deterministic_checks': checks,
            'semantic_review': {'status': 'not_applicable_fixed_retrieval'}}


def _data_snapshot(conn):
    # Capture actual extraction side effects instead of silently replacing extraction.
    return {table: {str(row['_eval_rowid']): dict(row) for row in conn.execute(
        'SELECT rowid AS _eval_rowid,* FROM ' + table + ' ORDER BY rowid')}
        for table in ('person', 'person_alias', 'person_field', 'memory_item', 'pending_item',
                      'interaction', 'interaction_person', 'relation_edge', 'promise', 'ledger_entry',
                      'reminder', 'applied_patch', 'memory_correction', 'evidence')}


def _changes(before, after):
    changes = {}
    for table in before:
        old, new = before[table], after[table]
        added = [value for key, value in new.items() if key not in old]
        removed = [value for key, value in old.items() if key not in new]
        updated = [{'before': old[key], 'after': new[key]} for key in old.keys() & new.keys()
                   if old[key] != new[key]]
        if added or removed or updated:
            changes[table] = {'added': added, 'removed': removed, 'updated': updated}
    return changes


def _write_checks(app, conn, result, expect, people, changes):
    """Compare actual persistent changes, independently of the model's intent label."""
    checks = []
    if 'intent' in expect:
        actual = result.get('turn_intent', {}).get('intent')
        checks.append(_check('classified_turn_intent', actual == expect['intent'],
                             expected=expect['intent'], actual=actual))
    if expect.get('no_memory_writes'):
        unexpected = sorted(table for table in changes if table != 'evidence')
        checks.append(_check('query_has_no_memory_writes', not unexpected, changed_tables=unexpected))
    for field in expect.get('saved_fields', []):
        current = app['repo'].current_fields(conn, people[field['person']]).get(field['field'])
        checks.append(_check('stored_field:' + field['person'] + ':' + field['field'],
                             current is not None and current['value'] == field['value'],
                             expected=field['value'], actual=current['value'] if current else None))
    return checks


def _material(app, conn, records):
    material = {}
    for key, record in records.items():
        kind, entity = record['type'], record['id']
        if kind == 'memory':
            source = app['sources'].memory_source(conn, entity)
            stored = conn.execute('SELECT * FROM memory_item WHERE id=?', (entity,)).fetchone()
        elif kind == 'interaction':
            source = app['sources'].interaction_source(conn, entity)
            stored = conn.execute('SELECT * FROM interaction WHERE id=?', (entity,)).fetchone()
        else:
            identity = json.loads(entity)
            source = next((source for source in app['sources'].field_sources(conn, identity[0], limit=100)
                           if _canonical_entity(source) == entity), None)
            stored = conn.execute('SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from=?',
                                  identity).fetchone()
        material[key] = {'visible': source is not None, 'source': source,
                         'stored': dict(stored) if stored else None}
    return material


def _answer_checks(app, conn, result, expect, people, records):
    checks = []
    for index, source in enumerate(result.get('sources', [])):
        failures = source_checks(conn, source)
        checks.append(_check('cited_source_snapshot:' + str(index), not failures, failures=failures))
    dependencies = result.get('source_dependencies', result.get('sources', []))
    for index, source in enumerate(dependencies):
        if source.get('type') != 'profile_context':
            failures = source_checks(conn, source)
            checks.append(_check('exposed_source_snapshot:' + str(index), not failures, failures=failures))
        key = record_key(source, records)
        checks.append(_check('forbidden_source_not_exposed:' + str(index),
                             key not in expect.get('forbidden_source_records', []), record=key))
        if 'allowed_persons' in expect:
            owners = {(source.get('person') or {}).get('id')} - {None}
            owners.update(person['id'] for person in source.get('people', []))
            if source.get('type') == 'profile_context':
                owners.add(source['entity_id'])
            checks.append(_check('exposed_person_scope:' + str(index),
                                 owners <= {people[key] for key in expect['allowed_persons']}))
    warning_codes = [warning['code'] for warning in result.get('source_warnings', [])]
    checks.append(_check('no_fabricated_citation', 'invalid_citation' not in warning_codes))
    return checks


def _dialogue(app, conn, case, people, records):
    cid = app['repo'].create_conversation(conn, title='Evaluation ' + case['id'])
    conn.commit()
    turns, prior = [], []
    for number, turn in enumerate(case['turns'], 1):
        started = time.monotonic()
        _actions(app, conn, turn.get('actions', []), people, records)
        expect, checks, prior_states = turn['expect'], [], []
        for old in prior:
            current_row = conn.execute('SELECT content,tool_calls FROM message WHERE id=?', (old['id'],)).fetchone()
            checks.append(_check('old_answer_immutable:' + old['id'], current_row is not None
                                 and dict(current_row) == old['row']))
            for source in old['dependencies']:
                state = app['sources'].source_state(conn, source)
                key = record_key(source, records)
                prior_states.append({'answer_id': old['id'], 'record': key,
                                     'source': source, 'current_state': state})
                wanted = expect.get('previous_source_states', {}).get(key)
                if wanted:
                    checks.append(_check('old_source_state:' + str(key), state['status'] == wanted,
                                         expected=wanted, actual=state['status']))
        material, before = _material(app, conn, records), _data_snapshot(conn)
        before_messages = {row['id'] for row in app['repo'].list_messages(conn, cid, limit=-1)}
        try:
            result = app['harness'].run_turn(conn, cid, turn['question'])
            rows = [dict(row) for row in app['repo'].list_messages(conn, cid, limit=-1)
                    if row['id'] not in before_messages]
            trace = [{'id': row['id'], 'role': row['role'], 'content': row['content'],
                      'metadata': json.loads(row['tool_calls']) if row['tool_calls'] else None}
                     for row in rows if row['role'] in {'assistant', 'tool'}]
            checks.extend(_answer_checks(app, conn, result, expect, people, records))
            changes = _changes(before, _data_snapshot(conn))
            checks.extend(_write_checks(app, conn, result, expect, people, changes))
            answer_row = conn.execute('SELECT content,tool_calls FROM message WHERE id=?',
                                      (result['assistant_message_id'],)).fetchone()
            prior.append({'id': result['assistant_message_id'], 'row': dict(answer_row),
                          'dependencies': result.get('source_dependencies', result.get('sources', []))})
            relevant = set(expect.get('source_records', []))
            exposed = {record_key(source, records) for source in result.get('source_dependencies', [])}
            cited = {record_key(source, records) for source in result.get('sources', [])}
            detail = {'turn': number, 'question': turn['question'], 'actions': turn.get('actions', []),
                      'expected': expect, 'expected_material': material, 'result': result, 'tool_trace': trace,
                      'data_changes': changes, 'prior_answer_states': prior_states,
                      'deterministic_checks': checks,
                      'limited_fact_checks': limited_fact_checks(result['answer'], expect.get('limited_fact_rules', [])),
                      'evidence_access': {'relevant_record_labels': sorted(relevant),
                                          'exposed_relevant_records': sorted(relevant & exposed),
                                          'cited_relevant_records': sorted(relevant & cited),
                                          'not_exposed_relevant_records': sorted(relevant - exposed),
                                          'meaning': 'Diagnostic evidence access only, not required citations or semantic correctness'},
                      'semantic_review': {'status': 'review_pending', 'ratings': None,
                                          'review_points': expect.get('review_points', [])}}
        except Exception as exc:
            detail = {'turn': number, 'question': turn['question'], 'actions': turn.get('actions', []),
                      'expected': expect, 'expected_material': material, 'error': _safe_error(app, exc),
                      'deterministic_checks': checks,
                      'limited_fact_checks': limited_fact_checks('', expect.get('limited_fact_rules', [])),
                      'semantic_review': {'status': 'not_generated', 'ratings': None},
                      'data_changes': _changes(before, _data_snapshot(conn)), 'prior_answer_states': prior_states}
        detail['elapsed_seconds'] = round(time.monotonic() - started, 3)
        turns.append(detail)
    return {'id': case['id'], 'tags': case.get('tags', []), 'turns': turns}


def _safe_error(app, exc):
    message = str(exc)
    key = app['settings'].llm_api_key
    if key:
        message = message.replace(key, '[redacted]')
    return {'type': type(exc).__name__, 'message': message[:1000]}


def _load_app(app_root, directory, track):
    root, directory = Path(app_root).resolve(), Path(directory)
    sys.path.insert(0, str(root))
    config = importlib.import_module('app.config')
    imported_root = Path(config.__file__).resolve().parents[1]
    if imported_root != root:
        raise RuntimeError('Selected app root differs from imported implementation; use an isolated process')
    kwargs = {'data_dir': str(directory), 'db_path': str(directory / 'eval.db'),
              'blob_dir': str(directory / 'blobs'), 'model_embed': ''}
    if track == 'retrieval':
        kwargs['llm_api_key'] = ''
    settings = config.Settings(_env_file=RUNNER_ROOT / '.env' if track == 'answers' else None, **kwargs)
    # This worker has its own environment, module cache and SQLite connection.
    # No global monkeypatches or shared connections cross concurrent workers.
    for key, value in settings.model_dump().items():
        os.environ['HOME_' + key.upper()] = str(value)
    config.get_settings.cache_clear()
    if track == 'retrieval':
        def forbidden(*args, **kwargs):
            raise AssertionError('Network forbidden in fixed FTS retrieval evaluation')
        socket.socket.connect = forbidden
        socket.socket.connect_ex = forbidden
        socket.create_connection = forbidden
    modules = {'db': 'app.store.db', 'repo': 'app.store.repo', 'tools': 'app.agent.tools',
               'harness': 'app.agent.harness', 'sources': 'app.agent.memory_sources',
               'corrections': 'app.memory.memory_corrections', 'util': 'app.util'}
    return {key: importlib.import_module(module) for key, module in modules.items()} | {'settings': settings}


def _worker(job):
    with TemporaryDirectory(prefix='home-question-worker-') as directory:
        app = _load_app(job['app_root'], directory, job['track'])
        conn = app['db']._connect(Path(directory) / 'eval.db')
        try:
            app['db'].migrate(conn)
            people, records = _seed(app, conn, job['fixture'])
            if job['track'] == 'retrieval':
                detail = _retrieval_case(app, conn, job['case'], people, records)
            else:
                detail = _dialogue(app, conn, job['case'], people, records)
            return {'detail': detail, 'model': app['settings'].model_mid if job['track'] == 'answers' else None,
                    'configured_model_tiers': {'mini': app['settings'].model_mini, 'mid': app['settings'].model_mid}
                        if job['track'] == 'answers' else None,
                    'source_hash_after': app_source_hash(job['app_root'])}
        finally:
            conn.close()


def _execute(job, timeout=600):
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--_worker'],
                                 input=json.dumps(job, ensure_ascii=False), text=True, capture_output=True,
                                 timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'detail': {'id': job['case']['id'], 'error': {'type': 'WorkerTimeout',
                            'message': 'Scenario exceeded ' + str(timeout) + ' seconds; worker terminated'}}}
    if process.returncode:
        return {'detail': {'id': job['case']['id'], 'error': {'type': 'WorkerError',
                            'message': 'Isolated worker exited with code ' + str(process.returncode)}}}
    try:
        return json.loads(process.stdout)
    except ValueError:
        return {'detail': {'id': job['case']['id'], 'error': {'type': 'WorkerProtocolError',
                            'message': 'Isolated worker did not return a JSON report'}}}


def summarize(track, details):
    units = details if track == 'retrieval' else [turn for item in details for turn in item.get('turns', [])]
    checks = [check for item in units for check in item.get('deterministic_checks', [])]
    facts = [check for item in units for check in item.get('limited_fact_checks', [])]
    errors = sum('error' in item for item in units)
    if track == 'answers':
        errors += sum('error' in item and not item.get('turns') for item in details)
    summary = {'scenarios': len(details), 'units': len(units), 'runtime_errors': errors,
               'scenario_errors': sum('error' in item for item in details),
               'failed_turns': sum('error' in item for item in units) if track == 'answers' else None,
               'deterministic_checks': {'total': len(checks), 'passed': sum(check['passed'] for check in checks),
                                        'failed': sum(not check['passed'] for check in checks)},
               'limited_fact_checks': {'total': len(facts), 'passed': sum(check['passed'] for check in facts),
                                        'failed': sum(not check['passed'] for check in facts),
                                        'meaning': 'Limited string rules, not semantic answer accuracy'},
               'semantic_review': {'status': 'review_pending' if track == 'answers' else 'not_applicable',
                                   'pending_turns': sum(item.get('semantic_review', {}).get('status') == 'review_pending'
                                                        for item in units), 'reviewed_turns': 0, 'score': None}}
    if track == 'retrieval':
        positives = [item for item in details if item.get('metrics', {}).get('recall_at_5') is not None]
        summary['retrieval'] = {'positive_queries': len(positives),
                                'nonranking_cases': len(details) - len(positives),
                                **{metric: sum(item['metrics'][metric] for item in positives) / len(positives)
                                   if positives else None
                                   for metric in ('recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr_at_5')}}
    return summary


def evaluate(fixture=None, *, track='retrieval', app_root=None, repeat=1, tags=None, limit=None,
             progress=None, checkpoint=None, worker_timeout=600):
    fixture = fixture or load_fixture()
    validate_fixture(fixture)
    if track not in {'retrieval', 'answers'} or repeat < 1 or limit is not None and limit < 1 or worker_timeout <= 0:
        raise ValueError('Invalid track, repeat or limit')
    app_root = Path(app_root or os.environ.get('HOME_EVAL_APP_ROOT') or RUNNER_ROOT).resolve()
    before_hash = app_source_hash(app_root)
    app_git, runner_git = _git_info(app_root), _git_info(RUNNER_ROOT)
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    created_at = datetime.now(timezone.utc)
    report_id = created_at.strftime('%Y%m%d-%H%M%S-%f') + '-' + str(os.getpid())
    cases = fixture['retrieval_cases' if track == 'retrieval' else 'dialogues']
    if tags:
        cases = [case for case in cases if set(tags) & set(case.get('tags', []))]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError('No evaluation cases selected')
    details, models, source_changed = [], set(), False
    report = {'evaluation': 'memory-questions-v08', 'schema_version': 1, 'track': track,
              'created_at': created_at.isoformat(), 'report_id': report_id, 'run_status': 'running',
              'fixture_version': fixture['version'], 'fixture_hash': fixture_hash(fixture),
              'corpus_provenance': fixture['provenance'], 'corpus_description': fixture.get('description'),
              'app_root': str(app_root), 'app_source_hash': before_hash,
              'runner_source_hash': runner_hash, 'app_source_changed_during_run': False,
              'app_git': app_git, 'runner_git': runner_git,
              'configuration': {'repeat': repeat, 'tags': tags or [], 'limit': limit,
                                'worker_timeout_seconds': worker_timeout,
                                'models': [], 'configured_model_tiers': [], 'embedding': 'disabled', 'cache': False,
                                'runtime': 'separate process and temporary SQLite per scenario and repeat',
                                'retrieval_mode': 'FTS only; network forbidden' if track == 'retrieval'
                                    else 'real harness including extraction; embeddings disabled'},
              'limitations': ['Fictional development corpus, not real users or an unseen semantic benchmark',
                              'String checks and source integrity do not establish answer correctness',
                              'Answer semantics require human review; no LLM self-grading'],
              'expectation_semantics': {
                  'relevant': 'Fixed-query relevance labels used only for retrieval ranking metrics',
                  'source_records': 'Potentially relevant evidence for human review, not mandatory citations or an answer score',
                  'limited_fact_rules': 'Literal string probes only; negation and entailment require human review',
                  'deterministic_checks': 'Stored provenance, identity scope, source status and lifecycle contracts',
                  'no_memory_writes': 'Actual domain-table changes excluding the original audit evidence, not the intent label',
                  'saved_fields': 'Exact current stored field value, not a model claim or a pending proposal'},
              'human_review_rubric': fixture.get('human_review_rubric', []),
              'selected_scenarios': len(cases) * repeat, 'details': details, 'summary': summarize(track, [])}
    if checkpoint:
        checkpoint(report)
    for repetition in range(1, repeat + 1):
        for case in cases:
            if progress:
                progress('Running ' + track + ' ' + case['id'] + ' repeat ' + str(repetition))
            result = _execute({'fixture': fixture, 'case': case, 'track': track, 'app_root': str(app_root)},
                              timeout=worker_timeout)
            detail = {'question': case.get('question'), 'tags': case.get('tags', []),
                      'expected': case.get('expect'), **result['detail'], 'repeat': repetition}
            if 'error' in detail:
                if track == 'retrieval':
                    detail['metrics'] = ranking_metrics([], case['expect'].get('relevant', []))
                else:
                    detail['turns'] = [{'turn': index, 'question': turn['question'], 'expected': turn['expect'],
                                        'error': detail['error'], 'deterministic_checks': [],
                                        'limited_fact_checks': limited_fact_checks('', turn['expect'].get('limited_fact_rules', [])),
                                        'semantic_review': {'status': 'not_generated', 'ratings': None}}
                                       for index, turn in enumerate(case['turns'], 1)]
            details.append(detail)
            if result.get('model'):
                models.add(result['model'])
            tiers = result.get('configured_model_tiers')
            if tiers and tiers not in report['configuration']['configured_model_tiers']:
                report['configuration']['configured_model_tiers'].append(tiers)
            if result.get('source_hash_after') and result['source_hash_after'] != before_hash:
                source_changed = True
            report['configuration']['models'] = sorted(models)
            report['app_source_changed_during_run'] = source_changed
            report['summary'] = summarize(track, details)
            if checkpoint:
                checkpoint(report)
    source_changed = source_changed or app_source_hash(app_root) != before_hash
    report.update(run_status='complete', app_source_changed_during_run=source_changed)
    if checkpoint:
        checkpoint(report)
    return report


def review_markdown(report):
    lines = ['# Memory questions evaluation', '',
             'Corpus: fictional records with natural-language questions.',
             'Semantic status: human review pending. No semantic accuracy score.', '',
             '- Track: ' + report['track'], '- Fixture SHA256: `' + report['fixture_hash'] + '`',
             '- Application SHA256: `' + report['app_source_hash'] + '`',
             '- Models: ' + ', '.join(report['configuration']['models']), '', '## Summary', '',
             '```json', json.dumps(report['summary'], ensure_ascii=False, indent=2), '```', '', '## Human rubric', '']
    lines.extend('- ' + point for point in report['human_review_rubric'])
    for scenario in report['details']:
        lines.extend(['', '## ' + scenario['id'] + ' (repeat ' + str(scenario['repeat']) + ')', ''])
        units = scenario.get('turns', [scenario])
        for unit in units:
            lines.extend(['### Turn ' + str(unit.get('turn', 1)), '', unit.get('question', ''), ''])
            result = unit.get('result', {})
            if report['track'] == 'answers':
                lines.extend(['#### Actual answer', '', result.get('answer', '(No answer generated)'), '',
                              'Review rating: PENDING', '', 'Reviewer notes: ', ''])
                lines.extend('- ' + point for point in unit.get('expected', {}).get('review_points', []))
            blocks = {'Expected labels': unit.get('expected'), 'Expected material before the answer': unit.get('expected_material'),
                      'Actual sources': result.get('sources'), 'Exposed dependencies': result.get('source_dependencies'),
                      'Tool trace': unit.get('tool_trace'), 'Actual outcomes': result.get('outcomes'),
                      'Data changes during real extraction and tools': unit.get('data_changes'),
                      'Prior answer source states': unit.get('prior_answer_states'),
                      'Deterministic checks': unit.get('deterministic_checks'),
                      'Limited fact string rules': unit.get('limited_fact_checks'),
                      'Evidence access diagnostics': unit.get('evidence_access'),
                      'Fixed retrieval result': result if report['track'] == 'retrieval' else None,
                      'Runtime error': unit.get('error')}
            for title, value in blocks.items():
                if value is not None:
                    lines.extend(['#### ' + title, '', '```json', json.dumps(value, ensure_ascii=False, indent=2), '```', ''])
    return '\n'.join(lines) + '\n'


def write_report(report, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = report['track'] + '-' + report['report_id'] + '-' + report['app_source_hash'][:8]
    json_path, review_path = directory / (stem + '.json'), directory / (stem + '-review.md')
    for path, content in ((json_path, json.dumps(report, ensure_ascii=False, indent=2) + '\n'),
                          (review_path, review_markdown(report))):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)
    return json_path, review_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=('retrieval', 'answers'), default='retrieval')
    parser.add_argument('--cases', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--app-root', type=Path, default=None,
                        help='Implementation directory; also HOME_EVAL_APP_ROOT. Selected before importing app.')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--tag', action='append', default=[])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--worker-timeout', type=float, default=600,
                        help='Maximum seconds per scenario; timed-out model workers are terminated')
    parser.add_argument('--output-dir', type=Path, default=RUNNER_ROOT / 'eval/results/memory-questions-v08')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        job = json.load(sys.stdin)
        try:
            with redirect_stdout(sys.stderr):
                result = _worker(job)
        except Exception as exc:
            result = {'detail': {'id': job['case']['id'], 'error': {'type': type(exc).__name__,
                               'message': 'Worker setup failed; check implementation and model configuration'}}}
        print(json.dumps(result, ensure_ascii=False))
        return 0
    try:
        report = evaluate(load_fixture(args.cases), track=args.track, app_root=args.app_root,
                          repeat=args.repeat, tags=args.tag, limit=args.limit,
                          worker_timeout=args.worker_timeout,
                          checkpoint=lambda partial: write_report(partial, args.output_dir),
                          progress=lambda message: print(message, file=sys.stderr, flush=True))
        paths = write_report(report, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    print('Report: ' + str(paths[0]))
    print('Review: ' + str(paths[1]))
    return int(bool(report['summary']['runtime_errors'] or report['summary']['deterministic_checks']['failed']
                    or report['summary']['limited_fact_checks']['failed'] or report['app_source_changed_during_run']))


if __name__ == '__main__':
    raise SystemExit(main())
