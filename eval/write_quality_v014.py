"""Observe real write/retry/draft paths in isolated, fictional development cases.

No model substitutes, semantic self-grading, or automatic proposal confirmation.
The CLI is opt-in to real model calls. Importing and validating fixtures is offline.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from eval import memory_questions as base
from eval.relationship_workflow_v013 import _audit_model_responses

RUNNER_VERSION = '0.14.0-write-quality-1'
DEFAULT_FIXTURE = base.RUNNER_ROOT / 'eval/fixtures/write_quality_v014.json'
COHORTS = {'known_regression', 'new_wording'}


def validate_fixture(fixture):
    if fixture.get('provenance') != base.PROVENANCE:
        raise ValueError('Fictional provenance must be explicit')
    people = [person['key'] for person in fixture.get('persons', [])]
    if not people or len(set(people)) != len(people):
        raise ValueError('People must have unique keys')
    people = set(people)
    record_keys = [item['key'] for group in ('fields', 'memories', 'interactions')
                   for item in fixture.get(group, [])]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError('Record keys must be unique')
    for group in ('fields', 'memories', 'interactions'):
        if any(item.get('person') not in people for item in fixture.get(group, [])):
            raise ValueError('Unknown record owner')
    cases = fixture.get('scenarios', [])
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Scenarios must have unique IDs')
    for case in cases:
        if case.get('cohort') not in COHORTS or not case.get('steps'):
            raise ValueError('Explicit cohort and nonempty steps are required')
        for index, step in enumerate(case['steps'], 1):
            op = step.get('op')
            if op not in {'chat', 'retry', 'prepare'}:
                raise ValueError('Unsupported step operation')
            if op == 'retry':
                source = step.get('source_step')
                if type(source) is not int or not 1 <= source < index \
                        or case['steps'][source - 1].get('op') != 'chat':
                    raise ValueError('Retry must refer to an earlier original chat step')
            elif not isinstance(step.get('text' if op == 'chat' else 'goal'), str) \
                    or not step['text' if op == 'chat' else 'goal'].strip():
                raise ValueError('Nonempty user text is required')
            if op == 'prepare' and step.get('person') not in people:
                raise ValueError('Unknown preparation person')
            if any(key in step for key in ('person_id', 'person_ids')):
                raise ValueError('Hidden chat person IDs are forbidden')
            expect = step.get('expect')
            if not isinstance(expect, dict) or not expect.get('review_points'):
                raise ValueError('Every step requires an explicit review rubric')
            if expect.get('intent') not in {None, 'query', 'record', 'mixed', 'unknown'}:
                raise ValueError('Unsupported intent expectation')
            for key in ('allowed_persons', 'unchanged_persons', 'ambiguity'):
                if not set(expect.get(key, [])) <= people:
                    raise ValueError('Unknown person expectation')
            for key in ('saved_fields', 'absent_fields'):
                for field in expect.get(key, []):
                    if field.get('person') not in people or not field.get('field') \
                            or key == 'saved_fields' and 'value' not in field:
                        raise ValueError('Explicit field contract is required')
            for key in ('required_dependency_records', 'forbidden_source_records'):
                if not set(expect.get(key, [])) <= set(record_keys):
                    raise ValueError('Unknown source record expectation')


def load_fixture(path=DEFAULT_FIXTURE):
    fixture = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_fixture(fixture)
    return fixture


def _check(name, passed, **details):
    return {'name': name, 'passed': bool(passed), **details}


def _owned_state(conn, person_id):
    """Stored personal rows, including same-count updates and historical fields."""
    return {table: [dict(row) for row in conn.execute(
        'SELECT * FROM ' + table + ' WHERE ' + column + '=? ORDER BY rowid', (person_id,))]
        for table, column in [('person', 'id'), ('person_alias', 'person_id'),
                              ('person_field', 'person_id'), ('memory_item', 'person_id')]}


def _stored_state(conn):
    return {table: [dict(row) for row in conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
            for table in ('person', 'person_alias', 'person_field', 'memory_item', 'pending_item',
                          'interaction', 'interaction_person', 'applied_patch', 'promise', 'evidence')}


def _extra_checks(app, conn, result, expect, people, owned_before, changes=None):
    checks = []
    if expect.get('saved_fields'):
        allowed = {people[field['person']] for field in expect['saved_fields']}
        foreign = []
        for table, column in [('person', 'id'), ('person_alias', 'person_id'), ('person_field', 'person_id'),
                              ('memory_item', 'person_id'), ('interaction_person', 'person_id')]:
            delta = (changes or {}).get(table, {})
            rows = [*delta.get('added', []), *delta.get('removed', []),
                    *(row[version] for row in delta.get('updated', []) for version in ('before', 'after'))]
            foreign.extend({'table': table, 'person_id': row.get(column)}
                           for row in rows if row.get(column) not in allowed)
        checks.append(_check('write_person_scope', not foreign, foreign_changes=foreign))
    if 'allowed_persons' in expect:
        scope = result.get('conversation_scope', {})
        checks.append(_check('resolved_scope_persons', scope.get('blocked') is False
                             and set(scope.get('person_ids', [])) == {people[key] for key in expect['allowed_persons']},
                             actual=scope))
    for field in expect.get('absent_fields', []):
        value = app['repo'].current_fields(conn, people[field['person']]).get(field['field'])
        checks.append(_check('absent_field:' + field['person'] + ':' + field['field'], value is None,
                             actual=value))
    for person in expect.get('unchanged_persons', []):
        after = _owned_state(conn, people[person])
        checks.append(_check('unrelated_person_unchanged:' + person, after == owned_before[person]))
    if expect.get('no_pending'):
        pending = [dict(row) for row in conn.execute('SELECT * FROM pending_item WHERE resolved_at IS NULL')]
        checks.append(_check('no_unexpected_pending', not pending,
                             pending_ids=[row['id'] for row in pending]))
    if expect.get('ambiguity'):
        scope = result.get('conversation_scope', {})
        candidates = {item.get('person_id') for item in scope.get('candidates', [])}
        checks.append(_check('exact_ambiguity_candidates', scope.get('blocked') is True
                             and candidates == {people[key] for key in expect['ambiguity']},
                             actual=scope))
    return checks


def _prepare(conn, person_id, goal, request_id):
    """Use production HTTP submission and durable task, synchronously dispatched."""
    from fastapi.testclient import TestClient
    from app.api import main
    from app.jobs import tasks
    with patch.object(tasks, 'start', lambda tid: None):
        client = TestClient(main.app)
        try:
            response = client.post(f'/api/persons/{person_id}/followups',
                                   json={'goal': goal, 'request_id': request_id})
            queued = response.json()
            if response.status_code != 202:
                raise RuntimeError('Followup submission returned HTTP ' + str(response.status_code))
            tasks._run(queued['id'])
            finished = client.get('/api/tasks/' + queued['id']).json()
            record_response = client.get('/api/followups/' + queued['payload']['followup_id'])
            record = record_response.json()
            if record_response.status_code != 200 or finished.get('status') != 'done':
                raise RuntimeError('Followup durable task failed: ' + str(finished.get('error')))
            return {'answer': record.get('draft', ''), 'followup': record, 'task': finished,
                    'queued': queued}
        finally:
            client.close()


def run_scenario(app, conn, fixture, case, *, model_audit=None):
    people, records = base._seed(app, conn, fixture)
    cid = app['repo'].create_conversation(conn, title='v0.14 evaluation ' + case['id'])
    conn.commit()
    steps = []
    for number, step in enumerate(case['steps'], 1):
        before = base._data_snapshot(conn)
        owned = {key: _owned_state(conn, pid) for key, pid in people.items()}
        old_messages = {row['id'] for row in app['repo'].list_messages(conn, cid, limit=-1)}
        first_call = len(model_audit or [])
        started = time.monotonic()
        entry = {'step': number, 'operation': step['op'], 'text': step.get('text', step.get('goal')),
                 'expected': step['expect'], 'stored_before': _stored_state(conn),
                 'deterministic_checks': [], 'semantic_review': {'status': 'review_pending',
                    'score': None, 'review_points': step['expect']['review_points']}}
        steps.append(entry)
        try:
            op = step['op']
            if op == 'chat':
                result = app['harness'].run_turn(conn, cid, step['text'])
            elif op == 'retry':
                original = steps[step['source_step'] - 1]
                user_id = original.get('user_message_id')
                if not user_id:
                    raise RuntimeError('Original chat did not persist a user message; retry unavailable')
                entry.update(text=original['text'], source_step=step['source_step'], user_message_id=user_id)
                # Production durable tasks retry with this original ID. Do not
                # create a successor message or borrow the latest person's ID.
                result = app['harness'].run_turn(conn, cid, original['text'], user_message_id=user_id)
            else:
                result = _prepare(conn, people[step['person']], step['goal'], case['id'] + ':' + str(number))
            entry['result'] = result
            changes = base._changes(before, base._data_snapshot(conn))
            checks = base._write_checks(app, conn, result, step['expect'], people, changes)
            checks.extend(_extra_checks(app, conn, result, step['expect'], people, owned, changes))
            if op != 'prepare':
                checks.extend(base._answer_checks(app, conn, result, step['expect'], people, records))
            else:
                record = result['followup']
                valid_ids = {source['id'] for source in record['brief']['sources']}
                checks.extend([
                    _check('draft_ready_and_nonempty', record['status'] == 'ready'
                           and isinstance(result.get('answer'), str) and bool(result['answer'].strip())),
                    _check('draft_source_ids_are_real', bool(record['draft_source_ids'])
                           and set(record['draft_source_ids']) <= valid_ids)])
            checks.append(_check('nonempty_answer', isinstance(result.get('answer'), str)
                                 and bool(result['answer'].strip())))
            if op == 'retry':
                checks.append(_check('retry_creates_no_user_message', not any(
                    row['id'] not in old_messages and row['role'] == 'user'
                    for row in app['repo'].list_messages(conn, cid, limit=-1))))
                checks.append(_check('retry_keeps_original_user_binding',
                                     result.get('conversation_scope', {}).get('user_message_id') == user_id))
            entry['deterministic_checks'] = checks
        except Exception as error:
            entry['error'] = base._safe_error(app, error)
            entry['semantic_review']['status'] = 'not_generated'
        finally:
            rows = [dict(row) for row in app['repo'].list_messages(conn, cid, limit=-1)
                    if row['id'] not in old_messages]
            user_rows = [row for row in rows if row['role'] == 'user']
            if step['op'] == 'chat' and len(user_rows) == 1:
                entry['user_message_id'] = user_rows[0]['id']
            entry.update(stored_after=_stored_state(conn), data_changes=base._changes(before, base._data_snapshot(conn)),
                         message_trace=[{'id': row['id'], 'role': row['role'], 'content': row['content'],
                             'metadata': json.loads(row['tool_calls']) if row['tool_calls'] else None} for row in rows],
                         elapsed_seconds=round(time.monotonic() - started, 3),
                         model_calls=list(range(first_call + 1, len(model_audit or []) + 1)))
    return {'id': case['id'], 'cohort': case['cohort'], 'tags': case.get('tags', []),
            'conversation_id': cid, 'people': people, 'records': records, 'steps': steps}


def summarize(details):
    steps = [step for detail in details for step in detail.get('steps', [])]
    checks = [check for step in steps for check in step.get('deterministic_checks', [])]
    errors = sum('error' in step for step in steps)
    return {'scenarios': len(details), 'steps': len(steps), 'generated_steps': sum('result' in step for step in steps),
            'runtime_errors': errors, 'checks': len(checks), 'checks_passed': sum(c['passed'] for c in checks),
            'checks_failed': sum(not c['passed'] for c in checks),
            'all_checks_passed': bool(steps) and not errors and bool(checks) and all(c['passed'] for c in checks),
            'semantic_review': {'status': 'review_pending', 'score': None,
                                'meaning': 'DB and source checks do not establish semantic task completion'}}


def _worker(job):
    with TemporaryDirectory(prefix='home-write-quality-v014-') as directory:
        app = base._load_app(job['app_root'], directory, 'answers')
        conn = app['db'].get_conn()
        audit = []
        try:
            with _audit_model_responses(audit):
                detail = run_scenario(app, conn, job['fixture'], job['case'], model_audit=audit)
            return {'detail': detail, 'model_response_audit': audit,
                    'configured_model_tiers': {'mini': app['settings'].model_mini, 'mid': app['settings'].model_mid},
                    'app_source_hash_after': base.app_source_hash(job['app_root'])}
        finally:
            app['db'].reset_for_tests(Path(directory) / 'eval.db')


def _execute(job, timeout):
    try:
        process = subprocess.run([sys.executable, '-m', 'eval.write_quality_v014', '--_worker'],
                                 input=json.dumps(job, ensure_ascii=False), text=True, capture_output=True,
                                 cwd=str(base.RUNNER_ROOT), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'error': {'type': 'WorkerTimeout', 'message': 'Scenario exceeded timeout; worker terminated'}}
    if process.returncode:
        return {'error': {'type': 'WorkerError', 'message': 'Worker exited with code ' + str(process.returncode)}}
    try:
        return json.loads(process.stdout)
    except ValueError:
        return {'error': {'type': 'WorkerProtocolError', 'message': 'Worker did not produce JSON'}}


def evaluate(fixture=None, *, app_root=None, cohort='all', timeout=900, progress=None, checkpoint=None):
    fixture = fixture or load_fixture()
    validate_fixture(fixture)
    if cohort not in COHORTS | {'all'} or timeout <= 0:
        raise ValueError('Invalid cohort or timeout')
    root = Path(app_root or base.RUNNER_ROOT).resolve()
    cases = [case for case in fixture['scenarios'] if cohort == 'all' or case['cohort'] == cohort]
    if not cases:
        raise ValueError('No scenarios selected')
    source_hash = base.app_source_hash(root)
    report = {'evaluation': 'write-quality-v014', 'runner_version': RUNNER_VERSION,
              'created_at': datetime.now(timezone.utc).isoformat(),
              'report_id': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f') + '-' + str(os.getpid()),
              'run_status': 'running', 'fixture_version': fixture['version'],
              'fixture_hash': base.fixture_hash(fixture), 'app_source_hash': source_hash,
              'runner_source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_source_hashes': {name: hashlib.sha256((base.RUNNER_ROOT / name).read_bytes()).hexdigest()
                  for name in ('eval/memory_questions.py', 'eval/relationship_workflow_v013.py')},
              'app_git': base._git_info(root), 'app_source_changed_during_run': False,
              'configuration': {'cohort': cohort, 'repeat': 1, 'runtime': 'separate process and temporary SQLite per scenario',
                  'timeout_seconds': timeout, 'embedding': 'disabled', 'cache': False, 'configured_model_tiers': []},
              'limitations': [fixture['description'],
                  'No semantic accuracy score; deterministic checks and agent evidence review are separate',
                  'LLM.chat returned objects are captured; chat_stream aggregates remain in saved answer_review, not provider HTTP bytes or chunks',
                  'Retries call the real harness with original user_message_id; no transport failure is manufactured'],
              'human_review_rubric': fixture['human_review_rubric'], 'details': []}
    if checkpoint:
        checkpoint(report)
    for case in cases:
        if progress:
            progress('Running ' + case['cohort'] + ' ' + case['id'])
        result = _execute({'app_root': str(root), 'fixture': fixture, 'case': case}, timeout)
        if result.get('error'):
            result['detail'] = {'id': case['id'], 'cohort': case['cohort'], 'steps': [
                {'step': number, 'operation': step['op'], 'text': step.get('text', step.get('goal')),
                 'expected': step['expect'], 'error': result['error'], 'deterministic_checks': [],
                 'semantic_review': {'status': 'not_generated', 'score': None}}
                for number, step in enumerate(case['steps'], 1)]}
        detail = result['detail']
        detail['model_response_audit'] = result.get('model_response_audit', [])
        report['details'].append(detail)
        tiers = result.get('configured_model_tiers')
        if tiers and tiers not in report['configuration']['configured_model_tiers']:
            report['configuration']['configured_model_tiers'].append(tiers)
        report['app_source_changed_during_run'] |= result.get('app_source_hash_after', source_hash) != source_hash
        report['summary'] = summarize(report['details'])
        if checkpoint:
            checkpoint(report)
    report['run_status'] = 'complete'
    report['app_source_changed_during_run'] |= base.app_source_hash(root) != source_hash
    if checkpoint:
        checkpoint(report)
    return report


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = 'write-quality-' + report['report_id'] + '-' + report['app_source_hash'][:8]
    path = directory / (stem + '.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    lines = ['# Write quality v0.14', '', '内部虚构开发语料；语义逐步待审阅；不计算准确率。', '',
             '- App SHA256: `' + report['app_source_hash'] + '`',
             '- Fixture SHA256: `' + report['fixture_hash'] + '`',
             '- Runner SHA256: `' + report['runner_source_hash'] + '`', '',
             '```json', json.dumps(report.get('summary'), ensure_ascii=False, indent=2), '```', '']
    for case in report['details']:
        lines.extend(['## ' + case['id'] + ' (' + case['cohort'] + ')', ''])
        for step in case['steps']:
            lines.extend(['### Step ' + str(step['step']), '', step.get('text') or '(retry original message)', '',
                          step.get('result', {}).get('answer', '(No answer generated)'), '',
                          '代理证据审阅：PENDING', ''])
            lines.extend('- ' + point for point in step['expected'].get('review_points', []))
    review = directory / (stem + '-review.md')
    review.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path, review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--app-root', type=Path, default=base.RUNNER_ROOT)
    parser.add_argument('--output-dir', type=Path, default=base.RUNNER_ROOT / 'data/qa/write-quality-v014')
    parser.add_argument('--cohort', choices=sorted(COHORTS | {'all'}), default='all')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        with redirect_stdout(sys.stderr):
            result = _worker(json.loads(sys.stdin.read()))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    fixture = load_fixture(args.fixture)
    if args.validate_only:
        print(json.dumps({'fixture_hash': base.fixture_hash(fixture),
                          'fixture_file_hash': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                          'scenarios': len(fixture['scenarios']),
                          'steps': sum(len(case['steps']) for case in fixture['scenarios'])}))
        return 0
    report = evaluate(fixture, app_root=args.app_root, cohort=args.cohort, timeout=args.timeout,
                      progress=lambda text: print(text, file=sys.stderr),
                      checkpoint=lambda value: write_report(value, args.output_dir))
    paths = write_report(report, args.output_dir)
    print(json.dumps({'paths': [str(path) for path in paths], 'summary': report['summary']}, ensure_ascii=False))
    return 0 if report['summary']['all_checks_passed'] and not report['app_source_changed_during_run'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
