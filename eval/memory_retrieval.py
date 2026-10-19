"""Offline FTS retrieval and lifecycle evaluation, without an LLM quality score.

Run from the repository root:
    .venv/bin/python eval/memory_retrieval.py
    .venv/bin/python eval/memory_retrieval.py --json --output /tmp/retrieval.json
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import tools  # noqa: E402
from app.config import Settings  # noqa: E402
from app.memory import memory_corrections  # noqa: E402
from app.store import db, repo  # noqa: E402
from app.util import new_id, now_ms  # noqa: E402

DEFAULT_FIXTURE = ROOT / 'eval' / 'fixtures' / 'memory_retrieval.json'


def load_fixture(path=DEFAULT_FIXTURE):
    fixture = json.loads(Path(path).read_text(encoding='utf-8'))
    ids = [case['id'] for case in fixture['cases']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Evaluation requires nonempty, unique case IDs')
    for case in fixture['cases']:
        if not case.get('expect') or case['tool'] not in {'query_memory', 'get_person'}:
            raise ValueError('Each case requires an explicit expectation and supported tool')
        if case['tool'] == 'query_memory' and not any(
                key in case['expect'] for key in ('ids', 'error', 'needs_confirmation')):
            raise ValueError('Retrieval expectations must identify hits, exclusion or ambiguity')
    return fixture


@contextmanager
def offline_runtime(directory):
    """Isolate data and settings, and reject socket connections even on accidental code paths."""
    directory = Path(directory)
    settings = Settings(_env_file=None, llm_api_key='', model_embed='',
                        db_path=str(directory / 'unused.db'), blob_dir=str(directory / 'blobs'))
    with ExitStack() as stack:
        stack.enter_context(patch('app.config.get_settings', return_value=settings))
        stack.enter_context(patch('app.memory.compile_card.get_settings', return_value=settings))
        stack.enter_context(patch('app.llm.get_llm', side_effect=RuntimeError('Offline FTS evaluation')))
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')))
        yield


def seed_scenario(conn, fixture):
    identities, records = {}, {}
    for item in fixture['persons']:
        identities[item['key']] = repo.create_person(conn, item['name'], status='active')
    for item in fixture['memories']:
        eid = repo.add_evidence(conn, 'text', source_ref='eval:' + item['key'], excerpt=item['text'])
        records[item['key']] = repo.add_memory(conn, 'semantic', item['text'],
            identities[item['person']] if item['person'] else None, .95, eid)
    distractors = fixture.get('distractors', {})
    for index in range(distractors.get('count', 0)):
        text = distractors['text'] + str(index)
        eid = repo.add_evidence(conn, 'text', source_ref=f'eval:distractor:{index}', excerpt=text)
        records[f'distractor:{index}'] = repo.add_memory(conn, 'semantic', text,
            identities[distractors['person']], 1., eid)
    for item in fixture.get('fields', []):
        eid = repo.add_evidence(conn, 'text', source_ref='eval:city:' + item['value'], excerpt=item['value'])
        repo.upsert_field(conn, identities[item['person']], item['field'], item['value'], 1.,
                          eid, valid_from=item['valid_from'])
    for item in fixture.get('interactions', []):
        source_ref = 'eval:interaction:' + item['key']
        repo.add_evidence(conn, 'text', source_ref=source_ref, excerpt=item['text'])
        records[item['key']], _ = repo.insert_interaction(conn, now_ms(), item['channel'],
            item['text'], source_ref, [(identities[item['person']], 'participant')], kind=item['kind'])
    conn.commit()
    return identities, records


def apply_actions(conn, actions, identities, records):
    for action in actions:
        operation = action['op']
        if operation in {'archive_person', 'restore_person'}:
            getattr(repo, operation)(conn, identities[action['person']])
            conn.commit()
            continue
        if operation not in {'edit', 'withdraw', 'restore'}:
            raise ValueError('Unsupported fixture action: ' + operation)
        mid = records[action['memory']]
        revision = conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0]
        body = {'action': operation, 'expected_revision': revision, 'request_id': new_id()}
        if 'text' in action:
            body['text'] = action['text']
        if 'person_id' in action:
            body['person_id'] = identities[action['person_id']] if action['person_id'] else None
        memory_corrections.correct_memory(conn, mid, body)


def _source_failures(conn, source, *, hit=None):
    failures = []
    if not isinstance(source, dict):
        return ['missing source snapshot']
    if source.get('type') not in {'memory', 'interaction', 'field'}:
        return ['unsupported source type']
    if type(source.get('date')) is not int or type(source.get('captured_at')) is not int:
        failures.append('source lacks date or capture timestamp')
    evidence = source.get('evidence')
    stored_evidence = conn.execute('SELECT * FROM evidence WHERE id=?',
        (evidence.get('id') if isinstance(evidence, dict) else None,)).fetchone()
    if not stored_evidence:
        failures.append('source evidence is not a real fixture record')
    elif any(evidence.get(key) != stored_evidence[key] for key in
             ('id', 'kind', 'excerpt', 'source_ref', 'blob_hash', 'created_at')):
        failures.append('source evidence content differs from the stored evidence')
    if hit:
        for key, expected in (('type', hit['type']), ('entity_id', hit['id']), ('text', hit['text'])):
            if source.get(key) != expected:
                failures.append('source ' + key + ' differs from retrieved record')
    if source['type'] == 'memory':
        row = conn.execute('SELECT * FROM memory_item WHERE id=?', (source.get('entity_id'),)).fetchone()
        if not row:
            failures.append('source memory does not exist')
        else:
            if source.get('revision') != row['revision'] or source.get('text') != row['text']:
                failures.append('source does not quote the current memory revision')
            if (source.get('person') or {}).get('id') != row['person_id']:
                failures.append('source person differs from current memory owner')
            if source.get('date') != row['ts'] or source.get('kind') != row['kind']:
                failures.append('source date or kind differs from the memory')
            if (evidence or {}).get('id') != row['evidence_id']:
                failures.append('source evidence does not belong to the memory')
            if source.get('temporal_status') != 'historical_background':
                failures.append('memory source must remain historical background')
    elif source['type'] == 'field':
        identity = json.loads(source['entity_id'])
        row = conn.execute('SELECT * FROM person_field WHERE person_id=? AND field=? AND valid_from=?',
                           identity).fetchone()
        if row is None:
            failures.append('source field version does not exist')
        elif source.get('text') != row['value'] or source.get('valid_to') != row['valid_to'] \
                or (evidence or {}).get('id') != row['evidence_id']:
            failures.append('source field content, validity or evidence differs from the version')
        elif source.get('temporal_status') != ('current_field' if row['valid_to'] is None else 'historical_field'):
            failures.append('source field temporal state is incorrect')
    elif source['type'] == 'interaction':
        row = conn.execute('SELECT * FROM interaction WHERE id=?', (source.get('entity_id'),)).fetchone()
        if not row or any(source.get(key) != row[column] for key, column in
                          (('text', 'content'), ('date', 'ts'), ('kind', 'kind'), ('source_ref', 'source_ref'))):
            failures.append('interaction snapshot differs from the stored record')
    return failures


def check_result(conn, case, result, identities, records, top_k):
    expected, failures = case['expect'], []
    hits = result.get('results', [])
    ranked = [hit['id'] for hit in hits[:top_k]]
    recall, reciprocal_rank = None, None
    if 'needs_confirmation' in expected:
        wanted = {identities[key] for key in expected['needs_confirmation']}
        got = {item['person_id'] for item in result.get('candidates', [])}
        if result.get('ok') is not False or result.get('needs_confirmation') is not True or got != wanted:
            failures.append('ambiguous name did not return exactly the expected candidates')
        if hits:
            failures.append('ambiguous name returned potentially misattributed hits')
    elif expected.get('error'):
        if result.get('ok') is not False or hits:
            failures.append('invalid person did not stop retrieval')
    else:
        if result.get('ok') is not True:
            failures.append('tool failed: ' + str(result.get('error')))
        if 'ids' in expected:
            wanted = {records[key] for key in expected['ids']}
            recalled = wanted & set(ranked)
            if wanted:
                recall = len(recalled) / len(wanted)
                reciprocal_rank = next((1 / (index + 1) for index, oid in enumerate(ranked) if oid in wanted), 0.)
            if recalled != wanted:
                failures.append('expected records missing from top-k')
            if expected.get('only') and set(ranked) != wanted:
                failures.append('unexpected or misattributed records in top-k')
        text = '\n'.join(hit['text'] for hit in hits)
        for needle in expected.get('contains', []):
            if needle not in text:
                failures.append('missing expected content: ' + needle)
        for needle in expected.get('absent', []):
            if needle in text:
                failures.append('obsolete or foreign content returned: ' + needle)
        for needle in expected.get('card_contains', []):
            if needle not in result.get('card', ''):
                failures.append('current profile lacks: ' + needle)
        for needle in expected.get('card_absent', []):
            if needle in result.get('card', ''):
                failures.append('historical field appears as current: ' + needle)
        for field in expected.get('fields', []):
            if not any(all(source.get(key) == value for key, value in field.items())
                       for source in result.get('sources', [])):
                failures.append('missing field source with temporal state: ' + json.dumps(field, ensure_ascii=False))
        for key in expected.get('memory_sources', []):
            if not any(source.get('type') == 'memory' and source.get('entity_id') == records[key]
                       for source in result.get('sources', [])):
                failures.append('profile lacks saved background source: ' + key)
    sources = [hit.get('source') for hit in hits] if case['tool'] == 'query_memory' else result.get('sources', [])
    source_ok = 0
    for index, source in enumerate(sources):
        source_errors = _source_failures(conn, source, hit=hits[index] if hits else None)
        source_ok += not source_errors
        failures.extend(source_errors)
    reverse = {value: key for key, value in records.items()}
    return {'id': case['id'], 'label': case.get('label', case['id']), 'passed': not failures,
            'failures': failures, 'ranked_records': [reverse.get(oid, oid) for oid in ranked],
            'top_k': top_k if case['tool'] == 'query_memory' else None,
            'recall_at_k': recall, 'reciprocal_rank': reciprocal_rank,
            'source_snapshots': len(sources), 'valid_source_snapshots': source_ok,
            'expected': expected}


def evaluate(fixture=None):
    fixture = fixture or load_fixture()
    details = []
    with TemporaryDirectory(prefix='home-memory-retrieval-') as directory, offline_runtime(directory):
        for index, case in enumerate(fixture['cases']):
            conn = db._connect(Path(directory) / f'case-{index}.db')
            try:
                db.migrate(conn)
                identities, records = seed_scenario(conn, fixture)
                apply_actions(conn, case.get('actions', []), identities, records)
                args = dict(case['args'])
                if args.get('person_id'):
                    args['person_id'] = identities[args['person_id']]
                if case['tool'] == 'query_memory':
                    args.setdefault('top_k', fixture['top_k'])
                result = tools.execute_tool(conn, case['tool'], args)
                details.append(check_result(conn, case, result, identities, records,
                                             args.get('top_k', fixture['top_k'])))
            except Exception as exc:  # A broken case must be visible in the same failure report.
                positive = bool(case['expect'].get('ids'))
                details.append({'id': case['id'], 'label': case.get('label', case['id']), 'passed': False,
                    'failures': [type(exc).__name__ + ': ' + str(exc)], 'ranked_records': [],
                    'top_k': case['args'].get('top_k', fixture['top_k']) if case['tool'] == 'query_memory' else None,
                    'recall_at_k': 0. if positive else None, 'reciprocal_rank': 0. if positive else None,
                    'source_snapshots': 0,
                    'valid_source_snapshots': 0, 'expected': case['expect']})
            finally:
                conn.close()
    positives = [item for item in details if item['recall_at_k'] is not None]
    snapshot_count = sum(item['source_snapshots'] for item in details)
    passed = sum(item['passed'] for item in details)
    return {'evaluation': 'memory-retrieval-lifecycle', 'fixture_version': fixture['version'],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'channel': 'real FTS5, embeddings disabled, isolated SQLite database per case, network forbidden',
        'limitations': ['Synthetic regression fixtures, not an unseen semantic benchmark',
                        'Does not measure LLM answer correctness or embedding quality'],
        'summary': {'cases': len(details), 'passed': passed, 'failed': len(details) - passed,
            'default_top_k': fixture['top_k'],
            'pass_rate': passed / len(details) if details else 0., 'positive_retrieval_cases': len(positives),
            'recall_at_k': sum(item['recall_at_k'] for item in positives) / len(positives) if positives else None,
            'mrr': sum(item['reciprocal_rank'] for item in positives) / len(positives) if positives else None,
            'source_snapshot_accuracy': sum(item['valid_source_snapshots'] for item in details) / snapshot_count
                if snapshot_count else None}, 'details': details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--json', action='store_true', help='Print the complete machine-readable report')
    args = parser.parse_args(argv)
    report = evaluate(load_fixture(args.cases))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    if args.json:
        print(encoded)
    else:
        for item in report['details']:
            print(('PASS ' if item['passed'] else 'FAIL ') + item['id'] + ': ' + item['label'])
            for failure in item['failures']:
                print('  ' + failure)
        print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
        print('Scope: deterministic FTS and lifecycle regression; no LLM or embedding quality score.')
    return 0 if report['summary']['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
