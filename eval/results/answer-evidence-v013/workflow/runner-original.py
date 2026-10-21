"""Exercise a fictional relationship workflow through real HTTP routes and tasks.

Only task dispatch is made synchronous. Chat, follow-up preparation, recording an
actual outcome, proposal generation and explicit confirmation use production
service code and a temporary SQLite database. No web server is started.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from eval.memory_questions import (
    RUNNER_ROOT, _changes, _data_snapshot, _git_info, _load_app, _safe_error,
    app_source_hash, fixture_hash, source_checks,
)

DEFAULT_FIXTURE = RUNNER_ROOT / 'eval/fixtures/relationship_workflow_v013.json'


def load_fixture(path=DEFAULT_FIXTURE):
    fixture = json.loads(Path(path).read_text(encoding='utf-8'))
    if fixture.get('provenance') != 'fictional_records_with_natural_questions':
        raise ValueError('Fictional provenance must be explicit')
    if not fixture.get('person', {}).get('name') or not fixture.get('review_points'):
        raise ValueError('Workflow needs an explicit person and review rubric')
    for key in ('initial_query', 'goal', 'final_query'):
        if not isinstance(fixture.get(key), str) or not fixture[key].strip():
            raise ValueError('Missing workflow text: ' + key)
    if not all(fixture.get('expected', {}).get(key) for key in
               ('company', 'city', 'memory_evidence', 'explicit_promise')):
        raise ValueError('Expected field, memory and promise checks must be explicit')
    return fixture


def _check(name, passed, **detail):
    return {'name': name, 'passed': bool(passed), **detail}


def _counts(conn):
    return {table: conn.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
            for table in ('person_field', 'memory_item', 'interaction', 'promise',
                          'pending_item', 'followup', 'followup_outcome', 'task')}


def _seed(app, conn, fixture):
    repo, person = app['repo'], fixture['person']
    pid = repo.create_person(conn, person['name'], status='active')
    for alias in person.get('aliases', []):
        repo.add_alias(conn, pid, alias, source='eval-v013-workflow')
    company = fixture['seed']['company']
    eid = repo.add_evidence(conn, 'text', source_ref='eval-v013-workflow:company', excerpt=company['value'])
    repo.upsert_field(conn, pid, 'company', company['value'], 1., eid, company['valid_from'])
    memory = fixture['seed']['memory']
    eid = repo.add_evidence(conn, 'text', source_ref='eval-v013-workflow:memory', excerpt=memory['text'])
    mid = repo.add_memory(conn, 'semantic', memory['text'], pid, 1., eid, ts=memory['ts'])
    conn.commit()
    return pid, mid


def run_workflow(app, conn, fixture, *, checkpoint=None):
    """Run real routes; report failures without manufacturing replacement proposals.

    A test may supply a scripted model, but it must keep the same HTTP/service
    path. The actual model run never stubs harness, follow-up or memory services.
    """
    from fastapi.testclient import TestClient
    from app.api import main
    from app.jobs import tasks

    pid, seeded_memory_id = _seed(app, conn, fixture)
    report = {'person_id': pid, 'seeded_memory_id': seeded_memory_id, 'steps': [],
              'review_points': fixture['review_points'], 'semantic_review': {
                  'status': 'agent_review_pending', 'semantic_accuracy': None,
                  'limitation': 'Development-constructed fictional workflow; agent evidence review is not independent human scoring'}}
    before = _data_snapshot(conn)
    started = time.monotonic()

    def step(name, fn):
        entry = {'name': name, 'counts_before': _counts(conn)}
        report['steps'].append(entry)
        began = time.monotonic()
        try:
            entry.update(fn())
        except Exception as error:
            entry['error'] = _safe_error(app, error)
            raise
        finally:
            entry['counts_after'] = _counts(conn)
            entry['elapsed_seconds'] = round(time.monotonic() - began, 3)
            if checkpoint:
                checkpoint(report)
        return entry

    def request(client, method, url, expected_status=200, **kwargs):
        response = client.request(method, url, **kwargs)
        try:
            data = response.json()
        except ValueError:
            data = {'non_json_response': True}
        record = {'method': method, 'url': url, 'status_code': response.status_code, 'response': data}
        if response.status_code != expected_status:
            raise RuntimeError('Unexpected HTTP status ' + str(response.status_code) + ' for ' + method + ' ' + url)
        return record

    def task(client, submitted):
        data = submitted['response']
        tasks._run(data['id'])
        finished = request(client, 'GET', '/api/tasks/' + data['id'])
        if finished['response']['status'] != 'done':
            raise RuntimeError('Durable task did not finish: ' + str(finished['response'].get('error')))
        return finished

    def unchanged_domain(old):
        return sorted(table for table in _changes(old, _data_snapshot(conn)) if table != 'evidence')

    def chat(client, cid, text, label):
        old = _data_snapshot(conn)
        queued = request(client, 'POST', f'/api/conversations/{cid}/turns', 202,
                         json={'text': text, 'request_id': label})
        finished = task(client, queued)
        result = finished['response']['result']
        checks = [_check('query_has_no_domain_writes', not unchanged_domain(old),
                         changed_tables=unchanged_domain(old)),
                  _check('query_classified_readonly', result.get('turn_intent', {}).get('intent') == 'query'
                         and result.get('turn_intent', {}).get('allow_writes') is False),
                  _check('nonempty_answer', bool(result.get('answer', '').strip()))]
        for number, source in enumerate(result.get('source_dependencies', []), 1):
            if isinstance(source, dict) and source.get('type') == 'profile_context':
                from app.memory.compile_card import build_card
                person = app['repo'].get_person(conn, source.get('entity_id'))
                errors = [] if person is not None and person['deleted_at'] is None \
                    and person['status'] != 'merged' and source.get('text') == build_card(conn, person['id']) \
                    else ['profile context differs from current person/card']
            else:
                errors = source_checks(conn, source)
            checks.append(_check('current_source:' + str(number), not errors, errors=errors))
        return {'queued': queued, 'finished': finished, 'result': result, 'checks': checks}

    try:
        # No TestClient lifespan context: scheduler recovery is unrelated to this
        # workflow. Production task execution remains intact and synchronous.
        with patch.object(tasks, 'start', lambda tid: None):
            client = TestClient(main.app)
            try:
                created = request(client, 'POST', '/api/conversations', json={'title': 'v0.13 fictional relationship workflow'})
                cid = created['response']['id']
                report['conversation_id'] = cid
                initial = step('query_person', lambda: chat(client, cid, fixture['initial_query'], 'workflow-query-initial'))
                sources = initial['result'].get('source_dependencies', [])
                initial['checks'].extend([
                    _check('initial_company_really_read', any(s.get('type') == 'field' and s.get('field') == 'company'
                           and s.get('text') == fixture['seed']['company']['value'] for s in sources)),
                    _check('initial_preference_really_read', any(s.get('type') == 'memory'
                           and s.get('entity_id') == seeded_memory_id for s in sources))])

                def prepare():
                    old = _data_snapshot(conn)
                    queued = request(client, 'POST', f'/api/persons/{pid}/followups', 202,
                                     json={'goal': fixture['goal'], 'request_id': 'workflow-prepare-initial'})
                    finished = task(client, queued)
                    fid = queued['response']['payload']['followup_id']
                    record = request(client, 'GET', '/api/followups/' + fid)['response']
                    valid_ids = {source['id'] for source in record['brief']['sources']}
                    return {'queued': queued, 'finished': finished, 'record': record,
                            'checks': [_check('preparation_writes_no_domain_facts', not unchanged_domain(old),
                                              changed_tables=unchanged_domain(old)),
                                       _check('draft_is_ready_and_nonempty', record['status'] == 'ready'
                                              and bool(record.get('draft', '').strip())),
                                       _check('draft_source_ids_are_real', bool(record['draft_source_ids'])
                                              and set(record['draft_source_ids']) <= valid_ids)]}

                prepared = step('draft_communication', prepare)
                record, fid = prepared['record'], prepared['record']['id']
                body = {'request_id': 'workflow-outcome', 'expected_revision': record['revision'], **fixture['outcome']}

                def outcome():
                    old = _counts(conn)
                    saved = request(client, 'POST', f'/api/followups/{fid}/outcomes', json=body)
                    stored = saved['response']['record']['outcomes'][-1]
                    after = _counts(conn)
                    replay = request(client, 'POST', f'/api/followups/{fid}/outcomes', json=body)
                    contact = conn.execute('SELECT * FROM interaction WHERE id=?', (stored['contact_id'],)).fetchone()
                    party = conn.execute('SELECT role FROM interaction_person WHERE interaction_id=? AND person_id=?',
                                         (stored['contact_id'], pid)).fetchone()
                    promises = [dict(row) for row in app['repo'].open_promises(conn, pid)]
                    return {'saved': saved, 'replay': replay, 'outcome': stored, 'checks': [
                        _check('one_actual_contact_saved', after['interaction'] == old['interaction'] + 1
                               and contact is not None and contact['kind'] == 'contact'
                               and contact['ts'] == body['contacted_at'] and contact['content'] == body['notes']
                               and party is not None and party['role'] == 'participant'),
                        _check('only_explicit_promise_saved', after['promise'] == old['promise'] + 1
                               and any(row['content'] == fixture['expected']['explicit_promise'] for row in promises)),
                        _check('outcome_does_not_auto_update_memory', all(after[key] == old[key]
                               for key in ('person_field', 'memory_item', 'pending_item'))),
                        _check('outcome_replay_is_idempotent', _counts(conn) == after
                               and replay['response']['record']['outcomes'][-1]['id'] == stored['id'])]}

                saved = step('record_actual_result', outcome)
                oid = saved['outcome']['id']

                def analyze():
                    old = _counts(conn)
                    queued = request(client, 'POST', f'/api/followup-outcomes/{oid}/memory-analysis', 202, json={})
                    finished = task(client, queued)
                    listing = request(client, 'GET', f'/api/followups/{fid}/memory-updates')
                    proposals = next(item['proposals'] for item in listing['response']['outcomes'] if item['outcome_id'] == oid)
                    after = _counts(conn)
                    expected = fixture['expected']
                    selected = [proposal for proposal in proposals if
                                proposal['kind'] == 'field' and proposal['field'] in {'company', 'city'}
                                and proposal['content'] == expected[proposal['field']]
                                or proposal['kind'] == 'memory'
                                and expected['memory_evidence'] in proposal['evidence']
                                and all(word in proposal['content'] for word in ('讨论报价', '邮件', '清单', '电话'))]
                    return {'queued': queued, 'finished': finished, 'listing': listing,
                            'proposals': proposals, 'selected_proposals': selected, 'checks': [
                        _check('analysis_does_not_apply_proposals', all(after[key] == old[key]
                               for key in ('person_field', 'memory_item', 'interaction', 'promise'))),
                        _check('all_proposals_have_same_person_and_exact_notes_evidence', bool(proposals)
                               and all(p.get('person', {}).get('id') == pid and p['evidence'] in body['notes']
                                       and p['content'] in p['evidence'] for p in proposals)),
                        _check('model_proposed_required_company', any(p['kind'] == 'field' and p['field'] == 'company' for p in selected)),
                        _check('model_proposed_required_city', any(p['kind'] == 'field' and p['field'] == 'city' for p in selected)),
                        _check('model_proposed_required_memory', any(p['kind'] == 'memory' for p in selected))]}

                analyzed = step('analyze_memory_proposals', analyze)

                def confirm():
                    selected = analyzed['selected_proposals']
                    resolutions, guards, replays = [], [], []
                    counts_before = _counts(conn)
                    checks = []
                    for proposal in selected:
                        url = '/api/memory-updates/' + proposal['id'] + '/resolve'
                        old = _counts(conn)
                        rejected = request(client, 'POST', url, 409, json={
                            'action': 'confirm', 'expected_revision': proposal['revision'] + 1})
                        guards.append(rejected)
                        checks.append(_check('stale_revision_does_not_write:' + proposal['id'], _counts(conn) == old))
                        request_body = {'action': 'confirm', 'expected_revision': proposal['revision']}
                        accepted = request(client, 'POST', url, json=request_body)
                        resolutions.append(accepted)
                        after = _counts(conn)
                        replay = request(client, 'POST', url, json=request_body)
                        replays.append(replay)
                        checks.append(_check('confirm_replay_is_idempotent:' + proposal['id'], _counts(conn) == after))
                        checks.append(_check('confirmation_has_real_receipt:' + proposal['id'],
                                             accepted['response']['status'] == 'confirmed'
                                             and accepted['response'].get('receipt', {}).get('status') == 'ok'))
                    current = app['repo'].current_fields(conn, pid)
                    memories = [dict(row) for row in conn.execute("SELECT * FROM memory_item WHERE person_id=? AND status='active'", (pid,))]
                    confirmed_ids = [r['response']['receipt']['id'] for r in resolutions if r['response']['kind'] == 'memory']
                    visible = [request(client, 'GET', '/api/memories/' + memory_id) for memory_id in confirmed_ids]
                    from app.agent.tools import execute_tool
                    retrieved = execute_tool(conn, 'query_memory', {
                        'query': fixture['expected']['memory_evidence'],
                        'person': fixture['person']['name'], 'top_k': 5})
                    retrieved_ids = {item.get('id') for item in retrieved.get('results', []) if item.get('type') == 'memory'}
                    checks.extend([
                        _check('confirm_does_not_duplicate_contacts_or_promises', all(_counts(conn)[key] == counts_before[key]
                               for key in ('interaction', 'promise', 'followup_outcome'))),
                        _check('confirmed_company_is_current', current.get('company', {}).get('value') == fixture['expected']['company']),
                        _check('confirmed_city_is_current', current.get('city', {}).get('value') == fixture['expected']['city']),
                        _check('confirmed_memory_persisted', bool(confirmed_ids)
                               and all(any(m['id'] == mid for m in memories) for mid in confirmed_ids)),
                        _check('confirmed_memory_indexed', bool(confirmed_ids) and all(conn.execute(
                            "SELECT 1 FROM memory_fts WHERE owner_type='memory' AND owner_id=?", (mid,)).fetchone()
                            for mid in confirmed_ids)),
                        _check('confirmed_memory_retrievable_through_query_tool', bool(confirmed_ids)
                               and set(confirmed_ids) <= retrieved_ids),
                        _check('confirmed_memory_visible_with_proposal_and_outcome_source', bool(visible) and all(
                            item['response']['proposal_id'] in {p['id'] for p in selected}
                            and item['response']['source']['source_ref'] == 'followup-outcome:' + oid
                            for item in visible))])
                    return {'revision_guards': guards, 'resolutions': resolutions, 'replays': replays,
                            'stored_current_fields': current, 'stored_memories': memories,
                            'confirmed_memory_ids': confirmed_ids, 'visible_memory_details': visible,
                            'retrieved_confirmed_memory': retrieved, 'checks': checks}

                confirmed = step('confirm_memory_proposals', confirm)
                final = step('query_again', lambda: chat(client, cid, fixture['final_query'], 'workflow-query-final'))
                dependencies = final['result'].get('source_dependencies', [])
                final['checks'].extend([
                    _check('final_query_reads_confirmed_company', any(s.get('type') == 'field' and s.get('field') == 'company'
                           and s.get('text') == fixture['expected']['company'] and s.get('temporal_status') == 'current_field' for s in dependencies)),
                    _check('final_query_reads_confirmed_city', any(s.get('type') == 'field' and s.get('field') == 'city'
                           and s.get('text') == fixture['expected']['city'] and s.get('temporal_status') == 'current_field' for s in dependencies)),
                    _check('final_query_reads_confirmed_memory', bool(confirmed['confirmed_memory_ids']) and
                           set(confirmed['confirmed_memory_ids']) <= {s.get('entity_id') for s in dependencies if s.get('type') == 'memory'})])

                def prepare_again():
                    old = _data_snapshot(conn)
                    queued = request(client, 'POST', f'/api/persons/{pid}/followups', 202,
                                     json={'goal': fixture['goal'], 'request_id': 'workflow-prepare-after-confirmation'})
                    finished = task(client, queued)
                    record = request(client, 'GET', '/api/followups/' + queued['response']['payload']['followup_id'])['response']
                    brief = record['brief']
                    return {'queued': queued, 'finished': finished, 'record': record, 'checks': [
                        _check('preparation_after_confirmation_remains_readonly', not unchanged_domain(old), changed_tables=unchanged_domain(old)),
                        _check('new_brief_reads_confirmed_company', any(f['field'] == 'company' and f['value'] == fixture['expected']['company'] for f in brief['current_fields'])),
                        _check('new_brief_reads_confirmed_memory', bool(confirmed['confirmed_memory_ids']) and
                               set(confirmed['confirmed_memory_ids']) <= {m['id'] for m in brief['memories']}),
                        _check('new_brief_reads_actual_contact', brief['last_contact']['id'] == saved['outcome']['contact_id'])]}

                step('prepare_again_from_confirmed_memory', prepare_again)
            finally:
                client.close()
    except Exception as error:
        report['error'] = _safe_error(app, error)
    checks = [check for entry in report['steps'] for check in entry.get('checks', [])]
    report.update(elapsed_seconds=round(time.monotonic() - started, 3), data_changes=_changes(before, _data_snapshot(conn)),
                  summary={'steps_completed': sum('error' not in entry for entry in report['steps']),
                           'expected_steps': 7, 'checks': len(checks), 'checks_passed': sum(check['passed'] for check in checks),
                           'failed_checks': [check['name'] for check in checks if not check['passed']],
                           'workflow_contract_passed': len(report['steps']) == 7 and not report.get('error')
                               and all(check['passed'] for check in checks), 'semantic_accuracy': None})
    if checkpoint:
        checkpoint(report)
    return report


def _worker(job):
    with TemporaryDirectory(prefix='home-relationship-v013-') as directory:
        app = _load_app(job['app_root'], directory, 'answers')
        conn = app['db'].get_conn()
        try:
            result = run_workflow(app, conn, job['fixture'])
            return {'workflow': result, 'configured_model_tiers': {
                'mini': app['settings'].model_mini, 'mid': app['settings'].model_mid},
                'app_source_hash_after': app_source_hash(job['app_root'])}
        finally:
            app['db'].reset_for_tests(Path(directory) / 'eval.db')


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = 'workflow-' + report['report_id'] + '-' + report['app_source_hash'][:8]
    path = directory / (name + '.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    review = directory / (name + '-review.md')
    lines = ['# Relationship workflow v0.13', '',
             '虚构开发构造；实际使用 HTTP 路由与持久任务，在临时 SQLite 中运行。',
             '代理证据审阅待完成；不计算语义准确率。', '',
             '- Fixture SHA256: `' + report['fixture_hash'] + '`',
             '- App SHA256: `' + report['app_source_hash'] + '`', '',
             '```json', json.dumps(report.get('workflow', {}).get('summary'), ensure_ascii=False, indent=2), '```', '']
    for entry in report.get('workflow', {}).get('steps', []):
        lines.extend(['## ' + entry['name'], '', '```json', json.dumps(entry, ensure_ascii=False, indent=2), '```', '',
                      '代理证据审阅：PENDING', ''])
    review.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path, review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--app-root', type=Path, default=RUNNER_ROOT)
    parser.add_argument('--output-dir', type=Path, default=RUNNER_ROOT / 'data/qa/answer-evidence-v013/workflow')
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        job = json.loads(sys.stdin.read())
        with redirect_stdout(sys.stderr):
            result = _worker(job)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    fixture = load_fixture(args.fixture)
    root = args.app_root.resolve()
    before_hash = app_source_hash(root)
    report = {'report_id': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
              'fixture_version': fixture['version'], 'fixture_hash': fixture_hash(fixture),
              'fixture_file_hash': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              'app_root': str(root), 'app_source_hash': before_hash,
              'runner_source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'app_git': _git_info(root), 'run_status': 'running',
              'configuration': {'runtime': 'isolated process; temporary SQLite; real HTTP routes and durable tasks; synchronous dispatch',
                                'embedding': 'disabled', 'cache': False, 'repeat': 1},
              'limitations': ['Fictional development workflow held out only during this implementation, not an external benchmark',
                              'Deterministic DB/HTTP checks do not measure semantic answer accuracy',
                              'Agent evidence review is not independent human scoring']}
    write_report(report, args.output_dir)
    job = {'app_root': str(root), 'fixture': fixture}
    try:
        completed = subprocess.run([sys.executable, '-m', 'eval.relationship_workflow_v013', '--_worker'],
                                   input=json.dumps(job, ensure_ascii=False), text=True, capture_output=True,
                                   cwd=str(RUNNER_ROOT), timeout=args.timeout)
        if completed.returncode:
            report['error'] = {'type': 'WorkerError', 'message': 'Isolated worker exited with code ' + str(completed.returncode)}
        else:
            report.update(json.loads(completed.stdout))
    except subprocess.TimeoutExpired:
        report['error'] = {'type': 'WorkerTimeout', 'message': 'Isolated workflow exceeded timeout; worker terminated'}
    except ValueError:
        report['error'] = {'type': 'WorkerProtocolError', 'message': 'Worker did not produce a JSON report'}
    report.update(run_status='complete', app_source_changed_during_run=app_source_hash(root) != before_hash
                  or report.get('app_source_hash_after', before_hash) != before_hash)
    paths = write_report(report, args.output_dir)
    print(json.dumps({'paths': [str(path) for path in paths],
                      'summary': report.get('workflow', {}).get('summary'), 'error': report.get('error')}, ensure_ascii=False))
    return 1 if report.get('error') or report['app_source_changed_during_run'] \
        or not report.get('workflow', {}).get('summary', {}).get('workflow_contract_passed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
