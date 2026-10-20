"""Exercise a fictional relationship workflow through real HTTP routes and tasks.

Only task dispatch is made synchronous. Chat, follow-up preparation, recording an
actual outcome, proposal generation and explicit confirmation use production
service code and a temporary SQLite database. No web server is started.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
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
RUNNER_VERSION = '0.13.0-relationship-workflow-user-review-2'
ORIGINAL_RUNNER_HASH = '595352328090a9926cdc11d557177849eab75c202dca8e2ad0def01a62119123'


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


def _check(name, passed, *, category='workflow_contract', **detail):
    return {'name': name, 'passed': bool(passed), 'category': category, **detail}


def _outcome_source_checks(conn, source):
    """Validate the original outcome chain, including sources without evidence IDs."""
    errors = []
    reference = source.get('source_ref', '')
    oid = reference.removeprefix('followup-outcome:')
    outcome = conn.execute('SELECT o.*,f.person_id FROM followup_outcome o '
                           'JOIN followup f ON f.id=o.followup_id WHERE o.id=?', (oid,)).fetchone()
    if outcome is None:
        return ['source outcome or followup does not exist']
    contact = conn.execute('SELECT * FROM interaction WHERE id=?', (outcome['contact_id'],)).fetchone()
    if contact is None or outcome['contact_id'] != source.get('entity_id'):
        return ['source outcome contact identity mismatch']
    try:
        payload = json.loads(outcome['payload'])
        notes, contacted_at, channel = (payload[key] for key in ('notes', 'contacted_at', 'channel'))
        if not isinstance(notes, str) or type(contacted_at) is not int or not isinstance(channel, str):
            raise ValueError('invalid outcome payload')
    except (ValueError, TypeError, KeyError):
        return ['source outcome payload is invalid']
    if contact['kind'] != 'contact' or source.get('kind') != 'contact' \
            or source.get('temporal_status') != 'contact':
        errors.append('source outcome is not an actual contact')
    if contact['source_ref'] != reference or reference != 'followup-outcome:' + outcome['id']:
        errors.append('source outcome reference mismatch')
    if contact['content'] != notes.strip() or source.get('text') != notes.strip():
        errors.append('source outcome notes mismatch')
    if contact['ts'] != contacted_at or source.get('date') != contacted_at or source.get('ts') != contacted_at:
        errors.append('source outcome event time mismatch')
    if contact['channel'] != channel.strip() or source.get('channel') != channel.strip():
        errors.append('source outcome channel mismatch')
    if source.get('record_context') != {'recorded_at': contact['created_at']}:
        errors.append('source outcome recording time mismatch')
    parties = [dict(row) for row in conn.execute(
        'SELECT ip.person_id,ip.role,p.display_name,p.deleted_at,p.status FROM interaction_person ip '
        'JOIN person p ON p.id=ip.person_id WHERE ip.interaction_id=? ORDER BY ip.person_id', (contact['id'],))]
    expected_people = [{'id': row['person_id'], 'display_name': row['display_name'], 'role': row['role']}
                       for row in parties]
    if source.get('people') != expected_people:
        errors.append('source outcome participant snapshot mismatch')
    if not any(row['person_id'] == outcome['person_id'] and row['role'] == 'participant' for row in parties):
        errors.append('source outcome followup owner is not a contact participant')
    if any(row['deleted_at'] is not None or row['status'] == 'merged' for row in parties):
        errors.append('source outcome participant is unavailable')
    expected_person = ({key: expected_people[0][key] for key in ('id', 'display_name')}
                       if len(expected_people) == 1 else None)
    if source.get('person') != expected_person:
        errors.append('source outcome owner snapshot mismatch')
    return errors


def workflow_source_checks(conn, source):
    """Keep generic evidence checks and independently verify legitimate outcome links."""
    errors = source_checks(conn, source)
    if isinstance(source, dict) and source.get('type') == 'interaction' \
            and isinstance(source.get('source_ref'), str) \
            and source['source_ref'].startswith('followup-outcome:'):
        outcome_errors = _outcome_source_checks(conn, source)
        # No exemption for a missing/forged evidence object or an invalid chain.
        if not outcome_errors and source.get('evidence') is None:
            errors = [error for error in errors if error != 'source evidence does not exist']
        errors.extend(outcome_errors)
    return errors


def _user_review_selection(proposals, person_id, outcome_id, followup_id, fixture):
    """Select existing unambiguous proposals; expected text is an explicit user edit.

    The frozen fixture supplies the simulated user's reviewed text. It never
    supplies a replacement model proposal, and unsupported/ambiguous candidates
    remain unconfirmed. Strict model quality checks are computed separately.
    """
    selected, decisions = [], []
    notes, expected = fixture['outcome']['notes'], fixture['expected']
    for kind, field, expected_key in [('field', 'company', 'company'), ('field', 'city', 'city'),
                                      ('memory', None, 'memory_evidence')]:
        candidates = [p for p in proposals if p.get('kind') == kind and p.get('field') == field]
        reviewed = expected[expected_key]
        decision = {'kind': kind, 'field': field, 'candidate_count': len(candidates),
                    'reviewed_content': reviewed, 'text_origin': 'predeclared_fixture_expected',
                    'accepted': False}
        decisions.append(decision)
        if len(candidates) != 1:
            decision['reason'] = 'no_unique_existing_proposal'
            continue
        proposal = candidates[0]
        content, evidence = proposal.get('content'), proposal.get('evidence')
        decision.update(proposal_id=proposal['id'], original_content=content, original_evidence=evidence)
        if proposal.get('person', {}).get('id') != person_id or proposal.get('outcome_id') != outcome_id \
                or proposal.get('followup_id') != followup_id or proposal.get('status') != 'pending' \
                or proposal.get('stale') or proposal.get('conflict'):
            decision['reason'] = 'proposal_owner_source_or_state_mismatch'
            continue
        if not isinstance(content, str) or not content or not isinstance(evidence, str) or not evidence \
                or content not in evidence or evidence not in notes or reviewed not in notes:
            decision['reason'] = 'proposal_or_reviewed_text_not_supported_by_original_notes'
            continue
        related = (reviewed in evidence and reviewed in content) if kind == 'field' else \
            ((evidence in reviewed or reviewed in evidence) and (content in reviewed or reviewed in content))
        if not related:
            decision['reason'] = 'proposal_evidence_is_unrelated_to_reviewed_text'
            continue
        decision.update(accepted=True, edited=reviewed != content, reason='unique_source_supported_proposal')
        selected.append(proposal)
    return selected, decisions


@contextmanager
def _audit_model_responses(records):
    """Observe real LLM.chat returns without substituting or retrying any call."""
    from app.llm import LLM, failure_metadata
    original = LLM.chat

    def traced(client, messages, *args, **kwargs):
        entry = {'call': len(records) + 1, 'tier': kwargs.get('tier', args[0] if args else 'mid'),
                 'message_count': len(messages), 'system_message_hash': hashlib.sha256(
                     str(messages[0].get('content', '') if messages else '').encode()).hexdigest()}
        records.append(entry)
        started = time.monotonic()
        try:
            result = original(client, messages, *args, **kwargs)
            entry['response'] = deepcopy(result)
            raw = result.get('content')
            entry['content_sha256'] = hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None
            return result
        except Exception as error:
            entry['error'] = failure_metadata(error)
            raise
        finally:
            entry['elapsed_seconds'] = round(time.monotonic() - started, 3)

    with patch.object(LLM, 'chat', traced):
        yield


def _counts(conn):
    return {table: conn.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
            for table in ('person_field', 'memory_item', 'interaction', 'promise',
                          'pending_item', 'followup', 'followup_outcome', 'task')}


def _database_snapshot(conn):
    """Compare every stored row, including same-count updates and FTS state."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    # FTS shadow tables can be WITHOUT ROWID. Sort stored column values instead
    # of requiring a rowid that those real SQLite tables do not expose.
    return {table: sorted((tuple(row) for row in conn.execute(
        'SELECT * FROM "' + table.replace('"', '""') + '"')), key=repr) for table in tables}


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


def run_workflow(app, conn, fixture, *, checkpoint=None, user_review=False, model_audit=None):
    """Run real routes; report failures without manufacturing replacement proposals.

    A test may supply a scripted model, but it must keep the same HTTP/service
    path. The actual model run never stubs harness, follow-up or memory services.
    """
    from fastapi.testclient import TestClient
    from app.api import main
    from app.jobs import tasks

    pid, seeded_memory_id = _seed(app, conn, fixture)
    report = {'person_id': pid, 'seeded_memory_id': seeded_memory_id, 'steps': [],
              'review_mode': 'simulated_user_fixture_review' if user_review else 'unmodified_model_proposals',
              'review_points': fixture['review_points'], 'semantic_review': {
                  'status': 'agent_review_pending', 'semantic_accuracy': None,
                  'limitation': 'Development-constructed fictional workflow; agent evidence review is not independent human scoring'}}
    before = _data_snapshot(conn)
    started = time.monotonic()

    def step(name, fn):
        entry = {'name': name, 'counts_before': _counts(conn)}
        if model_audit is not None:
            entry['first_model_call'] = len(model_audit) + 1
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
            if model_audit is not None:
                entry['model_calls'] = list(range(entry['first_model_call'], len(model_audit) + 1))
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
                errors = workflow_source_checks(conn, source)
            checks.append(_check('current_source:' + str(number), not errors, errors=errors,
                                 validation='followup_outcome_backlink' if isinstance(source, dict)
                                 and source.get('type') == 'interaction'
                                 and str(source.get('source_ref', '')).startswith('followup-outcome:')
                                 else 'stored_source'))
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
                    after_snapshot = _database_snapshot(conn)
                    replay = request(client, 'POST', f'/api/followups/{fid}/outcomes', json=body)
                    contact = conn.execute('SELECT * FROM interaction WHERE id=?', (stored['contact_id'],)).fetchone()
                    party = conn.execute('SELECT role FROM interaction_person WHERE interaction_id=? AND person_id=?',
                                         (stored['contact_id'], pid)).fetchone()
                    promises = [dict(row) for row in app['repo'].open_promises(conn, pid)]
                    from app.agent.memory_sources import interaction_source
                    contact_source = interaction_source(conn, stored['contact_id'])
                    source_errors = workflow_source_checks(conn, contact_source)
                    return {'saved': saved, 'replay': replay, 'outcome': stored,
                            'contact_source': contact_source, 'checks': [
                        _check('one_actual_contact_saved', after['interaction'] == old['interaction'] + 1
                               and contact is not None and contact['kind'] == 'contact'
                               and contact['ts'] == body['contacted_at'] and contact['content'] == body['notes']
                               and party is not None and party['role'] == 'participant'),
                        _check('only_explicit_promise_saved', after['promise'] == old['promise'] + 1
                               and any(row['content'] == fixture['expected']['explicit_promise'] for row in promises)),
                        _check('outcome_does_not_auto_update_memory', all(after[key] == old[key]
                               for key in ('person_field', 'memory_item', 'pending_item'))),
                        _check('outcome_replay_is_idempotent', _database_snapshot(conn) == after_snapshot
                               and replay['response']['record']['outcomes'][-1]['id'] == stored['id']),
                        _check('actual_contact_has_valid_outcome_source', not source_errors,
                               validation='followup_outcome_backlink', errors=source_errors)]}

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
                    strict = [proposal for proposal in proposals if
                                proposal['kind'] == 'field' and proposal['field'] in {'company', 'city'}
                                and proposal['content'] == expected[proposal['field']]
                                or proposal['kind'] == 'memory'
                                and expected['memory_evidence'] in proposal['evidence']
                                and all(word in proposal['content'] for word in ('讨论报价', '邮件', '清单', '电话'))]
                    selected, decisions = _user_review_selection(proposals, pid, oid, fid, fixture) \
                        if user_review else (strict, [])
                    checks = [
                        _check('analysis_does_not_apply_proposals', all(after[key] == old[key]
                               for key in ('person_field', 'memory_item', 'interaction', 'promise'))),
                        _check('all_proposals_have_same_person_and_exact_notes_evidence', bool(proposals)
                               and all(p.get('person', {}).get('id') == pid and p['evidence'] in body['notes']
                                       and p['content'] in p['evidence'] for p in proposals)),
                        _check('model_proposed_required_company', any(p['kind'] == 'field' and p['field'] == 'company'
                               for p in strict), category='model_proposal_quality'),
                        _check('model_proposed_required_city', any(p['kind'] == 'field' and p['field'] == 'city'
                               for p in strict), category='model_proposal_quality'),
                        _check('model_proposed_required_memory', any(p['kind'] == 'memory'
                               for p in strict), category='model_proposal_quality')]
                    if user_review:
                        checks.extend(_check('user_review_has_unique_supported_' + (item['field'] or 'memory'),
                                             item['accepted'], decision=item) for item in decisions)
                    return {'queued': queued, 'finished': finished, 'listing': listing,
                            'proposals': proposals, 'strict_quality_matches': strict,
                            'selected_proposals': selected, 'user_review_decisions': decisions, 'checks': checks}

                analyzed = step('analyze_memory_proposals', analyze)

                def confirm():
                    selected = analyzed['selected_proposals']
                    resolutions, guards, replays = [], [], []
                    counts_before = _counts(conn)
                    checks = []
                    reviews = {item['proposal_id']: item for item in analyzed['user_review_decisions'] if item['accepted']}
                    audit_evidence = []
                    for proposal in selected:
                        url = '/api/memory-updates/' + proposal['id'] + '/resolve'
                        old = _database_snapshot(conn)
                        request_body = {'action': 'confirm', 'expected_revision': proposal['revision']}
                        if user_review:
                            request_body['content'] = reviews[proposal['id']]['reviewed_content']
                        stale_body = request_body | {'expected_revision': proposal['revision'] + 1}
                        rejected = request(client, 'POST', url, 409, json=stale_body)
                        rejected['submitted_body'] = stale_body
                        guards.append(rejected)
                        checks.append(_check('stale_revision_does_not_write:' + proposal['id'], _database_snapshot(conn) == old))
                        accepted = request(client, 'POST', url, json=request_body)
                        accepted['submitted_body'] = request_body
                        resolutions.append(accepted)
                        after = _database_snapshot(conn)
                        replay = request(client, 'POST', url, json=request_body)
                        replay['submitted_body'] = request_body
                        replays.append(replay)
                        checks.append(_check('confirm_replay_is_idempotent:' + proposal['id'], _database_snapshot(conn) == after))
                        checks.append(_check('confirmation_has_real_receipt:' + proposal['id'],
                                             accepted['response']['status'] == 'confirmed'
                                             and accepted['response'].get('receipt', {}).get('status') == 'ok'))
                        if user_review:
                            evidence_id = accepted['response']['receipt']['evidence_id']
                            evidence = conn.execute('SELECT * FROM evidence WHERE id=?', (evidence_id,)).fetchone()
                            evidence = dict(evidence) if evidence else None
                            audit_evidence.append({'proposal_id': proposal['id'], 'evidence': evidence})
                            reviewed = reviews[proposal['id']]
                            if reviewed['edited']:
                                envelope = json.loads(evidence['excerpt']) if evidence else {}
                                original_evidence = conn.execute('SELECT * FROM evidence WHERE id=?',
                                    (envelope.get('source_evidence_id'),)).fetchone()
                                valid_audit = evidence is not None \
                                    and evidence['source_ref'] == 'outcome-memory-review:' + proposal['id'] \
                                    and envelope.get('outcome_id') == oid and envelope.get('person_id') == pid \
                                    and envelope.get('original_quote') == proposal['evidence'] \
                                    and envelope.get('edited_by_user') is True \
                                    and envelope.get('user_confirmed') == {
                                        'content': reviewed['reviewed_content'],
                                        'valid_from': accepted['response']['valid_from']} \
                                    and original_evidence is not None \
                                    and original_evidence['source_ref'] == 'followup-outcome:' + oid \
                                    and original_evidence['excerpt'] == body['notes']
                            else:
                                valid_audit = evidence is not None \
                                    and evidence['source_ref'] == 'followup-outcome:' + oid \
                                    and evidence['excerpt'] == body['notes']
                            checks.append(_check('user_review_confirmation_has_original_source_audit:' + proposal['id'],
                                                 valid_audit, edited=reviewed['edited']))
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
                            and any((item['response'].get(key) or {}).get('source_ref') == 'followup-outcome:' + oid
                                    for key in ('source', 'original_source'))
                            for item in visible))])
                    if user_review:
                        checks.append(_check('user_reviewed_memory_text_persisted', bool(confirmed_ids)
                            and all(any(m['id'] == mid and m['text'] == fixture['expected']['memory_evidence']
                                        for m in memories) for mid in confirmed_ids)))
                    return {'revision_guards': guards, 'resolutions': resolutions, 'replays': replays,
                            'confirmation_source_audit': audit_evidence,
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
    quality = [check for check in checks if check['category'] == 'model_proposal_quality']
    lifecycle = [check for check in checks if check['category'] == 'workflow_contract']
    completed = len(report['steps']) == 7 and not report.get('error')
    report.update(elapsed_seconds=round(time.monotonic() - started, 3), data_changes=_changes(before, _data_snapshot(conn)),
                  summary={'steps_completed': sum('error' not in entry for entry in report['steps']),
                           'expected_steps': 7, 'checks': len(checks), 'checks_passed': sum(check['passed'] for check in checks),
                           'failed_checks': [check['name'] for check in checks if not check['passed']],
                           'all_checks_passed': completed and all(check['passed'] for check in checks),
                           'workflow_contract_passed': completed and all(check['passed'] for check in lifecycle),
                           'workflow_contract_checks': len(lifecycle),
                           'workflow_contract_checks_passed': sum(check['passed'] for check in lifecycle),
                           'model_proposal_quality_passed': len(quality) == 3 and all(check['passed'] for check in quality),
                           'model_proposal_quality_checks': len(quality),
                           'model_proposal_quality_checks_passed': sum(check['passed'] for check in quality),
                           'semantic_accuracy': None})
    if checkpoint:
        checkpoint(report)
    return report


def _worker(job):
    with TemporaryDirectory(prefix='home-relationship-v013-') as directory:
        app = _load_app(job['app_root'], directory, 'answers')
        conn = app['db'].get_conn()
        try:
            audit = []
            with _audit_model_responses(audit):
                result = run_workflow(app, conn, job['fixture'], user_review=job.get('user_review', False),
                                      model_audit=audit)
            return {'workflow': result, 'configured_model_tiers': {
                'mini': app['settings'].model_mini, 'mid': app['settings'].model_mid},
                'model_response_audit': audit,
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
             '- Runner SHA256: `' + report['runner_source_hash'] + '`',
             '- Review mode: `' + report['configuration'].get('user_review', 'unmodified_model_proposals') + '`', '',
             '```json', json.dumps(report.get('workflow', {}).get('summary'), ensure_ascii=False, indent=2), '```', '']
    if report['configuration'].get('user_review') == 'simulated_user_fixture_review':
        lines[4:4] = ['本次为同一已知开发 fixture 的后续验证，保留首次失败。',
                      '用户审阅由 fixture 预先期望文字模拟；修订后闭环不等于原模型提案质量通过。', '']
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
    parser.add_argument('--user-review', action='store_true',
                        help='Simulate explicit user review/edits from frozen fixture text; retain model quality failures')
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
              'runner_version': RUNNER_VERSION,
              'fixture_version': fixture['version'], 'fixture_hash': fixture_hash(fixture),
              'fixture_file_hash': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              'app_root': str(root), 'app_source_hash': before_hash,
              'runner_source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'app_git': _git_info(root), 'run_status': 'running',
              'configuration': {'runtime': 'isolated process; temporary SQLite; real HTTP routes and durable tasks; synchronous dispatch',
                                'embedding': 'disabled', 'cache': False, 'repeat': 1,
                                'user_review': 'simulated_user_fixture_review' if args.user_review else 'unmodified_model_proposals',
                                'model_response_audit': 'read-only LLM.chat response capture; original call once; no substitution'},
              'limitations': ['Fictional development workflow, not an external benchmark or an independent accuracy evaluation',
                              'Deterministic DB/HTTP checks do not measure semantic answer accuracy',
                              'Agent evidence review is not independent human scoring']}
    if args.user_review:
        report['followup_validation'] = {
            'known_development_fixture': True, 'original_failure_preserved': True,
            'original_runner_source_hash': ORIGINAL_RUNNER_HASH,
            'original_failure_report_sha256': 'e9f36f00bf336233f0c46dd0a3ab7fd1b06c02eb43518322f3ff2d3a8b7c81de',
            'same_fixture_as_original_failure': fixture_hash(fixture)
                == 'fe6d223d909236cf284b4449d47ad3136374af2c75b7acbfd5b9f090b1d7ca13',
            'review_text_origin': 'predeclared fixture expected, checked against original outcome notes',
            'model_proposal_quality_kept_separate': True}
        report['limitations'].extend([
            'Follow-up validation on the same known development fixture after an observed failure; not blind evaluation',
            'User review is simulated from predeclared fixture text; reviewed confirmations do not prove model proposal completeness'])
    write_report(report, args.output_dir)
    job = {'app_root': str(root), 'fixture': fixture, 'user_review': args.user_review}
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
        or not report.get('workflow', {}).get('summary', {}).get('all_checks_passed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
