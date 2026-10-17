"""Corrections must update reads atomically while retaining evidence and replay protection."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.memory import corrections, compile_card
from app.store import repo, db
from app.util import now_ms


@pytest.fixture
def client(conn, monkeypatch):
    from app.api import workspace, main
    monkeypatch.setattr(workspace, 'get_conn', lambda: conn)
    monkeypatch.setattr(main, 'get_conn', lambda: conn)
    return TestClient(main.app)


def seed_field(conn, name='林舟', value='杭州星河科技'):
    pid = repo.create_person(conn, name, status='active')
    eid = repo.add_evidence(conn, 'text', excerpt=f'{name}在{value}工作')
    repo.upsert_field(conn, pid, 'company', value, .95, eid, now_ms() - 10000)
    conn.commit()
    return pid, eid, repo.current_fields(conn, pid)['company']['valid_from']


def post_field(client, pid, **data):
    return client.post(f'/api/persons/{pid}/fields/correct', json={'field': 'company', **data})


def test_delete_and_restore_change_current_reads_without_erasing_history(client, conn):
    pid, eid, old = seed_field(conn)
    assert post_field(client, pid, action='delete', expected_valid_from=old).status_code == 200
    assert not repo.current_fields(conn, pid)
    assert '杭州星河科技' not in compile_card.build_card(conn, pid)
    assert repo.field_at(conn, pid, 'company', old)['evidence_id'] == eid
    assert post_field(client, pid, action='restore', version=old, expected_valid_from=None).status_code == 200
    restored = repo.current_fields(conn, pid)['company']
    assert restored['evidence_id'] == eid and restored['valid_from'] > old
    h = client.get(f'/api/persons/{pid}/history').json()
    assert len(h['history']) == 2
    assert [r['action'] for r in h['corrections']] == ['restore', 'delete']
    assert h['corrections'][0]['before']['selected']['value'] == '杭州星河科技'


def test_manual_edit_uses_new_evidence_and_rejects_stale_page(client, conn):
    pid, eid, old = seed_field(conn)
    url = f'/api/persons/{pid}/fields'
    body = {'field': 'company', 'value': '新的公司', 'expected_valid_from': old, 'reason': '换工作'}
    assert client.post(url, json=body).status_code == 200
    current = repo.current_fields(conn, pid)['company']
    assert current['evidence_id'] != eid
    assert '用户手动录入' in conn.execute('SELECT excerpt FROM evidence WHERE id=?', (current['evidence_id'],)).fetchone()[0]
    assert client.post(url, json={**body, 'value': '旧页面覆盖'}).status_code == 409
    assert repo.current_fields(conn, pid)['company']['value'] == '新的公司'
    assert len(corrections.logs(conn, 'person', pid)) == 1


def test_restore_does_not_rewrite_the_gap_after_deletion(client, conn, monkeypatch):
    pid, _, old = seed_field(conn)
    end = now_ms()
    monkeypatch.setattr(corrections, 'now_ms', lambda: end)
    assert post_field(client, pid, action='delete', expected_valid_from=old).status_code == 200
    monkeypatch.setattr(corrections, 'now_ms', lambda: end + 5000)
    assert post_field(client, pid, action='restore', version=old, expected_valid_from=None).status_code == 200
    assert repo.field_at(conn, pid, 'company', end + 1000) is None
    assert repo.field_at(conn, pid, 'company', end + 5000)['value'] == '杭州星河科技'


def test_transfer_requires_explicit_replacement_and_retains_both_histories(client, conn):
    src, eid, version = seed_field(conn)
    dst, _, dst_version = seed_field(conn, '陈岚', '接收者原公司')
    body = {'action': 'transfer', 'target_person_id': dst, 'expected_valid_from': version,
            'target_expected_valid_from': dst_version}
    assert post_field(client, src, **body).status_code == 409
    assert repo.current_fields(conn, src)['company']['value'] == '杭州星河科技'
    assert post_field(client, src, **body, replace_target=True).status_code == 200
    assert 'company' not in repo.current_fields(conn, src)
    assert repo.current_fields(conn, dst)['company']['evidence_id'] == eid
    assert len(repo.field_history(conn, dst, 'company')) == 2
    assert corrections.logs(conn, 'person', dst)[0]['before']['target']['value'] == '接收者原公司'


def test_transfer_audit_failure_rolls_back_both_people(client, conn, monkeypatch):
    src, _, version = seed_field(conn)
    dst = repo.create_person(conn, '陈岚', status='active')
    conn.commit()
    def fail(*args, **kwargs):
        raise RuntimeError('audit storage unavailable')
    monkeypatch.setattr(corrections, 'audit', fail)
    with pytest.raises(RuntimeError):
        post_field(client, src, action='transfer', target_person_id=dst,
                   expected_valid_from=version, target_expected_valid_from=None)
    assert repo.current_fields(conn, src)['company']['valid_from'] == version
    assert not repo.current_fields(conn, dst)


def test_transfer_rejects_archived_and_stale_destination(client, conn):
    src, _, version = seed_field(conn)
    dst, _, target_version = seed_field(conn, '陈岚')
    body = {'action': 'transfer', 'target_person_id': dst, 'expected_valid_from': version,
            'target_expected_valid_from': None, 'replace_target': True}
    assert post_field(client, src, **body).status_code == 409
    repo.archive_person(conn, dst)
    conn.commit()
    assert post_field(client, src, **{**body, 'target_expected_valid_from': target_version}).status_code == 400
    assert repo.current_fields(conn, src)['company']['valid_from'] == version


def seed_ledger(conn):
    pid, eid, _ = seed_field(conn)
    lid, _ = repo.add_ledger(conn, 'out', 10000, pid, '礼金', now_ms(), 'confirmed', 'source-dedupe', eid)
    conn.commit()
    return pid, eid, lid


def test_ledger_edit_and_void_update_totals_and_keep_replay_protection(client, conn):
    pid, eid, lid = seed_ledger(conn)
    dst = repo.create_person(conn, '陈岚', status='active')
    conn.commit()
    body = {'expected_revision': 0, 'person_id': dst, 'direction': 'in', 'amount_cents': 3567,
            'occurred_on': '2026-09-30', 'note': '更正为收款', 'reason': '误记'}
    assert client.patch(f'/api/ledger/{lid}', json=body).status_code == 200
    assert repo.ledger_totals(conn, pid)['count'] == 0
    assert repo.ledger_totals(conn, dst) == {'in_cents': 3567, 'out_cents': 0, 'count': 1}
    assert client.patch(f'/api/ledger/{lid}', json=body).status_code == 409
    assert repo.add_ledger(conn, 'out', 10000, pid, '礼金', now_ms(), 'confirmed', 'source-dedupe', eid) == (lid, False)
    conn.commit()
    assert client.post(f'/api/ledger/{lid}/void', json={'expected_revision': 1}).status_code == 200
    assert repo.ledger_totals(conn)['count'] == 0
    assert not client.get('/api/ledger').json()['entries']
    assert client.get('/api/ledger?include_void=true').json()['entries'][0]['status'] == 'void'
    detail = client.get(f'/api/ledger/{lid}').json()
    assert detail['entry']['evidence_id'] == eid
    assert detail['entry']['revision'] == 2
    assert len(detail['corrections']) == 2
    assert detail['corrections'][1]['before']['amount_cents'] == 10000
    assert client.patch(f'/api/ledger/{lid}', json={**body, 'expected_revision': 2}).status_code == 409
    assert repo.add_ledger(conn, 'out', 10000, pid, '礼金', now_ms(), 'confirmed', 'source-dedupe', eid) == (lid, False)


@pytest.mark.parametrize('field,value', [('amount_cents', 0), ('amount_cents', -1),
                                        ('amount_cents', '123'), ('amount_cents', True),
                                        ('occurred_on', '2026-02-30'), ('direction', 'other')])
def test_invalid_ledger_edits_leave_totals_unchanged(client, conn, field, value):
    pid, _, lid = seed_ledger(conn)
    body = {'expected_revision': 0, 'person_id': pid, 'direction': 'in', 'amount_cents': 120,
            'occurred_on': '2026-10-01', field: value}
    assert client.patch(f'/api/ledger/{lid}', json=body).status_code == 422
    assert repo.ledger_totals(conn)['out_cents'] == 10000
    assert not corrections.logs(conn, 'ledger', lid)


def test_purge_removes_correction_snapshots(client, conn):
    pid, _, version = seed_field(conn)
    assert post_field(client, pid, action='delete', expected_valid_from=version).status_code == 200
    repo.purge_person(conn, pid)
    assert not conn.execute('SELECT * FROM correction_log').fetchall()


def test_correction_history_follows_merge_and_split(client, conn):
    from app.memory.apply import merge_persons, split_merge
    pid, _, version = seed_field(conn)
    survivor = repo.create_person(conn, '陆星宁', status='active')
    conn.commit()
    assert post_field(client, pid, action='delete', expected_valid_from=version).status_code == 200
    mid = merge_persons(conn, survivor, pid)
    assert len(corrections.logs(conn, 'person', survivor)) == 1
    split_merge(conn, mid)
    assert not corrections.logs(conn, 'person', survivor)
    assert len(corrections.logs(conn, 'person', pid)) == 1


def test_migration_keeps_existing_ledger_and_allows_corrections(tmp_path):
    c = db._connect(tmp_path / 'old.db')
    root = Path(__file__).resolve().parents[1]
    c.executescript((root / 'app/store/schema.sql').read_text(encoding='utf-8'))
    for f in sorted((root / 'migrations').glob('*.sql')):
        if int(f.name[:3]) < 4:
            c.executescript(f.read_text(encoding='utf-8'))
    c.execute('PRAGMA user_version=3')
    lid, _ = repo.add_ledger(c, 'out', 100, None, '旧账', now_ms(), 'confirmed', 'old')
    c.commit()
    db.migrate(c)
    latest = max(int(f.name.split('_')[0]) for f in (root / 'migrations').glob('*.sql'))
    assert c.execute('PRAGMA user_version').fetchone()[0] == latest
    assert corrections.ledger_row(c, lid)['revision'] == 0
    corrections.change_ledger(c, lid, 0, void=True)
    assert repo.ledger_totals(c)['count'] == 0
    c.close()
