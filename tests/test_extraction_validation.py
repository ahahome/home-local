"""Malformed model outputs must neither lose valid facts nor create guessed money."""
import json

import pytest
from fastapi.testclient import TestClient

from app.agent import tools
from app.memory import apply
from app.memory.extract import sanitize
from app.memory.pending import resolve
from app.money import MAX_AMOUNT_CENTS, amount_to_cents
from app.store import repo


def raw_person(**items):
    return {'name': '张晨', 'canonical_name': '张晨', **items}


@pytest.mark.parametrize('amount', [
    float('inf'), float('-inf'), float('nan'), 'NaN', 'Infinity', '1e9999',
    True, {}, [], '金额未知', 0, -1, '0.001', '19.995',
    '10.000000000000000000000000000001', '1000000000000.01',
])
def test_invalid_amount_does_not_lose_other_valid_information(amount):
    patch = sanitize({'people': [raw_person(
        fields=[{'field': 'city', 'value': '杭州', 'confidence': .95}],
        ledger=[{'direction': 'in', 'amount': amount}, {'direction': 'out', 'amount': '1.13'}],
    )]})
    person = patch['people'][0]
    assert person['fields'][0]['value'] == '杭州'
    assert len(person['ledger']) == 1 and person['ledger'][0]['amount_cents'] == 113


@pytest.mark.parametrize('amount,cents', [
    (0.29, 29), (1.13, 113), ('10.00', 1000), ('1e-2', 1),
    ('900719925474.09', 90071992547409), ('1000000000000', MAX_AMOUNT_CENTS),
])
def test_valid_currency_is_exact(amount, cents):
    assert amount_to_cents(amount) == cents
    patch = sanitize({'people': [raw_person(ledger=[{'direction': 'in', 'amount': amount}])]})
    assert patch['people'][0]['ledger'][0]['amount_cents'] == cents


@pytest.mark.parametrize('confidence', ['nan', float('inf'), '不确定', True, {}])
def test_invalid_confidence_cannot_auto_save_facts_or_money(conn, confidence):
    patch = sanitize({'people': [raw_person(
        fields=[{'field': 'city', 'value': '杭州', 'confidence': confidence}],
        facts=[{'text': '张晨喜欢登山', 'confidence': confidence}],
        ledger=[{'direction': 'in', 'amount': 10, 'confidence': confidence}],
    )], 'my_notes': [{'text': '我喜欢登山', 'confidence': confidence}]})
    apply.apply_extraction(conn, patch, 'bad-confidence', '测试来源')
    assert not repo.current_fields(conn, repo.all_active_persons(conn)[0]['id'])
    assert conn.execute('SELECT COUNT(*) FROM memory_item').fetchone()[0] == 0
    assert repo.ledger_totals(conn)['count'] == 0


def test_malformed_collections_and_object_values_do_not_poison_siblings():
    patch = sanitize({'people': [raw_person(fields=123, aliases=True, relations={}, facts='bad',
                                           promises=42, ledger=False),
        {'name': '江遥', 'fields': [{'field': 'city', 'value': {'guess': '杭州'}},
                                {'field': 'company', 'value': '星桥软件'}]}],
        'my_notes': [{'text': {'guess': '我喜欢登山'}}, {'text': '我喜欢游泳'}]})
    assert [p['name'] for p in patch['people']] == ['张晨', '江遥']
    assert [f['field'] for f in patch['people'][1]['fields']] == ['company']
    assert [n['text'] for n in patch['my_notes']] == ['我喜欢游泳']
    assert sanitize({'people': 123, 'my_notes': False}) == {'people': [], 'my_notes': []}


@pytest.mark.parametrize('amount', ['NaN', float('inf'), '19.995', True, '1e9999'])
def test_transfer_tool_rejects_invalid_amount_before_any_write(conn, amount):
    result = tools.execute_tool(conn, 'record_transfer', {
        'person': '张晨', 'direction': 'out', 'amount_yuan': amount,
    })
    assert result['ok'] is False and result['error']
    assert conn.execute('SELECT COUNT(*) FROM person').fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM ledger_entry').fetchone()[0] == 0


def unknown_transfer(conn, monkeypatch, source='张晨的餐费是20元，收支方向需要核对。'):
    monkeypatch.setattr(apply, 'today_local', lambda: '2026-10-03')
    patch = sanitize({'people': [raw_person(ledger=[{
        'amount': 20, 'note': '餐费', 'confidence': .99,
    }])]}, source=source)
    result = apply.apply_extraction(conn, patch, 'unclear-transfer', source)
    return patch, result


@pytest.mark.parametrize('direction', ['in', 'out'])
def test_direction_choice_requires_explicit_input_and_survives_replay(conn, monkeypatch, direction):
    from app.api.main import app
    patch, result = unknown_transfer(conn, monkeypatch)
    client = TestClient(app)
    assert len(result['pending_ids']) == 1
    assert conn.execute('SELECT COUNT(*) FROM ledger_entry').fetchone()[0] == 0
    item = client.get('/api/pending').json()['pending'][0]
    assert item['payload']['direction'] is None
    assert '餐费是20元' in item['payload']['evidence']
    url = f"/api/pending/{item['id']}/resolve"
    assert client.post(url, json={'action': 'confirm'}).status_code == 400
    assert client.post(url, json={'action': 'confirm', 'direction': 'sideways'}).status_code == 422
    assert not repo.get_pending(conn, item['id'])['resolved_at']
    assert client.post(url, json={'action': 'confirm', 'direction': direction}).status_code == 200
    assert repo.ledger_totals(conn) == {
        'in_cents': 2000 if direction == 'in' else 0,
        'out_cents': 2000 if direction == 'out' else 0, 'count': 1,
    }
    monkeypatch.setattr(apply, 'today_local', lambda: '2026-10-04')
    replay = apply.apply_extraction(conn, patch, 'unclear-transfer', '张晨的餐费是20元。')
    assert any(o['status'] == 'duplicate' for o in replay['outcomes'])
    assert not repo.list_pending(conn, 'ledger')
    # A later correctly directed extraction uses the ordinary event key too.
    patch['people'][0]['ledger'][0].update(direction=direction, occurred='2026-10-03')
    apply.apply_extraction(conn, patch, 'clear-transfer', '张晨的餐费是20元。')
    assert conn.execute('SELECT COUNT(*) FROM ledger_entry').fetchone()[0] == 1


def test_discarded_unknown_direction_does_not_reappear_next_day(conn, monkeypatch):
    patch, result = unknown_transfer(conn, monkeypatch)
    resolve(conn, result['pending_ids'][0], 'discard')
    monkeypatch.setattr(apply, 'today_local', lambda: '2026-10-04')
    apply.apply_extraction(conn, patch, 'unclear-transfer', '张晨的餐费是20元。')
    assert not repo.list_pending(conn)
    assert conn.execute('SELECT COUNT(*) FROM ledger_entry').fetchone()[0] == 0


@pytest.mark.parametrize('source,direction', [('张晨给我转了20元餐费。', 'in'), ('我给张晨转了20元餐费。', 'out')])
def test_explicit_original_direction_can_resolve_missing_model_value(conn, monkeypatch, source, direction):
    _, result = unknown_transfer(conn, monkeypatch, source)
    assert not result['pending_ids']
    assert conn.execute('SELECT direction FROM ledger_entry').fetchone()[0] == direction
    assert repo.ledger_totals(conn)['count'] == 1


def test_confirm_unknown_transfer_matches_existing_known_event(conn, monkeypatch):
    _, result = unknown_transfer(conn, monkeypatch)
    payload = json.loads(repo.get_pending(conn, result['pending_ids'][0])['payload'])
    repo.add_ledger(conn, 'in', 2000, payload['person_id'], '已有收款',
                    apply._date_ms('2026-10-03'), 'confirmed',
                    apply.ledger_key(payload['person_id'], 'in', 2000, '2026-10-03'))
    conn.commit()
    resolve(conn, result['pending_ids'][0], 'confirm', direction='in')
    assert conn.execute('SELECT COUNT(*) FROM ledger_entry').fetchone()[0] == 1
    assert repo.ledger_totals(conn)['in_cents'] == 2000


def test_unknown_transfer_confirmation_cannot_revive_void_event(conn, monkeypatch):
    _, result = unknown_transfer(conn, monkeypatch)
    item_id = result['pending_ids'][0]
    payload = json.loads(repo.get_pending(conn, item_id)['payload'])
    repo.add_ledger(conn, 'in', 2000, payload['person_id'], '已撤销收款',
                    apply._date_ms('2026-10-03'), 'void',
                    apply.ledger_key(payload['person_id'], 'in', 2000, '2026-10-03'))
    conn.commit()
    with pytest.raises(ValueError, match='已撤销'):
        resolve(conn, item_id, 'confirm', direction='in')
    assert not repo.get_pending(conn, item_id)['resolved_at']
    assert repo.ledger_totals(conn)['count'] == 0
    assert conn.execute('SELECT COUNT(*) FROM applied_patch WHERE field=?', ('ledger',)).fetchone()[0] == 0


def test_pending_unknown_transfer_is_stable_across_days(conn, monkeypatch):
    patch, result = unknown_transfer(conn, monkeypatch)
    monkeypatch.setattr(apply, 'today_local', lambda: '2026-10-04')
    replay = apply.apply_extraction(conn, patch, 'unclear-transfer', '张晨的餐费是20元。')
    assert replay['pending_ids'] == result['pending_ids']
    assert len(repo.list_pending(conn, 'ledger')) == 1
    payload = json.loads(repo.get_pending(conn, result['pending_ids'][0])['payload'])
    assert payload['occurred'] == '2026-10-03'


def test_transfer_tool_and_extraction_keep_decimal_cents_identical(conn):
    result = tools.execute_tool(conn, 'record_transfer', {
        'person': '张晨', 'direction': 'in', 'amount_yuan': .29, 'occurred_date': '2026-10-03',
    })
    assert result['ok']
    patch = sanitize({'people': [raw_person(ledger=[{
        'direction': 'in', 'amount': '0.29', 'occurred': '2026-10-03', 'confidence': .95,
    }])]})
    apply.apply_extraction(conn, patch, 'decimal-transfer', '张晨给我转了0.29元。')
    assert repo.ledger_totals(conn) == {'in_cents': 29, 'out_cents': 0, 'count': 1}


@pytest.mark.parametrize('action', ['confirm', 'discard', 'pending'])
def test_later_model_direction_cannot_override_manual_review(conn, monkeypatch, action):
    patch, result = unknown_transfer(conn, monkeypatch)
    if action != 'pending':
        resolve(conn, result['pending_ids'][0], action, direction='in' if action == 'confirm' else None)
    patch['people'][0]['ledger'][0]['direction'] = 'out'
    monkeypatch.setattr(apply, 'today_local', lambda: '2026-10-04')
    replay = apply.apply_extraction(conn, patch, 'unclear-transfer', '张晨的餐费是20元。')
    assert repo.ledger_totals(conn) == {
        'in_cents': 2000 if action == 'confirm' else 0, 'out_cents': 0,
        'count': 1 if action == 'confirm' else 0,
    }
    assert len(repo.list_pending(conn, 'ledger')) == (1 if action == 'pending' else 0)
    assert replay['pending_ids'] == (result['pending_ids'] if action == 'pending' else [])
