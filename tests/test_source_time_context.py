"""SQLite-backed model context keeps availability distinct from fact and audit time."""
import json

from app.agent.memory_sources import field_sources, interaction_source, memory_source, source_state
from app.agent.source_registry import SourceRegistry, compact_source
from app.memory import memory_corrections
from app.store import repo
from app.util import new_id


def seed_memory(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    eid = repo.add_evidence(conn, 'text', excerpt='林舟之前偏好电话')
    mid = repo.add_memory(conn, 'semantic', '林舟之前偏好电话', pid, .9, eid, ts=1000)
    conn.execute('UPDATE memory_item SET created_at=2000,updated_at=9000 WHERE id=?', (mid,))
    conn.execute('UPDATE evidence SET created_at=3000 WHERE id=?', (eid,))
    return pid, mid


def correct(conn, mid, monkeypatch, stamp, action='edit', **changes):
    monkeypatch.setattr(memory_corrections, 'now_ms', lambda: stamp)
    revision = conn.execute('SELECT revision FROM memory_item WHERE id=?', (mid,)).fetchone()[0]
    conn.commit()
    return memory_corrections.correct_memory(conn, mid, {
        'action': action, 'expected_revision': revision, 'request_id': new_id(), **changes})


def test_memory_dated_background_is_available_but_not_verified_as_current(conn):
    _, mid = seed_memory(conn)
    # A system update/revision and even an edit-looking evidence excerpt do not
    # supply a user-edit audit or an event date.
    conn.execute('UPDATE memory_item SET revision=7 WHERE id=?', (mid,))
    conn.execute('UPDATE evidence SET excerpt=? WHERE id=(SELECT evidence_id FROM memory_item WHERE id=?)',
                 (json.dumps({'edited_by_user': True, 'valid_from': '2000-01-01'}), mid))
    snapshot = memory_source(conn, mid)
    model = compact_source(snapshot)
    assert '未撤销' in model['记录状态']
    assert '当前事实是否仍成立尚未核实' in model['事实时效']
    assert model['时间信息']['最近人工编辑时间'] == '未知'
    assert model['时间信息']['记忆所标日期（不代表录入或修改日期）'].endswith('08:00:01+08:00')
    assert model['时间信息']['首次录入时间'].endswith('08:00:02+08:00')
    assert not {'revision', 'temporal_status', 'date', 'valid_from', 'ts', 'record_context'} & model.keys()
    # Full quotations remain compatible with the original source inspector.
    assert snapshot['revision'] == 7 and snapshot['temporal_status'] == 'historical_background'
    assert snapshot['date'] == snapshot['ts'] == 1000


def test_restored_record_keeps_real_edit_date_separate_from_restore_date(conn, monkeypatch):
    _, mid = seed_memory(conn)
    correct(conn, mid, monkeypatch, 4000, text='林舟偏好短邮件')
    correct(conn, mid, monkeypatch, 5000, action='withdraw')
    assert memory_source(conn, mid) is None
    correct(conn, mid, monkeypatch, 6000, action='restore')
    snapshot = memory_source(conn, mid)
    model = compact_source(snapshot)
    assert '未撤销' in model['记录状态']
    assert '尚未核实' in model['事实时效']
    assert model['时间信息']['最近人工编辑时间'].endswith('08:00:04+08:00')
    assert model['时间信息']['最近撤销或恢复操作'] == {
        '操作': '恢复', '操作时间': '1970-01-01T08:00:06+08:00'}
    assert model['时间信息']['记忆所标日期（不代表录入或修改日期）'].endswith('08:00:01+08:00')
    assert model['时间信息']['首次录入时间'].endswith('08:00:02+08:00')
    assert snapshot['revision'] == 3 and snapshot['text'] == '林舟偏好短邮件'
    assert source_state(conn, snapshot)['status'] == 'current'


def test_archive_restore_system_updates_do_not_claim_a_user_edit(conn):
    pid, mid = seed_memory(conn)
    repo.soft_delete_person(conn, pid)
    assert memory_source(conn, mid) is None
    repo.restore_person(conn, pid)
    model = compact_source(memory_source(conn, mid))
    assert '未撤销' in model['记录状态']
    assert model['时间信息']['最近人工编辑时间'] == '未知'
    assert '最近撤销或恢复操作' not in model['时间信息']


def test_field_effective_recorded_and_edit_dates_are_not_interchangeable(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    repo.upsert_field(conn, pid, 'city', '杭州', 1., None, valid_from=1000)
    before = field_sources(conn, pid)[0]
    repo.upsert_field(conn, pid, 'city', '上海', 1., None, valid_from=5000)
    conn.execute('UPDATE person_field SET recorded_at=9000 WHERE person_id=?', (pid,))
    current, historical = [compact_source(source) for source in field_sources(conn, pid)]
    assert current['事实时效'] == '当前档案字段，尚无结束记录'
    assert historical['事实时效'] == '已结束的历史档案字段，不代表现状'
    assert current['时间信息']['字段生效时间'].endswith('08:00:05+08:00')
    assert historical['时间信息']['字段结束时间'].endswith('08:00:05+08:00')
    assert current['时间信息']['录入时间'].endswith('08:00:09+08:00')
    assert current['时间信息']['最近人工编辑时间'] == '未知'
    assert source_state(conn, before)['status'] == 'updated'


def test_interaction_occurrence_does_not_reuse_import_or_evidence_date(conn):
    pid = repo.create_person(conn, '林舟', status='active')
    iid, _ = repo.insert_interaction(conn, 1000, 'chat', '林舟说准备搬家', 'import',
                                    [(pid, 'participant')], kind='contact')
    conn.execute('UPDATE interaction SET created_at=9000 WHERE id=?', (iid,))
    model = compact_source(interaction_source(conn, iid))
    assert model['时间信息']['发生时间'].endswith('08:00:01+08:00')
    assert model['时间信息']['录入时间'].endswith('08:00:09+08:00')
    assert model['时间信息']['最近人工编辑时间'] == '未知'
    assert '不证明' in model['事实时效']


def test_old_snapshot_is_not_marked_updated_by_new_metadata_but_real_changes_still_are(conn, monkeypatch):
    _, mid = seed_memory(conn)
    old = memory_source(conn, mid)
    old.pop('record_context')
    old['id'] = 'S1'
    assert source_state(conn, old)['status'] == 'current'
    correct(conn, mid, monkeypatch, 4000, text='林舟偏好短邮件')
    assert source_state(conn, old)['status'] == 'updated'
    correct(conn, mid, monkeypatch, 5000, action='withdraw')
    assert source_state(conn, old)['status'] == 'withdrawn'


def test_new_time_metadata_change_is_a_source_change_and_preserves_registered_quote(conn, monkeypatch):
    _, mid = seed_memory(conn)
    correct(conn, mid, monkeypatch, 4000, text='林舟偏好短邮件')
    registry = SourceRegistry(conn)
    snapshot = registry.register(memory_source(conn, mid))
    compact = registry.tool_result({'sources': [memory_source(conn, mid)]})['sources'][0]
    assert compact['时间信息']['最近人工编辑时间'].endswith('08:00:04+08:00')
    assert registry.sources['S1'] == snapshot
    conn.execute("UPDATE memory_correction SET created_at=8000 WHERE memory_id=? AND action='edit'", (mid,))
    assert source_state(conn, snapshot)['status'] == 'updated'
    assert snapshot['record_context']['last_user_edit_at'] == 4000


def test_legacy_model_source_does_not_infer_withdrawal_from_background_tag(conn):
    _, mid = seed_memory(conn)
    old = memory_source(conn, mid)
    old.pop('record_context')
    model = compact_source(old)
    assert '不可据此推断记录已撤销或已恢复' in model['记录状态']
    assert model['时间信息']['首次录入时间'] == '未知'
    assert '背景记录不表示已撤销' in model['事实时效']
