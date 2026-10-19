"""资料可检索，只有明确参与的实际联系影响关系跟进。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import followup, harness
from app.ingest import chatlog, normalize
from app.jobs import scheduler, tasks
from app.memory import apply as apply_mod
from app.memory import compile_card, retrieve
from app.store import db, repo
from app.util import now_ms


def person(conn, name):
    return repo.create_person(conn, name, status="active")


def people_patch(*names):
    return {"people": [{"name": name, "canonical_name": name, "fields": [],
                        "aliases": [], "relations": [], "facts": [],
                        "promises": [], "ledger": []} for name in names], "my_notes": []}


def test_note_preserves_existing_contact_and_remains_queryable(conn):
    pid = person(conn, "林舟")
    contacted_at = now_ms() - 20 * 86400000
    contact_id, _ = repo.insert_interaction(conn, contacted_at, "manual", "实际通话", None,
                                            [(pid, "participant")], kind="contact")
    note_id, _ = repo.insert_interaction(conn, now_ms(), "chat", "林舟的产品资料", None,
                                        [(pid, "participant")])

    assert repo.last_contact(conn, pid) == contacted_at
    assert [r["id"] for r in repo.recent_contacts(conn, pid)] == [contact_id]
    assert [r["id"] for r in repo.recent_interactions(conn, pid)] == [note_id, contact_id]
    assert [r["id"] for r in repo.recent_interactions(conn, pid, kind="note")] == [note_id]
    assert [r["id"] for r in repo.recent_interactions(conn, pid, kind="contact")] == [contact_id]
    assert any(r["id"] == note_id for r in retrieve.hybrid_search(conn, "产品资料"))
    assert "资料记录[" in compile_card.build_card(conn, pid)
    assert "联系[" in compile_card.build_card(conn, pid)


def test_contact_counts_only_participant_even_for_same_named_people(conn):
    participants = [person(conn, "小张"), person(conn, "王琳")]
    mentioned = person(conn, "小张")
    from_person, to_person = person(conn, "赵明"), person(conn, "李响")
    ts = now_ms()
    repo.insert_interaction(conn, ts, "manual", "与两位参与者开会，提及另外三个人", None,
                            [(participants[0], "mentioned"), (participants[0], "participant"),
                             (participants[1], "participant"), (participants[1], "mentioned"),
                             (mentioned, "mentioned"), (from_person, "from"), (to_person, "to")],
                            kind="contact")

    for pid in participants:
        assert repo.last_contact(conn, pid) == ts
        assert len(repo.recent_contacts(conn, pid)) == 1
        assert repo.get_person(conn, pid)["importance"] > 0
    for pid in (mentioned, from_person, to_person):
        assert repo.last_contact(conn, pid) is None
        assert repo.recent_contacts(conn, pid) == []
        assert repo.get_person(conn, pid)["importance"] == 0
        assert "联系[" not in compile_card.build_card(conn, pid)


def test_replaying_note_cannot_promote_it_or_add_participants(conn):
    pid = person(conn, "陈岚")
    iid, created = repo.insert_interaction(conn, 1000, "chat", "同一份记录", "first",
                                           [(pid, "mentioned")])
    assert created
    repeated, created = repo.insert_interaction(conn, 9000, "chat", "同一份记录", "retry",
                                                [(pid, "participant")], kind="contact")
    assert repeated == iid and not created
    row = conn.execute("SELECT * FROM interaction WHERE id=?", (iid,)).fetchone()
    assert (row["kind"], row["ts"]) == ("note", 1000)
    assert repo.last_contact(conn, pid) is None
    assert repo.recent_interactions(conn, pid)[0]["person_role"] == "mentioned"


def test_replaying_contact_cannot_demote_or_refresh_it(conn):
    pid = person(conn, "陈岚")
    iid, _ = repo.insert_interaction(conn, 1000, "manual", "已发生的联系", None,
                                     [(pid, "participant")], kind="contact")
    repeated, created = repo.insert_interaction(conn, 9000, "manual", "已发生的联系", None,
                                                [(pid, "mentioned")])
    assert (repeated, created) == (iid, False)
    assert repo.last_contact(conn, pid) == 1000
    assert repo.recent_contacts(conn, pid)[0]["kind"] == "contact"


def test_source_keys_keep_same_worded_contacts_separate_and_retries_stable(conn):
    first, second = person(conn, "陈岚"), person(conn, "李响")
    ids = []
    for ts, pid, key in ((1000, first, "outcome:1"), (2000, first, "outcome:2"),
                         (3000, second, "outcome:3")):
        iid, created = repo.insert_interaction(conn, ts, "manual", "已沟通", None,
                                               [(pid, "participant")], kind="contact", dedupe_key=key)
        assert created
        ids.append(iid)
    repeated, created = repo.insert_interaction(conn, 9000, "manual", "重试改变的文本", None,
                                                [(second, "participant")], kind="contact", dedupe_key="outcome:1")
    assert (repeated, created) == (ids[0], False)
    assert len(set(ids)) == 3
    assert repo.last_contact(conn, first) == 2000
    assert repo.last_contact(conn, second) == 3000
    assert [r["id"] for r in repo.recent_contacts(conn, first)] == [ids[1], ids[0]]


def test_quiet_people_and_importance_ignore_new_notes(conn):
    old, recent, only_note = [person(conn, n) for n in ("旧联系", "新联系", "资料人物")]
    ts = now_ms()
    repo.insert_interaction(conn, ts - 30 * 86400000, "manual", "旧联系通话", None,
                            [(old, "participant")], kind="contact")
    repo.insert_interaction(conn, ts, "manual", "新联系通话", None,
                            [(recent, "participant"), (only_note, "mentioned")], kind="contact")
    for index, pid in enumerate((old, only_note)):
        repo.insert_interaction(conn, ts, "chat", f"最新资料{index}", None, [(pid, "mentioned")])
    rows = {r["person_id"]: r for r in retrieve.quiet_people(conn, days=14)}
    assert set(rows) == {old, only_note}
    assert rows[old]["last_contact"] == ts - 30 * 86400000
    assert rows[old]["last_interaction"] == rows[old]["last_contact"]
    assert rows[only_note]["last_contact"] is None
    assert repo.get_person(conn, only_note)["importance"] == 0
    assert repo.get_person(conn, recent)["importance"] > repo.get_person(conn, old)["importance"]


@pytest.mark.parametrize("query", ["林舟最近怎么样？", "帮我起草发给林舟的消息"])
def test_agent_queries_and_drafts_keep_only_audit_without_refreshing_contact(conn, monkeypatch, query):
    pid = person(conn, "林舟")
    ts = now_ms() - 20 * 86400000
    repo.insert_interaction(conn, ts, "manual", "此前实际联系", None,
                            [(pid, "participant")], kind="contact")
    cid = repo.create_conversation(conn)
    monkeypatch.setattr(harness, "get_llm", lambda: SimpleNamespace(chat=lambda *a, **kw: {"content": "回答"}))
    monkeypatch.setattr(harness.turn_intent, 'classify_turn', lambda text, **kwargs:
        harness.turn_intent.validate_intent({'intent': 'query', 'record_spans': [], 'reason': 'query fixture'}, text))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test_double', 'semantic_verified': False}, 'warnings': []})
    monkeypatch.setattr(harness.extract, "extract_from_text", lambda *a: pytest.fail('query must not enter extraction'))

    result = harness.run_turn(conn, cid, query)

    assert repo.last_contact(conn, pid) == ts
    interactions = repo.recent_interactions(conn, pid)
    assert len(interactions) == 1 and interactions[0]['kind'] == 'contact'
    assert result['saved'] == result['pending'] == []
    assert conn.execute('SELECT COUNT(*) FROM pending_item').fetchone()[0] == 0
    source_id = result['conversation_scope']['user_message_id']
    assert conn.execute('SELECT excerpt FROM evidence WHERE source_ref=?', (source_id,)).fetchone()[0] == query


@pytest.mark.parametrize("entry", ["file", "url"])
def test_normal_document_and_webpage_ingestion_is_note(conn, monkeypatch, entry):
    pid = person(conn, "林舟")
    monkeypatch.setattr(normalize.extract, "extract_from_text", lambda *a: people_patch("林舟"))
    monkeypatch.setattr(normalize.documents, "extract_text", lambda *a: "林舟的公司资料")
    monkeypatch.setattr(normalize.webpage, "fetch_and_clean", lambda *a: {"title": "资料", "text": "林舟"})

    if entry == "file":
        normalize.ingest_file(conn, "profile.pdf", b"fake document")
    else:
        normalize.ingest_url(conn, "https://example.com/profile")

    assert repo.last_contact(conn, pid) is None
    assert repo.recent_interactions(conn, pid)[0]["kind"] == "note"


def test_chat_import_contacts_speakers_but_not_third_party_mentions(conn, monkeypatch):
    alice, bob, third = [person(conn, n) for n in ("陈岚", "李响", "林舟")]
    text = """[2026-10-01 10:00:00] 陈岚: 林舟的方案我看过了
[2026-10-01 10:01:00] 我: 我也看过，下午讨论
[2026-10-01 10:10:00] 李响: 收到，晚点再回复
"""
    monkeypatch.setattr(chatlog.extract, "extract_from_text", lambda *a: people_patch("陈岚", "李响", "林舟"))

    chatlog.import_chatlog(conn, "group.txt", text.encode())

    messages = chatlog.parse(text)["messages"]
    assert repo.last_contact(conn, alice) == messages[1]["ts"]
    assert repo.last_contact(conn, bob) == messages[2]["ts"]
    assert repo.last_contact(conn, third) is None
    assert len(repo.recent_contacts(conn, alice)) == len(repo.recent_contacts(conn, bob)) == 1
    assert all(r["person_role"] == "mentioned" for r in repo.recent_interactions(conn, third))


def test_chat_import_requires_user_participation_in_each_day(conn, monkeypatch):
    pid = person(conn, "陈岚")
    text = """[2026-10-01 10:00:00] 陈岚: 今天一起讨论
[2026-10-01 10:01:00] 我: 好
[2026-10-02 10:00:00] 陈岚: 今天我和别人聊了
"""
    monkeypatch.setattr(chatlog.extract, "extract_from_text", lambda *a: people_patch("陈岚"))

    chatlog.import_chatlog(conn, "days.txt", text.encode())

    assert repo.last_contact(conn, pid) == chatlog.parse(text)["messages"][1]["ts"]
    assert [r["kind"] for r in repo.recent_interactions(conn, pid)] == ["note", "contact"]


@pytest.mark.parametrize("me", [None, "不存在的用户"])
def test_guessed_or_absent_self_does_not_create_contacts(conn, monkeypatch, me):
    first, second = person(conn, "陈岚"), person(conn, "李响")
    text = """[2026-10-01 10:00:00] 陈岚: 你好
[2026-10-01 10:01:00] 李响: 你好
"""
    monkeypatch.setattr(chatlog.extract, "extract_from_text", lambda *a: people_patch("陈岚", "李响"))

    chatlog.import_chatlog(conn, "others.txt", text.encode(), me=me)

    assert repo.last_contact(conn, first) is None
    assert repo.last_contact(conn, second) is None
    assert {r[0] for r in conn.execute("SELECT kind FROM interaction")} == {"note"}


def test_unknown_chat_date_is_not_replaced_with_import_time_as_contact(conn, monkeypatch):
    pid = person(conn, "陈岚")
    monkeypatch.setattr(chatlog.extract, "extract_from_text", lambda *a: people_patch("陈岚"))

    chatlog.import_chatlog(conn, "undated.txt", "陈岚: 你好\n我: 你好".encode())

    assert repo.last_contact(conn, pid) is None
    assert repo.recent_interactions(conn, pid)[0]["kind"] == "note"


def test_ambiguous_speaker_is_never_bound_to_one_same_named_person(conn, monkeypatch):
    first, second = person(conn, "小张"), person(conn, "小张")
    text = """[2026-10-01 10:00:00] 小张: 你好
[2026-10-01 10:01:00] 我: 收到
"""
    monkeypatch.setattr(chatlog.extract, "extract_from_text", lambda *a: people_patch("小张"))

    result = chatlog.import_chatlog(conn, "names.txt", text.encode())

    assert result["pending_ids"]
    assert repo.last_contact(conn, first) is repo.last_contact(conn, second) is None
    assert conn.execute("SELECT kind FROM interaction").fetchone()[0] == "note"


@pytest.mark.parametrize("channel", ["chat", "chatlog", "import"])
def test_migration_keeps_historical_interactions_as_notes(tmp_path, channel):
    root = Path(__file__).resolve().parents[1]
    with sqlite3.connect(tmp_path / "old.db") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript((root / "app/store/schema.sql").read_text())
        for migration in sorted((root / "migrations").glob("*.sql")):
            if int(migration.name.split("_")[0]) <= 4:
                conn.executescript(migration.read_text())
        conn.execute("PRAGMA user_version=4")
        pid = person(conn, "历史人物")
        conn.execute("UPDATE person SET importance=0.9 WHERE id=?", (pid,))
        conn.execute("INSERT INTO interaction VALUES(?,?,?,?,?,?,?)",
                     ("old", 1000, channel, "old-source", "旧记录", "old-hash", 2000))
        conn.execute("INSERT INTO interaction_person VALUES(?,?,?)", ("old", pid, "from"))
        conn.commit()

        db.migrate(conn)

        row = conn.execute("SELECT * FROM interaction WHERE id='old'").fetchone()
        assert row["kind"] == "note"
        assert (row["ts"], row["channel"], row["content_hash"]) == (1000, channel, "old-hash")
        assert repo.get_person(conn, pid)["importance"] == 0
        assert repo.last_contact(conn, pid) is None
        assert repo.recent_interactions(conn, pid)[0]["person_role"] == "from"
        db.migrate(conn)
        assert len(repo.recent_interactions(conn, pid)) == 1


@pytest.mark.parametrize("survivor_role", [None, "mentioned", "participant"])
def test_merge_and_split_restore_each_person_contact_role(conn, survivor_role):
    survivor, merged = person(conn, "林舟"), person(conn, "另一位林舟")
    ts = now_ms() - 60000
    older, _ = repo.insert_interaction(conn, ts - 1000, "电话", "存活人物此前的联系", None,
                                       [(survivor, "participant")], kind="contact")
    links = [(merged, "participant")]
    if survivor_role:
        links.append((survivor, survivor_role))
    shared, _ = repo.insert_interaction(conn, ts, "微信", "合并前另一人的实际联系", None,
                                        links, kind="contact")

    log_id = apply_mod.merge_persons(conn, survivor, merged)

    assert repo.last_contact(conn, survivor) == ts
    assert [r["id"] for r in repo.recent_contacts(conn, survivor)] == [shared, older]
    assert repo.recent_contacts(conn, survivor)[0]["person_role"] == "participant"
    assert repo.recent_interactions(conn, merged) == []

    apply_mod.split_merge(conn, log_id)

    assert repo.last_contact(conn, merged) == ts
    assert [r["id"] for r in repo.recent_contacts(conn, merged)] == [shared]
    assert repo.recent_contacts(conn, merged)[0]["person_role"] == "participant"
    expected_last = ts if survivor_role == "participant" else ts - 1000
    assert repo.last_contact(conn, survivor) == expected_last
    shared_rows = [r for r in repo.recent_interactions(conn, survivor) if r["id"] == shared]
    assert [r["person_role"] for r in shared_rows] == ([survivor_role] if survivor_role else [])
    expected_contacts = [shared, older] if survivor_role == "participant" else [older]
    assert [r["id"] for r in repo.recent_contacts(conn, survivor)] == expected_contacts
    assert repo.get_person(conn, merged)["importance"] > 0


def ready_followup(conn, monkeypatch, pid, request_id="prepare-1"):
    monkeypatch.setattr(tasks, "start", lambda tid: None)
    monkeypatch.setattr(followup, "get_llm", lambda: SimpleNamespace(
        chat_json=lambda *args, **kwargs: {"draft": "最近方便聊聊吗？", "source_ids": []}))
    task = followup.create_followup(conn, pid, "确认下一次沟通时间", request_id)
    return followup.prepare_followup(conn, task["payload"]["followup_id"])


def confirm_contact(conn, record, ts, **changes):
    return followup.record_outcome(conn, record["id"], {
        "request_id": "outcome-1", "expected_revision": record["revision"],
        "contacted_at": ts, "channel": "电话", "notes": "实际沟通后等待回复",
        "state": "awaiting_reply", "next_followup_at": None,
        "complete_promise_ids": [], "new_promises": [], **changes})["record"]


def test_followup_and_its_contact_return_to_original_person_after_split(conn, monkeypatch):
    survivor, merged = person(conn, "林舟"), person(conn, "陈岚")
    own_record = ready_followup(conn, monkeypatch, survivor)
    other_record = ready_followup(conn, monkeypatch, merged)
    ts = now_ms() - 60000
    other_record = confirm_contact(conn, other_record, ts, next_followup_at=now_ms() + 86400000,
        new_promises=[{"who": "them", "content": "合并前约定发送资料", "due_date": "2026-12-20"}])
    outcome = other_record["outcomes"][0]
    repo.add_reminder(conn, now_ms() + 172800000, "独立提醒", merged)
    unrelated_reminder = repo.add_reminder(conn, now_ms() + 172800000, "存活人物提醒", survivor)
    reminders = conn.execute("SELECT * FROM reminder WHERE person_id=?", (merged,)).fetchall()
    reminder_ids = {r["id"] for r in reminders}
    assert len(reminder_ids) == 3
    scan_at = max(r["fire_at"] for r in reminders) + 1

    log_id = apply_mod.merge_persons(conn, survivor, merged)

    records = {r["id"]: r for r in followup.list_followups(conn, survivor)}
    assert set(records) == {own_record["id"], other_record["id"]}
    assert records[other_record["id"]]["person_id"] == survivor
    assert records[other_record["id"]]["outcomes"][0]["contact_id"] == outcome["contact_id"]
    assert repo.last_contact(conn, survivor) == ts
    assert {r["id"] for r in conn.execute("SELECT id FROM reminder WHERE person_id=?", (survivor,))} == (
        reminder_ids | {unrelated_reminder})
    assert reminder_ids <= {r["id"] for r in repo.due_reminders(conn, scan_at)}

    apply_mod.split_merge(conn, log_id)

    assert [r["id"] for r in followup.list_followups(conn, survivor)] == [own_record["id"]]
    restored = followup.list_followups(conn, merged)
    assert len(restored) == 1 and restored[0]["id"] == other_record["id"]
    assert restored[0]["person_id"] == restored[0]["person"]["id"] == merged
    assert restored[0]["outcomes"] == other_record["outcomes"]
    assert repo.last_contact(conn, survivor) is None
    assert repo.last_contact(conn, merged) == ts
    assert [r["id"] for r in repo.recent_contacts(conn, merged)] == [outcome["contact_id"]]
    assert {r["id"] for r in conn.execute("SELECT id FROM reminder WHERE person_id=?", (merged,))} == reminder_ids
    assert {r["id"] for r in conn.execute("SELECT id FROM reminder WHERE person_id=?", (survivor,))} == {
        unrelated_reminder}
    assert reminder_ids <= {r["id"] for r in repo.due_reminders(conn, scan_at)}


def test_purge_cascades_followup_outcomes_and_keeps_other_person_records(conn, monkeypatch):
    removed, retained = person(conn, "林舟"), person(conn, "陈岚")
    ts = now_ms() - 60000
    records = []
    for pid in (removed, retained):
        record = ready_followup(conn, monkeypatch, pid)
        records.append(confirm_contact(conn, record, ts))
    removed_record, retained_record = records
    contact_id = removed_record["outcomes"][0]["contact_id"]

    result = repo.purge_person(conn, removed)

    assert result["ok"] and repo.verify_purged(conn, removed, result["aliases"]) == []
    assert not conn.execute("SELECT 1 FROM followup WHERE id=?", (removed_record["id"],)).fetchone()
    assert not conn.execute("SELECT 1 FROM followup_outcome WHERE followup_id=?",
                            (removed_record["id"],)).fetchone()
    assert not conn.execute("SELECT 1 FROM interaction WHERE id=?", (contact_id,)).fetchone()
    assert not conn.execute("SELECT 1 FROM memory_fts WHERE owner_type='interaction' AND owner_id=?",
                            (contact_id,)).fetchone()
    assert followup.list_followups(conn, retained) == [retained_record]
    assert repo.last_contact(conn, retained) == ts


def test_merge_failure_restores_followup_promise_and_reminder_ownership(conn, monkeypatch):
    survivor, merged = person(conn, "林舟"), person(conn, "陈岚")
    record = ready_followup(conn, monkeypatch, merged)
    confirm_contact(conn, record, now_ms() - 60000, next_followup_at=now_ms() + 86400000,
        new_promises=[{"who": "them", "content": "稍后发送资料", "due_date": "2026-12-20"}])
    tables = ("person", "person_alias", "followup", "interaction_person", "promise", "reminder", "merge_log")
    before = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
              for t in tables}
    conn.execute("CREATE TRIGGER fail_merge BEFORE UPDATE OF person_id ON reminder "
                 "WHEN NEW.person_id != OLD.person_id BEGIN "
                 "SELECT RAISE(ABORT,'reminder transfer unavailable'); END")
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="reminder transfer unavailable"):
        apply_mod.merge_persons(conn, survivor, merged)

    assert before == {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
                      for t in tables}
    assert followup.read_followup(conn, record["id"])["person_id"] == merged
    assert repo.last_contact(conn, survivor) is None
    assert repo.last_contact(conn, merged) is not None


def test_journal_input_distinguishes_notes_contacts_and_third_party_mentions(conn, monkeypatch):
    participant, mentioned = person(conn, "陈岚"), person(conn, "林舟")
    day = "2026-10-01"
    start, _ = scheduler._day_range_ms(day)
    repo.insert_interaction(conn, start + 1000, "文档", "只整理了资料", None,
                            [(mentioned, "participant")], kind="note")
    repo.insert_interaction(conn, start + 2000, "电话", "只和陈岚实际通话", None,
                            [(participant, "participant"), (mentioned, "mentioned")], kind="contact")
    repo.insert_interaction(conn, start + 3000, "导入", "没有明确参与者的记录", None,
                            [(mentioned, "mentioned")], kind="contact")
    repo.insert_interaction(conn, start - 1, "电话", "前一天的联系不应出现", None,
                            [(participant, "participant")], kind="contact")
    mid = repo.add_memory(conn, "episodic", "历史事件只作为记忆", mentioned)
    conn.execute("UPDATE memory_item SET ts=? WHERE id=?", (start + 4000, mid))
    captured = []

    def chat(messages, **kwargs):
        captured.extend(messages)
        return {"content": "今日联系陈岚，另整理了一些资料。"}

    monkeypatch.setattr(scheduler, "get_conn", lambda: conn)
    monkeypatch.setattr(scheduler, "get_llm", lambda: SimpleNamespace(chat=chat))

    result = scheduler.generate_journal(day)

    assert repo.get_journal(conn, day)["text"] == result
    assert "只有标注为实际沟通的条目才能归入今天联系了谁" in captured[0]["content"]
    prompt = captured[1]["content"]
    assert "资料记录（涉及：林舟）：只整理了资料" in prompt
    assert "实际沟通（涉及：陈岚）：只和陈岚实际通话" in prompt
    assert "资料记录（涉及：林舟）：没有明确参与者的记录" in prompt
    assert "- 记忆：历史事件只作为记忆" in prompt
    assert "前一天的联系不应出现" not in prompt
    assert sum("实际沟通" in line for line in prompt.splitlines()) == 1


def test_person_api_exposes_actual_contact_time_and_each_records_role(conn, monkeypatch):
    from app.api import main

    participant, mentioned = person(conn, "陈岚"), person(conn, "林舟")
    ts = now_ms() - 60000
    own, _ = repo.insert_interaction(conn, ts, "电话", "林舟实际参加的联系", None,
                                     [(mentioned, "participant")], kind="contact")
    third_party, _ = repo.insert_interaction(conn, ts + 1000, "微信", "陈岚联系中提及林舟", None,
                                             [(participant, "participant"), (mentioned, "mentioned")],
                                             kind="contact")
    note, _ = repo.insert_interaction(conn, ts + 2000, "文档", "林舟的新资料", None,
                                      [(mentioned, "participant")], kind="note")
    unknown = person(conn, "仅有资料")
    repo.insert_interaction(conn, ts + 3000, "文档", "尚无实际沟通", None, [(unknown, "mentioned")])
    conn.commit()
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    client = TestClient(main.app)

    response = client.get(f"/api/persons/{mentioned}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["last_contact"] == ts
    assert [(r["id"], r["kind"], r["person_role"]) for r in payload["interactions"]] == [
        (note, "note", "participant"), (third_party, "contact", "mentioned"),
        (own, "contact", "participant")]
    assert client.get(f"/api/persons/{participant}").json()["last_contact"] == ts + 1000
    assert client.get(f"/api/persons/{unknown}").json()["last_contact"] is None


@pytest.mark.parametrize("fail_first", [False, True])
def test_split_restores_new_followup_assets_without_moving_unrelated_assets(conn, monkeypatch, fail_first):
    from app.memory.writes import promise_key

    survivor, merged = person(conn, "林舟"), person(conn, "陈岚")
    record = ready_followup(conn, monkeypatch, merged)
    old_promise = repo.add_promise(conn, "me", "存活人物原有承诺", survivor)
    conn.commit()
    log_id = apply_mod.merge_persons(conn, survivor, merged)
    ts = now_ms() - 60000
    record = confirm_contact(conn, followup.read_followup(conn, record["id"]), ts,
        next_followup_at=now_ms() + 86400000, complete_promise_ids=[old_promise],
        new_promises=[{"who": "them", "content": "发送新的项目资料", "due_date": "2026-12-20"}])
    outcome = record["outcomes"][0]
    promised = conn.execute("SELECT * FROM promise WHERE content='发送新的项目资料'").fetchone()
    evidence_id = promised["evidence_id"]
    source_ref = "followup-outcome:" + outcome["id"]
    own_record = ready_followup(conn, monkeypatch, survivor)
    own_record = confirm_contact(conn, own_record, ts + 1000,
        next_followup_at=now_ms() + 172800000,
        new_promises=[{"who": "me", "content": "存活人物新承诺", "due_date": "2026-12-21"}])
    unrelated_ids = {r["id"] for r in repo.open_promises(conn, survivor)} - {promised["id"]}

    if fail_first:
        tables = ("person", "followup", "interaction_person", "promise", "reminder",
                  "applied_patch", "evidence", "merge_log")
        before = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
                  for t in tables}
        conn.execute("CREATE TRIGGER fail_split BEFORE UPDATE OF person_id ON reminder "
                     "WHEN NEW.person_id != OLD.person_id BEGIN "
                     "SELECT RAISE(ABORT,'reminder transfer unavailable'); END")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="reminder transfer unavailable"):
            apply_mod.split_merge(conn, log_id)
        after = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
                 for t in tables}
        assert after == before
        conn.execute("DROP TRIGGER fail_split")
        conn.commit()

    apply_mod.split_merge(conn, log_id)

    assert followup.read_followup(conn, record["id"])["person_id"] == merged
    assert followup.read_followup(conn, own_record["id"])["person_id"] == survivor
    assert repo.last_contact(conn, merged) == ts
    assert repo.last_contact(conn, survivor) == ts + 1000
    assert [r["id"] for r in repo.recent_contacts(conn, merged)] == [outcome["contact_id"]]
    restored_promise = conn.execute("SELECT * FROM promise WHERE id=?", (promised["id"],)).fetchone()
    assert restored_promise["person_id"] == merged
    restored_reminders = conn.execute("SELECT * FROM reminder WHERE id=? OR promise_id=?",
                                      (outcome["reminder_id"], promised["id"])).fetchall()
    assert len(restored_reminders) == 2
    assert all(r["person_id"] == merged and "陈岚" in r["text"] for r in restored_reminders)
    proof = conn.execute("SELECT excerpt FROM evidence WHERE id=?", (evidence_id,)).fetchone()[0]
    assert json.loads(proof)["person_id"] == merged
    patches = conn.execute("SELECT * FROM applied_patch WHERE source_msg_id=?", (source_ref,)).fetchall()
    assert len(patches) == 1 and patches[0]["person_id"] == merged
    assert patches[0]["fingerprint"] == promise_key(merged, "them", promised["content"], "2026-12-20")
    assert {r["id"] for r in repo.open_promises(conn, survivor)} == unrelated_ids
    assert conn.execute("SELECT person_id,status FROM promise WHERE id=?", (old_promise,)).fetchone()[:] == (
        survivor, "done")
    own_reminder = conn.execute("SELECT * FROM reminder WHERE id=?", (own_record["reminder_id"],)).fetchone()
    assert own_reminder["person_id"] == survivor and "林舟" in own_reminder["text"]
    audit = json.loads(repo.get_merge_log(conn, log_id)["snapshot"])
    assert any(a["outcome_id"] == outcome["id"] and a["promise_ids"] == [promised["id"]]
               for a in audit["restored_followup_assets"])
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("interaction", "promise", "reminder", "followup_outcome")}
    replayed = followup.record_outcome(conn, record["id"], {
        key: value for key, value in outcome.items()
        if key not in {"id", "contact_id", "reminder_id", "receipts", "created_at"}})
    assert replayed["record"]["person_id"] == merged
    assert counts == {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in counts}
