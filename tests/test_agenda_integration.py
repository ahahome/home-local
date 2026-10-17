"""Agenda behavior across the main HTTP app, outcomes and person lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import followup
from app.jobs import tasks
from app.store import db, repo
from app.util import now_ms

DAY = 86400000


@pytest.fixture
def api(conn, monkeypatch):
    from app import llm
    from app.agent import agenda
    from app.api import agenda as agenda_api, followups, main

    clock = {"now": now_ms()}
    connection = {"conn": conn}
    for module in (agenda_api, followups, main):
        monkeypatch.setattr(module, "get_conn", lambda: connection["conn"])
    monkeypatch.setattr(agenda, "now_ms", lambda: clock["now"])
    monkeypatch.setattr(tasks, "start", lambda tid: None)
    monkeypatch.setattr(llm, "get_llm", lambda: pytest.fail("integration tests must not call a model"))
    monkeypatch.setattr(followup, "get_llm", lambda: SimpleNamespace(
        chat_json=lambda *args, **kwargs: {"draft": "最近方便聊聊吗？", "source_ids": []}))
    client = TestClient(main.app)
    yield SimpleNamespace(client=client, connection=connection, clock=clock, app=main.app,
                          agenda_api=agenda_api)
    client.close()


def person(conn, name):
    pid = repo.create_person(conn, name, status="active")
    conn.commit()
    return pid


def contact(conn, pid, timestamp):
    iid, _ = repo.insert_interaction(conn, timestamp, "电话", "此前已经实际联系", None,
                                     [(pid, "participant")], kind="contact", dedupe_key="seed:" + pid)
    conn.commit()
    return iid


def get_agenda(client, include_hidden=False):
    response = client.get("/api/agenda", params={"include_hidden": include_hidden})
    assert response.status_code == 200, response.text
    return response.json()


def set_policy(client, pid, enabled=True, interval=14):
    response = client.get(f"/api/persons/{pid}/contact-policy")
    assert response.status_code == 200, response.text
    response = client.patch(f"/api/persons/{pid}/contact-policy", json={
        "enabled": enabled, "interval_days": interval, "expected_revision": response.json()["revision"]})
    assert response.status_code == 200, response.text
    return response.json()


def action(client, item, command, **changes):
    return client.post("/api/agenda/actions", json={
        "item_id": item["id"], "fingerprint": item["fingerprint"], "action": command, **changes})


def ready_followup(client, conn, pid):
    response = client.post(f"/api/persons/{pid}/followups", json={
        "goal": "确认合作进展", "request_id": "integration-prepare"})
    assert response.status_code == 202, response.text
    return followup.prepare_followup(conn, response.json()["payload"]["followup_id"])


def record_outcome(client, record, timestamp, **changes):
    response = client.post(f"/api/followups/{record['id']}/outcomes", json={
        "request_id": "integration-outcome", "expected_revision": record["revision"],
        "contacted_at": timestamp, "channel": "电话", "notes": "已实际确认合作进展",
        "state": "awaiting_reply", "next_followup_at": None,
        "complete_promise_ids": [], "new_promises": [], **changes})
    assert response.status_code == 200, response.text
    return response.json()["record"]


def test_main_app_exposes_agenda_with_actual_contacts_and_unknown_people(api, conn):
    known, unknown = person(conn, "陈岚"), person(conn, "林舟")
    iid = contact(conn, known, api.clock["now"] - 40 * DAY)
    repo.insert_interaction(conn, api.clock["now"], "文档", "今天补充了两人的资料", None,
                            [(known, "mentioned"), (unknown, "participant")], kind="note")
    conn.commit()

    payload = get_agenda(api.client)

    assert payload["date"] and payload["timezone"]
    assert [(i["person"]["id"], i["kind"], i["source"]["id"]) for i in payload["items"]] == [
        (known, "contact_gap", iid)]
    assert [p["id"] for p in payload["unknown_people"]] == [unknown]
    assert payload["counts"]["active"] == payload["counts"]["unknown"] == 1


def test_policy_patch_serializes_competing_clients_without_lost_updates(api, conn, monkeypatch):
    pid = person(conn, "陈岚")
    initial = set_policy(api.client, pid)
    database = Path(conn.execute("PRAGMA database_list").fetchone()["file"])
    connections = []
    connection_lock = Lock()

    def independent_connection():
        connection = db._connect(database)
        with connection_lock:
            connections.append(connection)
        return connection

    monkeypatch.setattr(api.agenda_api, "get_conn", independent_connection)
    start = Barrier(2)

    def update(client, enabled, interval):
        start.wait(timeout=5)
        return client.patch(f"/api/persons/{pid}/contact-policy", json={
            "enabled": enabled, "interval_days": interval, "expected_revision": initial["revision"]})

    first, second = TestClient(api.app), TestClient(api.app)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(update, first, False, 7), pool.submit(update, second, True, 30)]
            responses = [f.result(timeout=10) for f in futures]
        assert sorted(r.status_code for r in responses) == [200, 409]
        winner = next(r.json() for r in responses if r.status_code == 200)
        assert winner["revision"] == initial["revision"] + 1
        assert first.get(f"/api/persons/{pid}/contact-policy").json() == winner
        row = conn.execute("SELECT enabled,interval_days,revision FROM contact_policy WHERE person_id=?",
                            (pid,)).fetchone()
        assert tuple(row) == (int(winner["enabled"]), winner["interval_days"], winner["revision"])
    finally:
        first.close()
        second.close()
        for connection in connections:
            connection.close()


def test_outcome_immediately_replaces_agenda_sources_and_resets_periodic_contact(api, conn, monkeypatch):
    pid = person(conn, "陈岚")
    past = api.clock["now"] - 40 * DAY
    monkeypatch.setattr(followup, "now_ms", lambda: past)
    record = ready_followup(api.client, conn, pid)
    record = record_outcome(api.client, record, past, next_followup_at=api.clock["now"] - DAY)
    old_reminder = record["reminder_id"]
    promise = repo.add_promise(conn, "me", "发送项目资料", pid, api.clock["now"] - 2 * DAY)
    repo.add_reminder(conn, api.clock["now"] - 2 * DAY, "约定到期提醒", pid, promise)
    conn.commit()
    set_policy(api.client, pid, interval=7)
    before = get_agenda(api.client)
    assert {(i["source"]["kind"], i["source"]["id"]) for i in before["items"]} == {
        ("promise", promise), ("reminder", old_reminder)}

    monkeypatch.setattr(followup, "now_ms", lambda: api.clock["now"])
    record = record_outcome(api.client, record, api.clock["now"], request_id="integration-outcome-2",
                            state="completed", complete_promise_ids=[promise])

    after = get_agenda(api.client, include_hidden=True)
    assert after["items"] == after["hidden_items"] == []
    assert repo.last_contact(conn, pid) == api.clock["now"]
    assert record["status"] == "completed"
    assert conn.execute("SELECT status FROM reminder WHERE id=?", (old_reminder,)).fetchone()[0] == "cancelled"
    api.clock["now"] += 8 * DAY
    refreshed = get_agenda(api.client)["items"]
    assert len(refreshed) == 1 and refreshed[0]["kind"] == "contact_gap"
    assert refreshed[0]["source"]["id"] == record["outcomes"][-1]["contact_id"]


@pytest.mark.parametrize("state,remains", [("follow_up", True), ("awaiting_reply", False)])
def test_new_contact_preserves_explicit_followup_but_resolves_waiting_signal(api, conn, state, remains):
    pid = person(conn, "陈岚")
    record = ready_followup(api.client, conn, pid)
    record = record_outcome(api.client, record, api.clock["now"] - 4 * DAY, state=state)
    before = get_agenda(api.client)["items"]
    assert len(before) == 1 and before[0]["kind"] == state
    assert before[0]["followup_id"] == record["id"]

    contact(conn, pid, api.clock["now"])

    after = get_agenda(api.client)["items"]
    assert len(after) == int(remains)
    if remains:
        assert after[0]["id"] == "followup:" + record["id"]
        assert after[0]["kind"] == "follow_up"
        assert after[0]["fingerprint"] != before[0]["fingerprint"]
        record_outcome(api.client, record, api.clock["now"], request_id="integration-completed", state="completed")
        assert get_agenda(api.client)["items"] == []


def test_snooze_survives_a_new_client_and_database_connection(api, conn):
    pid = person(conn, "陈岚")
    promise = repo.add_promise(conn, "me", "发送资料", pid, api.clock["now"] - DAY)
    conn.commit()
    item = get_agenda(api.client)["items"][0]
    until = api.clock["now"] + 3600000
    assert item["source"]["id"] == promise
    assert action(api.client, item, "snooze", until=until).status_code == 200
    api.client.close()
    database = Path(conn.execute("PRAGMA database_list").fetchone()["file"])

    try:
        with closing(db._connect(database)) as reopened:
            api.connection["conn"] = reopened
            client = TestClient(api.app)
            try:
                assert get_agenda(client)["items"] == []
                hidden = get_agenda(client, include_hidden=True)["hidden_items"]
                assert len(hidden) == 1 and hidden[0]["fingerprint"] == item["fingerprint"]
                assert hidden[0]["disposition"] == {"state": "snoozed", "until": until}
                api.clock["now"] = until + 1
                assert [i["id"] for i in get_agenda(client)["items"]] == [item["id"]]
            finally:
                client.close()
    finally:
        api.connection["conn"] = conn


def test_purge_cascades_agenda_preferences_without_touching_another_person(api, conn):
    removed, retained = person(conn, "陈岚"), person(conn, "林舟")
    for pid in (removed, retained):
        contact(conn, pid, api.clock["now"] - 40 * DAY)
        set_policy(api.client, pid)
    items = get_agenda(api.client)["items"]
    for item in items:
        assert action(api.client, item, "dismiss").status_code == 200
    assert conn.execute("SELECT COUNT(*) FROM agenda_action").fetchone()[0] == 2

    response = api.client.post(f"/api/persons/{removed}/purge")

    assert response.status_code == 200, response.text
    assert not conn.execute("SELECT 1 FROM contact_policy WHERE person_id=?", (removed,)).fetchone()
    assert not conn.execute("SELECT 1 FROM agenda_action WHERE person_id=?", (removed,)).fetchone()
    assert conn.execute("SELECT COUNT(*) FROM contact_policy WHERE person_id=?", (retained,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM agenda_action WHERE person_id=?", (retained,)).fetchone()[0] == 1
    payload = get_agenda(api.client, include_hidden=True)
    assert [i["person"]["id"] for i in payload["hidden_items"]] == [retained]


def test_archived_and_merged_people_are_excluded_and_policies_remain_independent(api, conn):
    survivor, merged, archived = [person(conn, name) for name in ("陈岚", "林舟", "李响")]
    for pid in (survivor, merged, archived):
        contact(conn, pid, api.clock["now"] - 40 * DAY)
    survivor_policy = set_policy(api.client, survivor, enabled=False, interval=7)
    merged_policy = set_policy(api.client, merged, enabled=True, interval=30)
    set_policy(api.client, archived)
    assert api.client.delete(f"/api/persons/{archived}").status_code == 200
    assert [i["person"]["id"] for i in get_agenda(api.client)["items"]] == [merged]

    response = api.client.post("/api/persons/merge", json={"survivor_id": survivor, "merged_id": merged})

    assert response.status_code == 200, response.text
    payload = get_agenda(api.client, include_hidden=True)
    assert payload["items"] == payload["hidden_items"] == payload["unknown_people"] == []
    assert api.client.get(f"/api/persons/{survivor}/contact-policy").json() == survivor_policy
    for pid in (merged, archived):
        assert api.client.get(f"/api/persons/{pid}/contact-policy").status_code == 404
    dormant = conn.execute("SELECT enabled,interval_days,revision FROM contact_policy WHERE person_id=?",
                           (merged,)).fetchone()
    assert tuple(dormant) == (1, 30, merged_policy["revision"])

    assert api.client.post("/api/persons/split", json={
        "merge_log_id": response.json()["merge_log_id"]}).status_code == 200
    assert api.client.get(f"/api/persons/{merged}/contact-policy").json() == merged_policy
    assert [i["person"]["id"] for i in get_agenda(api.client)["items"]] == [merged]


def test_disposition_follows_source_and_person_fingerprint_across_merge_and_split(api, conn):
    survivor, merged = person(conn, "陈岚"), person(conn, "林舟")
    promise = repo.add_promise(conn, "me", "发送方案", merged, api.clock["now"] - DAY)
    conn.commit()
    original = get_agenda(api.client)["items"][0]
    assert action(api.client, original, "dismiss").status_code == 200
    response = api.client.post("/api/persons/merge", json={"survivor_id": survivor, "merged_id": merged})
    assert response.status_code == 200, response.text

    moved = get_agenda(api.client)["items"]

    assert len(moved) == 1 and moved[0]["source"]["id"] == promise
    assert moved[0]["person"]["id"] == survivor
    assert moved[0]["id"] == original["id"] and moved[0]["fingerprint"] != original["fingerprint"]
    assert action(api.client, original, "snooze", until=api.clock["now"] + DAY).status_code == 409
    assert action(api.client, moved[0], "dismiss").status_code == 200

    assert api.client.post("/api/persons/split", json={
        "merge_log_id": response.json()["merge_log_id"]}).status_code == 200
    payload = get_agenda(api.client, include_hidden=True)
    assert payload["items"] == []
    restored = payload["hidden_items"]
    assert len(restored) == 1 and restored[0]["fingerprint"] == original["fingerprint"]
    assert restored[0]["person"]["id"] == merged
    assert restored[0]["disposition"]["state"] == "dismissed"
    assert action(api.client, moved[0], "restore").status_code == 409
