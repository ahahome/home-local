"""Outcome memory across real HTTP routes, durable tasks and person lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import followup
from app.config import tz
from app.jobs import tasks
from app.store import db, repo
from app.util import new_id, now_ms

DAY = 86400000
NOTES = "现在在星桥软件工作，住在杭州，喜欢周末骑车。我答应给他发送白皮书。"


@pytest.fixture
def api(conn, monkeypatch):
    from app import llm
    from app.api import followups, main, outcome_memory as memory_api, workspace
    from app.agent import outcome_memory

    model = {"result": {"proposals": []}, "calls": [], "during_call": None}

    def chat_json(messages, **kwargs):
        model["calls"].append(deepcopy(messages))
        if model["during_call"]:
            model["during_call"]()
        return deepcopy(model["result"])

    for module in (followups, main, memory_api, workspace, tasks):
        monkeypatch.setattr(module, "get_conn", lambda: conn)
    monkeypatch.setattr(tasks, "start", lambda tid: None)
    monkeypatch.setattr(llm, "get_llm", lambda: pytest.fail("integration tests must not call a model"))
    monkeypatch.setattr(followup, "get_llm", lambda: SimpleNamespace(chat_json=lambda *a, **k: {
        "draft": "最近方便聊聊近况吗？", "source_ids": []}))
    monkeypatch.setattr(outcome_memory, "get_llm", lambda: SimpleNamespace(chat_json=chat_json))
    client = TestClient(main.app)
    yield SimpleNamespace(client=client, app=main.app, model=model, memory_api=memory_api,
                          domain=outcome_memory)
    client.close()


@pytest.fixture
def independent_clients(api, conn, monkeypatch):
    database = Path(conn.execute("PRAGMA database_list").fetchone()["file"])
    connections, connection_lock = [], Lock()

    def connection():
        current = db._connect(database)
        with connection_lock:
            connections.append(current)
        return current

    monkeypatch.setattr(api.memory_api, "get_conn", connection)
    first, second = TestClient(api.app), TestClient(api.app)
    yield first, second
    first.close()
    second.close()
    for current in connections:
        current.close()


def person(conn, name="陈岚"):
    pid = repo.create_person(conn, name, status="active")
    conn.commit()
    return pid


def outcome(api, conn, pid, *, notes=NOTES, contacted_at=None):
    response = api.client.post(f"/api/persons/{pid}/followups", json={
        "goal": "了解近况并跟进白皮书", "request_id": new_id()})
    assert response.status_code == 202, response.text
    task = response.json()
    tasks._run(task["id"])
    assert tasks.read(conn, task["id"])["status"] == "done"
    fid = task["payload"]["followup_id"]
    record = api.client.get(f"/api/followups/{fid}").json()
    default_contact = datetime.fromtimestamp(now_ms() / 1000, tz()).replace(
        hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
    response = api.client.post(f"/api/followups/{fid}/outcomes", json={
        "request_id": new_id(), "expected_revision": record["revision"],
        "contacted_at": contacted_at or int(default_contact.timestamp() * 1000),
        "channel": "电话", "notes": notes,
        "state": "follow_up", "next_followup_at": None, "complete_promise_ids": [],
        "new_promises": [{"who": "me", "content": "发送白皮书", "due_date": None}]})
    assert response.status_code == 200, response.text
    saved = response.json()["record"]["outcomes"][-1]
    return SimpleNamespace(pid=pid, fid=fid, oid=saved["id"], contact_id=saved["contact_id"],
                           contacted_at=saved["contacted_at"], prepare_task=task["id"])


def analyze(api, conn, record):
    response = api.client.post(f"/api/followup-outcomes/{record.oid}/memory-analysis", json={})
    assert response.status_code == 202, response.text
    task = response.json()
    assert task["kind"] == "outcome_memory"
    tasks._run(task["id"])
    finished = tasks.read(conn, task["id"])
    assert finished["status"] == "done", finished["error"]
    response = api.client.get(f"/api/followups/{record.fid}/memory-updates")
    assert response.status_code == 200, response.text
    saved = next(row for row in response.json()["outcomes"] if row["outcome_id"] == record.oid)
    return task, saved["proposals"]


def resolve(api, proposal, action="confirm", **changes):
    return api.client.post(f"/api/memory-updates/{proposal['id']}/resolve", json={
        "action": action, "expected_revision": proposal["revision"], **changes})


def counts(conn):
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("interaction", "promise", "memory_item", "person_field")}


def field_proposal(pid, content="星桥软件", field="company", evidence="在星桥软件工作", valid_from=None):
    return {"person_id": pid, "kind": "field", "field": field, "content": content,
            "memory_kind": None, "valid_from": valid_from, "evidence": evidence}


def memory_proposal(pid):
    return {"person_id": pid, "kind": "memory", "field": None, "content": "喜欢周末骑车",
            "memory_kind": "semantic", "valid_from": None, "evidence": "喜欢周末骑车"}


def proposals_for(api, fid):
    response = api.client.get(f"/api/followups/{fid}/memory-updates")
    assert response.status_code == 200, response.text
    return [proposal for row in response.json()["outcomes"] for proposal in row["proposals"]]


def test_main_analysis_is_read_only_until_confirmation_then_next_brief_and_fts_see_updates(api, conn):
    pid = person(conn)
    record = outcome(api, conn, pid)
    before = counts(conn)
    api.model["result"] = {"proposals": [field_proposal(pid), memory_proposal(pid)]}

    task, proposals = analyze(api, conn, record)

    assert counts(conn) == before
    assert len(proposals) == 2
    assert len(api.model["calls"]) == 1
    assert api.client.get(f"/api/tasks/{task['id']}").json()["status"] == "done"
    for proposal in proposals:
        response = api.client.get(f"/api/memory-updates/{proposal['id']}")
        assert response.status_code == 200, response.text
        assert response.json()["id"] == proposal["id"]
        confirmed = resolve(api, proposal)
        assert confirmed.status_code == 200, confirmed.text
    assert counts(conn) == {**before, "person_field": before["person_field"] + 1,
                           "memory_item": before["memory_item"] + 1}
    assert repo.current_fields(conn, pid)["company"]["value"] == "星桥软件"
    memory = conn.execute("SELECT * FROM memory_item WHERE person_id=?", (pid,)).fetchone()
    assert memory["text"] == "喜欢周末骑车" and memory["evidence_id"]
    assert conn.execute("SELECT 1 FROM memory_fts WHERE owner_type='memory' AND owner_id=?",
                        (memory["id"],)).fetchone()

    response = api.client.post(f"/api/persons/{pid}/followups", json={
        "goal": "继续聊骑车与工作近况", "request_id": new_id()})
    assert response.status_code == 202, response.text
    tasks._run(response.json()["id"])
    fresh = api.client.get(f"/api/followups/{response.json()['payload']['followup_id']}").json()
    assert any(item["value"] == "星桥软件" for item in fresh["brief"]["current_fields"])
    assert any(item["text"] == memory["text"] for item in fresh["brief"]["memories"])
    assert counts(conn)["interaction"] == before["interaction"]
    assert counts(conn)["promise"] == before["promise"]


@pytest.mark.parametrize("action", ["confirm", "discard"])
def test_generic_pending_endpoint_cannot_bypass_memory_revision_and_resolution(api, conn, action):
    pid = person(conn)
    record = outcome(api, conn, pid)
    api.model["result"] = {"proposals": [field_proposal(pid)]}
    _, proposals = analyze(api, conn, record)
    pending = api.client.get("/api/pending").json()["pending"]
    pending = [item for item in pending if item["kind"] == "outcome_memory"]
    assert len(pending) == 1

    response = api.client.post(f"/api/pending/{pending[0]['id']}/resolve", json={"action": action})

    assert response.status_code == 400, response.text
    assert repo.get_pending(conn, pending[0]["id"])["resolved_at"] is None
    assert proposals_for(api, record.fid) == proposals
    assert repo.current_fields(conn, pid) == {}


def test_older_outcome_field_joins_history_without_overwriting_a_later_current_value(api, conn):
    pid = person(conn)
    old_date = int(datetime.fromisoformat("2026-01-10T12:00:00+08:00").timestamp() * 1000)
    current_date = int(datetime.fromisoformat("2026-01-20T12:00:00+08:00").timestamp() * 1000)
    repo.upsert_field(conn, pid, "company", "目前公司", 1, None, current_date)
    conn.commit()
    record = outcome(api, conn, pid, contacted_at=old_date)
    api.model["result"] = {"proposals": [field_proposal(pid)]}
    before = counts(conn)
    _, proposals = analyze(api, conn, record)

    response = resolve(api, proposals[0])

    assert response.status_code == 200, response.text
    assert repo.current_fields(conn, pid)["company"]["value"] == "目前公司"
    assert repo.field_at(conn, pid, "company", old_date)["value"] == "星桥软件"
    history = repo.field_history(conn, pid, "company")
    assert len(history) == 2 and history[0]["valid_to"] == current_date
    assert counts(conn) == {**before, "person_field": before["person_field"] + 1}


def test_changed_field_requires_rebase_before_confirmation_and_retains_newer_value(api, conn):
    pid = person(conn)
    repo.upsert_field(conn, pid, "company", "原公司", 1, None, now_ms() - 10 * DAY)
    conn.commit()
    record = outcome(api, conn, pid)
    api.model["result"] = {"proposals": [field_proposal(pid)]}
    _, proposals = analyze(api, conn, record)
    proposal = proposals[0]
    repo.upsert_field(conn, pid, "company", "刚刚人工修正的公司", 1, None)
    conn.commit()
    before = counts(conn)

    rejected = resolve(api, proposal)

    assert rejected.status_code == 409, rejected.text
    assert counts(conn) == before
    rebased = resolve(api, proposal, "rebase")
    assert rebased.status_code == 200, rebased.text
    refreshed = api.client.get(f"/api/memory-updates/{proposal['id']}").json()
    assert refreshed["revision"] > proposal["revision"]
    assert resolve(api, proposal).status_code == 409
    confirmed = resolve(api, refreshed)
    assert confirmed.status_code == 200, confirmed.text
    assert repo.current_fields(conn, pid)["company"]["value"] == "刚刚人工修正的公司"
    assert repo.field_at(conn, pid, "company", record.contacted_at)["value"] == "星桥软件"
    assert counts(conn)["interaction"] == before["interaction"]
    assert counts(conn)["promise"] == before["promise"]


def test_archive_blocks_analysis_and_confirmation_until_person_is_restored(api, conn):
    pid = person(conn)
    record = outcome(api, conn, pid)
    api.model["result"] = {"proposals": [field_proposal(pid)]}
    _, proposals = analyze(api, conn, record)
    assert api.client.delete(f"/api/persons/{pid}").status_code == 200
    before = counts(conn)

    assert resolve(api, proposals[0]).status_code == 404
    assert api.client.post(f"/api/followup-outcomes/{record.oid}/memory-analysis", json={}).status_code == 404
    assert counts(conn) == before
    response = api.client.post(f"/api/persons/{pid}/restore")
    assert response.status_code == 200, response.text
    confirmed = resolve(api, proposals[0])
    assert confirmed.status_code == 200, confirmed.text
    assert repo.current_fields(conn, pid)["company"]["value"] == "星桥软件"


def test_merge_rejects_old_subject_confirmation_and_rebase_then_split_restores_original_subject(api, conn):
    survivor, original = person(conn, "同名陈岚"), person(conn, "同名陈岚")
    record = outcome(api, conn, original)
    api.model["result"] = {"proposals": [memory_proposal(original)]}
    _, proposals = analyze(api, conn, record)
    response = api.client.post("/api/persons/merge", json={"survivor_id": survivor, "merged_id": original})
    assert response.status_code == 200, response.text
    merge_id = response.json()["merge_log_id"]

    for action in ("confirm", "rebase"):
        rejected = resolve(api, proposals[0], action)
        assert rejected.status_code == 409, rejected.text
    assert not repo.memories_of(conn, survivor)
    assert api.client.post("/api/persons/split", json={"merge_log_id": merge_id}).status_code == 200
    confirmed = resolve(api, proposals[0])
    assert confirmed.status_code == 200, confirmed.text
    assert not repo.memories_of(conn, survivor)
    assert [row["text"] for row in repo.memories_of(conn, original)] == ["喜欢周末骑车"]
    assert conn.execute("SELECT person_id FROM followup WHERE id=?", (record.fid,)).fetchone()[0] == original


def test_competing_analysis_requests_share_one_durable_task_across_connections(api, conn, independent_clients):
    pid = person(conn)
    record = outcome(api, conn, pid)
    start = Barrier(2)

    def submit(client):
        start.wait(timeout=5)
        return client.post(f"/api/followup-outcomes/{record.oid}/memory-analysis", json={})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, client) for client in independent_clients]
        responses = [future.result(timeout=10) for future in futures]

    assert [response.status_code for response in responses] == [202, 202]
    assert responses[0].json()["id"] == responses[1].json()["id"]
    assert conn.execute("SELECT COUNT(*) FROM task WHERE kind='outcome_memory'").fetchone()[0] == 1
    tasks._run(responses[0].json()["id"])
    assert len(api.model["calls"]) == 1
    assert tasks.read(conn, responses[0].json()["id"])["status"] == "done"


def test_confirm_and_discard_compete_atomically_across_independent_connections(api, conn, independent_clients):
    pid = person(conn)
    record = outcome(api, conn, pid)
    api.model["result"] = {"proposals": [field_proposal(pid)]}
    _, proposals = analyze(api, conn, record)
    proposal = proposals[0]
    start = Barrier(2)

    def decide(client, action):
        start.wait(timeout=5)
        return action, client.post(f"/api/memory-updates/{proposal['id']}/resolve", json={
            "action": action, "expected_revision": proposal["revision"]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(decide, client, action) for client, action in zip(
            independent_clients, ("confirm", "discard"))]
        results = [future.result(timeout=10) for future in futures]

    assert sorted(response.status_code for _, response in results) == [200, 409]
    winner = next(action for action, response in results if response.status_code == 200)
    assert ("company" in repo.current_fields(conn, pid)) == (winner == "confirm")
    assert conn.execute("SELECT COUNT(*) FROM person_field WHERE person_id=?", (pid,)).fetchone()[0] == int(
        winner == "confirm")
    refreshed = api.client.get(f"/api/memory-updates/{proposal['id']}").json()
    assert refreshed["revision"] > proposal["revision"]
    assert all(row["resolved_at"] is not None for row in conn.execute(
        "SELECT * FROM pending_item WHERE kind='outcome_memory'"))


def test_purge_removes_unconfirmed_analysis_evidence_pending_and_tasks_but_preserves_other_subject(api, conn):
    removed, retained = person(conn, "要清除的陈岚"), person(conn, "保留的林舟")
    first, second = outcome(api, conn, removed), outcome(api, conn, retained)
    api.model["result"] = {"proposals": [field_proposal(removed)]}
    first_task, first_proposals = analyze(api, conn, first)
    api.model["result"] = {"proposals": [field_proposal(retained)]}
    second_task, second_proposals = analyze(api, conn, second)
    evidence = conn.execute("SELECT evidence_id FROM outcome_memory_analysis WHERE outcome_id=?",
                            (first.oid,)).fetchone()[0]
    assert evidence and conn.execute("SELECT 1 FROM evidence WHERE id=?", (evidence,)).fetchone()
    assert repo.current_fields(conn, removed) == {}
    proposal_ids = {item["id"] for item in first_proposals}
    pending_ids = [row["id"] for row in repo.list_pending(conn) if any(
        proposal_id in json.dumps(json.loads(row["payload"])) for proposal_id in proposal_ids)]
    assert len(pending_ids) == len(first_proposals) == 1

    response = api.client.post(f"/api/persons/{removed}/purge")

    assert response.status_code == 200, response.text
    for tid in (first.prepare_task, first_task["id"]):
        assert not conn.execute("SELECT 1 FROM task WHERE id=?", (tid,)).fetchone()
        assert not conn.execute("SELECT 1 FROM task_event WHERE task_id=?", (tid,)).fetchone()
    assert not conn.execute("SELECT 1 FROM evidence WHERE id=?", (evidence,)).fetchone()
    assert not conn.execute("SELECT 1 FROM outcome_memory_analysis WHERE outcome_id=?", (first.oid,)).fetchone()
    for pending_id in pending_ids:
        assert repo.get_pending(conn, pending_id) is None
    for proposal_id in proposal_ids:
        assert api.client.get(f"/api/memory-updates/{proposal_id}").status_code == 404
    assert api.client.get(f"/api/tasks/{second_task['id']}").status_code == 200
    assert [proposal["id"] for proposal in proposals_for(api, second.fid)] == [
        proposal["id"] for proposal in second_proposals]
    assert not repo.get_person(conn, removed)
    assert repo.get_person(conn, retained)


def test_purge_during_model_call_cannot_recreate_analysis_evidence_pending_or_task_events(api, conn):
    pid = person(conn)
    record = outcome(api, conn, pid)
    api.model["result"] = {"proposals": [field_proposal(pid)]}
    response = api.client.post(f"/api/followup-outcomes/{record.oid}/memory-analysis", json={})
    assert response.status_code == 202, response.text

    def purge():
        response = api.client.post(f"/api/persons/{pid}/purge")
        assert response.status_code == 200, response.text

    api.model["during_call"] = purge
    tasks._run(response.json()["id"])

    assert len(api.model["calls"]) == 1
    for table in ("person", "followup", "followup_outcome", "task", "task_event", "pending_item",
                  "evidence", "outcome_memory_analysis", "memory_item", "person_field", "interaction"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


@pytest.mark.parametrize("change", ["archive", "merge", "notes", "contact", "notes_and_contact"])
def test_source_changes_during_model_call_never_publish_proposals_for_the_old_snapshot(api, conn, change):
    pid = person(conn)
    other = person(conn, "同名的另一个陈岚")
    record = outcome(api, conn, pid)
    before = counts(conn)
    api.model["result"] = {"proposals": [field_proposal(pid)]}
    response = api.client.post(f"/api/followup-outcomes/{record.oid}/memory-analysis", json={})
    assert response.status_code == 202, response.text
    tid = response.json()["id"]

    def change_source():
        if change == "archive":
            assert api.client.delete(f"/api/persons/{pid}").status_code == 200
        elif change == "merge":
            response = api.client.post("/api/persons/merge", json={"survivor_id": other, "merged_id": pid})
            assert response.status_code == 200, response.text
        elif change in {"notes", "notes_and_contact"}:
            payload = json.loads(conn.execute("SELECT payload FROM followup_outcome WHERE id=?",
                                               (record.oid,)).fetchone()[0])
            payload["notes"] = "原始结果已更正，不能继续使用旧的模型快照。"
            conn.execute("UPDATE followup_outcome SET payload=? WHERE id=?",
                         (json.dumps(payload, ensure_ascii=False), record.oid))
            if change == "notes_and_contact":
                conn.execute("UPDATE interaction SET content=? WHERE id=?", (payload["notes"], record.contact_id))
            conn.commit()
        else:
            conn.execute("UPDATE interaction SET content=? WHERE id=?", ("实际联系的原文已更正", record.contact_id))
            conn.commit()

    api.model["during_call"] = change_source
    tasks._run(tid)

    finished = tasks.read(conn, tid)
    assert finished["status"] == "failed", finished
    assert len(api.model["calls"]) == 1
    assert counts(conn) == before
    assert not conn.execute("SELECT 1 FROM pending_item WHERE kind='outcome_memory'").fetchone()
    assert conn.execute("SELECT analyzed_at FROM outcome_memory_analysis WHERE outcome_id=?",
                        (record.oid,)).fetchone()[0] is None


def test_interrupted_after_proposals_commit_retries_without_model_or_duplicate_domain_writes(api, conn,
                                                                                           monkeypatch):
    pid = person(conn)
    record = outcome(api, conn, pid)
    api.model["result"] = {"proposals": [memory_proposal(pid)]}
    response = api.client.post(f"/api/followup-outcomes/{record.oid}/memory-analysis", json={})
    assert response.status_code == 202, response.text
    tid = response.json()["id"]
    original_event = tasks.event

    class Shutdown(BaseException):
        pass

    def stop_before_worker_result(conn, task_id, kind, data):
        if task_id == tid and kind == "result":
            raise Shutdown()
        return original_event(conn, task_id, kind, data)

    monkeypatch.setattr(tasks, "event", stop_before_worker_result)
    with pytest.raises(Shutdown):
        tasks._run(tid)
    assert tasks.read(conn, tid)["status"] == "running"
    proposals = proposals_for(api, record.fid)
    assert len(proposals) == len(api.model["calls"]) == 1
    monkeypatch.setattr(tasks, "event", original_event)
    tasks.recover()
    assert tasks.read(conn, tid)["status"] == "interrupted"
    confirmed = resolve(api, proposals[0])
    assert confirmed.status_code == 200, confirmed.text
    before = counts(conn)
    pending_before = conn.execute("SELECT COUNT(*) FROM pending_item").fetchone()[0]
    evidence_before = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]

    retried = api.client.post(f"/api/tasks/{tid}/retry")
    assert retried.status_code == 202, retried.text
    tasks._run(tid)

    finished = tasks.read(conn, tid)
    assert finished["status"] == "done", finished["error"]
    assert len(api.model["calls"]) == 1
    assert counts(conn) == before
    assert before["interaction"] == before["promise"] == before["memory_item"] == 1
    assert conn.execute("SELECT COUNT(*) FROM pending_item").fetchone()[0] == pending_before
    assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == evidence_before
    refreshed = proposals_for(api, record.fid)
    assert [proposal["id"] for proposal in refreshed] == [proposal["id"] for proposal in proposals]
    assert refreshed[0]["revision"] > proposals[0]["revision"]
