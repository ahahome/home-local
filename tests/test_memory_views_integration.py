"""Saved memories and proposal history through real routes and communication tasks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import followup, outcome_memory
from app.jobs import tasks
from app.store import repo
from app.util import new_id, now_ms

NOTES = "在星桥软件工作，喜欢周末骑车。" + "这次交流还记录了骑车路线和当地天气。" * 30
MEMORY = "喜欢周末骑车"
EDITED_MEMORY = "周末喜欢骑车，偏好清晨出发"


@pytest.fixture
def api(conn, monkeypatch):
    from app import llm
    from app.api import followups, main, memories, outcome_memory as proposal_api, workspace

    model = {"result": {"proposals": []}}
    for module in (followups, main, memories, proposal_api, workspace, tasks):
        monkeypatch.setattr(module, "get_conn", lambda: conn)
    monkeypatch.setattr(tasks, "start", lambda tid: None)
    monkeypatch.setattr(llm, "get_llm", lambda: pytest.fail("memory view tests must not call a model"))
    monkeypatch.setattr(followup, "get_llm", lambda: SimpleNamespace(chat_json=lambda *a, **k: {
        "draft": "最近方便聊聊近况吗？", "source_ids": []}))
    monkeypatch.setattr(outcome_memory, "get_llm", lambda: SimpleNamespace(
        chat_json=lambda *a, **k: deepcopy(model["result"])))
    client = TestClient(main.app)
    yield SimpleNamespace(client=client, model=model)
    client.close()


def person(conn, name="陈岚"):
    pid = repo.create_person(conn, name, status="active")
    conn.commit()
    return pid


def memory_candidate(pid):
    return {"person_id": pid, "kind": "memory", "field": None, "content": MEMORY,
            "memory_kind": "semantic", "valid_from": None, "evidence": MEMORY}


def analyzed_outcome(api, conn, pid, *, include_field=False):
    response = api.client.post(f"/api/persons/{pid}/followups", json={
        "goal": "聊工作和骑车近况", "request_id": new_id()})
    assert response.status_code == 202, response.text
    task = response.json()
    tasks._run(task["id"])
    assert tasks.read(conn, task["id"])["status"] == "done"
    fid = task["payload"]["followup_id"]
    record = api.client.get(f"/api/followups/{fid}").json()
    response = api.client.post(f"/api/followups/{fid}/outcomes", json={
        "request_id": new_id(), "expected_revision": record["revision"],
        "contacted_at": now_ms() - 60000, "channel": "电话", "notes": NOTES,
        "state": "completed", "next_followup_at": None,
        "complete_promise_ids": [], "new_promises": []})
    assert response.status_code == 200, response.text
    saved = response.json()["record"]["outcomes"][-1]
    api.model["result"] = {"proposals": [memory_candidate(pid)]}
    if include_field:
        api.model["result"]["proposals"].append({"person_id": pid, "kind": "field", "field": "company",
            "content": "星桥软件", "memory_kind": None, "valid_from": None, "evidence": "在星桥软件工作"})
    response = api.client.post(f"/api/followup-outcomes/{saved['id']}/memory-analysis", json={})
    assert response.status_code == 202, response.text
    tasks._run(response.json()["id"])
    finished = tasks.read(conn, response.json()["id"])
    assert finished["status"] == "done", finished["error"]
    response = api.client.get(f"/api/followups/{fid}/memory-updates")
    assert response.status_code == 200, response.text
    proposals = response.json()["outcomes"][0]["proposals"]
    return SimpleNamespace(pid=pid, fid=fid, oid=saved["id"], proposals=proposals,
                           memory=next(item for item in proposals if item["kind"] == "memory"))


def collection(api, path="/api/memories", **params):
    response = api.client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def resolve(api, proposal, action="confirm", **changes):
    response = api.client.post(f"/api/memory-updates/{proposal['id']}/resolve", json={
        "action": action, "expected_revision": proposal["revision"], **changes})
    assert response.status_code == 200, response.text
    return response.json()


def confirmed_memory(api, conn, pid, *, edited=False):
    record = analyzed_outcome(api, conn, pid)
    resolve(api, record.memory, content=EDITED_MEMORY if edited else MEMORY)
    saved = collection(api, person_id=pid)["memories"]
    assert len(saved) == 1
    return record, saved[0]


def test_review_collection_separates_proposals_from_saved_memories_and_preserves_edited_source(api, conn):
    pid = person(conn)
    assert collection(api)["total"] == 0

    record = analyzed_outcome(api, conn, pid, include_field=True)

    pending = collection(api, "/api/memory-updates", person_id=pid, status="pending")
    assert pending["total"] == 2
    assert {proposal["id"] for proposal in pending["proposals"]} == {
        proposal["id"] for proposal in record.proposals}
    assert collection(api)["memories"] == []
    field = next(proposal for proposal in record.proposals if proposal["kind"] == "field")
    resolve(api, field)
    assert collection(api)["total"] == 0
    assert collection(api, "/api/memory-updates", person_id=pid, kind="field", status="confirmed")["total"] == 1

    resolved = resolve(api, record.memory, content=EDITED_MEMORY)

    page = collection(api, person_id=pid, q="清晨", kind="semantic", limit=1, offset=0)
    assert page["total"] == 1 and page["has_more"] is False
    assert page["limit"] == 1 and page["offset"] == 0
    memory = page["memories"][0]
    assert memory["text"] == EDITED_MEMORY
    assert memory["person"]["id"] == pid
    assert memory["proposal_id"] == record.memory["id"]
    assert memory["followup_id"] == record.fid
    assert resolved["receipt"]["id"] == memory["id"]
    response = api.client.get(f"/api/memories/{memory['id']}")
    assert response.status_code == 200, response.text
    detail = response.json()
    assert detail["id"] == memory["id"] and detail["text"] == EDITED_MEMORY
    review = json.loads(detail["source"]["excerpt"])
    assert detail["source"]["source_ref"] == "outcome-memory-review:" + record.memory["id"]
    assert review["edited_by_user"] is True
    assert review["original_quote"] == MEMORY
    assert review["user_confirmed"]["content"] == EDITED_MEMORY
    assert review["outcome_id"] == record.oid
    assert detail["original_source"]["id"] == review["source_evidence_id"]
    assert detail["original_source"]["source_ref"] == "followup-outcome:" + record.oid
    assert detail["original_source"]["excerpt"] == NOTES
    assert len(memory["original_source"]["excerpt"]) <= 300 < len(detail["original_source"]["excerpt"])
    history = collection(api, "/api/memory-updates", person_id=pid, kind="memory", status="confirmed", q="清晨")
    assert history["total"] == 1
    assert history["proposals"][0]["id"] == record.memory["id"]
    assert history["proposals"][0]["receipt"]["id"] == detail["id"]
    assert collection(api, "/api/memory-updates", status="pending")["total"] == 0


def test_discarded_proposal_remains_in_history_without_creating_a_saved_memory(api, conn):
    pid = person(conn)
    record = analyzed_outcome(api, conn, pid)

    resolve(api, record.memory, "discard")

    assert collection(api)["memories"] == []
    assert collection(api, "/api/memory-updates", status="pending")["total"] == 0
    history = collection(api, "/api/memory-updates", person_id=pid, status="discarded", q="骑车")
    assert history["total"] == 1 and history["has_more"] is False
    assert history["proposals"][0]["id"] == record.memory["id"]
    assert history["proposals"][0]["status"] == "discarded"
    assert not conn.execute("SELECT 1 FROM memory_item WHERE person_id=?", (pid,)).fetchone()


def test_merge_and_split_move_saved_memory_owner_without_redirecting_historical_proposal_subject(api, conn):
    survivor, original = person(conn, "同名陈岚"), person(conn, "同名陈岚")
    record, memory = confirmed_memory(api, conn, original)
    response = api.client.post("/api/persons/merge", json={"survivor_id": survivor, "merged_id": original})
    assert response.status_code == 200, response.text
    merge_id = response.json()["merge_log_id"]

    merged = collection(api, person_id=survivor)

    assert merged["total"] == 1
    assert merged["memories"][0]["id"] == memory["id"]
    assert merged["memories"][0]["person"]["id"] == survivor
    assert merged["memories"][0]["proposal_id"] == record.memory["id"]
    assert collection(api, "/api/memory-updates", q=MEMORY)["total"] == 0
    proposal = api.client.get(f"/api/memory-updates/{record.memory['id']}").json()
    assert proposal["person"]["id"] == original and proposal["stale"] is True
    detail = api.client.get(f"/api/memories/{memory['id']}").json()
    assert detail["person"]["id"] == survivor and detail["proposal_id"] == record.memory["id"]

    response = api.client.post("/api/persons/split", json={"merge_log_id": merge_id})

    assert response.status_code == 200, response.text
    assert collection(api, person_id=survivor)["total"] == 0
    restored = collection(api, person_id=original)
    assert restored["total"] == 1 and restored["memories"][0]["id"] == memory["id"]
    assert restored["memories"][0]["person"]["id"] == original
    history = collection(api, "/api/memory-updates", person_id=original, status="confirmed")
    assert history["total"] == 1 and history["proposals"][0]["id"] == record.memory["id"]
    assert history["proposals"][0]["person"]["id"] == original
    assert history["proposals"][0]["stale"] is False


def test_archive_restore_and_purge_change_visibility_without_removing_personal_memories(api, conn):
    pid = person(conn)
    eid = repo.add_evidence(conn, "text", source_ref="personal-source", excerpt="用户自己的阅读偏好")
    personal_id = repo.add_memory(conn, "semantic", "我喜欢阅读传记", evidence_id=eid)
    conn.commit()
    record, memory = confirmed_memory(api, conn, pid, edited=True)
    assert collection(api)["total"] == 2
    assert [item["id"] for item in collection(api, scope="personal")["memories"]] == [personal_id]

    assert api.client.delete(f"/api/persons/{pid}").status_code == 200

    assert [item["id"] for item in collection(api)["memories"]] == [personal_id]
    assert api.client.get(f"/api/memories/{memory['id']}").status_code == 404
    assert collection(api, "/api/memory-updates", q=EDITED_MEMORY)["total"] == 0
    archived = api.client.get(f"/api/memory-updates/{record.memory['id']}")
    assert archived.status_code == 200 and archived.json()["stale"] is True

    assert api.client.post(f"/api/persons/{pid}/restore").status_code == 200

    assert collection(api)["total"] == 2
    restored = api.client.get(f"/api/memories/{memory['id']}")
    assert restored.status_code == 200 and restored.json()["text"] == EDITED_MEMORY
    assert collection(api, "/api/memory-updates", person_id=pid, status="confirmed")["total"] == 1

    response = api.client.post(f"/api/persons/{pid}/purge")

    assert response.status_code == 200, response.text
    assert api.client.get(f"/api/memories/{memory['id']}").status_code == 404
    assert api.client.get(f"/api/memory-updates/{record.memory['id']}").status_code == 404
    assert collection(api, "/api/memory-updates")["total"] == 0
    assert [item["id"] for item in collection(api)["memories"]] == [personal_id]
    personal = api.client.get(f"/api/memories/{personal_id}")
    assert personal.status_code == 200 and personal.json()["person"] is None
    assert personal.json()["source"]["id"] == eid
