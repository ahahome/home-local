"""Cross-entry regression tests for the defects found in the project audit."""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent import harness, tools, research
from app.jobs import tasks
from app.memory import apply, retrieve
from app.memory.pending import resolve
from app.store import repo
from app.util import now_ms


def person_patch(name="张晨", **data):
    return {"name": name, "canonical_name": name, "fields": [], "aliases": [],
            "relations": [], "facts": [], "promises": [], "ledger": [], **data}


def test_drafting_instruction_never_becomes_memory_through_extract_or_tool(conn):
    from app.memory.extract import sanitize
    source='帮我给沈悦起草一条跟进消息，语气轻松。只起草，不发送。'
    text='用户要求给沈悦起草一条跟进消息，语气轻松，且只起草不发送'
    extraction={'people': [], 'my_notes': [{'text':text,'kind':'episodic','confidence':1}]}
    assert sanitize(extraction, source=source)['my_notes'] == []
    apply.apply_extraction(conn, extraction, 'draft-only', source)
    result=tools.execute_tool(conn, 'note', {'text':text}, source_text=source, source_msg_id='draft-tool')
    assert result['ok']
    assert not repo.my_profile_memories(conn)


def test_negative_facts_and_durable_preferences_survive_command_filter(conn):
    from app.memory.extract import sanitize
    samples=[('我没有给陈岚转账，也没有答应发合同。','用户没有给陈岚转账，也没有答应发合同'),
             ('我希望通过分析改善睡眠。','用户希望通过分析改善睡眠'),
             ('请记住：我喜欢简短自然的回复。','用户喜欢简短自然的回复')]
    for i,(source,text) in enumerate(samples):
        obj=sanitize({'people':[],'my_notes':[{'text':text,'kind':'semantic','confidence':1}]},source=source)
        assert len(obj['my_notes'])==1
        apply.apply_extraction(conn,obj,f'personal-{i}',source)
    assert len(repo.my_profile_memories(conn))==3


def test_duplicate_names_have_company_context_in_list_and_confirmation(conn, monkeypatch):
    from app.api import main
    monkeypatch.setattr(main,'get_conn',lambda:conn)
    a=repo.create_person(conn,'张晨',status='active')
    b=repo.create_person(conn,'张晨',status='active')
    repo.upsert_field(conn,a,'company','青禾商贸',1,None)
    repo.upsert_field(conn,b,'company','星桥软件',1,None)
    repo.add_pending(conn,'person_identity',{'candidates':[{'person_id':a,'display_name':'张晨'},
                     {'person_id':b,'display_name':'张晨'}]},'同名待确认')
    conn.commit()
    client=TestClient(main.app)
    assert {p['summary'] for p in client.get('/api/persons').json()['persons']}=={'青禾商贸','星桥软件'}
    candidates=client.get('/api/pending').json()['pending'][0]['payload']['candidates']
    assert {p['hint'] for p in candidates}=={'青禾商贸','星桥软件'}


def test_contradictory_transfer_direction_and_date_prefixed_promise_never_double_write(conn):
    source='2026-10-01，我给江遥转了47.23元餐费。我答应2026-10-21前给江遥发送验收文件。'
    extraction={'people':[person_patch('江遥',
        ledger=[{'direction':'in','amount_cents':4723,'occurred':'2026-10-01','note':'餐费','confidence':.95}],
        promises=[{'who':'me','content':'2026-10-21前给江遥发送验收文件','due':'2026-10-21','confidence':.95}])]}
    apply.apply_extraction(conn,extraction,'cross-path',source)
    tools.execute_tool(conn,'record_transfer',{'person':'江遥','direction':'out','amount_yuan':47.23,
        'occurred_date':'2026-10-01','note':'餐费'},source_text=source,source_msg_id='cross-path')
    tools.execute_tool(conn,'create_promise',{'person':'江遥','who':'me','content':'给江遥发送验收文件',
        'due_date':'2026-10-21'},source_text=source,source_msg_id='cross-path')
    assert repo.ledger_totals(conn)=={'in_cents':0,'out_cents':4723,'count':1}
    assert conn.execute('SELECT COUNT(*) FROM promise').fetchone()[0]==1
    assert conn.execute('SELECT COUNT(*) FROM reminder').fetchone()[0]==1


def test_existing_legacy_promise_and_distinct_same_day_actions(conn):
    from app.memory.writes import promise_key
    pid=repo.create_person(conn,'江遥',status='active')
    repo.add_promise(conn,'me','2026-10-21前给江遥发送验收文件',pid,apply._date_ms('2026-10-21'))
    for content in ('给江遥发送验收文件','给江遥发送报价单'):
        apply.apply_extraction(conn,{'people':[person_patch('江遥',promises=[
            {'who':'me','content':content,'due':'2026-10-21','confidence':.95}])]},content,content)
    assert conn.execute('SELECT COUNT(*) FROM promise').fetchone()[0]==2
    assert promise_key(pid,'me','10月21日前给江遥发送验收文件','2026-10-21')==promise_key(pid,'me','给江遥发送验收文件','2026-10-21')


@pytest.mark.parametrize('first,second', [
    ('2026-10-21前给江遥发送验收文件', '发送验收文件'),
    ('发送验收文件', '给江遥发送验收文件'),
    ('给小江发送验收文件', '给江遥发送验收文件'),
])
def test_promise_recipient_variants_deduplicate_across_extract_and_tool(conn, first, second):
    pid = repo.create_person(conn, '江遥', status='active')
    repo.add_alias(conn, pid, '小江', 'manual', 1)
    apply.apply_extraction(conn, {'people': [person_patch('江遥', promises=[
        {'who': 'me', 'content': first, 'due': '2026-10-21', 'confidence': .95}])]},
        'recipient-variant', first)
    result = tools.execute_tool(conn, 'create_promise', {'person_id': pid, 'who': 'me',
        'content': second, 'due_date': '2026-10-21'}, source_text=second,
        source_msg_id='recipient-variant')
    assert result['ok']
    assert conn.execute('SELECT COUNT(*) FROM promise').fetchone()[0] == 1
    assert conn.execute('SELECT COUNT(*) FROM reminder').fetchone()[0] == 1


def test_promise_recipient_normalization_preserves_distinct_commitments(conn):
    for who, content, due in [('me', '给江遥发送验收文件', '2026-10-21'),
                              ('me', '给陈岚发送验收文件', '2026-10-21'),
                              ('me', '给江遥发送报价单', '2026-10-21'),
                              ('me', '发送验收文件', '2026-10-22'),
                              ('them', '发送验收文件', '2026-10-21')]:
        apply.apply_extraction(conn, {'people': [person_patch('江遥', promises=[
            {'who': who, 'content': content, 'due': due, 'confidence': .95}])]},
            content + due + who, content)
    assert conn.execute('SELECT COUNT(*) FROM promise').fetchone()[0] == 5


def test_direction_override_requires_named_parties_and_exact_amount():
    from app.memory.writes import explicit_transfer_direction
    assert explicit_transfer_direction('江遥给我转了47.23元餐费。',['江遥'],4723)=='in'
    assert explicit_transfer_direction('我收到潘念转账：68.42元。',['潘念'],6842)=='in'
    assert explicit_transfer_direction('我没有给江遥转47.23元。',['江遥'],4723) is None
    assert explicit_transfer_direction('09:01 潘念：我给江遥转了47.23元。',['江遥'],4723) is None
    assert explicit_transfer_direction('我给江遥转了48.23元。',['江遥'],4723) is None
    assert explicit_transfer_direction('我给江遥转了47.23元。江遥给我转了47.23元。',['江遥'],4723) is None


def test_archived_people_are_hidden_from_global_ledger_but_restorable(conn):
    pid = repo.create_person(conn, '李明', status='active')
    tools.execute_tool(conn, 'record_transfer', {'person_id': pid, 'direction': 'out', 'amount_yuan': 100})
    assert repo.ledger_totals(conn)['out_cents'] == 10000
    repo.archive_person(conn, pid)
    assert repo.ledger_totals(conn)['count'] == 0
    assert not repo.ledger_entries(conn)
    repo.restore_person(conn, pid)
    assert repo.ledger_totals(conn)['count'] == 1


def test_ledger_pages_reach_older_entries_and_keep_global_totals(conn, monkeypatch):
    from app.api import main, workspace
    monkeypatch.setattr(workspace, 'get_conn', lambda: conn)
    pid = repo.create_person(conn, '模拟分页人物', status='active')
    archived = repo.create_person(conn, '模拟归档人物', status='active')
    ids = set()
    for i in range(105):
        lid, _ = repo.add_ledger(conn, 'out', 100, pid, str(i), 1000, 'confirmed', f'page-{i}')
        ids.add(lid)
    repo.add_ledger(conn, 'in', 9900, archived, '归档', 2000, 'confirmed', 'archived')
    repo.archive_person(conn, archived)
    void_id, _ = repo.add_ledger(conn, 'out', 500, pid, '撤销', 3000, 'void', 'void-page')
    conn.commit()
    client = TestClient(main.app)
    first = client.get('/api/ledger').json()
    second = client.get('/api/ledger?offset=100').json()
    assert len(first['entries']) == 100 and first['has_more']
    assert len(second['entries']) == 5 and not second['has_more']
    assert {e['id'] for e in first['entries']} | {e['id'] for e in second['entries']} == ids
    assert not ({e['id'] for e in first['entries']} & {e['id'] for e in second['entries']})
    assert first['totals'] == second['totals'] == {'in_cents': 0, 'out_cents': 10500, 'count': 105}
    visible = client.get('/api/ledger?include_void=true&limit=1').json()
    assert visible['entries'][0]['id'] == void_id and visible['has_more']
    assert client.get('/api/ledger?offset=-1').status_code == 422
    assert client.get('/api/ledger?limit=101').status_code == 422


def test_purge_removes_exclusive_interactions_and_their_search_copies(conn):
    pid = repo.create_person(conn, '林木', status='active')
    iid, _ = repo.insert_interaction(conn, now_ms(), 'chat', '林木的独占互动', 'test', [(pid, 'mentioned')])
    repo.purge_person(conn, pid)
    assert not conn.execute('SELECT 1 FROM interaction WHERE id=?', (iid,)).fetchone()
    assert not conn.execute('SELECT 1 FROM memory_fts WHERE owner_id=?', (iid,)).fetchone()


def test_journal_can_recover_promises_and_money_without_raw_conversation(conn, monkeypatch):
    from app.jobs import scheduler
    from app.util import today_local
    tools.execute_tool(conn, 'create_promise', {'person': '张晨', 'who': 'me', 'content': '发送设计稿', 'due_date': '2026-10-09'})
    tools.execute_tool(conn, 'record_transfer', {'person': '张晨', 'direction': 'out', 'amount_yuan': 35, 'note': '餐费'})
    seen = []
    fake = type('LLM', (), {'chat': lambda self, messages, **kwargs: seen.append(messages[-1]['content']) or {'content': '完整日志'}})()
    monkeypatch.setattr(scheduler, 'get_conn', lambda: conn)
    monkeypatch.setattr(scheduler, 'get_llm', lambda: fake)
    assert scheduler.generate_journal(today_local()) == '完整日志'
    assert '发送设计稿' in seen[0] and '2026-10-09' in seen[0]
    assert '35.00' in seen[0] and '餐费' in seen[0]


def test_worker_retry_reuses_user_message_and_saves_partial_receipts(conn):
    cid = repo.create_conversation(conn)
    t = tasks.submit(conn, 'chat', 'test', {'conversation_id': cid, 'text': 'hello'}, dispatch=False)
    seen = []
    def fake_run(c, cid, text, emit, user_message_id):
        seen.append(user_message_id)
        emit('delta', {'text': 'answer'})
        return {'answer': 'answer', 'outcomes': [{'status': 'failed', 'detail': 'failed fact'}]}
    with patch.object(tasks, 'get_conn', return_value=conn), patch.object(harness, 'run_turn', side_effect=fake_run):
        tasks._run(t['id'])
        assert tasks.read(conn, t['id'])['status'] == 'partial'
        with patch.object(tasks, 'start'):
            tasks.retry(conn, t['id'])
        tasks._run(t['id'])
    assert seen[0] == seen[1]
    assert len(repo.list_messages(conn, cid)) == 1
    assert conn.execute("SELECT COUNT(*) FROM task_event WHERE task_id=? AND kind='result'", (t['id'],)).fetchone()[0] == 2


def test_worker_honors_cooperative_cancellation(conn):
    cid = repo.create_conversation(conn)
    t = tasks.submit(conn, 'chat', 'cancel', {'conversation_id': cid, 'text': 'hello'}, dispatch=False)
    def fake_run(c, cid, text, emit, user_message_id):
        c.execute('UPDATE task SET cancel_requested=1 WHERE id=?', (t['id'],))
        c.commit()
        emit('delta', {'text': 'must not be published'})
    with patch.object(tasks, 'get_conn', return_value=conn), patch.object(harness, 'run_turn', side_effect=fake_run):
        tasks._run(t['id'])
    assert tasks.read(conn, t['id'])['status'] == 'cancelled'
    assert not conn.execute("SELECT 1 FROM task_event WHERE kind='delta'").fetchone()


def test_recovery_interrupts_running_work_and_dispatches_queued_work(conn):
    first = tasks.submit(conn, 'journal', 'running', {}, dispatch=False)
    second = tasks.submit(conn, 'journal', 'queued', {}, dispatch=False)
    conn.execute("UPDATE task SET status='running' WHERE id=?", (first['id'],))
    conn.commit()
    with patch.object(tasks, 'start') as start, patch.object(tasks, 'get_conn', return_value=conn):
        tasks.recover()
    assert tasks.read(conn, first['id'])['status'] == 'interrupted'
    start.assert_called_once_with(second['id'])


def test_database_rejects_two_active_turns_for_one_conversation(conn):
    cid = repo.create_conversation(conn)
    tasks.submit(conn, 'chat', 'one', {'conversation_id': cid, 'text': 'one'}, dispatch=False)
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        tasks.submit(conn, 'chat', 'two', {'conversation_id': cid, 'text': 'two'}, dispatch=False)
    conn.rollback()


def test_research_keeps_user_supplied_public_url_without_search(conn):
    assert research.direct_sources('调研 https://marsdata.ai 的公开产品') == [
        {'title': 'marsdata.ai', 'url': 'https://marsdata.ai'}]
    with patch.object(research, 'websearch', return_value=[]), patch.object(research, 'fetch_text', return_value='官方资料'), \
         patch.object(research, 'get_llm') as llm:
        llm.return_value.chat_json.return_value = {'claims': [{'claim': '官方资料', 'source': 'https://marsdata.ai'}]}
        r = research._run_subtask('https://marsdata.ai', {'scope': '产品'})
    assert len(r['sources']) == 1
    assert r['claims'][0]['source'] == 'https://marsdata.ai'


def test_ambiguous_tools_never_write_to_first_match(conn):
    for name in ("王甲", "王乙"):
        pid = repo.create_person(conn, name, status="active")
        repo.add_alias(conn, pid, "老王", "chat")
    conn.commit()
    result = tools.execute_tool(conn, "save_person_fact", {"person": "老王", "field": "city", "value": "杭州"})
    assert result["needs_confirmation"]
    assert len(result["candidates"]) == 2
    assert conn.execute("SELECT COUNT(*) FROM person_field").fetchone()[0] == 0
    assert repo.get_person_by_name(conn, "老王") is None


def test_explicit_person_id_can_write_without_re_disambiguating_same_name(conn):
    a = repo.create_person(conn, '王明', status='active')
    b = repo.create_person(conn, '王明', status='active')
    result = tools.execute_tool(conn, 'save_person_fact', {'person_id': b, 'field': 'city', 'value': '杭州'})
    assert result['ok'] and not result['pending_ids']
    assert repo.current_fields(conn, b)['city']['value'] == '杭州'
    assert not repo.current_fields(conn, a)


def test_unverifiable_field_evidence_requires_confirmation_even_with_high_confidence(conn):
    from app.memory.extract import sanitize
    x = sanitize({'people': [person_patch(fields=[{'field': 'city', 'value': '杭州',
                                                  'confidence': .99, 'evidence': '虚构原文'}])]}, source='张晨说今天下雨')
    assert x['people'][0]['fields'][0]['confidence'] == .99
    r = apply.apply_extraction(conn, x, 'source', '张晨说今天下雨')
    assert r['pending_ids']
    assert not repo.current_fields(conn, r['person_ids'][0])


@pytest.mark.parametrize('name', ['妈妈', '爸爸', '姐姐', '岳母', '爷爷', '婆婆'])
def test_standalone_family_roles_do_not_create_independent_people(conn, name):
    from app.memory.extract import sanitize
    extraction = sanitize({'people': [person_patch(name=name)]}, source=name)
    assert not extraction['people']
    r = apply.apply_extraction(conn, {'people': [person_patch(name=name)]}, 'role-source', name)
    assert not r['person_ids'] and not repo.all_active_persons(conn)


def test_discard_is_preserved_when_same_source_task_retries(conn):
    extraction = {'people': [person_patch(fields=[{'field': 'city', 'value': '杭州', 'confidence': .7, 'evidence': '听说'}])]}
    r = apply.apply_extraction(conn, extraction, 'same-source', '听说张晨在杭州')
    resolve(conn, r['pending_ids'][0], 'discard')
    replay = apply.apply_extraction(conn, extraction, 'same-source', '听说张晨在杭州')
    assert not replay['pending_ids'] and not repo.list_pending(conn)
    assert any(o['status'] == 'skipped' for o in replay['outcomes'])
    # A fresh source can still present new information for confirmation.
    fresh = apply.apply_extraction(conn, extraction, 'new-source', '听说张晨在杭州')
    assert fresh['pending_ids']


def test_long_history_summary_advances_after_five_hundred_messages(conn, monkeypatch):
    cid = repo.create_conversation(conn)
    for i in range(600):
        repo.add_message(conn, cid, 'user', f'历史{i:04d}' + 'x' * 80)
    repo.set_kv(conn, f'summ_count_{cid}', '580')
    repo.set_kv(conn, f'summ_text_{cid}', '已总结早期历史')
    fake = type('LLM', (), {'chat': lambda self, messages, **kwargs: {'content': messages[-1]['content']}})()
    monkeypatch.setattr(harness, 'get_llm', lambda: fake)
    # Ten new old messages trigger incremental summarization.
    for i in range(600, 610):
        repo.add_message(conn, cid, 'user', f'历史{i:04d}' + 'x' * 80)
    result = harness._history_messages(conn, cid)
    assert '已总结早期历史' in result[0]['content']
    assert '历史0597' in result[0]['content']
    assert result[-1]['content'].startswith('历史0609')
    assert repo.get_kv(conn, f'summ_count_{cid}') == '598'


def test_transfer_tool_and_extraction_share_event_key(conn):
    tools.execute_tool(conn, "save_person_fact", {"person": "张晨", "field": "city", "value": "杭州"})
    tools.execute_tool(conn, "record_transfer", {"person": "张晨", "direction": "out", "amount_yuan": 100})
    le = {"direction": "out", "amount_cents": 10000, "note": "礼金", "occurred": None, "confidence": .95}
    apply.apply_extraction(conn, {"people": [person_patch(ledger=[le])]}, "same-message", "给张晨100元")
    assert repo.ledger_totals(conn)["out_cents"] == 10000
    assert repo.ledger_totals(conn)["count"] == 1


@pytest.mark.parametrize("action,amount,status", [("confirm", 10000, "confirmed"), ("discard", 0, "void")])
def test_pending_ledger_is_actionable_and_does_not_duplicate(conn, action, amount, status):
    le = {"direction": "out", "amount_cents": 10000, "note": "礼金", "occurred": None, "confidence": .7}
    a = apply.apply_extraction(conn, {"people": [person_patch(ledger=[le])]}, "ledger-message", "礼金")
    item = repo.list_pending(conn, "ledger")[0]
    assert a["pending_ids"] and repo.ledger_totals(conn)["count"] == 0
    resolve(conn, item["id"], action)
    assert conn.execute("SELECT COUNT(*) FROM ledger_entry").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM ledger_entry").fetchone()[0] == status
    assert repo.ledger_totals(conn)["out_cents"] == amount
    with pytest.raises(ValueError):
        resolve(conn, item["id"], action)


def test_fact_confidence_and_confirmation(conn):
    facts = [{"text": "喜欢网球", "confidence": .1}, {"text": "可能喜欢登山", "confidence": .7}]
    apply.apply_extraction(conn, {"people": [person_patch(facts=facts)]}, "facts", "来源")
    assert conn.execute("SELECT COUNT(*) FROM memory_item").fetchone()[0] == 0
    p = repo.list_pending(conn, "memory")[0]
    resolve(conn, p["id"], "confirm")
    assert conn.execute("SELECT text FROM memory_item").fetchone()[0] == "可能喜欢登山"


def test_confirmed_promise_has_reminder_and_fingerprint(conn):
    pr = {"who": "me", "content": "发送方案", "due": "2026-10-07", "confidence": .7}
    apply.apply_extraction(conn, {"people": [person_patch(promises=[pr])]}, "promise-msg", "方案")
    assert conn.execute("SELECT COUNT(*) FROM promise").fetchone()[0] == 0
    item = repo.list_pending(conn, "promise")[0]
    resolve(conn, item["id"], "confirm")
    assert conn.execute("SELECT COUNT(*) FROM reminder").fetchone()[0] == 1
    apply.apply_extraction(conn, {"people": [person_patch(promises=[pr])]}, "promise-msg", "方案")
    assert conn.execute("SELECT COUNT(*) FROM promise").fetchone()[0] == 1
    promise = conn.execute("SELECT id FROM promise").fetchone()[0]
    repo.set_promise_status(conn, promise, "done")
    assert conn.execute("SELECT status FROM reminder").fetchone()[0] == "cancelled"


def test_archived_interactions_hidden_from_search_and_journal(conn):
    p = repo.create_person(conn, "归档测试", status="active")
    repo.insert_interaction(conn, now_ms(), "chat", "归档测试专属秘密", None, [(p, "mentioned")])
    conn.commit()
    assert retrieve.hybrid_search(conn, "归档测试专属秘密")
    repo.archive_person(conn, p)
    conn.commit()
    assert retrieve.hybrid_search(conn, "归档测试专属秘密") == []
    assert repo.interactions_since(conn, 0) == []
    repo.restore_person(conn, p)
    assert retrieve.hybrid_search(conn, "归档测试专属秘密")


def test_history_window_keeps_newest_and_stable_order(conn):
    cid = repo.create_conversation(conn)
    for i in range(7):
        repo.add_message(conn, cid, "user", str(i))
    assert [r["content"] for r in repo.list_messages(conn, cid, 3)] == ["4", "5", "6"]


def test_backdated_field_does_not_replace_current_value(conn):
    pid = repo.create_person(conn, "张晨", status="active")
    repo.upsert_field(conn, pid, "company", "现公司", 1.0, None, valid_from=3000)
    repo.upsert_field(conn, pid, "company", "老公司", 1.0, None, valid_from=1000)
    repo.upsert_field(conn, pid, "company", "中间公司", 1.0, None, valid_from=2000)
    assert repo.current_fields(conn, pid)["company"]["value"] == "现公司"
    assert repo.field_at(conn, pid, "company", 1500)["value"] == "老公司"
    assert repo.field_at(conn, pid, "company", 2500)["value"] == "中间公司"


def test_chat_sends_current_message_once_and_reports_failure(conn, monkeypatch):
    class Fake:
        def __init__(self): self.calls = []
        def chat(self, messages, **kwargs):
            self.calls.append(list(messages))
            return {"role": "assistant", "content": "已回答"}
    fake = Fake()
    monkeypatch.setattr(harness, "get_llm", lambda: fake)
    monkeypatch.setattr(harness.turn_intent, 'classify_turn', lambda text, **kwargs:
        harness.turn_intent.validate_intent({'intent': 'record', 'record_spans': [text],
                                            'reason': 'explicit recording fixture'}, text))
    monkeypatch.setattr(harness.answer_review, 'review_answer', lambda llm, question, draft, sources, **kwargs:
        {'answer': draft, 'metadata': {'status': 'test_double', 'semantic_verified': False}, 'warnings': []})
    monkeypatch.setattr(harness.extract, "extract_from_text", lambda *a: (_ for _ in ()).throw(RuntimeError("抽取失败")))
    cid = repo.create_conversation(conn)
    result = harness.run_turn(conn, cid, "unique text")
    assert len(fake.calls) == 1  # Classifier/reviewer doubles do not consume answer scripts.
    assert sum(m["content"] == "unique text" for m in fake.calls[0]) == 1
    assert result["failed"][0]["detail"] == "抽取失败"
    final = repo.list_messages(conn, cid)[-1]
    assert json.loads(final["tool_calls"])["failed"]


def test_failed_ingest_can_retry(conn):
    jid, _ = repo.create_job(conn, "file", "sample.txt", "unique-file")
    repo.update_job(conn, jid, "failed", error="failed once")
    conn.commit()
    second, created = repo.create_job(conn, "file", "sample.txt", "unique-file")
    assert second == jid and created
    assert repo.get_job(conn, jid)["status"] == "queued"


def test_research_cannot_accept_unfetched_sources(conn, monkeypatch):
    fake = type("Fake", (), {"chat_json": lambda self,*a,**k: {"claims": [
        {"claim": "伪造", "source": "https://fabricated.invalid", "evidence": "确切原文"},
        {"claim": "正确", "source": "https://example.com", "evidence": "确切原文"},
        {"claim": "错引", "source": "https://example.com", "evidence": "源文没有这句"}], "conflicts": []}})()
    monkeypatch.setattr(research, "get_llm", lambda: fake)
    result = research._review("topic", [{"sources": [{"url": "https://example.com", "excerpt": "有确切原文内容"}],
        "claims": [{"claim": "正确", "source": "https://example.com"}]}])
    assert [r["claim"] for r in result["claims"]] == ["正确"]


def test_review_indexes_actual_original_text_and_rejects_invalid_passage(monkeypatch):
    original = 'Mars remembers everything about the people who matter.'
    fake = type('LLM', (), {'chat_json': lambda self, *a, **k: {'claims': [
        {'claim': '网站宣称记住重要人物', 'source': 'https://marsdata.ai', 'evidence_index': 0},
        {'claim': '越界片段', 'source': 'https://marsdata.ai', 'evidence_index': 999},
        {'claim': '未抓取网站', 'source': 'https://unfetched.invalid', 'evidence_index': 0}]}})()
    monkeypatch.setattr(research, 'get_llm', lambda: fake)
    reviewed = research._review('产品', [{'claims': [{'source': 'https://marsdata.ai', 'claim': '网站宣称'}],
                                         'sources': [{'url': 'https://marsdata.ai', 'excerpt': original}]}])
    assert len(reviewed['claims']) == 1
    assert reviewed['claims'][0]['evidence'] == original


@pytest.fixture()
def client(conn, monkeypatch):
    from app.api.main import app
    monkeypatch.setattr(tasks, "start", lambda tid: None)
    return TestClient(app)


def test_home_and_model_status_do_not_expose_key(client):
    assert client.get('/api/home').status_code == 200
    result = client.get('/api/status').json()
    assert 'llm_api_key' not in json.dumps(result)
    assert result['timezone'] == 'Asia/Shanghai'


def test_async_chat_submission_is_idempotent_and_busy_protected(client):
    cid = client.post('/api/conversations', json={}).json()['id']
    first = client.post(f'/api/conversations/{cid}/turns', json={'text':'hello','request_id':'same'})
    assert first.status_code == 202
    second = client.post(f'/api/conversations/{cid}/turns', json={'text':'hello','request_id':'same'})
    assert first.json()['id'] == second.json()['id']
    assert client.post(f'/api/conversations/{cid}/turns', json={'text':'other'}).status_code == 409
    assert client.post('/api/conversations/missing/turns',json={'text':'hello'}).status_code == 404


def test_multipart_chat_identity_is_received(client, monkeypatch):
    from app.ingest import chatlog
    seen = {}
    def fake(conn,filename,data,me=None):
        seen['me'] = me
        return {'ok':True}
    monkeypatch.setattr(chatlog, 'import_chatlog', fake)
    r = client.post('/api/ingest/chatlog', files={'file':('chat.txt',b'test')},data={'me':'我的名字'})
    assert r.status_code == 200 and seen['me'] == '我的名字'
    r = client.post('/api/tasks/file', files={'file':('chat.txt',b'test')},data={'mode':'chatlog','me':'我的名字'})
    assert r.status_code == 202 and r.json()['payload']['me'] == '我的名字'


def test_task_cancel_retry_and_validation(client):
    r = client.post('/api/tasks/url',json={'url':'https://example.com'}).json()
    tid = r['id']
    assert client.post(f'/api/tasks/{tid}/cancel').json()['status'] == 'cancelled'
    assert client.post(f'/api/tasks/{tid}/retry').json()['status'] == 'queued'
    assert client.post('/api/tasks/url',json={'url':'file:///secret'}).status_code == 400
    assert client.post('/api/pending/none/resolve',json={'action':'anything'}).status_code == 422


def test_sse_replays_after_cursor_and_closes_when_task_finishes(client, conn):
    task = tasks.submit(conn, 'journal', 'SSE test', {}, dispatch=False)
    cursor = conn.execute('SELECT MAX(seq) FROM task_event WHERE task_id=?', (task['id'],)).fetchone()[0]
    tasks.event(conn, task['id'], 'progress', {'message': 'working'})
    tasks.event(conn, task['id'], 'result', {'text': 'done'})
    conn.execute("UPDATE task SET status='done' WHERE id=?", (task['id'],))
    conn.commit()
    response = client.get(f"/api/tasks/{task['id']}/events?after={cursor}")
    assert response.status_code == 200
    assert 'event: queued' not in response.text
    assert 'event: progress' in response.text and 'event: result' in response.text
    assert 'event: end' in response.text and '"status": "done"' in response.text


def test_proposal_requires_confirmation_before_research(client, monkeypatch):
    r = client.post('/api/research',json={'topic':'测试主题'}).json()
    rid = r['payload']['research_id']
    assert r['kind'] == 'research_prepare'
    assert client.post(f'/api/research/{rid}/start').status_code == 409


def test_skill_and_project_crud(client):
    assert client.post('/api/skills',json={'name':'跟进','body':'简短问候'}).status_code == 200
    sid = client.get('/api/skills').json()['skills'][0]['id']
    assert client.delete(f'/api/skills/{sid}').status_code == 200
    p = client.post('/api/projects',json={'name':'合作','description':'测试'}).json()['id']
    c = client.post('/api/conversations',json={'project_id':p})
    assert c.status_code == 200
    assert client.get('/api/projects').json()['projects'][0]['conversations']
    assert client.delete(f'/api/projects/{p}').status_code == 200
    assert client.get('/api/projects').json()['projects'] == []


def test_export_is_snapshot_without_credentials(client):
    import io, zipfile
    response = client.get('/api/export')
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert 'home.db' in archive.namelist()
        assert '.env' not in archive.namelist()


def test_evidence_download_is_original_bytes_and_rejects_invalid_blob_path(client, conn):
    from app.store import blob
    original = b'original fixture attachment'
    h = blob.put(original)
    eid = repo.add_evidence(conn, 'blob', blob_hash=h, source_ref='test.txt')
    bad = repo.add_evidence(conn, 'blob', blob_hash='../../.env', source_ref='bad.txt')
    conn.commit()
    response = client.get(f'/api/evidence/{eid}/file')
    assert response.status_code == 200 and response.content == original
    assert 'attachment' in response.headers['content-disposition']
    assert client.get(f'/api/evidence/{bad}/file').status_code == 404
    assert client.get('/api/evidence/none/file').status_code == 404
