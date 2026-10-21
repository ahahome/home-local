"""Offline controlled reproduction, not a recovered production trace.
Uses report-preserved extraction patch and actual harness/write/registry code.
No repository source/fixture/runner mutation and no network model requests.
"""
import copy
import difflib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path('/Users/emdoor/Documents/projects/integrate/home-local')
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'eval/results/answer-evidence-v013/final-followups/answers-20261020-104913-948740-87843-d8d0fc28.json'
report = json.loads(REPORT.read_text())
failed = report['details'][2]['turns'][1]
original_patch = json.loads(failed['data_changes']['pending_item']['added'][0]['payload'])['patch']
question = failed['question']
record_text = '帮我记下：乐团周汐现在住在长沙。'

os.environ['HOME_LLM_API_KEY'] = ''
from app.agent import harness, source_registry
from app.agent.memory_sources import source_state
from app.config import get_settings
from app.store import db, repo

class ScriptedModel:
    def __init__(self, mode, person_id):
        self.mode = mode
        self.person_id = person_id
        self.tool_sent = False
        self.calls = []
    def chat(self, messages, **kwargs):
        if '写入意图分类器' in messages[0]['content']:
            text = json.loads(messages[-1]['content'])['message']
            self.calls.append({'kind':'classifier','text':text})
            return {'content': json.dumps({'intent':'mixed' if text == question else 'query',
                'record_spans':[record_text] if text == question else [], 'reason':'controlled offline response'}, ensure_ascii=False)}
        self.calls.append({'kind':'draft'})
        if not self.tool_sent and self.mode != 'no_write_tool':
            self.tool_sent = True
            if self.mode == 'save_person_fact':
                name, args = 'save_person_fact', {'person_id':self.person_id, 'person':'乐团周汐', 'field':'city', 'value':'长沙'}
            else:
                name, args = 'note', {'text':record_text}
            self.calls[-1].update(tool=name,args=args)
            return {'role':'assistant','content':'','tool_calls':[{'id':'offline-write','type':'function',
                'function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]}
        return {'content':'资料原文：乐团周汐说演出邀约先邮件发场地与曲目范围，排练期间不接电话。 [[S1]]'}

results=[]
for mode in ('no_write_tool','save_person_fact','note'):
    with tempfile.TemporaryDirectory(prefix='home-v013-source-diagnostic-') as temp:
        path=Path(temp)
        os.environ['HOME_DB_PATH']=str(path/'isolated.db')
        os.environ['HOME_BLOB_DIR']=str(path/'blobs')
        get_settings.cache_clear()
        db.reset_for_tests(path/'isolated.db')
        conn=db.get_conn()
        pottery=repo.create_person(conn,'周汐',status='active')
        music=repo.create_person(conn,'周汐',status='active')
        repo.add_alias(conn,pottery,'陶艺周汐','offline-diagnostic')
        repo.add_alias(conn,music,'乐团周汐','offline-diagnostic')
        for pid,text in ((pottery,'陶艺周汐说接定制订单时先发器型参考图和尺寸，材料确认后才讨论工期。'),
                         (music,'乐团周汐说演出邀约先邮件发场地与曲目范围，排练期间不接电话。')):
            eid=repo.add_evidence(conn,'text',source_ref='offline-diagnostic:'+pid,excerpt=text)
            repo.add_memory(conn,'semantic',text,person_id=pid,confidence=.95,evidence_id=eid,ts=1768179600000)
        conn.commit()
        model=ScriptedModel(mode,music)
        captured=[]
        extracts=[]
        original_save=source_registry.SourceRegistry.save_answer
        def observe_save(self,save):
            states=[]
            for dependency in self.dependencies:
                state=source_state(self.conn,dependency)
                item={'type':dependency['type'],'entity_id':dependency['entity_id'],'status':state['status']}
                if state['status']!='current':
                    item['old']=dependency
                    item['current']=state['current']
                    item['diff']=list(difflib.unified_diff(dependency.get('text','').splitlines(),
                        (state.get('current') or {}).get('text','').splitlines(),fromfile='captured',tofile='current',lineterm=''))
                states.append(item)
            captured.append(states)
            return original_save(self,save)
        def extraction(text,*args):
            extracts.append(text)
            return {'people':[copy.deepcopy(original_patch)],'my_notes':[]}
        def review(*args,**kwargs):
            return {'answer':args[2], 'metadata':{'version':1,'status':'not_needed','reason':'offline-controlled-diagnostic',
                    'semantic_verified':False},'warnings':[]}
        with patch.object(harness,'get_llm',lambda:model),patch.object(harness.extract,'extract_from_text',extraction), \
             patch.object(harness.answer_review,'review_answer',review), \
             patch.object(source_registry.SourceRegistry,'save_answer',observe_save):
            cid=repo.create_conversation(conn)
            first=harness.run_turn(conn,cid,'周汐的邀约应该先怎么沟通？')
            assert first['conversation_scope']['mode']=='clarify'
            try:
                answer=harness.run_turn(conn,cid,question)
                error=None
            except Exception as exc:
                answer=None
                error={'type':type(exc).__name__,'message':str(exc)}
            rows=[dict(row) for row in conn.execute('SELECT * FROM message WHERE conversation_id=? ORDER BY rowid',(cid,))]
            domain={table:[dict(row) for row in conn.execute('SELECT * FROM '+table)] for table in ('person_field','pending_item','applied_patch')}
            results.append({'mode':mode,'error':error,'extract_calls':extracts,'model_calls':model.calls,
                'final_dependencies':captured[-1],'field_rows':domain['person_field'],
                'pending_rows':domain['pending_item'],
                'tool_rows':[dict(row) for row in rows if row['role']=='tool'],
                'answer_saved':answer is not None})
        db.reset_for_tests(path/'isolated.db')
get_settings.cache_clear()
out=Path('/tmp/home_v013_source_diagnostic.json')
out.write_text(json.dumps({'provenance':'controlled_offline_reproduction_not_recovered_real_trace',
    'report_path':str(REPORT),'results':results},ensure_ascii=False,indent=2))
for result in results:
    print(json.dumps({'mode':result['mode'],'error':result['error'],'extract_calls':len(result['extract_calls']),
        'city':[row['value'] for row in result['field_rows'] if row['field']=='city'],
        'stale_dependencies':[item for item in result['final_dependencies'] if item['status']!='current']},ensure_ascii=False,indent=2))
print('Report: '+str(out))
