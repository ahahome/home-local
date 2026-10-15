"""Real HTTP/tasks/media QA for a generated mock_scenarios database."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time

import httpx
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'data/qa/latest-mock.json').read_text())['run_dir']).resolve()
assert run.is_relative_to((ROOT/'data/qa').resolve())
os.environ['HOME_DB_PATH']=str(run/'home.db')
os.environ['HOME_BLOB_DIR']=str(run/'blobs')
BASE='http://127.0.0.1:8767'
HTTP=httpx.Client(base_url=BASE,timeout=30)
from app.store.db import get_conn
from app.store import repo
c=get_conn()
report={'synthetic':True,'tasks':[],'checks':[]}


def store():
    (run/'e2e.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def request(method,path,**kwargs):
    result=HTTP.request(method,'/api'+path,**kwargs)
    result.raise_for_status()
    return result.json()


def await_task(t):
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        t=request('GET','/tasks/'+t['id'])
        if t['status'] in ('done','partial','failed','cancelled','interrupted'):
            return t
        time.sleep(1)
    raise TimeoutError('QA task deadline exceeded')


def make_turn(item):
    cid=request('POST','/conversations',json={'title':'模拟验收：'+item['id']})['id']
    t=request('POST',f'/conversations/{cid}/turns',json={'text':item['text'],'request_id':'mock-'+item['id']})
    same=request('POST',f'/conversations/{cid}/turns',json={'text':item['text'],'request_id':'mock-'+item['id']})
    assert same['id']==t['id']
    final=await_task(t)
    (run/('turn-'+item['id']+'.json')).write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    msgs=request('GET',f'/conversations/{cid}/messages')['messages']
    return {'kind':'chat','id':item['id'],'task_id':t['id'],'status':final['status'],
            'error':final['error'],'message_roles':[m['role'] for m in msgs],
            'answer':(final.get('result') or {}).get('answer','')}


def make_image(filename,lines):
    img=Image.new('RGB',(1000,700),'#f4f7f1')
    draw=ImageDraw.Draw(img)
    font_paths=[os.environ.get('HOME_QA_FONT',''),
                'C:/Windows/Fonts/msyh.ttc',
                '/System/Library/Fonts/Hiragino Sans GB.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    font_path=next((p for p in font_paths if p and Path(p).is_file()),None)
    if not font_path:
        raise RuntimeError('Set HOME_QA_FONT to a Chinese font file for media QA')
    font=ImageFont.truetype(font_path,30)
    draw.rounded_rectangle((35,35,965,665),radius=20,fill='white',outline='#ccd8c2',width=2)
    for i,line in enumerate(lines):
        draw.text((65,65+i*78),line,font=font,fill='#223627')
    path=run/filename
    img.save(path)
    return path


def upload(path,mode='file'):
    t=request('POST','/tasks/file',files={'file':(path.name,path.read_bytes())},data={'mode':mode,'me':'我'})
    final=await_task(t)
    (run/('media-'+path.stem+'.json')).write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'kind':mode,'filename':path.name,'task_id':t['id'],'status':final['status'],
            'error':final['error'],'result':final['result']}


def probe(path):
    start=time.perf_counter()
    try:
        request('GET',path)
        return {'path':path,'milliseconds':round((time.perf_counter()-start)*1000,2),'ok':True}
    except Exception as e:
        return {'path':path,'ok':False,'error':type(e).__name__}


expected_ids={r[0] for r in c.execute("SELECT id FROM person WHERE deleted_at IS NULL AND status!='merged'")}
actual_ids={p['id'] for p in request('GET','/persons')['persons']}
if actual_ids != expected_ids:
    raise SystemExit('Port 8767 is not serving the latest mock fixture. Restart it with that fixture before QA.')
if c.execute('SELECT COUNT(*) FROM task').fetchone()[0]:
    raise SystemExit('Integration QA requires a fresh mock fixture; replay saved outputs first, then restart port 8767.')
paths=['/home','/persons','/pending','/ledger','/promises','/projects','/atlas']*20
with ThreadPoolExecutor(max_workers=8) as pool:
    probes=list(pool.map(probe,paths))
latencies=sorted(x['milliseconds'] for x in probes if x['ok'])
report['http']={'requests':len(probes),'concurrency':8,'failures':[x for x in probes if not x['ok']],
                'median_ms':round(statistics.median(latencies),2),
                'p95_ms':latencies[int(len(latencies)*.95)-1],'max_ms':max(latencies)}
store()
print('HTTP read checks: '+json.dumps(report['http']),flush=True)
turns=[{'id':'query','text':'用表格查询林舟、陈岚、许晴各自的公司、职位和城市。只查询已有记录，不新增事实。'},
       {'id':'draft','text':'帮我给沈悦起草一条自然、简短的合作跟进消息，结合已有资料。只起草，不发送。'},
       {'id':'money','text':'2026-10-01，我给江遥转了47.23元餐费。我答应2026-10-21前给江遥发送验收文件。请记录并告诉我结果。'}]
with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(make_turn,turns):
        report['tasks'].append(result)
        store()
        print('Chat task '+result['id']+': '+result['status'],flush=True)
ledger=c.execute("SELECT * FROM ledger_entry WHERE amount_cents=4723 AND note LIKE '%餐费%'").fetchall()
report['checks'].append({'check':'chat_transfer_exactly_once','passed':len(ledger)==1,'rows':len(ledger)})
promises=c.execute("SELECT * FROM promise WHERE content LIKE '%验收文件%'").fetchall()
report['checks'].append({'check':'chat_promise_exactly_once','passed':len(promises)==1,'rows':len(promises)})
images=[make_image('business-card.png',['夏初','公司：长风材料研究院','职位：材料工程师','城市：宁波','生日：1992-04-09']),
        make_image('receipt.png',['转账记录','2026-10-01 10:30','我收到潘念转账：68.42元','备注：归还垫付车费'])]
for path in images:
    result=upload(path)
    report['tasks'].append(result)
    store()
    print('Image task '+path.name+': '+result['status'],flush=True)
chatlog=run/'group-chat.txt'
chatlog.write_text('''[2026-10-01 09:00:00] 我: 大家好，今天确认一下近况。
[2026-10-01 09:01:00] 潘念: 我在长风物流做运营经理，目前住宁波。
[2026-10-01 09:02:00] 顾澜: 我在清溪设计做设计师，目前住厦门。
[2026-10-01 09:03:00] 我: 潘念，我答应2026-10-22前给你发运输清单。
[2026-10-01 09:04:00] 潘念: 好的，已经收到。
[2026-10-01 09:05:00] 顾澜: 我答应2026-10-23前把修改好的封面发给你。
[2026-10-01 09:06:00] 我: 顾澜，这次封面只是在报价，还没付款。
[2026-10-01 09:07:00] 潘念: 改天一起吃饭。
[2026-10-01 09:08:00] 我: 有空再聚。
''',encoding='utf-8')
first=upload(chatlog,'chatlog')
report['tasks'].append(first)
before=c.execute('SELECT COUNT(*) FROM ledger_entry').fetchone()[0]
second=upload(chatlog,'chatlog')
after=c.execute('SELECT COUNT(*) FROM ledger_entry').fetchone()[0]
report['checks'].append({'check':'chatlog_file_replay','passed':before==after and bool(second['result'].get('duplicate'))})
for name,expected in {'夏初':{'company':'长风材料研究院','title':'材料工程师','city':'宁波'},
                      '潘念':{'company':'长风物流','title':'运营经理','city':'宁波'},
                      '顾澜':{'company':'清溪设计','title':'设计师','city':'厦门'}}.items():
    person=repo.get_person_by_name(c,name)
    fields=repo.current_fields(c,person['id']) if person else {}
    actual={k:v['value'] for k,v in fields.items()}
    report['checks'].append({'check':'media_fields:'+name,'passed':all(actual.get(k)==v for k,v in expected.items()),
                            'expected':expected,'actual':actual})
receipt=c.execute("SELECT * FROM ledger_entry WHERE amount_cents=6842 AND direction='in'").fetchall()
report['checks'].append({'check':'receipt_amount_and_direction','passed':len(receipt)==1,'rows':len(receipt)})
report['checks'].append({'check':'final_user_assistant_messages',
    'passed':all(t['message_roles']==['user','assistant'] for t in report['tasks'] if t['kind']=='chat')})
report['final_counts']={t:c.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in
                        ('person','person_field','interaction','memory_item','promise','ledger_entry','pending_item','task')}
report['failed_checks']=[x['check'] for x in report['checks'] if not x['passed']]
store()
print('Integration failed checks: '+json.dumps(report['failed_checks']),flush=True)
print('Final counts: '+json.dumps(report['final_counts']),flush=True)
HTTP.close()
