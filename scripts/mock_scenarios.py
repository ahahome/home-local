"""Isolated synthetic QA: volume fixtures plus real extraction and replay checks.

Run: python -m scripts.mock_scenarios --live
All generated data stays under data/qa; the default application DB is never opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def cases():
    names = ['林舟', '陈岚', '许晴', '周宁', '赵岚', '沈悦', '陆星宁', '苏沐', '顾言',
             '江遥', '唐梨', '宋青', '叶澄', '方序', '谢川', '温禾', '程朔', '秦月']
    cities = ['杭州', '苏州', '南京', '上海', '深圳', '成都']
    jobs = ['产品经理', '设计师', '研究员', '采购经理', '工程师', '客户经理']
    out = []
    for i, name in enumerate(names):
        company = f'{["青禾", "星桥", "蓝湾", "云帆", "明川", "南岸"][i % 6]}第{i+1}工作室'
        city, title = cities[i % 6], jobs[i % 6]
        text = f'{name}目前在{company}工作，职位是{title}，常住{city}。'
        expected = {name: {'company': company, 'title': title, 'city': city}}
        promises, money = [], []
        if i % 3 == 0:
            due = f'2026-10-{7+i%18:02d}'
            text += f'我答应{due}前给{name}发送合作方案。'
            promises.append({'person': name, 'who': 'me', 'due': due})
        if i % 2 == 0:
            cents = 1234 + i * 137
            direction = 'in' if i % 4 == 0 else 'out'
            text += (f'2026-10-01，{name}给我转了{cents/100:.2f}元餐费。' if direction == 'in' else
                     f'2026-10-01，我给{name}转了{cents/100:.2f}元餐费。')
            money.append({'person': name, 'direction': direction, 'amount_cents': cents})
        out.append({'id': f'profile-{i+1:02d}', 'text': text, 'fields': expected,
                    'promises': promises, 'ledger': money})
    for i, pair in enumerate([('梁溪', '罗汐'), ('孟棠', '韩序'), ('蒋白', '魏澜'), ('徐珂', '郑屿')]):
        a, b = pair
        out.append({'id': f'multiple-{i+1}',
                    'text': f'{a}在清远实验室做研究员，住在武汉。{b}在云岭科技做工程师，住在西安。'
                            f'{a}和{b}是大学同学。',
                    'fields': {a: {'company': '清远实验室', 'title': '研究员', 'city': '武汉'},
                               b: {'company': '云岭科技', 'title': '工程师', 'city': '西安'}},
                    'promises': [], 'ledger': []})
    edges = [
        ('salary', '周宁年薪60万元，奖金5万元。这是收入情况，我们之间没有收付款。'),
        ('quote', '许晴说实验室设备报价18万元，目前只询价，还没有付款。'),
        ('negated', '我没有给陈岚转账，也没有答应发合同，暂时只是讨论。'),
        ('hypothetical', '如果叶澄下个月加入新公司，可能搬到广州，目前尚未确定。'),
        ('pleasantry', '我和林舟说：改天一起吃饭，有空再聚，保持联系。'),
        ('role', '妈妈说前台今天给那个客户打电话了，具体姓名不清楚。'),
        ('family', '沈悦的老公是主任医师，女儿在加拿大读书。沈悦本人仍然住在成都。'),
        ('ambiguous-1', '张晨住在北京，今天给我转了88元餐费。'),
        ('ambiguous-2', '小张答应2026-10-16前把报价单发给我。'),
        ('alias', '小赵就是赵岚，她的生日是1994-06-18。'),
        ('query-only', '请查询林舟的公司、陈岚的城市以及我最近的收支汇总，不要新增事实。'),
        ('draft-only', '帮我给沈悦起草一条跟进消息，语气轻松。只起草，不发送。'),
        ('correction', '更正：陆星宁目前常住宁波，之前说杭州是误记。'),
        ('history', '补记历史：2024-02-01时林舟常住合肥，这是过去的情况，不是现在。'),
    ]
    for key, text in edges:
        entry = {'id': key, 'text': text, 'fields': {}, 'promises': [], 'ledger': [], 'edge': True}
        if key in {'query-only', 'draft-only', 'role'}:
            entry['expect_no_new_data'] = True
        if key == 'family':
            entry['fields'] = {'沈悦': {'city': '成都'}}
        if key == 'alias':
            entry['fields'] = {'赵岚': {'birthday': '1994-06-18'}}
        if key == 'correction':
            entry['fields'] = {'陆星宁': {'city': '宁波'}}
        if key == 'history':
            entry['fields'] = {'林舟': {'city': '合肥'}}
        if key.startswith('ambiguous'):
            entry['ambiguous'] = True
        out.append(entry)
    return out


def seed(c):
    from app.store import repo
    from app.memory.apply import _date_ms
    from app.util import now_ms, sha256_text
    names = ['林舟', '陈岚', '许晴', '周宁', '赵岚', '沈悦', '安予', '白榆', '曹知', '戴宁',
             '杜青', '傅遥', '高禾', '贺澄', '胡悦', '季川', '金遥', '孔序', '赖星', '李棠',
             '刘屿', '卢白', '马晴', '苗沐', '彭语', '齐澜', '任溪', '邵言', '施禾', '孙朔',
             '田宁', '汪序', '吴遥', '夏知', '肖澄', '熊月', '杨青', '余川', '袁星', '曾悦']
    cities = ['杭州', '苏州', '南京', '上海', '深圳', '成都', '武汉', '西安']
    rng = random.Random(20261001)
    people = []
    for i, name in enumerate(names):
        pid = repo.create_person(c, name, status='active')
        people.append(pid)
        eid = repo.add_evidence(c, 'text', source_ref=f'mock-volume:{i}', excerpt=f'虚构数据：{name}的模拟档案。')
        for key, value in {'company': f'模拟机构{i+1:02d}', 'title': ['设计师','工程师','采购经理','研究员'][i%4],
                           'city': cities[i % 8]}.items():
            repo.upsert_field(c, pid, key, value, 1, eid, _date_ms('2026-08-01'))
        if name == '赵岚':
            repo.add_alias(c, pid, '小赵', 'manual', 1)
        if i >= 35:
            repo.archive_person(c, pid)
    for i in range(3):
        pid = repo.create_person(c, '张晨' if i < 2 else '张辰', status='active')
        people.append(pid)
        repo.add_alias(c, pid, '小张', 'manual', 1)
        repo.upsert_field(c, pid, 'company', ['青禾商贸','星桥软件','蓝湾实验室'][i], 1, None)
    for i in range(300):
        pid = people[i % len(people)]
        name = repo.get_person(c, pid)['display_name']
        ts = now_ms() - (i % 60) * 86400000
        text = f'【虚构交流 {i+1:03d}】与{name}讨论第{i%9+1}批合作方案，记录需求和下一次跟进。'
        iid, _ = repo.insert_interaction(c, ts, 'import', text, f'mock-interaction:{i}', [(pid,'mentioned')])
        if i % 3 == 0:
            repo.add_memory(c, 'episodic', text, pid, 1, ts=ts)
    for i in range(120):
        pid = people[i % len(people)]
        eid = repo.add_evidence(c, 'text', excerpt=f'虚构账目 #{i+1}，用于汇总验收')
        repo.add_ledger(c, 'in' if i%3==0 else 'out', rng.randint(100, 80000), pid,
                        f'模拟往来 #{i+1}', now_ms()-(i%30)*86400000,
                        'pending' if i%10==0 else 'void' if i%17==0 else 'confirmed',
                        sha256_text(f'mock-ledger:{i}'), eid)
    for i in range(60):
        pid = people[i % len(people)]
        due = _date_ms(f'2026-10-{1+i%25:02d}')
        promise = repo.add_promise(c, 'me' if i%2==0 else 'them', f'模拟承诺 #{i+1}：提交第{i%9+1}批方案', pid, due)
        if i%5==0:
            repo.set_promise_status(c, promise, 'done')
        elif i%7==0:
            repo.set_promise_status(c, promise, 'cancelled')
        else:
            repo.add_reminder(c, due, f'模拟跟进 #{i+1}', pid, promise)
    for i in range(12):
        pid=people[i]
        eid=repo.add_evidence(c, 'text', excerpt='虚构待确认：可能换城市，尚无明确消息')
        repo.add_pending(c, 'field', {'person_id':pid,'field':'city','value':cities[(i+2)%8],
                         'confidence':.7,'evidence_id':eid,'evidence':'可能换城市'}, '模拟待确认，尚未核实')
    for i in range(18):
        repo.upsert_edge(c, people[i], people[(i+3)%35], 'colleague', .7)
    for i, pid in enumerate(people[:35]):
        repo.recompute_importance(c, pid)
    from app.util import new_id
    for index, name in enumerate(['模拟秋季合作','模拟客户维护','模拟研究计划']):
        project = new_id()
        c.execute('INSERT INTO project VALUES(?,?,?,?,?)', (project,name,'全部内容均为虚构，供体验功能',now_ms(),now_ms()))
        for pid in people[index*5:index*5+5]:
            c.execute('INSERT INTO project_person VALUES(?,?)', (project,pid))
    repo.save_skill(c,'模拟温和跟进','先确认对方进度，再简短说明下一步；只起草，不发送。')
    complete_seed_pending(c)
    c.commit()


def complete_seed_pending(c):
    from app.store import repo
    for row in c.execute("SELECT * FROM ledger_entry WHERE status='pending'").fetchall():
        if not c.execute("SELECT 1 FROM pending_item WHERE kind='ledger' AND json_extract(payload,'$.ledger_id')=?",
                         (row['id'],)).fetchone():
            repo.add_pending(c, 'ledger', {'ledger_id':row['id'],'person_id':row['person_id'],
                'direction':row['direction'],'amount_cents':row['amount_cents'],'note':row['note'],
                'dedupe_hash':row['dedupe_hash'],'evidence_id':row['evidence_id']}, '模拟待确认账目，确认前不计入汇总')


def counts(c):
    tables=('person','person_field','interaction','memory_item','promise','ledger_entry','pending_item','evidence')
    return {t:c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in tables}


def evaluate(case, extraction, applied):
    from app.util import normalize_value
    actual={(p['canonical_name'],f['field']):f['value'] for p in extraction['people'] for f in p['fields']}
    missed=[]
    expected_count=0
    for name, fields in case['fields'].items():
        for key,value in fields.items():
            expected_count+=1
            if normalize_value(actual.get((name,key),''))!=normalize_value(value):
                missed.append({'person':name,'field':key,'expected':value,'actual':actual.get((name,key))})
    actual_money=[{'person':p['canonical_name'],'direction':x['direction'],'amount_cents':x['amount_cents']}
                  for p in extraction['people'] for x in p['ledger']]
    actual_promises=[{'person':p['canonical_name'],'who':x['who'],'due':x['due']}
                     for p in extraction['people'] for x in p['promises']]
    forbidden_names=[p['canonical_name'] for p in extraction['people'] if p['canonical_name'] in
                     {'妈妈','前台','那个客户','沈悦的老公','女儿'}]
    unexpected_money=[] if case.get('ambiguous') else [x for x in actual_money if x not in case['ledger']]
    unexpected_promises=[] if case.get('ambiguous') else [x for x in actual_promises if x not in case['promises']]
    return {'expected_fields':expected_count,'matched_fields':expected_count-len(missed),'missed':missed,
            'missing_ledger':[x for x in case['ledger'] if x not in actual_money],
            'unexpected_ledger':unexpected_money,'missing_promises':[x for x in case['promises'] if x not in actual_promises],
            'unexpected_promises':unexpected_promises,'forbidden_names':forbidden_names,
            'ambiguous_blocked':any(o['kind']=='person' and o['status']=='pending' for o in applied['outcomes']) if case.get('ambiguous') else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Use the configured real gateway for 36 synthetic cases')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    run_dir=(args.run_dir or ROOT/'data'/'qa'/('mock-'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    if not run_dir.is_relative_to((ROOT/'data'/'qa').resolve()):
        raise SystemExit('Mock output must stay under this project data/qa directory')
    if (run_dir/'home.db').exists():
        raise SystemExit('Use a fresh directory; existing databases are never replaced')
    run_dir.mkdir(parents=True,exist_ok=True)
    os.environ['HOME_DB_PATH']=str(run_dir/'home.db')
    os.environ['HOME_BLOB_DIR']=str(run_dir/'blobs')
    from app.store.db import get_conn
    from app.store import repo
    from app.memory import extract, apply
    c=get_conn()
    seed(c)
    dataset=cases()
    save(run_dir/'cases.json',dataset)
    report={'run_dir':str(run_dir),'synthetic':True,'seed_counts':counts(c),'cases':[]}
    save(ROOT/'data/qa/latest-mock.json',{'run_dir':str(run_dir)})
    print('Isolated volume fixture ready: '+str(run_dir),flush=True)
    print(json.dumps(report['seed_counts']),flush=True)
    if args.live:
        known=[p['display_name'] for p in repo.all_active_persons(c)]
        def infer(case):
            start=time.perf_counter()
            result=extract.extract_from_text(case['text'],known,'2026-10-01')
            return result,round(time.perf_counter()-start,2)
        inferred={}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(infer,case):case for case in dataset}
            for future in as_completed(futures):
                case=futures[future]
                try:
                    inferred[case['id']]=future.result()
                    save(run_dir/('extract-'+case['id']+'.json'),inferred[case['id']][0])
                    print('Extracted '+case['id'],flush=True)
                except Exception as error:
                    inferred[case['id']]={'error':type(error).__name__}
                    print('Extraction failed '+case['id']+': '+type(error).__name__,flush=True)
        # Apply in source order. Network completions must not reorder current/historical facts.
        for case in dataset:
            result=inferred[case['id']]
            if isinstance(result,dict):
                report['cases'].append({'id':case['id'],**result})
                continue
            extraction,elapsed=result
            before=counts(c)
            applied=apply.apply_extraction(c,extraction,'mock-live:'+case['id'],case['text'])
            first=counts(c)
            apply.apply_extraction(c,extraction,'mock-live:'+case['id'],case['text'])
            replay=counts(c)
            metrics=evaluate(case,extraction,applied)
            entry={'id':case['id'],'seconds':elapsed,**metrics,'outcomes':applied['outcomes'],
                   'replay_added_rows':{t:replay[t]-first[t] for t in first if replay[t]!=first[t]},
                   'unexpected_writes':{t:first[t]-before[t] for t in ('person','person_field','memory_item','promise','ledger_entry')
                                        if case.get('expect_no_new_data') and first[t]!=before[t]}}
            report['cases'].append(entry)
            repo.insert_interaction(c,now_ms_safe(),'chat',case['text'],'mock-live:'+case['id'],
                                    [(pid,'mentioned') for pid in applied['person_ids']])
            c.commit()
            save(run_dir/'report.json',report)
        eligible=[r for r in report['cases'] if 'expected_fields' in r]
        report['metrics']={'field_expected':sum(r['expected_fields'] for r in eligible),
                           'field_matched':sum(r['matched_fields'] for r in eligible),
                           'replay_failures':[r['id'] for r in eligible if r['replay_added_rows']],
                           'unexpected_write_cases':[r['id'] for r in eligible if r['unexpected_writes']],
                           'unexpected_ledger_cases':[r['id'] for r in eligible if r['unexpected_ledger']],
                           'unexpected_promise_cases':[r['id'] for r in eligible if r['unexpected_promises']],
                           'api_errors':[r['id'] for r in report['cases'] if 'error' in r]}
        report['metrics']['field_recall']=round(report['metrics']['field_matched']/report['metrics']['field_expected'],4)
    report['final_counts']=counts(c)
    save(run_dir/'report.json',report)
    print(json.dumps(report.get('metrics',report['final_counts'])),flush=True)


def now_ms_safe():
    from app.util import now_ms
    return now_ms()


if __name__=='__main__':
    main()
