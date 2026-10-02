"""Replay saved real-model outputs into a fresh volume fixture, without another API call."""
import json
import os
from pathlib import Path
from datetime import datetime

from scripts.mock_scenarios import ROOT, seed, cases, counts, evaluate, save

source=Path(json.loads((ROOT/'data/qa/latest-mock.json').read_text())['run_dir'])
run=ROOT/'data/qa'/('mock-replay-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
run.mkdir()
os.environ['HOME_DB_PATH']=str(run/'home.db')
os.environ['HOME_BLOB_DIR']=str(run/'blobs')
from app.store.db import get_conn
from app.store import repo
from app.memory.apply import apply_extraction
from app.util import now_ms
c=get_conn()
seed(c)
report={'synthetic':True,'mode':'cached real-model outputs; no new model sampling',
        'source_run':str(source),'run_dir':str(run),'seed_counts':counts(c),'cases':[]}
for case in cases():
    extraction=json.loads((source/('extract-'+case['id']+'.json')).read_text(encoding='utf-8'))
    save(run/('extract-'+case['id']+'.json'),extraction)
    before=counts(c)
    applied=apply_extraction(c,extraction,'mock-live:'+case['id'],case['text'])
    first=counts(c)
    apply_extraction(c,extraction,'mock-live:'+case['id'],case['text'])
    second=counts(c)
    entry={'id':case['id'],**evaluate(case,extraction,applied),'outcomes':applied['outcomes'],
           'replay_added_rows':{t:second[t]-first[t] for t in first if second[t]!=first[t]},
           'unexpected_writes':{t:first[t]-before[t] for t in ('person','person_field','memory_item','promise','ledger_entry')
                                if case.get('expect_no_new_data') and first[t]!=before[t]}}
    report['cases'].append(entry)
    repo.insert_interaction(c,now_ms(),'chat',case['text'],'mock-live:'+case['id'],[(p,'mentioned') for p in applied['person_ids']])
    c.commit()
report['final_counts']=counts(c)
save(run/'report.json',report)
save(run/'cases.json',cases())
save(ROOT/'data/qa/latest-mock.json',{'run_dir':str(run)})
assert not any(x['replay_added_rows'] or x['unexpected_writes'] for x in report['cases'])
print('Fresh cached replay fixture ready: '+str(run),flush=True)
print(json.dumps(report['final_counts']),flush=True)
