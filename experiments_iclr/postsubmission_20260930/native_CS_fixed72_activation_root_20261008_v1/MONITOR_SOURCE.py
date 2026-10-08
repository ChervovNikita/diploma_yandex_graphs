from pathlib import Path
import json,socket,hashlib
assert socket.gethostname()=='anogena-2-0'
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');D=P/'native_own_best_TRAIN_only_CS_fixed72_execution_root_20261008_v1';out={}
for name in ('LAUNCH.json','WORKER_STARTED.json','TERMINAL.json','EVALUATION_WORKER.json'):
 f=D/name
 if f.exists():out[name]=json.loads(f.read_text())
if 'TERMINAL.json' in out:
 out['log_tail']=(D/'WORKER.log').read_text()[-3000:]
 if out['TERMINAL.json']['complete'] is True:
  f=D/'reference72/RECEIPT.json';r=json.loads(f.read_text());assert r['complete'] is True
  out['reference_complete']=True;out['native_records']=r['native_records'];out['selection']=r['selection'];out['selected_records']=[x for x in r['records'] if x['configuration_id']==r['selection']['configuration_id']];out['receipt_sha256']=hashlib.sha256(f.read_bytes()).hexdigest();out['receipt_bytes']=f.stat().st_size
else:
 f=D/'reference72/RECEIPT.json'
 if f.exists():r=json.loads(f.read_text());out['work_only_progress']={status:sum(x['status']==status for x in r['records']) for status in ('complete','failed','blocked','pending')}
print(json.dumps(out))
