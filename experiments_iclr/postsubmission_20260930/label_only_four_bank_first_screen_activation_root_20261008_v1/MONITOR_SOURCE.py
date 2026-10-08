from pathlib import Path
import json,socket,subprocess,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_only_four_bank_first_screen_activation_root_20261008_v1';E=P/'label_only_four_bank_first_screen_execution_root_20261008_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def identity(pid):
 f=Path('/proc',str(pid),'stat')
 if not f.exists():return None
 s=f.read_text();v=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(v[19]),state=v[0])
x={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent':identity(533583),'scores_read':False,'files':{}}
if x['parent']:assert x['parent']['start_ticks']==6029258572
for q in (E/'PARENT_OWNER.json',E/'handles/seed6101.json',E/'FAMILY_FAILURE.json',E/'FAMILY_CLOSURE.json',E/'PARENT_TERMINAL.json'):
 if q.exists():x['files'][str(q.relative_to(E))]=json.loads(q.read_text())
q=E/'handles/seed6101.json'
if q.exists():
 h=json.loads(q.read_text());x['worker']=identity(h['child']['pid']) if h.get('child') else None
q=E/'seed6101/PROGRESS.json'
if q.exists():
 p=json.loads(q.read_text());x['progress']={k:p[k] for k in ('epoch','complete','work') if k in p}
if 'FAMILY_FAILURE.json' in x['files'] or x['parent'] is None:
 x['owner_log']=(D/'OWNER.log').read_text()[-4500:]
 q=E/'logs/seed6101.log'
 if q.exists():x['worker_log']=q.read_text()[-4500:]
x['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(x))
