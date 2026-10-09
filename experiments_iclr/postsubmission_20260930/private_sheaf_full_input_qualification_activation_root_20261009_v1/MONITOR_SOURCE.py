from pathlib import Path
import json,socket,subprocess,datetime
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');A=P/'private_sheaf_full_input_qualification_activation_root_20261009_v1';D=P/'private_sheaf_full_input_qualification_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text();f=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(f[19]),state=f[0])
 except FileNotFoundError:return None
launch=json.loads((A/'LAUNCH.json').read_text());x=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=ident(launch['parent']['pid']),records=[],scientific_fit=False)
if x['parent']:assert x['parent']['start_ticks']==launch['parent']['start_ticks']
f=A/'WORKER_OWNER.json'
if f.exists():
 o=json.loads(f.read_text());x['child']=ident(o['child']['pid'])
 if x['child']:assert x['child']['start_ticks']==o['child']['start_ticks']
for f in D.glob('*/RESULT.json'):
 r=json.loads(f.read_text());x['records'].append({k:r[k] for k in ('config_id','status','failure','forward_counts','complete_attempt_seconds','cuda_peak_allocated_bytes') if k in r})
for name in ('QUALIFICATION_SUMMARY.json','QUALIFICATION_FAILURE.json'):
 f=D/name
 if f.exists():x[name]=json.loads(f.read_text())
f=A/'TERMINAL.json'
if f.exists():x['terminal']=json.loads(f.read_text());x['worker_log']=(A/'WORKER.log').read_text()[-4000:]
print(json.dumps(x))
