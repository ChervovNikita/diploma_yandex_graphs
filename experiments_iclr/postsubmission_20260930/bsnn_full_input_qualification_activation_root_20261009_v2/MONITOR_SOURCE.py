from pathlib import Path
import socket,subprocess,json,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'bsnn_full_input_qualification_activation_root_20261009_v2';D=P/'bsnn_full_input_qualification_execution_root_20261009_v2'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text();v=s[s.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(v[19]),'state':v[0]}
 except FileNotFoundError:return None
x={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent':ident(556051),'VALID_metrics':False}
if x['parent']:assert x['parent']['start_ticks']==6034114364
if (A/'WORKER_OWNER.json').exists():
 h=json.loads((A/'WORKER_OWNER.json').read_text());x['saved_child']=h['child'];x['child']=ident(h['child']['pid'])
 if x['child']:assert x['child']['start_ticks']==h['child']['start_ticks']
for name in ('RESULT.json','COMPLETE.json'):
 f=D/name
 if f.exists():
  h=json.loads(f.read_text());x[name]={k:h[k] for k in ('status','qualification_passed','stage','counters','failure','reconstruction_diagnostics','complete_attempt_seconds','peak_CUDA_allocated_bytes') if k in h}
f=A/'TERMINAL.json'
if f.exists():x['terminal']=json.loads(f.read_text())
if x['parent'] is None:x['worker_log']=(A/'WORKER.log').read_text()[-4000:]
print(json.dumps(x))
