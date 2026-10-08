from pathlib import Path
import json,socket,subprocess,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'query_value_gate_full15_activation_root_20261009_v1';D=P/'query_value_gate_full15_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text();v=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(v[19]),state=v[0])
 except FileNotFoundError:return None
launch=json.loads((A/'LAUNCH.json').read_text());out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),saved_parent=launch['parent'],actual_parent=ident(launch['parent']['pid']),seeds={},terminal={},quality_scores_read=False)
if out['actual_parent']:assert out['actual_parent']['start_ticks']==launch['parent']['start_ticks']
for seed in (6101,6203,6307):
 z={};h=D/'handles'/('seed'+str(seed)+'.json')
 if h.exists():
  q=json.loads(h.read_text());z['saved_child']=q.get('child');z['actual_child']=ident(q['child']['pid']) if q.get('child') else None
 f=D/('seed'+str(seed))/'PROGRESS.json'
 if f.exists():
  q=json.loads(f.read_text());z['label_epoch']=q['label_epoch'];z['work']=q['work'];z['cuda_memory']=q['cuda_memory']
 f=D/'handles'/('seed'+str(seed)+'.EXIT.json')
 if f.exists():z['exit']=json.loads(f.read_text())
 if z:out['seeds'][str(seed)]=z
for name in ('FAMILY_FAILURE.json','FAMILY_CLOSURE.json','PARENT_TERMINAL.json'):
 f=D/name
 if f.exists():out['terminal'][name]=json.loads(f.read_text())
if 'FAMILY_FAILURE.json' in out['terminal'] or not out['actual_parent']:
 out['owner_log_tail']=(A/'OWNER.log').read_text()[-4500:]
 for f in (D/'logs').glob('*.log'):out[str(f.relative_to(D))]=f.read_text()[-4500:]
print(json.dumps(out))
