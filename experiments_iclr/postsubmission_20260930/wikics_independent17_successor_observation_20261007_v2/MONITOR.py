import json,socket,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def who(pid):
 p=Path('/proc')/str(pid)
 try:
  s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();return {'PID':pid,'start_ticks':int(f[19]),'state':f[0],'ppid':int(f[1])}
 except FileNotFoundError:return None
root=P/'wikics_independent17_successor_execution_root_20261007_v1';r={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller':who(500633),'producer':who(498539),'status':{},'quality_values_read':False}
if r['controller']:assert r['controller']['start_ticks']==6014579194
if r['producer']:assert r['producer']['start_ticks']==6014469084
for n in ('WAIT.json','DEPENDENCY_ADMITTED.json','COMPLETE.json','FAILURE.json'):
 if (root/n).exists():r['status'][n]=json.loads((root/n).read_text())
q=P/'wikics_private_graph_controls_execution_root_20261007_v2/runs/DONOR_INDEPENDENT_17/PROGRESS.json'
if q.exists():r['prefix_progress']={k:v for k,v in json.loads(q.read_text()).items() if k in ('native_seed','epoch','complete_target')}
print(json.dumps(r))
