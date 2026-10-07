import json,socket,subprocess
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 q=Path('/proc')/str(pid)
 try:
  s=(q/'stat').read_text();f=s[s.rfind(')')+2:].split();return {'PID':pid,'start_ticks':int(f[19]),'state':f[0],'ppid':int(f[1]),'argv':[x.decode() for x in (q/'cmdline').read_bytes().split(b'\0') if x]}
 except FileNotFoundError:return None
owner=ident(498539);assert owner is None or owner['start_ticks']==6014469084
D=P/'wikics_private_graph_controls_execution_root_20261007_v2';out={'owner':owner,'cells':{},'terminal_present':(D/'owner'/'COMPLETE.json').exists(),'failure_present':(D/'owner'/'FAILURE.json').exists(),'quality_values_read':False}
for d in sorted((D/'runs').glob('*')):
 v={};q=d/'PROGRESS.json'
 if q.exists():
  a=json.loads(q.read_text());v['progress']={k:a[k] for k in ('native_seed','epoch','complete_target','arm','correction_epoch','target_epochs') if k in a}
 v['failure']=(d/'FAILURE.json').exists();v['complete_artifact']=(d/'FREEZE.json').exists() or (d/'DONOR_FREEZE.json').exists();v['smoke_files']=[p.name for p in d.glob('**/TRAIN_SMOKE*.json')]
 log=D/'owner'/'logs'/(d.name+'.CHILD_STARTED.json')
 if log.exists():
  a=json.loads(log.read_text());ii=a.get('identity');v['child']=ident(ii['PID']) if ii else None
 out['cells'][d.name]=v
out['initialization_owner']=ident(495028)
print(json.dumps(out))
