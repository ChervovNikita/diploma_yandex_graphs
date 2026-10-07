import json,os,socket,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip();assert inventory.split(',')[0]=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def identity(pid):
 q=Path('/proc')/str(pid)
 try:
  st=(q/'stat').read_text();a=st[st.rfind(')')+2:].split();return {'PID':pid,'state':a[0],'start_ticks':int(a[19]),'ppid':int(a[1])}
 except FileNotFoundError:return None
roots={}
for name,pid,ticks in [('citeseer_initialization_full14_flat_execution_root_20261006_v2',495028,6013629570),('citeseer_growth_full15_flat_execution_root_20261007_v1',495546,6013797508),('wikics_private_graph_comparison_execution_root_20261007_v1',498058,6014363517)]:
 d=P/name;r={'owner':identity(pid),'expected_start_ticks':ticks,'owner_terminal_files':{},'cells':{}};roots[name]=r
 if r['owner']:assert r['owner']['start_ticks']==ticks
 for f in [d/'owner'/'COMPLETE.json',d/'owner'/'FAILURE.json',d/'OWNER_COMPLETE.json',d/'OWNER_FAILURE.json']:
  if f.is_file():
   a=json.loads(f.read_text());r['owner_terminal_files'][str(f.relative_to(d))]={'all_complete':a.get('all_complete',a.get('complete')),'cells':[{k:c.get(k) for k in ('cell_id','complete','error')} for c in a.get('completed',a.get('results',[]))]}
 for out in sorted((d/'runs').glob('*')):
  item={}
  for f in ['PROGRESS.json','FAILURE.json']:
   q=out/f
   if q.is_file():
    a=json.loads(q.read_text());item[f]={k:a[k] for k in ('cycle','counters','condition','epoch','target_epochs','correction_epoch','arm','error') if k in a}
  item['FREEZE_present']=(out/'FREEZE.json').is_file();item['DONOR_FREEZE_present']=(out/'DONOR_FREEZE.json').is_file()
  r['cells'][out.name]=item
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':inventory,'roots':roots,'quality_values_requested':False}))
