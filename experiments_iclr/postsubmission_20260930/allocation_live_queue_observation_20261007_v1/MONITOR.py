import json,socket,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip();assert inv.split(',')[0]=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def ident(pid):
 q=Path('/proc')/str(pid)
 try:
  s=(q/'stat').read_text();f=s[s.rfind(')')+2:].split();return {'PID':pid,'start_ticks':int(f[19]),'state':f[0],'ppid':int(f[1])}
 except FileNotFoundError:return None
out={}
for name,pid,ticks in [('citeseer_initialization_remaining_flat_execution_root_20261007_v1',495028,6013629570),('citeseer_growth_full15_flat_execution_root_20261007_v1',495546,6013797508),('wikics_private_graph_controls_execution_root_20261007_v2',498539,6014469084)]:
 D=P/name;r={'owner':ident(pid),'cells':{},'terminal':{}};out[name]=r
 if r['owner']:assert r['owner']['start_ticks']==ticks
 for f in [D/'owner'/'COMPLETE.json',D/'owner'/'FAILURE.json',D/'OWNER_COMPLETE.json',D/'OWNER_FAILURE.json']:
  if f.is_file():
   a=json.loads(f.read_text());r['terminal'][str(f.relative_to(D))]={'complete':a.get('all_complete',a.get('complete')),'cells':[{k:c.get(k) for k in ('cell_id','complete','error')} for c in a.get('completed',a.get('results',[]))]}
 for d in sorted((D/'runs').glob('*')):
  c={'freeze_present':(d/'FREEZE.json').is_file(),'failure_present':(d/'FAILURE.json').is_file()};q=d/'PROGRESS.json'
  if q.is_file():
   a=json.loads(q.read_text());c['progress']={k:a[k] for k in ('cycle','counters','condition','epoch','native_seed','complete_target','correction_epoch','target_epochs','arm') if k in a}
  r['cells'][d.name]=c
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':inv,'roots':out,'quality_values_requested':False}))

E=P/'learnable_internal_be_WikiCS_resource_execution_root_20261007_v2'
r={'owner':ident(506790),'complete':(E/'COMPLETE.json').is_file(),'failure':(E/'FAILURE.json').is_file(),'memory_wait_failure':(E/'MEMORY_WAIT_FAILURE.json').is_file(),'resources':[]}
if r['owner']:assert r['owner']['start_ticks']==6015082416
for f in sorted((E/'receipts').glob('*_RESOURCE.json')):
 a=json.loads(f.read_text());r['resources'].append({'path':str(f.relative_to(P)),'passed':a['passed'],'status':a['status'],'seed':a['seed'],'peak_GPU_bytes':a['peak_GPU_bytes']})
print(json.dumps({'resources':r,'predictive_scores_read':False}))
