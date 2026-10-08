from pathlib import Path
import datetime,json,socket,subprocess
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'internal_BE_molhiv18_family_execution_root_20261007_v1'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:
  s=Path('/proc/'+str(pid)+'/stat').read_text();f=s[s.rfind(')')+2:].split();return dict(PID=pid,start_ticks=int(f[19]),state=f[0])
 except FileNotFoundError:return None
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner':ident(523400),'progress':{},'receipts':{},'terminal':{},'comparative_scores_read':False}
if out['owner']:assert out['owner']['start_ticks']==6019318952
for f in D.glob('fits/*/PROGRESS.json'):
 a=json.loads(f.read_text());out['progress'][str(f.relative_to(D))]={k:v for k,v in a.items() if k in ('epoch','epochs','complete_epochs','batches','updates','steps','condition','seed','status','complete')}
for f in D.glob('receipts/*_OWNER.json'):
 a=json.loads(f.read_text());out['receipts'][str(f.relative_to(D))]={'saved':a,'actual':ident(a['pid'])}
for f in D.glob('receipts/*.log'):out['receipts'][str(f.relative_to(D))]={'bytes':f.stat().st_size,'mtime':f.stat().st_mtime}
for f in [D/'FAMILY_CLOSURE.json',D/'OWNER_FAILURE.json',D/'HANDLES.json']:
 if f.is_file():out['terminal'][f.name]=json.loads(f.read_text())
for f in D.glob('receipts/*_CELL.json'):
 a=json.loads(f.read_text())
 if a.get('status') not in ('complete',):
  out['terminal'][str(f.relative_to(D))]=a
  log=f.with_name(f.name.replace('_CELL.json','.log'))
  if log.is_file():out['terminal'][str(log.relative_to(D))]=log.read_text()[-5000:]
E=P/'label_only_four_bank_first_screen_execution_root_20261008_v2'
out['corrector']={'parent':ident(533819),'seeds':{},'terminal':{}}
if out['corrector']['parent']:assert out['corrector']['parent']['start_ticks']==6029309094
for seed in (6101,6203,6307):
 z={}
 f=E/'handles'/('seed'+str(seed)+'.json')
 if f.exists():
  h=json.loads(f.read_text());z['child_saved']=h.get('child');z['child_actual']=ident(h['child']['pid']) if h.get('child') else None
 f=E/('seed'+str(seed))/'PROGRESS.json'
 if f.exists():
  h=json.loads(f.read_text());z['epoch']=h['epoch'];z['work']=h['work']
 f=E/'handles'/('seed'+str(seed)+'.EXIT.json')
 if f.exists():z['terminal']=json.loads(f.read_text())
 if z:out['corrector']['seeds'][str(seed)]=z
for name in ('PROGRESS.json','FAMILY_CLOSURE.json','FAMILY_FAILURE.json','PARENT_TERMINAL.json'):
 f=E/name
 if f.exists():out['corrector']['terminal'][name]=json.loads(f.read_text())
out['GPU_usage']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(out))
