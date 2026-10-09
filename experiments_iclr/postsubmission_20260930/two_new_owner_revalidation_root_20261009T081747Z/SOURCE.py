from pathlib import Path
import json,socket,subprocess,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def actual(saved):
 if saved is None:return None
 try:s=Path('/proc',str(saved['pid']),'stat').read_text()
 except FileNotFoundError:return None
 f=s[s.rfind(')')+2:].split();x=dict(pid=saved['pid'],start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),state=f[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 assert all(x[k]==saved[k] for k in ('pid','start_ticks','group','session','boot_id'))
 return x
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),scores_read=False,studies={})
for name in ('native15_materiality_activation_root_20261009_v1','bsnn_full_three_seed_baseline_activation_root_20261009_v1'):
 A=P/name;row={}
 for n in ('LAUNCH.json','WORKER_OWNER.json','TERMINAL.json','RESOURCE_ADMISSION.json'):
  f=A/n
  if f.is_file():row[n]=json.loads(f.read_text())
 row['actual_parent']=actual(row['LAUNCH.json']['parent'])
 row['actual_worker']=actual(row.get('WORKER_OWNER.json',{}).get('child'))
 if row.get('TERMINAL.json',{}).get('error') or 'WORKER_OWNER.json' not in row:
  f=A/'OWNER.log'
  if f.is_file():row['owner_log_tail']=f.read_text()[-3000:]
 D=P/name.replace('_activation_','_execution_')
 row['progress']={}
 if D.exists():
  for f in D.glob('**/PROGRESS.json'):
   x=json.loads(f.read_text());row['progress'][str(f.relative_to(D))]={k:v for k,v in x.items() if k in ('epoch','epochs','complete_epochs','seed','steps','status','complete')}
 out['studies'][name]=row
out['GPU_usage']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(out))
