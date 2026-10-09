"""Read only complete-family status and epochs; no molecular score access."""
from pathlib import Path
import datetime,json,socket,subprocess
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
release=P/'internal_BE_molhiv18_family_wait_execution_root_20261007_v1/EXECUTOR_RELEASE.json'
cfg=json.loads(release.read_text())
def inside(s):
 p=Path(s)
 if not p.is_absolute():p=P/p
 p=p.resolve();assert p.is_relative_to(P);return p
def identity(pid,birth):
 try:s=Path('/proc',str(pid),'stat').read_text()
 except FileNotFoundError:return dict(pid=pid,expected_birth=birth,present=False)
 f=s[s.rfind(')')+2:].split();assert int(f[19])==birth,'Reused PID'
 return dict(pid=pid,start_ticks=int(f[19]),present=True,state=f[0],group=int(f[2]),session=int(f[3]))
d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),release_keys=list(cfg),scores_opened=False,owner=identity(523400,6019318952))
if d['owner']['present']:d['owner_argv']=Path('/proc/523400/cmdline').read_bytes().replace(b'\0',b' ').decode()
d['CUDA_owned']=[]
for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).splitlines():
 if row.split(',')[0].strip()=='523400':d['CUDA_owned'].append(row)
for key in ['output','output_directory','family_output','executor_output','fresh_output','family_directory','execution_directory']:
 if key not in cfg:continue
 q=inside(cfg[key]);d['output']=str(q);d['root_names']=sorted(x.name for x in q.iterdir()) if q.exists() else []
 for name in ['FAMILY_CLOSURE.json','EXECUTOR_TERMINAL.json','TERMINAL.json','COMPLETE.json']:
  d[name+'_present']=(q/name).exists()
 d['cells']=[]
 for path in sorted(q.glob('**/*_CELL.json')):
  c=json.loads(path.read_text());d['cells'].append({k:c[k] for k in ('cell','condition','seed','status','exit_code','fit_output','error') if k in c})
 d['progress']=[]
 for path in sorted(q.glob('**/PROGRESS.json')):
  c=json.loads(path.read_text());d['progress'].append(dict(path=str(path),**{k:c[k] for k in ('epoch','steps','status') if k in c}))
print(json.dumps(d))
