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
d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),release_keys=list(cfg),scores_opened=False)
for key in ['output','output_directory','family_output','executor_output','fresh_output','family_directory']:
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
