from pathlib import Path
import json,socket,subprocess,datetime,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'bsnn_full_three_seed_baseline_activation_root_20261009_v1';D=P/'bsnn_full_three_seed_baseline_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={},progress={},comparative_scores_read=False)
for name in ('LAUNCH.json','WORKER_OWNER.json','TERMINAL.json','RESOURCE_ADMISSION.json'):
 f=A/name
 if f.exists():
  b=f.read_bytes();out['files'][str(f.relative_to(P))]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=b.decode())
for f in D.glob('**/RESULT.json'):
 x=json.loads(f.read_text());out['progress'][str(f.relative_to(D))]={k:x[k] for k in ('status','seed','epochs_completed','last_attempted_epoch','selected_epoch','family','failure') if k in x}
for n in ('SUMMARY.json','SCREEN_SUMMARY.json','TERMINAL.json'):
 f=D/n
 if f.exists():out['summary_available']=True
print(json.dumps(out))
