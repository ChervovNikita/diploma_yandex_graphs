from pathlib import Path
import json,socket,subprocess,hashlib,datetime,os
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'bsnn_full_three_seed_baseline_activation_root_20261009_v1';D=P/'bsnn_full_three_seed_baseline_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
t=json.loads((A/'TERMINAL.json').read_text());assert t['complete'] and t['error'] is None and t['exit_code']==0 and t['reaped'] and t['actual_worker_absent'] and t['actual_worker_CUDA_absent']
launch=json.loads((A/'LAUNCH.json').read_text());worker=json.loads((A/'WORKER_OWNER.json').read_text())
for h in (launch['parent'],worker['child']):
 assert not Path('/proc',str(h['pid'])).exists()
 try:os.killpg(h['group'],0)
 except ProcessLookupError:pass
 else:raise AssertionError('Original owner group still present')
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines();assert str(worker['child']['pid']) not in [r.strip() for r in rows]
files={};overview=[]
for seed in (1103,2207,3301):
 F=D/('seed'+str(seed));x=json.loads((F/'RESULT.json').read_text());assert x['status']=='complete' and x['seed']==seed
 hist=[json.loads(line) for line in (F/'HISTORY.jsonl').read_text().splitlines() if line.strip()]
 overview.append(dict(seed=seed,history_rows=len(hist),first=hist[0],last=hist[-1]))
 for name in ('RESULT.json',):
  f=F/name;b=f.read_bytes();assert len(b)<500000;files[str(f.relative_to(P))]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=b.decode())
for f in [A/'TERMINAL.json',D/'SUMMARY.json',D/'COMPLETE.json',D/'RUNTIME.json']:
 if f.exists():
  b=f.read_bytes();assert len(b)<500000;files[str(f.relative_to(P))]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=b.decode())
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),all3_complete_before_read=True,actual_parent_worker_group_CUDA_absence_revalidated=True,files=files,learning_overview=overview,raw_checkpoints_or_logits_transferred=False,TEST_truth_accessed=False)))
