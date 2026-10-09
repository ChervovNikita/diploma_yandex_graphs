from pathlib import Path
import json,socket,subprocess,hashlib,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'geometry_only_core_full_input_qualification_activation_root_20261009_v1';D=P/'geometry_only_core_full_input_qualification_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
t=json.loads((A/'TERMINAL.json').read_text());assert not t['complete'] and t['actual_worker_absent'] and t['actual_worker_CUDA_absent'] and t['reaped']
assert not Path('/proc','559977').exists() and not Path('/proc','559980').exists()
f=D/'QUALIFICATION.json';b=f.read_bytes();assert len(b)<250000;x=json.loads(b)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),partial=dict(path=str(f.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=b.decode()),raw_model_or_predictions_transferred=False,original_failed_status_retained=True,BSNN_progress={})
B=P/'bsnn_full_three_seed_baseline_execution_root_20261009_v1'
for f in B.glob('**/RESULT.json'):
 z=json.loads(f.read_text());out['BSNN_progress'][str(f.relative_to(B))]={k:z[k] for k in ('status','seed','epochs_completed','last_attempted_epoch','selected_epoch','failure') if k in z}
for pid,start in ((557178,6034281493),(557181,6034281501)):
 try:
  s=Path('/proc',str(pid),'stat').read_text();v=s[s.rfind(')')+2:].split();assert int(v[19])==start;out.setdefault('actual_BSNN_handles',[]).append(dict(pid=pid,start_ticks=int(v[19]),state=v[0]))
 except FileNotFoundError:out.setdefault('actual_BSNN_handles',[]).append(dict(pid=pid,absent=True))
print(json.dumps(out))
