from pathlib import Path
import json,socket,subprocess,hashlib,datetime,os
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'geometry_only_core_sequential_full_input_assessment_activation_root_20261009_v3';D=P/'geometry_only_core_sequential_full_input_assessment_execution_root_20261009_v3'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
t=json.loads((A/'TERMINAL.json').read_text());assert t['complete'] and t['error'] is None and t['exit_code']==0 and t['reaped'] and t['actual_worker_absent'] and t['actual_worker_CUDA_absent']
for pid in (562030,562033):
 assert not Path('/proc',str(pid)).exists()
 try:os.killpg(pid,0)
 except ProcessLookupError:pass
 else:raise AssertionError('Original group still alive')
f=D/'QUALIFICATION.json';b=f.read_bytes();assert len(b)<500000;x=json.loads(b);assert x['status']=='complete' and x['qualification_passed'] and not x['validation_metric_access'] and not x['TEST_truth_present']
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),path=str(f.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=b.decode(),actual_absence_verified=True,raw_engineering_state_transferred=False)))
