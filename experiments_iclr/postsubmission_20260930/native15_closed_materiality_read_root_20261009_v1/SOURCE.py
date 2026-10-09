from pathlib import Path
import json,socket,subprocess,hashlib,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
A=P/'native15_materiality_activation_root_20261009_v1';t=json.loads((A/'TERMINAL.json').read_text());assert t['complete'] and t['error'] is None and t['actual_worker_absent'] and t['actual_worker_CUDA_absent']
launch=json.loads((A/'LAUNCH.json').read_text());assert not Path('/proc',str(launch['parent']['pid'])).exists()
D=P/'native15_materiality_execution_root_20261009_v1';files={}
for f in D.glob('**/*.json'):
 data=f.read_bytes();assert len(data)<500000;files[str(f.relative_to(P))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data=data.decode())
for name in ('LAUNCH.json','WORKER_OWNER.json','TERMINAL.json','RESOURCE_ADMISSION.json'):
 f=A/name;data=f.read_bytes();files[str(f.relative_to(P))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data=data.decode())
f=P/'private_sheaf_baseline_screen_execution_root_20261009_v1/SCREEN_SUMMARY.json'
assert f.is_file();data=f.read_bytes();files[str(f.relative_to(P))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data=data.decode())
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),raw_transferred=False,files=files)))
