from pathlib import Path
import json,subprocess,socket,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'sehgnn_independent4_full_input_qualification_activation_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
for name in ('STARTER.json','LAUNCH.json','TERMINAL.json','MONITOR_FAILURE.json'):
 f=A/name
 if f.exists():d[name]=json.loads(f.read_text())
for name in ('worker.stderr','owner.stderr','worker.stdout'):
 f=A/name
 if f.exists():d[name]=f.read_text()[-6000:]
print(json.dumps(d))
