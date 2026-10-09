from pathlib import Path
import json,hashlib,socket,subprocess
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');A=P/'private_sheaf_full_input_qualification_activation_root_20261009_v1';D=P/'private_sheaf_full_input_qualification_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
t=json.loads((A/'TERMINAL.json').read_text());assert t['reaped'] and t['actual_worker_absent'];files={}
for f in [*D.glob('*.json'),*D.glob('*/RESULT.json'),*D.glob('*/DIRECT_STATIC_INDEX_COMPARISON*.json'),A/'TERMINAL.json',A/'WORKER_OWNER.json']:
 assert f.is_file() and f.stat().st_size<2000000;v=f.read_text();files[str(f.relative_to(P))]=dict(text=v,sha256=hashlib.sha256(v.encode()).hexdigest(),bytes=len(v.encode()))
print(json.dumps({'files':files,'selected_state_tensors_fetched':False}))
