from pathlib import Path
import json,hashlib,socket,subprocess
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');D=P/'pre_sigmoid_qk_full_input_qualification_output_20261009_v2';A=P/'pre_sigmoid_qk_full_input_qualification_activation_root_20261009_v2'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert json.loads((A/'TERMINAL.json').read_text())['complete'] and json.loads((D/'REPORT.json').read_text())['complete']
files={}
for f in [*D.glob('*.json'),A/'TERMINAL.json',A/'WORKER_OWNER.json']:
 assert f.is_file() and not f.is_symlink() and f.stat().st_size<2000000
 data=f.read_text();files[str(f.relative_to(P))]=dict(text=data,bytes=f.stat().st_size,sha256=hashlib.sha256(data.encode()).hexdigest())
print(json.dumps({'files':files,'scientific_scores_read':False}))
