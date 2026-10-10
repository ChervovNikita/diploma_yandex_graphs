"""Fetch the hash-bound result of the already closed stored-only comparison."""
from pathlib import Path
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import hashlib,json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=P/'pubmed_own4_M_normalization_source_20261010_v1'
A=P/'pubmed_own4_M_normalization_root_activation_20261010_v1'
start=json.loads((A/'comparison/STARTER.json').read_text());assert not Path('/proc',str(start['PID'])).exists()
assert json.loads((H/'comparison/FAMILY_COMPLETE.json').read_text())['complete']
assert not (H/'comparison/FAMILY_FAILURE.json').exists()
C=H/'comparison/cells/own4_M_normalization_complete3_comparison/COMPLETE.json'
v=json.loads(C.read_text());assert v['complete'] and v['new_model_forwards']==0
D=Path(v['comparison']['path']);assert D.is_relative_to(C.parent)
assert hashlib.sha256(D.read_bytes()).hexdigest()==v['comparison']['sha256']
print(json.dumps(dict(path=str(D.relative_to(P)),sha256=v['comparison']['sha256'],text=D.read_text())))
'''
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
(HERE/'NORMALIZATION_TRANSPORT.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
assert r.returncode==0,r.stderr
v=json.loads(r.stdout);q=HERE/'fetched'/v['path'];assert q.resolve().is_relative_to((HERE/'fetched').resolve())
q.parent.mkdir(parents=True,exist_ok=True);q.write_text(v['text'])
assert hashlib.sha256(q.read_bytes()).hexdigest()==v['sha256']
print(json.dumps(dict(local_path=str(q),sha256=v['sha256'],means=json.loads(v['text'])['mean_contrasts'])))
