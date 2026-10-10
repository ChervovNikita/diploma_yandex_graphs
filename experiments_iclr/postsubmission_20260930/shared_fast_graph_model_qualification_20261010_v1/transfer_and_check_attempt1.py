"""Transfer two explicit project sources, then run one bounded check."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

PHASE = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location('_existing_project_transport', PHASE/'publication/publish_exact_inventory_v19.py')
helper = importlib.util.module_from_spec(loader)
loader.loader.exec_module(helper)
names = ['shared_fast_graph_model_interface_20261010_v1/common_routes.py', 'shared_fast_graph_model_qualification_20261010_v1/check.py']
rows = [dict(path=n, sha256=hashlib.sha256((PHASE/n).read_bytes()).hexdigest(), data=base64.b64encode((PHASE/n).read_bytes()).decode()) for n in names]
remote = r'''
import base64,hashlib,json,os,pathlib,socket,subprocess,zlib
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
os.chdir(R)
for row in payload:
 q=P/row['path'];assert q.resolve().is_relative_to(P)
 data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
 if q.exists():assert q.read_bytes()==data
 else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
O=P/'shared_fast_graph_model_qualification_20261010_v1/actual_full_TRAIN_v1'
assert not O.exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
q=subprocess.run(['timeout','--signal=TERM','--kill-after=10s','180s',str(R/'.venv/bin/python'),'-B',str(P/payload[1]['path']),'--output',str(O)],env=env,capture_output=True,text=True)
print(json.dumps(dict(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr,output=str(O.relative_to(P)),source_sha256=payload[0]['sha256'],qualification_sha256=payload[1]['sha256'],VALID_access=False,TEST_access=False)))
raise SystemExit(q.returncode)
'''
remote = 'import sys\n' + remote
argv = ['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote])]
encoded = base64.b64encode(zlib.compress(json.dumps(rows).encode())).decode()
result = helper.terminal_transport(argv,input=encoded,capture_output=True,text=True,timeout=220)
receipt = HERE/'ACTUAL_TRANSPORT.json'
assert not receipt.exists()
receipt.write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,source_rows=[{k:v for k,v in r.items() if k!='data'} for r in rows]),indent=2)+'\n')
print(result.stdout)
raise SystemExit(result.returncode)
