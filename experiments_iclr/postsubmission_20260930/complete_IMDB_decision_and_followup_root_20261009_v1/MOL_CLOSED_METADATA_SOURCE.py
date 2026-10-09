from pathlib import Path
import json,socket,subprocess,hashlib,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
Q=P/'internal_BE_molhiv18_family_execution_root_20261007_v1'
files=[]
for name in ['FAMILY_CLOSURE.json','TERMINAL_EVIDENCE.json','HANDLES.json','LEDGER.json','PARENT_OWNER.json','FAMILY_OWNER.json']:
 p=Q/name;assert p.is_file() and p.stat().st_size<1500000
 b=p.read_bytes();files.append(dict(path=str(p.relative_to(P)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),text=b.decode()))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),files=files,scores_opened=False)))