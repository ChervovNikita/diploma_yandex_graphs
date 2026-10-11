import json,hashlib,os,socket,subprocess
from pathlib import Path
h=socket.gethostname();g=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
assert h=='anogena-2-0' and g==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs';os.chdir(R);base=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_walk_full_input_qualification_root_20261011_v1')
result=subprocess.run([R+'/.venv/bin/python','-B','-u','/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_meanSAGE_tour_scan_execution_source_20261011_v1/owner.py',str(base/'CONFIG.json')],cwd=R,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=970)
def file(rel):
 p=base/rel;b=p.read_bytes();return dict(receipt=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),value=json.loads(b))
value=dict(exit_code=result.returncode,owner_stdout=result.stdout,owner_stderr=result.stderr,route=dict(hostname=h,sole_gpu=g))
for name,rel in [('qualification','qualification_v1/QUALIFICATION.json'),('owner_end','owner_v1/OWNER_END.json')]:
 if (base/rel).is_file():value[name]=file(rel)
if result.returncode and (base/'owner_v1/child.log').is_file():value['child_failure_log']=(base/'owner_v1/child.log').read_text()[-12000:]
print(json.dumps(value),flush=True)
