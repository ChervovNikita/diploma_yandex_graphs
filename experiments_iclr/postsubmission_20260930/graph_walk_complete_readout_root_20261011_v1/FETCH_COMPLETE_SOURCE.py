import json,hashlib,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0';assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
base=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_walk_acquisition_root_20261011_v1');values={}
for name,rel in [('panel','acquisition_v1/COMPLETE_PANEL.json'),('owner_end','owner_v1/OWNER_END.json')]:
 p=base/rel;b=p.read_bytes();values[name]=dict(receipt=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),text=b.decode())
print(json.dumps(values),flush=True)
