"""Read named preparation metadata only from the verified allocation."""
import json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REMOTE=r"""
import json,socket,subprocess
from pathlib import Path
assert socket.gethostname()=="anogena-2-0"
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True,timeout=10).splitlines()==["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
p=Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930")
value={}
for name in ("publication/README_training_preparation_20261012_v1/PUSH_RECEIPT_SSH22_V1.json","publication/README_training_preparation_20261012_v1/PUSH_RECEIPT_SSH22_V2.json","publication/README_training_preparation_20261012_v1/MAIN_PUSH_RECEIPT_V1.json"):
 f=p/name
 if f.exists():value[name]=json.loads(f.read_text())
value["node_custodies"]={}
for name in ("cora","citeseer","pubmed","wikics","amazon-computers","amazon-photo","coauthor-cs","coauthor-physics"):
 f=p/"training_pipeline_v2_20261012/prepared_data"/name/"CUSTODY.json"
 value["node_custodies"][name]=json.loads(f.read_text())
print(json.dumps(value))
"""
a=["ssh","-tt","-p","2222","-i","/Users/alex/.ssh/mlspace__private_key_anogena.txt","-o","IdentitiesOnly=yes","-o","BatchMode=yes","-o","ConnectTimeout=15","-o","StrictHostKeyChecking=yes","-o","UpdateHostKeys=no","anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru",shlex.join(["/usr/bin/python3","-I","-S","-B","-c",REMOTE])]
f=HERE/"COMPACT_METADATA_TRANSPORT_V1.json";assert not f.exists()
r=subprocess.run(a,capture_output=True,text=True,timeout=45)
f.write_text(json.dumps({"UTC":datetime.now(timezone.utc).isoformat(),"exit_code":r.returncode,"stdout":r.stdout,"stderr":r.stderr},indent=2)+"\n")
assert r.returncode==0
v=json.loads(r.stdout)
for name,receipt in v.items():
 if name=="node_custodies":continue
 target=HERE.parents[1]/name
 assert not target.exists()
 target.write_text(json.dumps(receipt,indent=2)+"\n")
(HERE/"NODE_CUSTODY_METADATA_V1.json").write_text(json.dumps(v["node_custodies"],indent=2)+"\n")
print(json.dumps({"receipts":len(v)-1,"datasets":len(v["node_custodies"]),"splits":sum(len(c["splits"]) for c in v["node_custodies"].values())}))
