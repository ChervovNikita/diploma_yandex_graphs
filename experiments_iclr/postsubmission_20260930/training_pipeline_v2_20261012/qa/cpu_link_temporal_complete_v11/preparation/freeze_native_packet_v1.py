"""Freeze named reviewed native runtime/metadata; prepare only, no science."""
import base64,hashlib,json,shlex,subprocess,zlib
from pathlib import Path
from datetime import datetime,timezone
PACKAGE=Path(__file__).resolve().parents[1]
PHASE=PACKAGE.parent
HERE=PACKAGE/"preparation"
NAME="massive_native_packet_20261012_v1"
RUNTIME=("engine.py","metrics.py","spec.py","registry.py","partition.py","run_study.py","run_endpoint.py","launcher.py","phase_closure.py","adapters/__init__.py","adapters/classifier.py","adapters/node.py","adapters/node_losses.py","adapters/node_endpoints.py","adapters/search.py","start_massive.sh")
files=[]
def add(path,relative):
 raw=path.read_bytes();files.append({"path":relative,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"data":base64.b64encode(raw).decode()})
for name in RUNTIME:add(PACKAGE/name,"runtime/"+name)
for name in ("MATRIX_AXES_DRAFT.json","METHOD_INVENTORY.csv","METHOD_SOURCE_ALIASES.csv"):
 add(PACKAGE/"docs"/name,"metadata/"+name)
add(PHASE/"common_wrapper_native_source_20261010_v1/models.py","native/models.py")
for name in ("profile_native_resources.py","prepare_node_data.py","compile_native_matrix.py"):
 add(PACKAGE/name,"qualification/"+name)
for label in ("cpu_multitask_source_v7","cpu_execution_receipts_v11"):
 add(PACKAGE/"qa"/label/"TRANSPORT.json","fixtures/"+label+".json")
capacity={"worker_capacities_cuda_bytes":{x:78*1024**3 for x in ("allocation/gpu0","server77/gpu0","server77/gpu1")},"server77_basis":"historical80GiBGPUs; no live access; reverify before launch"}
raw=json.dumps(capacity,indent=2).encode()
files.append({"path":"metadata/PLANNING_CAPACITIES.json","bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"data":base64.b64encode(raw).decode()})
add(PACKAGE/"preparation/HOST_HARDWARE_BINDINGS.json","metadata/HOST_BINDINGS.json")
payload=base64.b64encode(zlib.compress(json.dumps({"files":files}).encode())).decode()
local=HERE/"native_freeze_v1";assert not local.exists();local.mkdir()
(local/"SOURCE_INVENTORY.json").write_text(json.dumps([{k:v for k,v in x.items() if k!="data"} for x in files],indent=2)+"\n")
prefix=["ssh","-tt","-p","2222","-i","/Users/alex/.ssh/mlspace__private_key_anogena.txt","-o","IdentitiesOnly=yes","-o","BatchMode=yes","-o","ConnectTimeout=15","-o","ServerAliveInterval=15","-o","ServerAliveCountMax=2","-o","StrictHostKeyChecking=yes","-o","UpdateHostKeys=no","anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"]
GUARD="""
import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
phase=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
"""
TRANSFER=GUARD+"""
f=phase/'training_pipeline_v2_20261012/preparation/native_freeze_v1_transport/payload.b64'
offset=int(sys.argv[1])
if offset==0:assert not f.parent.exists();f.parent.mkdir(parents=True)
else:assert f.stat().st_size==offset
with f.open('ab') as stream:stream.write(sys.argv[2].encode())
print(f.stat().st_size)
"""
def call(body,args):
 return subprocess.run(prefix+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',body,*args])],capture_output=True,text=True,timeout=60)
for offset in range(0,len(payload),24000):
 r=call(TRANSFER,[str(offset),payload[offset:offset+24000]])
 (local/("TRANSFER_"+str(offset)+".json")).write_text(json.dumps({"exit_code":r.returncode,"stdout":r.stdout,"stderr":r.stderr},indent=2)+"\n")
 assert r.returncode==0,r.stderr
FINISH=GUARD+"""
f=phase/'training_pipeline_v2_20261012/preparation/native_freeze_v1_transport/payload.b64'
encoded=f.read_bytes();assert hashlib.sha256(encoded).hexdigest()==sys.argv[1]
v=json.loads(zlib.decompress(base64.b64decode(encoded)))
root=phase/'massive_native_packet_20261012_v1';assert not root.exists();root.mkdir()
for x in v['files']:
 rel=Path(x['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(x['data']);assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
 target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
value={'schema':'named-native-runtime-freeze-v1','files':[{k:v for k,v in x.items() if k!='data'} for x in v['files']],'project_relative_root':str(root.relative_to(phase)),'benchmark_fits':0,'optimizer_updates':0,'GO_created':False}
(root/'SOURCE_INVENTORY.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
"""
r=call(FINISH,[hashlib.sha256(payload.encode()).hexdigest()])
(local/"TRANSPORT.json").write_text(json.dumps({"UTC":datetime.now(timezone.utc).isoformat(),"exit_code":r.returncode,"stdout":r.stdout,"stderr":r.stderr},indent=2)+"\n")
assert r.returncode==0,r.stderr
print(json.dumps({"frozen_files":len(files),"bytes":sum(x['bytes'] for x in files),"root":NAME,"science_started":False}))
