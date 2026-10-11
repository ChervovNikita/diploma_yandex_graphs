"""Deploy one reviewed immutable packet and bound its TRAIN-only qualifier."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

P=Path('postsubmission_research_20260930').resolve()
H=P/'native_SAGE_fixed_context_key_pilot_root_20261011_v1'
S=P/'native_SAGE_fixed_context_key_source_20261011_v1'
REMOTE=r"""import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'native_SAGE_fixed_context_key_pilot_root_20261011_v1'
for row in payload:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 q=P/rel;assert q.resolve().is_relative_to(P);data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256'];q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==data
 else:q.write_bytes(data)
assert json.loads((H/'ACTUAL_QUALIFICATION_V1.json').read_text())['qualified']
assert not (H/'actual_family_v1').exists() and not (H/'OWNER_START.json').exists()
print(json.dumps({'deployed':len(payload),'scientific_fits':0,'resources':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()}))
"""

def main():
    paths=[H/name for name in ['CALIBRATION_POLICY.json','DECISION.md','PROSPECTIVE_PROTOCOL.json','run.py','owner.py']]
    rows=[]
    for path in paths:
        data=path.read_bytes()
        rows.append({'path':str(path.relative_to(P)),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()})
    spec=importlib.util.spec_from_file_location('reviewed_literal_transport',P/'publication/publish_exact_inventory_v19.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'qualify','literal-route','TRAIN-only'])]
    result=transport.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=240)
    (H/'METADATA_DEPLOYMENT_TRANSPORT_V1.json').write_text(json.dumps({'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
    assert result.returncode==0,result.stderr[-2000:]
    print(result.stdout.strip())

if __name__=='__main__':main()
