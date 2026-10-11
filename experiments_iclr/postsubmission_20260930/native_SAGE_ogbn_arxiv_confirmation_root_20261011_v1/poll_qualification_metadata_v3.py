"""Read own qualification custody and counters; no scientific predictive outputs."""
import argparse
from datetime import datetime,timezone
import base64
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H=Path(__file__).resolve().parent
P=H.parent
REMOTE=r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
P=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=P/'native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1'
def j(p):return json.loads(p.read_text()) if p.exists() else None
r=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_predictive_outputs_read=False,
 engine_owner=j(H/'engine_owner_v3/OWNER_END.json'),engine_qualification=j(H/'engine_qualification_v3/ENGINE_EQUIVALENCE.json'),
 resources_owner=j(H/'resources_owner_v3/OWNER_END.json'),resources_qualification=j(H/'resources_qualification_v3/ARXIV_RESOURCE_QUALIFICATION.json'),
 archive_receipt=j(H/'OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V3.json'),
 Penn_owner=j(P/'native_SAGE_Penn94_data_preparation_root_20261011_v1/native_acquisition_v1_owner/OWNER_END.json'),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines())
for stage in ('engine','resources'):
 if r[stage+'_owner'] and r[stage+'_owner']['status']!='COMPLETE_EXIT_0':
  q=H/(stage+'_owner_v3')/'stderr.log';r[stage+'_failure_log']=q.read_text()[-5000:] if q.exists() else None
print(json.dumps(r))
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,required=True);a=ap.parse_args()
    target=H/f'QUALIFICATION_METADATA_REGISTERED_V{a.version}.json';assert not target.exists()
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'metadata','qualification','existing-owners'])]
    r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60)
    (H/f'QUALIFICATION_METADATA_REGISTERED_TRANSPORT_V{a.version}.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stdout[-2000:]
    obj=json.loads(r.stdout);target.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({k:v for k,v in obj.items() if k not in ('engine_qualification','resources_qualification','archive_receipt')}))

if __name__=='__main__':main()
