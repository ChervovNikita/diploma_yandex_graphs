"""Transfer exact small sources and start one finite bounded engine owner."""
from datetime import datetime,timezone
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H=Path(__file__).resolve().parent
P=H.parent
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE=r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,hashlib,os,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[sys.argv[2]]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'
H=phase/'native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1'
for row in payload['files']:
 path=phase/row['relative'];assert path.resolve().is_relative_to(phase) and not path.is_symlink()
 data=base64.b64decode(row['payload']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==data
 else:path.write_bytes(data)
os.chdir(repo)
bound=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(H/'bind_engine_qualification_v1.py')],cwd=repo,text=True,capture_output=True,timeout=90)
assert bound.returncode==0,bound.stdout[-1000:]+bound.stderr[-3000:]
log=H/'engine_outer.log';assert not log.exists()
with log.open('x') as f:
 owner=subprocess.Popen([str(repo/'.venv/bin/python'),'-B',str(H/'qualification_owner_v1.py'),'--stage','engine'],cwd=repo,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_PID=owner.pid,binding=json.loads(bound.stdout),scientific_fits=0,TEST_scoring=False,external_finite_watchdog=True)
(H/'ENGINE_LAUNCH_V1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''

def main():
    files=[]
    for dirname in ('native_SAGE_sequential_own_CE_arxiv_source_20261011_v2','native_SAGE_endpoint_value_transport_source_20261011_v2'):
        for path in sorted((P/dirname).iterdir()):
            if path.is_file() and path.suffix in ('.py','.json','.md'):
                data=path.read_bytes();assert len(data)<150000
                files.append(dict(relative=str(path.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),payload=base64.b64encode(data).decode()))
    for name in ('run_qualification_v1.py','bind_engine_qualification_v1.py','qualification_owner_v1.py'):
        path=H/name;data=path.read_bytes()
        compile(data,str(path),'exec')
        files.append(dict(relative=str(path.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),payload=base64.b64encode(data).decode()))
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    args=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,R,UUID,'sequential-engine-only'])]
    r=mod.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps(dict(files=files)).encode())).decode(),capture_output=True,text=True,timeout=120)
    (H/'ENGINE_LAUNCH_TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stdout[-3000:]
    value=json.loads(r.stdout);(H/'ENGINE_LAUNCH_V1.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
