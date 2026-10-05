"""Fetch only the original complete-schedule/closure process metadata."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,json,socket,subprocess
from pathlib import Path
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
out={'host':socket.gethostname(),'files':[],'scores_read':False,'TEST_access':False,'launches':0}
for slot in ('v6_full_schedule_v1','v6_closure_v1'):
 for name in ('LAUNCH.json','TERMINAL.json','FREEZE.json','RESULT.json','FAILURE.json'):
  p=root/slot/name
  if not p.exists():continue
  assert p.is_file() and not p.is_symlink() and p.stat().st_size<262144
  raw=p.read_bytes();v=json.loads(raw)
  if name=='RESULT.json':
   assert v['schema'] in ('amazon_polynormer_full_schedule_exit_v2','amazon_polynormer_cohort_closure_v2')
  out['files'].append({'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'utf8':raw.decode()})
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()
assert len(free)==1;out['GPU_free_bytes']=int(free[0])*1024*1024
print(json.dumps(out))
'''

def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=HERE/('observation_'+stamp);out.mkdir()
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
             '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
             '-o','UpdateHostKeys=no','-o','ConnectTimeout=20',
             'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
             'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -B -c '+shlex.quote(REMOTE)]
    started=datetime.now(timezone.utc).isoformat()
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=50)
        receipt={'start_UTC':started,'terminal_UTC':datetime.now(timezone.utc).isoformat(),
                 'exit_code':r.returncode,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest()}
        if r.returncode:
            receipt['stdout']=r.stdout
        else:
            v=json.loads(r.stdout)
            for row in v['files']:
                relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
                assert relative.parts[0]=='amazon_polynormer_paired_family_execution_root_20261003_v3'
                b=row.pop('utf8').encode();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
                p=out/relative;p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(b)
            with (out/'OBSERVATION.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
            receipt['observation_sha256']=hashlib.sha256((out/'OBSERVATION.json').read_bytes()).hexdigest()
    except Exception as error:
        receipt={'start_UTC':started,'terminal_UTC':datetime.now(timezone.utc).isoformat(),
                 'failure_type':type(error).__name__,'failure':str(error),'automatic_redispatch':False}
    with (out/'TRANSPORT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'directory':str(out),'transport':receipt,
                      'complete_schedule_freeze':any(x['path'].endswith('/v6_full_schedule_v1/FREEZE.json') for x in v['files']) if 'v' in locals() else False,
                      'complete_closure_freeze':any(x['path'].endswith('/v6_closure_v1/FREEZE.json') for x in v['files']) if 'v' in locals() else False,
                      'GPU_free_bytes':v['GPU_free_bytes'] if 'v' in locals() else None}))
    if receipt.get('exit_code')!=0:raise SystemExit(1)

if __name__=='__main__':main()
