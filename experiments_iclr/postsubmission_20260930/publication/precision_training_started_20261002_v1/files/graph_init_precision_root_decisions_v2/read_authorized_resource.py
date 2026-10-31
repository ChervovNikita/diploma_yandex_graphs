"""Read resources and branch identity only on the authorized project route."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
CODE = '''import json, os, subprocess
from pathlib import Path
repo=Path(%r)
assert Path.cwd().resolve()==repo
q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
rows=[r.strip() for r in q.stdout.splitlines() if r.strip()]
assert len(rows)==1 and rows[0].split(',')[1].strip()==%r
s=os.statvfs(repo)
def git(args):
 return subprocess.run(['git']+args,cwd=repo,capture_output=True,text=True,check=True,timeout=15).stdout.strip()
print(json.dumps(dict(repository=str(repo),gpu_csv=rows[0],disk_available_bytes=s.f_bavail*s.f_frsize,branch=git(['branch','--show-current']),head=git(['rev-parse','HEAD']),study_v1_exists=(repo/'experiments_iclr/postsubmission_20260930/graph_init_precision_execution_root_v2/study_v2').exists())))
''' % (REPO, UUID)

if __name__ == '__main__':
    path = HERE/'LIVE_RESOURCE_v1.json'
    if path.exists():
        raise ValueError('Receipt already exists')
    command='cd '+shlex.quote(REPO)+' && .venv/bin/python -c '+shlex.quote(CODE)
    result=subprocess.run(SSH+[command],capture_output=True,text=True,timeout=60)
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),ssh_destination=LOGIN,
                 exit_code=result.returncode,stderr=result.stderr,
                 resource=json.loads(result.stdout) if result.returncode==0 else None)
    with path.open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))
    raise SystemExit(result.returncode)
