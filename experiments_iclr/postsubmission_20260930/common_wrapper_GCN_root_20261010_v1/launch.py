"""Launch the committed finite owner on the literal authorized allocation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = r'''
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[uuid]
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
h=r/'experiments_iclr/postsubmission_20260930/common_wrapper_GCN_root_20261010_v1'
os.chdir(r)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==sys.argv[1]
assert not (h/'LAUNCH.json').exists()
frozen=json.loads((h/'FREEZE.json').read_text())
for row in frozen['bound_files']:assert hashlib.sha256((r/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=16384,'Less than16GiB free; defer this launch without touching other jobs'
with (h/'owner.stdout.log').open('x') as out,(h/'owner.stderr.log').open('x') as err:
 child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(h/'owner.py')],cwd=r,stdout=out,stderr=err,start_new_session=True)
 fields=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
value=dict(PID=child.pid,start_ticks=int(fields[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),hostname=socket.gethostname(),gpu_uuid=uuid,free_MiB_before=free,source_commit=sys.argv[1],freeze_sha256=hashlib.sha256((h/'FREEZE.json').read_bytes()).hexdigest(),detached=True,TEST_access=False)
with (h/'LAUNCH.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(value))
'''

def main():
    receipt = json.loads((PHASE / 'publication/common_wrapper_closed_and_transfer_frozen_20261010_v1/COMMIT_TRANSPORT.json').read_text())
    head = receipt['result']['commit']
    cmd = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, head])
    args = ['ssh', '-tt', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
            'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru', cmd]
    result = subprocess.run(args, capture_output=True, text=True, timeout=45)
    value = dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    with (HERE / 'ACTUAL_LAUNCH_TRANSPORT.json').open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))
    return result.returncode

if __name__ == '__main__':
    raise SystemExit(main())
