"""Stream only SHA-bound eligible project bytes through the existing linked Mac."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
SOURCE_REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
TARGET_REPO = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'

SENDER = r'''
from pathlib import Path
import hashlib,json,os,socket,sys,subprocess,tty
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
tty.setraw(0)
os.chdir(REPO)
assert Path.cwd()==Path(REPO)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for row in PLAN:
 p=Path(REPO)/row['source_relative']
 assert p.resolve().is_relative_to(Path(REPO)) and p.is_file() and not p.is_symlink()
 assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
for row in PLAN:
 p=Path(REPO)/row['source_relative'];h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):
   n+=len(b);h.update(b);sys.stdout.buffer.write(b)
 assert n==row['bytes'] and h.hexdigest()==row['sha256']
sys.stdout.buffer.flush()
'''
RECEIVER = r'''
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,socket,sys,subprocess
assert socket.gethostname()=='peptide'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
os.chdir(REPO)
assert Path.cwd()==Path(REPO)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
rows=[]
for row in PLAN:
 p=Path(REPO)/row['target_relative']
 assert p.resolve().is_relative_to(Path(REPO)) and not p.is_symlink()
 present=os.path.lexists(p)
 if present:assert p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
 rows.append((row,p,present))
result=[]
for row,p,present in rows:
 p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 assert p.parent.resolve().is_relative_to(Path(REPO))
 sink=None if present else p.open('xb')
 h=hashlib.sha256();remaining=row['bytes']
 try:
  while remaining:
   chunk=sys.stdin.buffer.read(min(1048576,remaining))
   assert chunk,'Exact stream ended early'
   remaining-=len(chunk);h.update(chunk)
   if sink is not None:sink.write(chunk)
  assert h.hexdigest()==row['sha256'],'Incoming exact-byte hash differs'
  if sink is not None:
   sink.flush();os.fsync(sink.fileno())
 finally:
  if sink is not None:sink.close()
 if not present:p.chmod(0o444)
 assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
 result.append({'path':str(p),'bytes':p.stat().st_size,'sha256':row['sha256'],'mode':oct(p.stat().st_mode&0o777),'preexisting_exact_file_preserved':present,'created_by_this_stream':not present})
assert sys.stdin.buffer.read(1)==b'','Unexpected trailing stream bytes'
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'repository':REPO,'status':'EXACT_STREAM_DESTINATION_HASHES_VERIFIED','files':result,'payloads_decoded':False,'existing_files_overwritten':False,'scientific_jobs_untouched':True}),flush=True)
'''

def tcl_quote(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$').replace('[', '\\[') + '"'

def remote(repo, plan, program):
    script = 'REPO=' + repr(repo) + '\nPLAN=' + repr(plan) + '\n' + program
    return '/usr/bin/python3 -I -S -B -c ' + shlex.quote(script)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.id):raise RuntimeError('Single-use stream identity required')
    path = args.plan.resolve()
    if not path.is_relative_to(HERE):raise RuntimeError('Reviewed project plan required')
    plan = json.loads(path.read_text())
    if not isinstance(plan, list) or not plan:raise RuntimeError('Explicit nonempty eligible file list required')
    for row in plan:
        if set(row) != {'source_relative','target_relative','bytes','sha256'}:raise RuntimeError('Exact file descriptor fields required')
        if type(row['bytes']) is not int or row['bytes'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}',row['sha256']):raise RuntimeError('Invalid descriptor')
        for key in ('source_relative','target_relative'):
            p=Path(row[key])
            if p.is_absolute() or '..' in p.parts or p.parts[:2] != ('experiments_iclr','postsubmission_20260930'):raise RuntimeError('Explicit phase-only file required')
    out = HERE / 'commands' / args.id;out.mkdir(parents=True,exist_ok=False)
    source = ['/usr/bin/ssh','-tt','-p','2222','-i','/Users/shmelev/.maclink-reverse/anogena_identity','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',remote(SOURCE_REPO,plan,SENDER)]
    target = ['/usr/bin/ssh','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','PreferredAuthentications=password','-o','PubkeyAuthentication=no','shmelev@192.168.18.77',remote(TARGET_REPO,plan,RECEIVER)]
    pipeline = shlex.join(source) + ' | ' + shlex.join(target)
    expect = r'''
set timeout 240
set handle [open /Users/shmelev/Desktop/server-192.168.18.51-password.txt r]
set password [string trim [read $handle]]
close $handle
if {$password eq "" || [string first "\n" $password] >= 0} {puts stderr "Invalid single-line credential"; exit 2}
log_user 0
'''
    expect += 'spawn /bin/bash -o pipefail -c ' + tcl_quote(pipeline) + '\n'
    expect += r'''
expect {
 -re {(?i)password:} {send -- "$password\r"}
 timeout {puts stderr "Exact stream authentication timed out"; exit 3}
 eof {puts stderr "Exact stream exited before password authentication"; exit 4}
}
unset password
log_user 1
expect {
 -re {Permission denied, please try again} {puts stderr "Exact stream authentication rejected"; exit 5}
 timeout {puts stderr "Exact stream transfer timed out"; exit 6}
 eof {set status [wait]; exit [lindex $status 3]}
}
'''
    argv=[str(WORKSPACE/'reverse_maclink/maclink-env/bin/python'),'-B',str(WORKSPACE/'reverse_maclink/maclink.py'),'run','--','/usr/bin/expect','-c',expect]
    result=subprocess.run(argv,cwd=WORKSPACE,capture_output=True,text=True,timeout=290)
    record=dict(UTC=datetime.now(timezone.utc).isoformat(),id=args.id,exit_code=result.returncode,
        source_route='anogena-2 via linked-Mac saved identity',target_route='shmelev@192.168.18.77',
        transport='exact binary pipe through two explicit SSH sessions on linked Mac',
        plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files=plan,credential_value_recorded=False,credential_command_payload_logged=False,
        current_Mac_payload_copy_created=False,linked_Mac_payload_copy_created=False,
        seven_GPU_role='MacLink forwarding relay only',stdout=result.stdout,stderr=result.stderr)
    (out/'RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('exit_code','stdout','stderr')},sort_keys=True))
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
