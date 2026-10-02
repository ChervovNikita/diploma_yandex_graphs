"""Run one explicit root request on the authorized route under whole supervision."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

PHASE=Path(__file__).resolve().parents[1]
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE=REPO+'/experiments_iclr/postsubmission_20260930/'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','outer','inner','receipt'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    for value in (args.request,args.outer,args.inner,args.receipt):
        rel=PurePosixPath(value)
        if rel.is_absolute() or '..' in rel.parts or str(rel)!=value:raise ValueError('Require phase-relative paths')
    receipt=PHASE/args.receipt
    if receipt.exists():raise ValueError('Launch identity already used')
    request=PHASE/args.request;data=json.loads(request.read_text())
    interpreter=REPO+'/.venv/bin/python'
    argv=[interpreter,REMOTE+'protocols/bounded_run_v1.py','--output',REMOTE+args.outer,
          '--request',REMOTE+args.request,'--cap-seconds',str(data['whole_cap_seconds']),'--grace-seconds','5','--',
          interpreter,REMOTE+'protocols/run_authorized_v2.py','--output',REMOTE+args.inner,'--require-idle-gpu','--',
          interpreter,data.get('entry_script',REMOTE+'modern_teacher_execution_root_v1/qualification_entry_v1.py'),
          '--request',REMOTE+args.request,'--supervisor',REMOTE+args.inner]
    command='cd '+shlex.quote(REPO)+' && source experiments_iclr/postsubmission_20260930/protocols/repo_env.sh && GNNM_SSH_DESTINATION='+shlex.quote(LOGIN)+' '+shlex.join(argv)
    ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes',
         '-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN,command]
    started=datetime.now(timezone.utc).isoformat();result=subprocess.run(ssh,capture_output=True,check=False)
    value=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
               request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),remote_argv=argv,
               ssh_destination=LOGIN,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace'))
    with receipt.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(exit_code=result.returncode,receipt=str(receipt),stdout=value['stdout'])))
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
