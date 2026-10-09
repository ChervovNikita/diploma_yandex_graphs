"""Use GitHub official SSH port443, the named key and exact ref verification.

No passphrase is accepted as an argument, printed, written or committed. The
temporary SSH-agent socket is created in this repository's Git metadata.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
KEY=Path('/home/jovyan/.ssh/the_github')

def run(args,env=None):
    return subprocess.run(args,cwd=REPO,env=env,capture_output=True,text=True,check=True).stdout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit-receipt',required=True)
    parser.add_argument('--push-receipt',required=True)
    args=parser.parse_args()
    if Path.cwd().resolve()!=REPO or os.environ.get('GNNM_SSH_DESTINATION')!=LOGIN:
        raise ValueError('Wrong repository or calling route')
    if run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).splitlines()!=[UUID]:
        raise ValueError('Wrong allocation')
    commit_path=PHASE/args.commit_receipt;receipt=PHASE/args.push_receipt
    if not commit_path.resolve().is_relative_to(PHASE) or not receipt.resolve().is_relative_to(PHASE) or receipt.exists():
        raise ValueError('Require confined existing commit receipt and new push identity')
    commit=json.loads(commit_path.read_text());head=run(['git','rev-parse','HEAD']).strip()
    if head!=commit['commit'] or run(['git','branch','--show-current']).strip()!=commit['branch']:
        raise ValueError('Branch/head changed before push')
    if run(['git','remote','get-url','origin']).strip()!='git@github.com:ChervovNikita/diploma_yandex_graphs.git':
        raise ValueError('Unexpected remote')
    if run(['git','diff','--cached','--name-only']):
        raise ValueError('Unexpected staged changes')
    socket=REPO/'.git/a4.sock'
    if not socket.parent.resolve().is_relative_to(REPO) or socket.exists() or len(str(socket).encode())>=104:
        raise ValueError('Require a new confined short agent socket')
    agent=run(['ssh-agent','-a',str(socket),'-s'])
    sock_match=re.search(r'SSH_AUTH_SOCK=([^;]+);',agent)
    pid_match=re.search(r'SSH_AGENT_PID=(\d+);',agent)
    if not sock_match or not pid_match or sock_match[1]!=str(socket):
        raise ValueError('Unexpected SSH-agent startup')
    env=os.environ.copy();env['SSH_AUTH_SOCK']=sock_match[1];env['SSH_AGENT_PID']=pid_match[1]
    env['GIT_SSH_COMMAND']=shlex.join(['ssh','-p','443','-o','Hostname=ssh.github.com','-o','HostKeyAlias=github.com','-o','ConnectTimeout=15','-o','BatchMode=yes','-o','UpdateHostKeys=no',
                                      '-o','StrictHostKeyChecking=yes','-o','IdentitiesOnly=yes',
                                      '-o','IdentityAgent='+str(socket),'-i',str(KEY)])
    started=datetime.now(timezone.utc).isoformat();value=None
    try:
        added=subprocess.run(['ssh-add',str(KEY)],cwd=REPO,env=env,timeout=180,check=False)
        if added.returncode:
            raise RuntimeError('Named key was not loaded; no push attempted')
        ref='refs/heads/'+commit['branch']
        probe=subprocess.run(['git','ls-remote','--exit-code','origin',ref],cwd=REPO,env=env,
                             capture_output=True,text=True,timeout=30,check=True)
        print(json.dumps(dict(SSH_transport='ssh.github.com:443',prior_advertised_ref=probe.stdout.strip())),flush=True)
        push=subprocess.run(['git','push','--set-upstream','origin','HEAD:refs/heads/'+commit['branch']],
                            cwd=REPO,env=env,capture_output=True,text=True,timeout=180,check=False)
        if push.returncode:
            value=dict(verified=False,push_exit_code=push.returncode,stderr=push.stderr,stdout=push.stdout)
        else:
            ref='refs/heads/'+commit['branch']
            advertised=run(['git','ls-remote','--exit-code','origin',ref],env).strip().split()
            if advertised!=[head,ref]:
                raise RuntimeError('Remote advertised ref does not match committed head')
            value=dict(verified=True,push_exit_code=0,remote_commit=head,remote_ref=ref,
                       pushed_commit_receipt_sha256=hashlib.sha256(commit_path.read_bytes()).hexdigest())
    except Exception as error:
        value=dict(verified=False,failure_type=type(error).__name__,failure=str(error))
        raise
    finally:
        subprocess.run(['ssh-agent','-k'],cwd=REPO,env=env,capture_output=True,check=False)
        if value is not None:
            value.update(schema='gnnm-named-key-exact-ref-push-443-v3',start_UTC=started,
                         terminal_UTC=datetime.now(timezone.utc).isoformat(),branch=commit['branch'],
                         commit=head,ssh_destination=LOGIN,gpu_uuid=UUID,
                         passphrase_recorded=False,agent_socket_confined=True)
            with receipt.open('x') as stream:
                json.dump(value,stream,indent=2);stream.write('\n')
            print(json.dumps(value))
    if not value['verified']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
