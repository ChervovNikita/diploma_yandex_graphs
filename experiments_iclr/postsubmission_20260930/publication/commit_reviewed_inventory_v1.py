"""Commit only a coordinator-reviewed inventory on the authorized route."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    return subprocess.run(['git',*args],cwd=REPO,text=True,capture_output=True,check=True).stdout

def confined(root,value):
    relative=PurePosixPath(value)
    if relative.is_absolute() or '..' in relative.parts or str(relative)!=value:
        raise ValueError('Require normalized relative path')
    path=root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes allowed root')
    cursor=root
    for part in relative.parts:
        cursor/=part
        if cursor.is_symlink():
            raise ValueError('Symlink path forbidden')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    if Path.cwd().resolve()!=REPO or os.environ.get('GNNM_SSH_DESTINATION')!=LOGIN:
        raise ValueError('Wrong repository or calling route')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if gpu.stdout.splitlines()!=[UUID]:
        raise ValueError('Wrong allocation')
    plan_path=confined(PHASE,args.plan);receipt=confined(PHASE,args.receipt)
    if receipt.exists():
        raise ValueError('Publication identity already used')
    plan_bytes=plan_path.read_bytes();plan=json.loads(plan_bytes)
    if git('branch','--show-current').strip()!=plan['branch'] or git('rev-parse','HEAD').strip()!=plan['expected_head']:
        raise ValueError('Branch/head changed since review')
    if git('diff','--cached','--name-only'):
        raise ValueError('Preserve preexisting staging')
    prepared=[];paths=[]
    for row in plan['files']:
        target=confined(REPO,row['target']);source=confined(PHASE,row['source_snapshot'])
        if row['target']!='README.md' and not row['target'].startswith('experiments_iclr/postsubmission_20260930/'):
            raise ValueError('Unexpected Git target')
        data=source.read_bytes()
        if sha(data)!=row['sha256'] or len(data)!=row['bytes']:
            raise ValueError('Reviewed snapshot changed')
        if target.exists():
            current=sha(target.read_bytes())
            if current!=row['sha256'] and current!=row['expected_old_sha256']:
                raise ValueError('Existing target differs from reviewed state: '+row['target'])
        elif row['expected_old_sha256'] is not None:
            raise ValueError('Expected existing target missing')
        prepared.append((target,data));paths.append(row['target'])
    if len(paths)!=len(set(paths)):
        raise ValueError('Duplicate target')
    if sha(plan_path.read_bytes())!=sha(plan_bytes):
        raise ValueError('Plan changed during preflight')
    for target,data in prepared:
        if not target.exists() or target.read_bytes()!=data:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    git('add','-f','--',*paths)
    staged=git('diff','--cached','--name-only','-z').split('\0')
    staged=[p for p in staged if p]
    if not staged or not set(staged)<=set(paths):
        raise ValueError('Staging is outside reviewed inventory or empty')
    for row in plan['files']:
        blob=subprocess.run(['git','show',':'+row['target']],cwd=REPO,capture_output=True,check=True).stdout
        if sha(blob)!=row['sha256']:
            raise ValueError('Staged blob differs from reviewed bytes')
    result=git('commit','-m',plan['message'])
    head=git('rev-parse','HEAD').strip()
    changed=[p for p in git('diff-tree','--no-commit-id','--name-only','-r',head).splitlines() if p]
    if set(changed)!=set(staged) or git('diff','--cached','--name-only'):
        raise ValueError('Commit/staging verification failed')
    value=dict(schema='gnnm-explicit-reviewed-source-commit-receipt-v1',
               UTC=datetime.now(timezone.utc).isoformat(),branch=plan['branch'],
               prior_head=plan['expected_head'],commit=head,changed_paths=changed,
               plan_sha256=sha(plan_bytes),source_sha256=sha(Path(__file__).read_bytes()),
               ssh_destination=LOGIN,gpu_uuid=UUID,credentials_included=False,
               binary_artifacts_included=False,push_verified=False)
    with receipt.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(commit=head,changed_files=len(changed),receipt=str(receipt),commit_output=result)))

if __name__=='__main__':
    main()
