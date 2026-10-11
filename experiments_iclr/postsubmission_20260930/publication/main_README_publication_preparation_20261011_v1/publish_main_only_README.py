"""Publish one reviewed README blob on exact main parent without an index/checkout.

Root executes only after reviewed content/hash and exact publication admission.
Key unlock stays in the established interactive TTY, never args/env/files/logs.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
KEY=Path('/home/jovyan/.ssh/the_github')
REMOTE='git@github.com:ChervovNikita/diploma_yandex_graphs.git'
RESEARCH_REF='refs/heads/codex/postsubmission-research-20260930'

def sha(data):return hashlib.sha256(data).hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def main():
    # Route verification precedes project/config/source reads.
    require(socket.gethostname()=='anogena-2-0','Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID],'Excluded wrong sole GPU')
    require(Path.cwd()==REPO and os.environ.get('GNNM_SSH_DESTINATION')==LOGIN,'Literal authorized caller and allocation repository')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True)
    parser.add_argument('--readme',type=Path,required=True)
    parser.add_argument('--readme-sha256',required=True)
    parser.add_argument('--original-readme-sha256',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--socket-name',default='rm42.sock')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}',args.parent) and re.fullmatch('[0-9a-f]{64}',args.readme_sha256) and re.fullmatch('[0-9a-f]{64}',args.original_readme_sha256),'Exact parent and reviewed SHA256 authority')
    source=args.readme.resolve();receipt=args.receipt.resolve()
    require(source.is_relative_to(PHASE) and receipt.is_relative_to(PHASE) and source.is_file() and receipt.parent.is_dir() and not receipt.exists(),'Confined reviewed source and fresh receipt path')
    reviewed=source.read_bytes();require(sha(reviewed)==args.readme_sha256 and reviewed.decode('utf-8').startswith('# '),'Exact reviewed UTF8 README bytes')
    require(re.fullmatch('[a-z][a-z0-9]{1,8}\.sock',args.socket_name),'Short unique agent socket name')
    agent_socket=REPO/'.git'/args.socket_name
    require(not agent_socket.exists() and len(str(agent_socket).encode())<104 and agent_socket.parent.resolve()==REPO/'.git','Fresh repository-confined agent socket')
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    def git(*argv,input=None,timeout=30):
        return subprocess.run(['git',*argv],cwd=REPO,env=env,input=input,capture_output=True,check=True,timeout=timeout).stdout
    def string(*argv):return git(*argv).decode().strip()
    require(string('remote','get-url','origin')==REMOTE,'Exact original GitHub origin')
    before=dict(HEAD=string('rev-parse','HEAD'),branch=string('branch','--show-current'),index_sha256=sha((REPO/'.git/index').read_bytes()),working_README_sha256=sha((REPO/'README.md').read_bytes()),status_sha256=sha(git('status','--porcelain=v1','--untracked-files=no')))
    require(before['branch']=='codex/postsubmission-research-20260930','Preserve existing research checkout')
    record=dict(schema='reviewed-main-only-README-plain-fast-forward-v1',UTC=datetime.now(timezone.utc).isoformat(),ssh_destination=LOGIN,hostname=socket.gethostname(),GPU_UUID=UUID,source=dict(path=str(source),bytes=len(reviewed),sha256=sha(reviewed)),requested_parent=args.parent,original_README_sha256=args.original_readme_sha256,publish_authorized_by_root=args.publish,push_attempted=False,passphrase_recorded=False,fresh_repo_confined_agent_socket=str(agent_socket),before=before)
    startup=subprocess.check_output(['ssh-agent','-t','300','-a',str(agent_socket),'-s'],cwd=REPO,text=True,timeout=15)
    sock=re.search(r'SSH_AUTH_SOCK=([^;]+);',startup);pid=re.search(r'SSH_AGENT_PID=(\d+);',startup)
    require(sock and pid and sock[1]==str(agent_socket),'Exact fresh SSH-agent startup')
    env.update(SSH_AUTH_SOCK=sock[1],SSH_AGENT_PID=pid[1])
    env['GIT_SSH_COMMAND']=shlex.join(['ssh','-p','22','-o','Hostname=github.com','-o','HostKeyAlias=github.com','-o','ConnectTimeout=15','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','IdentitiesOnly=yes','-o','IdentityAgent='+str(agent_socket),'-i',str(KEY)])
    try:
        # Deliberately inherit the established secure TTY for the original key.
        # The passphrase is supplied only at ssh-add's prompt, by root.
        print(json.dumps(dict(stage='named_key_unlock',key=str(KEY),fresh_socket=str(agent_socket))),flush=True)
        added=subprocess.run(['ssh-add',str(KEY)],cwd=REPO,env=env,timeout=180,check=False)
        require(added.returncode==0,'Original named key must load; no GitHub operation on unlock failure')
        def refs():
            fields=string('ls-remote','--exit-code','origin','refs/heads/main',RESEARCH_REF).splitlines()
            return {line.split()[1]:line.split()[0] for line in fields}
        advertised=refs();require(advertised.get('refs/heads/main')==args.parent,'Current advertised main differs from reviewed parent; stop for new review')
        record['remote_refs_before']=advertised
        # Parent already exists in the allocation Git object store; no fetch,
        # remote tracking ref update, checkout, reset, merge or index is needed.
        require(string('cat-file','-t',args.parent)=='commit','Exact advertised main parent object present')
        original=git('show',args.parent+':README.md');require(sha(original)==args.original_readme_sha256,'Original main README differs from reviewed preserve-content authority')
        record.update(parent_tree=string('rev-parse',args.parent+'^{tree}'),original_README_blob=string('rev-parse',args.parent+':README.md'),original_README_bytes=len(original))
        if not args.publish:
            record.update(status='PREPARED_REVIEW_ONLY',main_modified=False,objects_or_refs_written=False)
        else:
            require(reviewed!=original,'Reviewed README must contain a change')
            blob=git('hash-object','-w','--stdin',input=reviewed).decode().strip()
            entries=git('ls-tree','-z',args.parent).split(bytes([0]));replacement=[];found=0
            for entry in entries:
                if not entry:continue
                header,name=entry.split(bytes([9]),1);mode,kind,old_blob=header.split()
                if name==b'README.md':
                    require(kind==b'blob' and mode in (b'100644',b'100755'),'Original root README regular blob mode')
                    entry=mode+b' blob '+blob.encode()+bytes([9])+name;found+=1
                replacement.append(entry)
            require(found==1,'Exactly one root README tree entry')
            tree=git('mktree','-z',input=bytes([0]).join(replacement)+bytes([0])).decode().strip()
            commit=git('commit-tree',tree,'-p',args.parent,input=b'Update README with current architecture, comparisons and experiment status\n').decode().strip()
            require(string('rev-list','--parents','-n','1',commit).split()==[commit,args.parent],'Exactly one advertised-main parent')
            require(git('diff-tree','--no-commit-id','--name-only','-r','-z',args.parent,commit)==b'README.md'+bytes([0]),'Commit changes only README.md')
            require(git('cat-file','blob',commit+':README.md')==reviewed,'Committed README bytes match root-reviewed source exactly')
            delta=git('diff-tree','--no-commit-id','--raw','-r',args.parent,commit).decode().strip()
            record.update(new_README_blob=blob,new_tree=tree,new_commit=commit,changed_paths=['README.md'],tree_change=delta,normal_single_parent_commit=True)
            require(refs().get('refs/heads/main')==args.parent,'Main advanced before push; no rebase/force/merge permitted')
            record['push_attempted']=True
            pushed=subprocess.run(['git','push','--porcelain','origin',commit+':refs/heads/main'],cwd=REPO,env=env,capture_output=True,text=True,timeout=180,check=False)
            record.update(push_exit_code=pushed.returncode,push_stdout=pushed.stdout,push_stderr=pushed.stderr,force=False,checkout_or_index_used=False)
            require(pushed.returncode==0,'Normal fast-forward main push failed; retain exact object/receipt')
            final=refs();require(final.get('refs/heads/main')==commit,'GitHub main must advertise exact new README commit')
            record.update(status='PUBLISHED_EXACT_MAIN_README',remote_refs_after=final,remote_main_verified=True,research_remote_ref_unchanged=final.get(RESEARCH_REF)==advertised.get(RESEARCH_REF),published_only_ref='refs/heads/main')
    except BaseException as error:
        record.update(status='FAILED_RETAINED',failure_type=type(error).__name__,failure=str(error));raise
    finally:
        cleanup=subprocess.run(['ssh-agent','-k'],cwd=REPO,env=env,capture_output=True,text=True,timeout=15)
        for _ in range(20):
            if not agent_socket.exists():break
            time.sleep(.05)
        after=dict(HEAD=string('rev-parse','HEAD'),branch=string('branch','--show-current'),index_sha256=sha((REPO/'.git/index').read_bytes()),working_README_sha256=sha((REPO/'README.md').read_bytes()),status_sha256=sha(git('status','--porcelain=v1','--untracked-files=no')))
        record.update(after=after,checkout_index_working_files_unchanged=after==before,agent_cleanup_exit_code=cleanup.returncode,agent_socket_absent=not agent_socket.exists(),terminal_UTC=datetime.now(timezone.utc).isoformat())
        with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps(record,sort_keys=True),flush=True)
        require(after==before,'Unexpected concurrent checkout/index/worktree change; see retained snapshots')
        require(cleanup.returncode==0 and not agent_socket.exists(),'Exact temporary agent cleanup')

if __name__=='__main__':main()
