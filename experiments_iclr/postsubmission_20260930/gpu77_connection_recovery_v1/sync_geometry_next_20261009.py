"""Fast-forward reviewed source only; retain running jobs and untracked evidence."""
from pathlib import Path
import datetime,json,os,socket,subprocess
R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
branch='codex/postsubmission-research-20260930'
expected='0fc698be57d4fc7bb762827680bb719035e1f118'
assert socket.gethostname()=='peptide'
assert set(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
def run(args,env=None):
 p=subprocess.run(args,cwd=R,env=env,capture_output=True,text=True,timeout=60)
 assert p.returncode==0,dict(args=args,exit_code=p.returncode,stderr=p.stderr)
 return p.stdout.strip()
assert run(['git','branch','--show-current'])==branch
assert not run(['git','status','--porcelain','--untracked-files=no'])
assert not run(['git','diff','--cached','--name-only'])
before=run(['git','rev-parse','HEAD'])
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UpdateHostKeys=no')
run(['git','fetch','--no-tags','origin','refs/heads/'+branch+':refs/remotes/origin/'+branch],env)
assert run(['git','rev-parse','refs/remotes/origin/'+branch])==expected
run(['git','merge-base','--is-ancestor',before,expected])
run(['git','merge','--ff-only',expected],env)
after=run(['git','rev-parse','HEAD']);assert after==expected
assert not run(['git','status','--porcelain','--untracked-files=no'])
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),head_before=before,head_after=after,verified=True,tracked_clean=True,mode='normal fast-forward',other_jobs_changed=False,untracked_evidence_deleted=False)))
