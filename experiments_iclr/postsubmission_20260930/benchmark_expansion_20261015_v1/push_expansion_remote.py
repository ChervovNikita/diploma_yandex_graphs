"""Interactive TTY named-key push; no credentials in source or arguments."""
import shlex
import subprocess
from remote_control import SSH, GUARD, REMOTE_ROOT, REMOTE_PHASE, REMOTE_EXPANSION

relative=REMOTE_EXPANSION.split(REMOTE_PHASE+'/')[1]
code=GUARD+'import os,subprocess; os.chdir('+repr(REMOTE_ROOT)+'); os.environ["GNNM_SSH_DESTINATION"]="anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"; subprocess.run(['+repr(REMOTE_ROOT+'/.venv/bin/python')+','+repr(REMOTE_PHASE+'/reference_wrapper_redesign_20261013_v1/smoke_20261014/push_smokes_allocation.py')+',"--commit-receipt",'+repr(relative+'/COMMIT_RECEIPT.json')+',"--push-receipt",'+repr(relative+'/PUSH_RECEIPT.json')+'],check=True)'
if __name__=='__main__':raise SystemExit(subprocess.call(SSH+['python3 -B -c '+shlex.quote(code)]))
