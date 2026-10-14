"""Interactive named-key Git push through the literal allocation route."""
import os, shlex, subprocess
from remote_ops import SSH, GUARD, REMOTE_ROOT, REMOTE_DESIGN
code = GUARD + 'import os,subprocess; os.chdir(' + repr(REMOTE_ROOT) + '); os.environ["GNNM_SSH_DESTINATION"]="anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"; subprocess.run([' + repr(REMOTE_ROOT+'/.venv/bin/python') + ',' + repr(REMOTE_DESIGN+'/smoke_20261014/push_smokes_allocation.py') + ',"--commit-receipt","reference_wrapper_redesign_20261013_v1/smoke_20261014/COMMIT_RECEIPT.json","--push-receipt","reference_wrapper_redesign_20261013_v1/smoke_20261014/PUSH_RECEIPT.json"],check=True)'
raise SystemExit(subprocess.call(SSH + ['python3 -B -c ' + shlex.quote(code)]))
