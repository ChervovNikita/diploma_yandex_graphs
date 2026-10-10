"""Publish the committed source through the existing named-key helper."""
import shlex
import subprocess

REMOTE = r'''
import hashlib,os,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
os.chdir(R)
os.environ['GNNM_SSH_DESTINATION']='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
helper=R/'experiments_iclr/postsubmission_20260930/publication/push_committed_branch_443_v3.py'
body=helper.read_text()
assert body.count("REPO/'.git/a4.sock'")==1
body=body.replace("REPO/'.git/a4.sock'","REPO/'.git/lx9.sock'")
sys.argv=[str(helper),'--commit-receipt','publication/live_exchange_full9_started_20261010_v1/COMMIT_RECEIPT.json','--push-receipt','publication/live_exchange_full9_started_20261010_v1/PUSH_RECEIPT.json']
exec(compile(body,str(helper)+':fresh-socket','exec'),{'__name__':'__main__','__file__':str(helper)})
'''

argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
raise SystemExit(subprocess.call(argv))
