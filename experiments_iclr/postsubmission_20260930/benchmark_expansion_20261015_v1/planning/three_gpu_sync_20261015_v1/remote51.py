"""Project-only commands through the authenticated MacLink/SSH control socket."""
from pathlib import Path
import subprocess,json,shlex,hashlib
W=Path(__file__).resolve().parents[4]
ROOT='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
GPU_UUIDS=['GPU-9c6c20fa-a620-479d-f72f-163ac74deef2','GPU-30283973-140d-9e7b-e780-0fcc487dec54']
GUARD="""import socket,subprocess
assert socket.gethostname()=='olig'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-9c6c20fa-a620-479d-f72f-163ac74deef2','GPU-30283973-140d-9e7b-e780-0fcc487dec54']
"""
def call(code,timeout=90):
 command='/usr/bin/python3 -B -c '+shlex.quote(GUARD+code)
 p=subprocess.run([str(W/'reverse_maclink/maclink-env/bin/python'),'-B',str(W/'reverse_maclink/maclink.py'),'run','--','/usr/bin/ssh','-S','/Users/shmelev/Downloads/reverse_maclink_daily/.gnnm51-control.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','shmelev@192.168.18.51',command],cwd=W,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 return p.stdout.strip()
