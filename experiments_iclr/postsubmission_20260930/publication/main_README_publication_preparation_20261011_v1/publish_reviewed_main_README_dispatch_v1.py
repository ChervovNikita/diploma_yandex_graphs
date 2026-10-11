"""Publish root-reviewed progress README only on main, using secure interactive key unlock."""
import shlex,subprocess
REMOTE=r"""
import os,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');os.chdir(r)
os.environ['GNNM_SSH_DESTINATION']='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
p=r/'experiments_iclr/postsubmission_20260930'
f=p/'publication/main_README_publication_preparation_20261011_v1/publish_main_only_README.py'
sys.argv=[str(f),'--parent','5fc577ece84f5c894797c9099ce43e8772b283c8','--readme',str(p/'publication/README_progress_update_20261011_v1/README_MAIN.md'),'--readme-sha256','b76bea6c4c516960ea71eb2054f18d0b98a70f74ee75399c44d53d7755f1dc26','--original-readme-sha256','5ec06132b718a51b2362573fa0bde08c3deb96a0efa37ff07468f0db8bf1579b','--receipt',str(p/'publication/main_README_publication_preparation_20261011_v1/MAIN_PUSH_RECEIPT_V1.json'),'--socket-name','rm44.sock','--publish']
exec(compile(f.read_text(),str(f),'exec'),{'__name__':'__main__','__file__':str(f)})
"""
a=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
raise SystemExit(subprocess.call(a))
