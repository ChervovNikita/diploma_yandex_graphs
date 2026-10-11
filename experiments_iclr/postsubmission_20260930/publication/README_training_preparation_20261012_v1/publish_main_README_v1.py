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
import hashlib
raw=(r/'README.md').read_bytes();assert hashlib.sha256(raw).hexdigest()=='ff7123f38d55a2a9af2cad0b5e4ff2ef27e3733b8a2cd3c0a16960b3aa27a7dc'
target=p/'publication/README_training_preparation_20261012_v1/README_SOURCE_V1.md'
if target.exists():assert target.read_bytes()==raw
else:target.write_bytes(raw)
f=p/'publication/main_README_publication_preparation_20261011_v1/publish_main_only_README.py'
sys.argv=[str(f),'--parent','6185eca2d332743db73aa7c7d0e7e5f08afd3686','--readme',str(p/'publication/README_training_preparation_20261012_v1/README_SOURCE_V1.md'),'--readme-sha256','ff7123f38d55a2a9af2cad0b5e4ff2ef27e3733b8a2cd3c0a16960b3aa27a7dc','--original-readme-sha256','d021dd6d858de4b612b4c63fd28794cc9310a0b18611b16c75406dc79ab2bd0b','--receipt',str(p/'publication/README_training_preparation_20261012_v1/MAIN_PUSH_RECEIPT_V1.json'),'--socket-name','rm46.sock','--publish']
exec(compile(f.read_text(),str(f),'exec'),{'__name__':'__main__','__file__':str(f)})
"""
a=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
raise SystemExit(subprocess.call(a))
