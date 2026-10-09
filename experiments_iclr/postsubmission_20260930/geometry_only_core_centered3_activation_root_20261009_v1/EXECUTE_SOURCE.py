from pathlib import Path
import os,subprocess
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');A=R/'experiments_iclr/postsubmission_20260930/geometry_only_core_centered3_activation_root_20261009_v1'
os.chdir(R)
subprocess.run(['/usr/bin/python3','-I','-S','-B',str(A/'CONTINUE_SOURCE.py'),'--execute','--release',str(A/'CONTINUATION.json'),'--published-commit','40b4a4efd4c219e5d7cdc57146df005954a9faf9'],check=True)
