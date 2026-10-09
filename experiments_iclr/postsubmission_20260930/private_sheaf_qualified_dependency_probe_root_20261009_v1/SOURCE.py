from pathlib import Path
import json,sys,os,socket,subprocess,importlib,importlib.metadata,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R)
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
import torch_householder
roles=json.loads((P/'private_sheaf_train_valid_roles_allocation_root_20261009_v1/ROLE.json').read_text())
out={'python':sys.executable,'package_versions':{k:importlib.metadata.version(k) for k in ('torch','numpy','scikit-learn','torch-geometric','torch-sparse','torch-scatter','torch-householder')},'torch_householder_module':str(Path(torch_householder.__file__).resolve()),'roles_metadata':roles,'model_or_array_or_scientific_score_reads':False}
print(json.dumps(out))
