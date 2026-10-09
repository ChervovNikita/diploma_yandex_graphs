from pathlib import Path
import json,sys,socket,subprocess,base64,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,stdin=subprocess.DEVNULL).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True,stdin=subprocess.DEVNULL).strip()=='2e913233f2ffac3e1b43d12f5559b47a95ab0471'
row=json.loads(sys.stdin.read());assert row['path']=='bsnn_full_three_seed_baseline_activation_root_20261009_v1/LAUNCH_SOURCE.py'
path=P/row['path'];assert not path.exists() and path.parent.is_dir()
b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
with path.open('xb') as f:f.write(b)
print(json.dumps(dict(staged_exact_missing_ancestry_source=True,numerical_source_changed=False,numerical_work=False)),flush=True)
