from pathlib import Path
import json,hashlib,socket,subprocess
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
counts={}
for name in ['private_sheaf_identity_centered_fast_prior_extension_source_20261009_v1','bsnn_richer_cayley_d2_f32_L4_full_input_engineering_qualifier_source_20261009_v1','bsnn_richer_cayley_d2_f32_L4_normal_host_launch_source_20261009_v1']:
 rows=json.loads((P/name/'SOURCE_BINDINGS.json').read_text())['files']
 for row in rows:
  path=(P/row['path']).resolve(strict=True);assert path.is_relative_to(P)
  b=path.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
 counts[name]=len(rows)
print(json.dumps(dict(exact_source_closure_passed=True,counts=counts,numerical_work=False)))
