from pathlib import Path
import json,hashlib,socket,subprocess
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,stdin=subprocess.DEVNULL).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
d={}
for name in ['private_sheaf_identity_centered_fast_prior_extension_source_20261009_v1','bsnn_richer_cayley_d2_f32_L4_full_input_engineering_qualifier_source_20261009_v1','bsnn_richer_cayley_d2_f32_L4_normal_host_launch_source_20261009_v1']:
 rows=json.loads((P/name/'SOURCE_BINDINGS.json').read_text())['files'];missing=[];different=[];matched=0
 for row in rows:
  path=P/row['path'];assert path.resolve().is_relative_to(P.resolve())
  if not path.exists():missing.append(row);continue
  b=path.read_bytes()
  if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:different.append(row['path'])
  else:matched+=1
 d[name]=dict(matched=matched,missing=missing,different=different)
print(json.dumps(d),flush=True)
