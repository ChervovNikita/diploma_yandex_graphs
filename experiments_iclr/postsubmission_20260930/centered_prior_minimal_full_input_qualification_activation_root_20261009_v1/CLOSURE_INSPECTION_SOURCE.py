from pathlib import Path
import json,socket,subprocess,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
source=P/'private_sheaf_identity_centered_fast_prior_extension_source_20261009_v1/SOURCE_BINDINGS.json'
missing=[];different=[];matched=0
for row in json.loads(source.read_text())['files']:
 p=P/row['path'];assert p.resolve().is_relative_to(P)
 if not p.exists():missing.append(row);continue
 b=p.read_bytes()
 if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:different.append(row['path'])
 else:matched+=1
print(json.dumps(dict(matched=matched,missing=missing,different=different,numerical_work=False)))
