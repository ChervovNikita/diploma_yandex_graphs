import json,base64,hashlib,socket,subprocess,sys
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
S=P/'learnable_internal_be_resource_qualifier_source_20261007_v1'
for r in json.loads(sys.stdin.read()):
 p=S/r['path'];assert p.resolve().is_relative_to(S)
 d=base64.b64decode(r['data']);assert len(d)==r['bytes'] and hashlib.sha256(d).hexdigest()==r['sha256']
 if p.exists():assert p.is_file() and p.read_bytes()==d
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(d)
assert hashlib.sha256((S/'MANIFEST.json').read_bytes()).hexdigest()=='3668747a8e27ab7eaa3754f56744697d6a71cb1e2e3a05284175362e64833f50'
print(json.dumps({'source_staged':True,'execution':False,'scientific_fits':0,'files':14,'hostname':socket.gethostname(),'qualifier_manifest_sha256':hashlib.sha256((S/'MANIFEST.json').read_bytes()).hexdigest()}))
