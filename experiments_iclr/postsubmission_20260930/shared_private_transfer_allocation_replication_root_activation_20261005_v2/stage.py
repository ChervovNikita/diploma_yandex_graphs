"""Stage a sealed disabled operational packet and exact small source/metadata dependencies."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'shared_private_transfer_allocation_replication_preparation_20261005_v2'
EXPECTED_MANIFEST = 'c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(SOURCE / 'MANIFEST.json') == EXPECTED_MANIFEST
manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
paths = {SOURCE / 'MANIFEST.json', SOURCE / 'SEAL.json'}
for row in manifest['files']:
    path = SOURCE / row['path']
    assert path.resolve().is_relative_to(SOURCE) and not path.is_symlink()
    assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    paths.add(path)
pending = list(paths)
seen_json = set()
unavailable_local = []
text_suffixes = {'.json', '.py', '.md', '.patch', '.txt', '.sha256', '.jsonl'}
closed_names = {'FREEZE.json', 'CONFIG.json', 'EXTERNAL_ANCHORS.json', 'HISTORY.json', 'SELECTION.json'}
while pending:
    path = pending.pop()
    if path.suffix != '.json' or path.name in closed_names or path in seen_json:
        continue
    seen_json.add(path)
    value = json.loads(path.read_text())
    def visit(node):
        if isinstance(node, dict):
            if isinstance(node.get('path'), str) and isinstance(node.get('sha256'), str) and len(node['sha256']) == 64:
                rel = Path(node['path'])
                if not rel.is_absolute() and '..' not in rel.parts:
                    q = PHASE / rel
                    if q.suffix in text_suffixes and q.name not in closed_names:
                        if q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(PHASE) and q.stat().st_size < 2_000_000 and sha(q) == node['sha256']:
                            if q not in paths:
                                paths.add(q); pending.append(q)
                        else:
                            unavailable_local.append(dict(path=str(rel), sha256=node['sha256'], remote_validation_required=True))
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)
    visit(value)
rows = []
for path in sorted(paths):
    data = path.read_bytes()
    assert len(data) < 2_000_000 and not path.is_symlink()
    rows.append(dict(path=str(path.relative_to(PHASE)), bytes=len(data), sha256=sha(path), data=base64.b64encode(data).decode()))
REMOTE = r'''
from datetime import datetime,timezone
import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
packet=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
prepared=[]
for row in packet['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 path=phase/rel;assert path.resolve().is_relative_to(phase)
 assert not any(q.is_symlink() for q in [path,*path.parents] if q.is_relative_to(phase))
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and len(data)<2_000_000 and hashlib.sha256(data).hexdigest()==row['sha256']
 if path.exists():assert path.is_file() and path.read_bytes()==data,'Refuse to overwrite changed existing source/metadata: '+str(rel)
 prepared.append((path,data))
for path,data in prepared:
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as handle:handle.write(data)
observed=[dict(path=str(path.relative_to(phase)),bytes=len(data),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,data in prepared]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),stage_only=True,scientific_children_started=0,prepared_source_imports=0,
                     scores_read=False,TEST_access=False,files=observed,existing_numeric_sources_overwritten=False)))
'''
packet = dict(files=rows)
command = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
           'cd '+REPO+' && exec python3 -I -B -c '+shlex.quote(REMOTE)]
r = subprocess.run(command, input=base64.b64encode(zlib.compress(json.dumps(packet).encode(),9)).decode(),
                   capture_output=True,text=True,timeout=55)
receipt = dict(exit_code=r.returncode, stderr=r.stderr, operational_manifest_sha256=EXPECTED_MANIFEST,
               remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
               unavailable_local_dependencies=unavailable_local)
if r.returncode == 0:
    receipt['result'] = json.loads(r.stdout)
else:
    receipt['stdout'] = r.stdout
with (HERE / 'STAGE_RECEIPT.json').open('x') as handle:
    json.dump(receipt,handle,indent=2);handle.write('\n')
assert r.returncode == 0, r.stderr
print(json.dumps(dict(staged_files=len(rows), staged_bytes=sum(row['bytes'] for row in rows),
                     scientific_children_started=0, source_imports=0, unavailable_local=len(unavailable_local))))
