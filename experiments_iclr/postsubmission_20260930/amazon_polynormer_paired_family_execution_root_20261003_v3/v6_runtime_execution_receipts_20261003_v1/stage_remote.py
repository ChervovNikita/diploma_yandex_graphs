"""Exact authorized source/metadata staging, exclusive writes, no launch."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import os
import subprocess
import sys

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
os.chdir(REPO)
gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
assert Path.cwd() == REPO and gpu.returncode == 0 and gpu.stdout.strip() == 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
value = json.load(sys.stdin)
prepared = []
for item in value['files']:
    row = item['descriptor']; relative = Path(row['path']); p = PHASE/relative
    assert not relative.is_absolute() and '..' not in relative.parts and p.is_relative_to(PHASE)
    assert p.suffix in ('.json','.jsonl','.py','.md','.txt','.log','.sh','.html','.patch','.diff')
    assert not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(PHASE))
    raw = base64.b64decode(item['base64'],validate=True);raw.decode('utf8')
    assert hashlib.sha256(raw).hexdigest() == row['sha256'] and len(raw) == row['bytes']
    exists = p.exists()
    if exists:
        actual = p.read_bytes()
        assert hashlib.sha256(actual).hexdigest() == row['sha256'] and len(actual) == row['bytes'],str(p)
    prepared.append((p,raw,row,exists))
for row in value['opaque']:
    relative = Path(row['path']);p = PHASE/relative
    assert not relative.is_absolute() and '..' not in relative.parts and p.suffix in ('.pt','.npz','.npy','.pth')
    assert p.is_file() and p.stat().st_size == row['bytes'] and not p.is_symlink()
created = []
existing = []
for p,raw,row,exists in prepared:
    if exists:
        existing.append(row);continue
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    actual = p.read_bytes()
    assert hashlib.sha256(actual).hexdigest() == row['sha256'] and len(actual) == row['bytes']
    created.append(row)
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'route':{'repository':str(Path.cwd()),'single_GPU_UUID':gpu.stdout.strip()},'existing_exact_files':existing,'created_exact_files':created,'opaque_existing_files_stat_checked_not_opened':value['opaque'],'existing_files_overwritten':False,'numerical_or_GPU_execution':False,'qualifier_fit_or_scoring_launch':False,'source_stage_complete':True},indent=2))
