"""Copy only absent authenticated source/receipt files; no numerical retry."""
from pathlib import Path
import base64
import hashlib
import json
import zlib
from prepare_native_run import HERE, PHASE, REMOTE_REPO, REMOTE_PHASE, run, write

transport = json.loads((HERE / 'SOURCE_STAGE_FAILURE_DIAGNOSTIC_TRANSPORT.json').read_text())
assert transport['exit_code'] == 0
outer = json.loads(transport['stdout'])
assert outer['exit_code'] == 0
diagnostic = json.loads(outer['stdout'])
rows = {r['path']: r for r in json.loads((HERE / 'SOURCE_STAGE_INVENTORY.json').read_text())}
payload = []
for issue in diagnostic['issues']:
    assert issue['present'] is False and issue['symlink'] is False
    row = rows[issue['path']]
    raw = (PHASE / row['path']).read_bytes()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    payload.append(dict(row, data=base64.b64encode(raw).decode()))

def pack(value):
    return base64.b64encode(zlib.compress(json.dumps(value).encode(), 9)).decode()

chunks, current = [], []
for row in payload:
    candidate = current + [row]
    if current and len(pack(candidate)) > 55000:
        chunks.append(current)
        current = [row]
    else:
        current = candidate
if current:
    chunks.append(current)

receipts = []
for index, chunk in enumerate(chunks, 1):
    packed = pack(chunk)
    assert len(packed) < 90000
    code = 'from pathlib import Path\nimport base64,hashlib,json,os,zlib\n'
    code += 'repo=Path(' + repr(str(REMOTE_REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ')\n'
    code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
    code += 'rows=json.loads(zlib.decompress(base64.b64decode(' + repr(packed) + ')))\n'
    code += "for r in rows:\n p=(phase/r['path']).resolve();assert p.is_relative_to(phase);b=base64.b64decode(r['data'],validate=True);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']\n if p.exists():assert not p.is_symlink() and p.read_bytes()==b\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open('xb') as f:f.write(b)\n"
    code += "print(json.dumps(dict(status='PASS_EXACT_SOURCE_METADATA_STAGE',files=len(rows),bytes=sum(r['bytes'] for r in rows),scientific_execution=False)))\n"
    receipts.append(run('pooled_native_missing_stage_20261004_part%02d' % index, code))
    print(json.dumps(dict(chunk=index, total_chunks=len(chunks), receipt=receipts[-1])), flush=True)
write(HERE / 'MISSING_SOURCE_STAGE_RECEIPTS.json', dict(status='PASS_SOURCE_METADATA_ONLY',receipts=receipts,original_failure_preserved=True,scientific_retry=False))
