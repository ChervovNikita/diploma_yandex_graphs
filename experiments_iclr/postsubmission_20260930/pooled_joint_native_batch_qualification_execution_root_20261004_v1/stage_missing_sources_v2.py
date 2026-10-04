"""Bounded byte chunks for exact source metadata, preserving v1 failure."""
from pathlib import Path
import base64
import hashlib
import json
import zlib
from prepare_native_run import HERE, PHASE, REMOTE_REPO, REMOTE_PHASE, REMOTE, run, write

transport = json.loads((HERE / 'SOURCE_STAGE_FAILURE_DIAGNOSTIC_TRANSPORT.json').read_text())
assert transport['exit_code'] == 0
outer = json.loads(transport['stdout'])
assert outer['exit_code'] == 0
diagnostic = json.loads(outer['stdout'])
inventory = {r['path']: r for r in json.loads((HERE / 'SOURCE_STAGE_INVENTORY.json').read_text())}
payload = []
for issue in diagnostic['issues']:
    assert issue['present'] is False and issue['symlink'] is False
    row = inventory[issue['path']]
    raw = (PHASE / row['path']).read_bytes()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    payload.append(dict(row, data=base64.b64encode(raw).decode()))
compressed = zlib.compress(json.dumps(payload).encode(), 9)
encoded = base64.b64encode(compressed).decode()
chunks = [encoded[n:n+40000] for n in range(0,len(encoded),40000)]
remote_chunks = REMOTE / 'missing_source_metadata_chunks_v2'
receipts = []
for number, chunk in enumerate(chunks,1):
    code = 'from pathlib import Path\nimport hashlib,json,os\n'
    code += 'repo=Path(' + repr(str(REMOTE_REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(remote_chunks)) + ')\n'
    code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide" and root.resolve().is_relative_to(phase)\n'
    code += 'root.mkdir(exist_ok=True);p=root/' + repr('part%03d.b64' % number) + ';b=' + repr(chunk) + '.encode()\n'
    code += 'with p.open("xb") as f:f.write(b)\n'
    code += 'print(json.dumps(dict(status="SOURCE_METADATA_CHUNK_STAGED",bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),scientific_execution=False)))\n'
    result = run('pooled_native_metadata_v2_chunk%03d_20261004' % number,code)
    assert result['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    receipts.append(result)
    print(json.dumps(dict(chunk=number,total=len(chunks),bytes=len(chunk))),flush=True)
code = 'from pathlib import Path\nimport base64,hashlib,json,os,zlib\n'
code += 'repo=Path(' + repr(str(REMOTE_REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(remote_chunks)) + ')\n'
code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'encoded=b"".join((root/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks)+1) + '));compressed=base64.b64decode(encoded,validate=True)\n'
code += 'assert hashlib.sha256(compressed).hexdigest()==' + repr(hashlib.sha256(compressed).hexdigest()) + '\n'
code += 'rows=json.loads(zlib.decompress(compressed))\n'
code += "for r in rows:\n p=(phase/r['path']).resolve();assert p.is_relative_to(phase);b=base64.b64decode(r['data'],validate=True);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']\n if p.exists():assert not p.is_symlink() and p.read_bytes()==b\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open('xb') as f:f.write(b)\n"
code += "print(json.dumps(dict(status='PASS_EXACT_SOURCE_METADATA_STAGE',files=len(rows),bytes=sum(r['bytes'] for r in rows),scientific_execution=False)))\n"
result = run('pooled_native_metadata_v2_join_20261004',code)
write(HERE/'MISSING_SOURCE_STAGE_RECEIPTS_V2.json',dict(status=result['status'],final=result,chunks=receipts,prior_failures_preserved=True,scientific_retry=False))
print(json.dumps(result))
