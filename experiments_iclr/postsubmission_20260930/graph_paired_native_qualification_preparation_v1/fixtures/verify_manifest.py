"""Stdlib-only seal and nonnumeric original source/metadata verification."""
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest_bytes = (packet/'MANIFEST.json').read_bytes()
manifest = json.loads(manifest_bytes)
seal = json.loads((packet/'SEAL.json').read_text())
assert seal['manifest_sha256'] == sha(manifest_bytes)
assert seal['payload_count'] == len(manifest['payload'])
for row in manifest['payload']:
    data = (packet/row['path']).read_bytes()
    assert sha(data) == row['sha256'] and len(data) == row['bytes'], row['path']
bound = json.loads((packet/'BOUND_INPUTS.json').read_text())
count = 0
for row in bound['original_records']:
    if Path(row['path']).suffix in ('.npy','.npz','.pt','.pth'):
        continue
    relative = Path(row['path']).relative_to(bound['canonical_research_root'])
    assert sha((packet.parent/relative).read_bytes()) == row['sha256'], str(relative)
    count += 1
print(json.dumps(dict(passed=True, payload_count=len(manifest['payload']),
                      nonnumeric_originals_unchanged=count, numeric_originals_opened=False,
                      manifest_sha256=sha(manifest_bytes))))
