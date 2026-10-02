"""Read-only stdlib verification of this source snapshot and active sources."""
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest_bytes = (packet / 'MANIFEST.json').read_bytes()
manifest = json.loads(manifest_bytes)
seal = json.loads((packet / 'SEAL.json').read_bytes())
assert seal['manifest_sha256'] == sha(manifest_bytes), 'manifest seal mismatch'
assert seal['payload_count'] == len(manifest['payload']), 'payload count mismatch'
assert len({row['path'] for row in manifest['payload']}) == len(manifest['payload'])
for row in manifest['payload']:
    data = (packet / row['path']).read_bytes()
    assert sha(data) == row['sha256'], row['path']
    assert len(data) == row['bytes'], row['path']
bindings = json.loads((packet / 'ACTIVE_SOURCE_BINDINGS.json').read_bytes())
active = packet.parent / bindings['active_packet']
for row in bindings['files']:
    assert sha((active / row['path']).read_bytes()) == row['sha256'], row['path']
print(json.dumps(dict(passed=True, payload_count=len(manifest['payload']),
                      active_files_unchanged=len(bindings['files']),
                      manifest_sha256=sha(manifest_bytes))))
