"""Read-only stdlib hash verification; no candidate import, Torch or model."""
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest_bytes = (packet / 'MANIFEST.json').read_bytes()
manifest = json.loads(manifest_bytes)
seal = json.loads((packet / 'SEAL.json').read_bytes())
assert seal['manifest_sha256'] == sha(manifest_bytes)
assert seal['payload_count'] == len(manifest['payload'])
for row in manifest['payload']:
    data = (packet / row['path']).read_bytes()
    assert sha(data) == row['sha256'] and len(data) == row['bytes'], row['path']
bindings = json.loads((packet / 'SOURCE_BINDINGS.json').read_bytes())
phase = Path(bindings['path_root'])
for key, root_key in [('preserved_v1_files', 'preserved_v1_packet'), ('active_files', 'active_packet')]:
    for row in bindings[key]:
        assert sha((phase / bindings[root_key] / row['path']).read_bytes()) == row['sha256'], row['path']
native = json.loads((packet / 'NATIVE_SOURCE_BINDINGS.json').read_bytes())
for row in native['source_files']+native['source_receipts']+[native['graph_input_metadata_receipt']]:
    assert sha((phase / row['path']).read_bytes()) == row['sha256'], row['path']
print(json.dumps(dict(passed=True, payload_count=len(manifest['payload']),
                      v1_files_unchanged=len(bindings['preserved_v1_files']),
                      active_files_unchanged=len(bindings['active_files']),
                      manifest_sha256=sha(manifest_bytes))))
