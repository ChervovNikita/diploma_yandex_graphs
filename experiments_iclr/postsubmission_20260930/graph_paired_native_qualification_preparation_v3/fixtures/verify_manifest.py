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
for version, manifest_sha256, seal_sha256 in [
        ('v1', 'f21826f941d39dd22a8f12cfb28168067d052c3541612e9de97d5835bbd923e9',
               '2aedcc0c82054ffda5e632f28ca985a8caed5ac46e0d8efcdde2b6a6b98092e6'),
        ('v2', '07665a4d75613dcfc6c7f6b7582982d5a297de0a7e98150cefe549ab12d2bde8',
               'bd7371eb29d83c6b9eaf60aadc5ff285522eee35ac43d147fa8b35ffa28fa283')]:
    previous = packet.parent/('graph_paired_native_qualification_preparation_'+version)
    previous_manifest = (previous/'MANIFEST.json').read_bytes()
    previous_seal = (previous/'SEAL.json').read_bytes()
    assert sha(previous_manifest) == manifest_sha256
    assert sha(previous_seal) == seal_sha256
    assert json.loads(previous_seal)['manifest_sha256'] == sha(previous_manifest)
    for row in json.loads(previous_manifest)['payload']:
        data = (previous/row['path']).read_bytes()
        assert sha(data) == row['sha256'] and len(data) == row['bytes'], row['path']
print(json.dumps(dict(passed=True, payload_count=len(manifest['payload']),
                      nonnumeric_originals_unchanged=count, numeric_originals_opened=False,
                      prior_preparation_v1_unchanged=True,
                      prior_preparation_v2_unchanged=True,
                      manifest_sha256=sha(manifest_bytes))))
