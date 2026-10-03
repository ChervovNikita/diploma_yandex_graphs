"""Basic syntax and source custody only; no Torch, labels, dataset or execution."""
import ast
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
provenance = json.loads((packet / 'PROVENANCE.json').read_text())
for row in provenance['inputs']:
    data = (packet.parent / row['path']).read_bytes()
    assert sha(data) == row['sha256'] and len(data) == row['bytes'], row['path']
for name in ('acm_inputs.py', 'families_acm.py', 'train_acm.py', 'verify_source.py'):
    data = (packet / name).read_text()
    ast.parse(data)
    compile(data, str(packet / name), 'exec')
manifest = (packet / 'MANIFEST.json').read_bytes()
assert sha(manifest) == json.loads((packet / 'SEAL.json').read_text())['manifest_sha256']
for row in json.loads(manifest)['payload']:
    data = (packet / row['path']).read_bytes()
    assert sha(data) == row['sha256'] and len(data) == row['bytes'], row['path']
print(json.dumps(dict(status='PASS', syntax_sources=4, pinned_inputs=len(provenance['inputs']),
    Torch_dataset_label_remote_GPU_or_training_execution=False, ACM_checkpoint_replay_execution_verified=False)))
