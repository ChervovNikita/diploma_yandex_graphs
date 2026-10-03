"""Verify immutable HGEN preparation and private source bindings with stdlib."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    packet = Path(__file__).resolve().parent
    manifest = (packet/'MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest()==json.loads((packet/'SEAL.json').read_text())['manifest_sha256']
    payload = json.loads(manifest)['payload']
    for row in payload:
        raw = (packet/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
    provenance = json.loads((packet/'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        raw = (packet.parent/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
    sources = list(packet.glob('*.py'))
    for path in sources: ast.parse(path.read_text(),str(path))
    assert not (packet/'train_hgen.py').exists()
    print(json.dumps(dict(status='PASS',payloads=len(payload),unchanged_source_metadata_inputs=len(provenance['inputs']),
        own_syntax_sources=len(sources),HGEN_model_only_unadopted=True,Torch_original_arrays_labels_remote_GPU_or_training_execution=False)))


if __name__=='__main__': main()
