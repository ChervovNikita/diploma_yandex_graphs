"""Stdlib-only custody and syntax verification, with known HGEN defects retained."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    packet = Path(__file__).resolve().parent
    seal = json.loads((packet/'SEAL.json').read_text()); manifest = (packet/'MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest()==seal['manifest_sha256']
    for row in json.loads(manifest)['payload']:
        data = (packet/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
    provenance = json.loads((packet/'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        data = (packet.parent/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
    own = sorted(packet.glob('*.py'))
    for path in own:
        ast.parse(path.read_text(),str(path))
    hgen = json.loads((packet/'HGEN_QUALIFICATION.json').read_text())
    for record in hgen['preserved_parse_failures']:
        try:
            ast.parse((packet/record['path']).read_text())
            raise AssertionError('Preserved source defect disappeared')
        except SyntaxError as error:
            assert type(error).__name__==record['type'] and error.lineno==record['line'] and error.msg==record['message']
    print(json.dumps(dict(status='PASS',payloads=len(json.loads(manifest)['payload']),original_nonnumeric_inputs_unchanged=len(provenance['inputs']),
        own_syntax_sources=len(own),preserved_HGEN_parse_failures=len(hgen['preserved_parse_failures']),
        Torch_real_data_labels_remote_GPU_or_training_execution=False)))


if __name__=='__main__':
    main()
