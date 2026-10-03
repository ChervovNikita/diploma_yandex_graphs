"""Stdlib custody/syntax verifier. Does not import Torch or open any dataset."""
import ast
import hashlib
import json
from pathlib import Path
packet = Path(__file__).resolve().parent
sha = lambda data:hashlib.sha256(data).hexdigest()
provenance = json.loads((packet/'PROVENANCE.json').read_text())
for row in provenance['inputs']:
    data = (packet.parent/row['path']).read_bytes()
    assert sha(data)==row['sha256'] and len(data)==row['bytes'],row['path']
names = ('dblp_inputs.py','families.py','train_dblp.py','freeze_splits.py','stdlib_fixtures.py','cpu_training_fixtures.py','verify_source.py')
for name in names:
    data = (packet/name).read_text(); ast.parse(data); compile(data,str(packet/name),'exec')
sealed = (packet/'SEAL.json').exists()
if sealed:
    data = (packet/'MANIFEST.json').read_bytes()
    assert sha(data)==json.loads((packet/'SEAL.json').read_text())['manifest_sha256']
    for row in json.loads(data)['payload']:
        body = (packet/row['path']).read_bytes()
        assert sha(body)==row['sha256'] and len(body)==row['bytes'],row['path']
print(json.dumps(dict(status='PASS',sealed=sealed,nonnumeric_originals_unchanged=len(provenance['inputs']),
                     syntax_sources=len(names),Torch_dataset_label_remote_GPU_or_training_execution=False)))
