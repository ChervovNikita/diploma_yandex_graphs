"""Read-only stdlib custody/syntax verification. Never imports Torch or DGL."""
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
for name in ('hgt_private.py','cpu_fixtures.py','verify_source.py'):
    source = (packet/name).read_text()
    ast.parse(source); compile(source,str(packet/name),'exec')
sealed = (packet/'SEAL.json').exists()
payload_count = None
if sealed:
    data = (packet/'MANIFEST.json').read_bytes()
    seal = json.loads((packet/'SEAL.json').read_text())
    assert sha(data)==seal['manifest_sha256']
    manifest = json.loads(data)
    for row in manifest['payload']:
        body = (packet/row['path']).read_bytes()
        assert sha(body)==row['sha256'] and len(body)==row['bytes'],row['path']
    payload_count = len(manifest['payload'])
print(json.dumps(dict(status='PASS',sealed=sealed,payload_count=payload_count,
                     bound_nonnumeric_inputs_unchanged=len(provenance['inputs']),
                     syntax_sources=3,Torch_DGL_or_numeric_execution=False)))
