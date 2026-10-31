"""Freeze proposal/feasibility source and small receipts, without execution."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path.insert(0, str(PHASE / "training_pipeline_v2_20261012/preparation"))
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

REMOTE = GUARD + r'''
encoded=(phase/(sys.argv[1]+'_transport')/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
roots=payload['roots'];assert all(not (phase/rel).exists() for rel in roots)
for rel in roots:
 target=phase/rel;assert target.resolve().is_relative_to(phase);target.mkdir()
files=[]
for row in payload['files']:
 target=phase/row['path'];assert target.resolve().is_relative_to(phase) and target.parent.name in roots
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target.write_bytes(raw);target.chmod(0o444);files.append({k:v for k,v in row.items() if k!='data'})
manifest={'schema':'shared-replay-feasibility-proposal-frozen-manifest-v1','files':files,
          'source_epochs_generated':0,'scientific_fits':0,'optimizer_updates':0,'packets_modified':False,
          'actual_input_equivalence_proven':False,'resource_admission_credit':False,'GO':False}
out=phase/roots[0]/'MANIFEST.json';out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');out.chmod(0o444)
print(json.dumps({'manifest':{'path':str(out.relative_to(phase)),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},'files':len(files),'scientific_fits':0,'source_epochs_generated':0,'actual_input_equivalence_proven':False}))
'''


if __name__ == "__main__":
    roots = [HERE.name, "massive_storage_feasibility_20261012_v1", "massive_storage_feasibility_20261012_v2"]
    files = []
    for root in roots:
        for path in sorted((PHASE / root).glob("*")):
            if not path.is_file() or path.name in {"FREEZE_RECEIPT.json", "ALLOCATION_FREEZE.json"}:
                continue
            raw = path.read_bytes()
            files.append({"path": str(path.relative_to(PHASE)), "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    encoded = base64.b64encode(zlib.compress(json.dumps({"roots": roots, "files": files}).encode(), 9)).decode()
    transport_name = HERE.name + "_metadata_v2"
    for offset in range(0, len(encoded), 48000):
        part = invoke(TRANSFER, [transport_name, str(offset), encoded[offset:offset+48000]],
                      HERE, "TRANSFER_"+str(offset), 60)
        assert part["exit_code"] == 0, "Preserve partial proposal transfer"
    result = invoke(REMOTE, [transport_name, hashlib.sha256(encoded.encode()).hexdigest()],
                    HERE, "ALLOCATION_FREEZE_ATTEMPT2", 180)
    assert result["exit_code"] == 0, "Preserve metadata freezing failure"
    receipt = json.loads(result["stdout"])
    (HERE / "FREEZE_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))
