"""Restore only two missing retained source excerpts with their pinned bytes."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
spec = importlib.util.spec_from_file_location('_own_allocation_control', HERE/'allocation_metadata_control.py')
control = importlib.util.module_from_spec(spec); spec.loader.exec_module(control)
ALLOWED = (
    'graph_link_competing_family_source_20261003_v1/excerpts/ncn__README.md.txt',
    'link_covariance_idea_v1/sources/buddy_readme.md',
)


def restore():
    binding = json.loads((PHASE/'benchmark_expansion_20261015_v1/link_policy_v1/BINDINGS.json').read_text())
    sources = {row['path_relative_to_phase']: row for row in binding['source_files']}
    packet = []
    for name in ALLOWED:
        raw = (PHASE/name).read_bytes(); expected = sources[name]
        assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
        packet.append({'relative': name, 'base64': base64.b64encode(raw).decode()})
    code = '''import base64,hashlib,json
from pathlib import Path
phase=Path(PHASE_LITERAL)
binding=json.loads((phase/"benchmark_expansion_20261015_v1/link_policy_v1/BINDINGS.json").read_text())
sources={row["path_relative_to_phase"]:row for row in binding["source_files"]}
packet=PACKET_LITERAL
rows=[]
for row in packet:
 assert row["relative"] in ALLOWED_LITERAL
 target=phase/row["relative"]
 assert target.resolve().is_relative_to(phase) and not target.is_symlink()
 raw=base64.b64decode(row["base64"]); expected=sources[row["relative"]]
 assert len(raw)==expected["bytes"] and hashlib.sha256(raw).hexdigest()==expected["sha256"]
 existed=target.exists()
 if existed:
  assert target.read_bytes()==raw, "Existing retained source differs; do not overwrite"
 else:
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open("xb") as stream: stream.write(raw)
 rows.append({"path":str(target),"existed_before":existed,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
print(json.dumps({"guard":"passed","restored_exact_retained_source_bytes":rows,"fits_started":0,"models_constructed":0,"source_policy_changed":False}))
'''.replace('PHASE_LITERAL', repr(control.guarded.REMOTE_ROOT+'/experiments_iclr/postsubmission_20260930')).replace('PACKET_LITERAL', repr(packet)).replace('ALLOWED_LITERAL', repr(ALLOWED))
    result = json.loads(control.guarded.call(code, timeout=60))
    (HERE/'SOURCE_CUSTODY_REPAIR_RECEIPT.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(restore(), sort_keys=True))
