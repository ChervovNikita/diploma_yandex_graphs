"""Publish fresh handoffs/input receipts; audit against frozen root controller."""
import base64
import hashlib
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE = Path(__file__).resolve().parents[1]
HERE = PACKAGE / "preparation"
CASES = [("polyformer_allocation_20261012_v2", "massive_polyformer_packet_20261012_v2", "POLYFORMER_BOUNDARY_V2_FINAL_HANDOFF_20261012.md"),
         ("polyformer_practical_allocation_20261012_v1", "massive_polyformer_practical_packet_20261012_v1", "POLYFORMER_PRACTICAL_V1_FINAL_HANDOFF_20261012.md")]

REMOTE = GUARD + r'''
payload=json.loads(base64.b64decode(sys.argv[1]))
for row in payload:
 stage=phase/row['stage'];assert stage.is_dir()
 for name in ['FINAL_HANDOFF.md','MASTER_INPUT_ENTRY.json']:
  raw=base64.b64decode(row['files'][name]['data']);assert hashlib.sha256(raw).hexdigest()==row['files'][name]['sha256']
  p=stage/name;assert not p.exists();p.write_bytes(raw);p.chmod(0o444)
controller=phase/'master_controller_source_20261012_v1/runtime';assert (controller/'compile_master_plan.py').is_file()
sys.path.insert(0,str(controller))
from compile_master_plan import load_packet
proof=[]
for row in payload:
 stage=phase/row['stage'];entry=json.loads((stage/'MASTER_INPUT_ENTRY.json').read_text())
 packet=load_packet(phase,entry)
 result={'packet_id':entry['id'],'root_frozen_controller_metadata_admitted':True,'plan_hash':packet['plan_hash'],
  'study_count':len(packet['studies']),'typed_counts':packet['counts'],
  'all_actual_reference_cohorts_matched':all(v['matched'] for values in packet['references'].values() for v in values),
  'controller':str(controller.relative_to(phase)),'science_fits':0,'GO_created':False,
  'resource_qualification':'PROVISIONAL_REQUIRES_ROOT_QUALIFICATION'}
 p=stage/'ROOT_CONTROLLER_METADATA_CHECK.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');p.chmod(0o444)
 for path in [stage/'matrix/PLAN.json',stage/'matrix/SUMMARY.json',stage/'FINAL_VERIFICATION.json']:
  path.chmod(0o444)
 for path in (stage/'matrix/studies').iterdir():path.chmod(0o444)
 proof.append({'stage':row['stage'],'result':result,'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
  'handoff_sha256':hashlib.sha256((stage/'FINAL_HANDOFF.md').read_bytes()).hexdigest()})
 del packet
print(json.dumps(proof))
'''


def main():
    payload = []
    for label, stage, handoff in CASES:
        files = {}
        for name, path in (("FINAL_HANDOFF.md", HERE / handoff), ("MASTER_INPUT_ENTRY.json", HERE / label / "MASTER_INPUT_ENTRY.json")):
            raw = path.read_bytes(); files[name] = {"data": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest()}
        payload.append({"stage": stage, "files": files})
    out = HERE / "polyformer_final_handoff_20261012_v1"
    if out.exists(): raise FileExistsError("Preserve prior attempt")
    out.mkdir()
    result = invoke(REMOTE, [base64.b64encode(json.dumps(payload).encode()).decode()], out, "ROOT_CONTROLLER_CHECK_AND_HANDOFF", 900)
    if result["exit_code"] != 0: print(result["stdout"], result["stderr"]); raise SystemExit(1)
    proof = json.loads(next(line for line in result["stdout"].splitlines() if line.startswith('[')))
    (out / "FINAL_PROOF.json").write_text(json.dumps(proof, indent=2) + "\n")
    for row, (label, stage, handoff) in zip(proof, CASES):
        mirror = HERE / label / "allocation_artifacts" / stage
        (mirror / "ROOT_CONTROLLER_METADATA_CHECK.json").write_text(json.dumps(row["result"], indent=2) + "\n")
        (mirror / "FINAL_HANDOFF.md").write_bytes((HERE / handoff).read_bytes())
        (mirror / "MASTER_INPUT_ENTRY.json").write_bytes((HERE / label / "MASTER_INPUT_ENTRY.json").read_bytes())
    print(json.dumps([{k: v for k, v in row.items() if k != "result"} | {"packet_id": row["result"]["packet_id"],
        "admitted": row["result"]["root_frozen_controller_metadata_admitted"], "counts": row["result"]["typed_counts"]["totals"]}
        for row in proof], indent=2))


if __name__ == "__main__": main()
