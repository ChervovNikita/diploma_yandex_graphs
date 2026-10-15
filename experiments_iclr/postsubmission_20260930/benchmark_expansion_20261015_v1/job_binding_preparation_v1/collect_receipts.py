"""Verify descriptor custody and collect only compact metadata receipts."""
import base64
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_own_control', HERE/'allocation_metadata_control.py')
control = importlib.util.module_from_spec(spec); spec.loader.exec_module(control)


def collect():
    code = '''import base64,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
output=root/"generated_v2"
phase=root.parents[1]
expansion=root.parent
def pin(path):
 raw=path.read_bytes()
 return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def write(path,value):
 raw=(json.dumps(value,sort_keys=True,indent=2)+"\\n").encode()
 if path.exists(): assert path.read_bytes()==raw
 else:
  with path.open("xb") as stream: stream.write(raw)
 return pin(path)
ready=json.loads((output/"READINESS.json").read_text())
index_path=output/"DESCRIPTOR_INDEX.json"
index_pin=pin(index_path)
assert index_pin==ready["artifacts"]["index"]
index=json.loads(index_path.read_text())["fits"]
assert len(index)==len({r["id"] for r in index})==2480
assert len(list((output/"jobs").glob("*.json")))==2480
samples={}; checked=0
for row in index:
 path=Path(row["descriptor"]["path"])
 assert path.resolve().is_relative_to(output/"jobs")
 assert path.is_file() and path.stat().st_size==row["descriptor"]["bytes"]
 # Every source recipe/procedure/track has a representative byte verification.
 procedure=row["id"].rsplit("/",1)[-1]
 key=(row["recipe_id"],row["track"],procedure)
 if key not in samples:
  assert pin(path)==row["descriptor"]
  value=json.loads(path.read_text())
  assert value["id"]==row["id"] and value["source_plan_entry"]["id"]==row["id"]
  assert value["source_plan_contract_hash"]==row["contract_hash"] and value["TEST_access"] is False
  samples[key]=True; checked+=1
for expected in ready["source_inputs"].values(): assert pin(Path(expected["path"]))=={k:expected[k] for k in ("path","bytes","sha256")}
link_root=expansion/"link"
link_manifest=json.loads((link_root/"MANIFEST.json").read_text())
link_rows=[pin(link_root/"MANIFEST.json")]
for row in link_manifest["files"]:
 if row["path"].endswith(".py") or row["path"]=="CANDIDATES.json":
  actual=pin(link_root/row["path"])
  assert all(actual[key]==row[key] for key in ("bytes","sha256"))
  link_rows.append(actual)
supplement={"schema":"predecessor-supplemental-link-dependency-receipts-v1","files":link_rows,"native_dependency_imported":False,"model_constructed":False,"fits_started":0}
supplement_pin=write(output/"SUPPLEMENTAL_LINK_DEPENDENCY_RECEIPTS.json",supplement)
verification={"schema":"predecessor-descriptor-custody-verification-v1","descriptor_files":2480,"unique_PLAN_fit_ids":2480,
 "all_index_paths_exist_and_sizes_match":True,"recipe_procedure_track_body_representative_SHA256_checks":checked,
 "frozen_PLAN_ROSTER_smoke_bytes_unchanged":True,"metadata_validate_job_passes":ready["metadata_validate_job_passes"],
 "metadata_validate_job_rejections":ready["metadata_validate_job_rejections"],"index_receipt":index_pin,
 "supplemental_link_dependencies":supplement_pin,"fits_started":0,"models_constructed":0,"TEST_predictions_or_metrics":False,
 "source_GNNM_original_scores_modified":False,"full_campaign_admitted":False,"server77_accessed":False}
write(output/"VERIFICATION.json",verification)
names=("CELL_BINDINGS.json","SOURCE_RECEIPTS.json","SUPPLEMENTAL_LINK_DEPENDENCY_RECEIPTS.json","VERIFICATION.json","READINESS.json")
rows=[]
for name in names:
 path=output/name; raw=path.read_bytes()
 assert len(raw)<262144
 rows.append({"name":name,"receipt":pin(path),"base64":base64.b64encode(raw).decode()})
print(json.dumps({"compact_files":rows,"full_descriptors_downloaded":0}))
'''.replace('ROOT_LITERAL', repr(control.REMOTE))
    result = json.loads(control.guarded.call(code, timeout=90))
    receipts = []
    for row in result['compact_files']:
        (HERE/row['name']).write_bytes(base64.b64decode(row['base64']))
        receipts.append(row['receipt'])
    (HERE/'COLLECTION_RECEIPT.json').write_text(json.dumps({'compact_remote_files': receipts, 'full_descriptors_downloaded': 0}, sort_keys=True, indent=2)+'\n')
    return {'compact_files':len(receipts), 'full_descriptors_downloaded':0}


if __name__ == '__main__':
    print(json.dumps(collect()))
