"""Hash-only final packet audit and local metadata mirror, no target decoding."""
import base64
import hashlib
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE = Path(__file__).resolve().parents[1]
HERE = PACKAGE / "preparation"

VERIFY = GUARD + r'''
import gzip
from collections import Counter
stage=phase/'massive_polyformer_packet_20261012_v2';old=phase/'massive_polyformer_packet_20261012_v1'
package=stage/'sources/training_pipeline_v2_20261012';sys.path.insert(0,str(package))
from compile_polyformer_boundary_matrix import verify_receipt,receipt
from spec import fingerprint,validate
plan=json.loads((stage/'matrix/PLAN.json').read_text());summary=json.loads((stage/'matrix/SUMMARY.json').read_text())
assert fingerprint(plan)==summary['plan_hash']=='1636eab494d4f07e710dedd0738ca672a07e05304a981c91a8f8a3e9cb00de69'
assert not plan['scientific_launch_enabled'] and plan['human_GO_required'] and not summary['GO_created'] and not summary['scientific_training_started']
qa=json.loads((stage/'POLYFORMER_BOUNDARY_CPU_QA_RESULT.json').read_text());assert qa['exit_code']==0 and 'Ran 15 tests' in qa['stderr'] and 'skipped' not in qa['stderr']
oldplan=json.loads((old/'matrix/PLAN.json').read_text());old_specs={}
for row in oldplan['studies']:
 value=json.loads(gzip.decompress((phase/row['path']).read_bytes()));assert fingerprint(value)==row['spec_hash'];old_specs[value['dataset']]=value
seen={};preserved=0;newfits=0;cohorts=[];counts=Counter();search=0;comparative=0
for row in plan['studies']:
 path=phase/row['path'];value=json.loads(gzip.decompress(path.read_bytes()));assert fingerprint(value)==row['spec_hash'];validate(value)
 for binding in value['source']['files']:
  if binding['path'] not in seen:verify_receipt(phase,binding);seen[binding['path']]=binding
  else:assert seen[binding['path']]==binding
 prior=old_specs[value['dataset']];actual={r['run_id']:r for r in value['runs']};prior_ids={r['run_id'] for r in prior['runs']}
 for run in prior['runs']:
  a=dict(run);b=dict(actual[run['run_id']]);a.pop('adapter');b.pop('adapter');assert a==b
  if run['kind']=='model_fit':preserved+=1
 for run in value['runs']:
  counts[run['kind']]+=1
  if run['kind']=='model_fit':
   if run['config']['stage']=='search':search+=1
   if run['config']['stage']=='comparative':comparative+=1
   if run['run_id'] not in prior_ids:newfits+=1
 methods={r['method'] for r in value['runs'] if r['kind']=='ensemble_assembly'}
 pairs={(r['split'],r['seed']) for r in value['runs'] if r['kind']=='ensemble_assembly'}
 for comparison in value['comparisons']:
  for name in comparison['references']+comparison['candidates']:
   assert name in methods
   assert {(r['split'],r['seed']) for r in value['runs'] if r['kind']=='ensemble_assembly' and r['method']==name}==pairs
 cohorts.append({'dataset':value['dataset'],'split_seed_pairs':len(pairs),'export_method_count':len(methods),'all_references_matched':True})
assert preserved==173488 and newfits==970 and search==168320 and comparative==6138
assert counts=={'model_fit':174458,'selection':144,'ensemble_assembly':3630,'calibration':3630,'evaluation':3630}
proof={'schema':'polyformer-boundary-v2-final-verification-v1','plan_hash':fingerprint(plan),'counts':dict(counts),
 'search_fits':search,'comparative_fits':comparative,'prior_learning_contracts_preserved':preserved,'new_boundary_fits':newfits,
 'unique_source_and_custody_files_rehashed':len(seen),'actual_reference_cohorts':cohorts,'all15_CPU_tests_passed':True,
 'heldout_target_tensors_decoded':False,'science_fits':0,'benchmark_scores':0,'GO_created':False,'resource_status':'PROVISIONAL_REQUIRES_ROOT_QUALIFICATION',
 'PLAN_file':receipt(phase,stage.relative_to(phase)/'matrix/PLAN.json'),'formal_totals':summary['totals']}
p=stage/'FINAL_VERIFICATION.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n')
paths=['FINAL_VERIFICATION.json','PREPARATION_SUMMARY.json','RUNTIME_ANCESTRY.json','SOURCE_INVENTORY.json',
 'POLYFORMER_BOUNDARY_CPU_QA_RESULT.json','DISABLED_POLYFORMER_BOUNDARY_COMPILER_RESULT.json','matrix/PLAN.json','matrix/SUMMARY.json']
paths += [str((phase/r['path']).relative_to(stage)) for r in plan['studies']]
print(json.dumps({'proof':proof,'files':[{'relative':r,'sha256':hashlib.sha256((stage/r).read_bytes()).hexdigest(),
 'data':base64.b64encode((stage/r).read_bytes()).decode()} for r in paths]}))
'''


def main():
    out = HERE / "polyformer_allocation_20261012_v2"
    result = invoke(VERIFY, [], out, "FINAL_VERIFY_AND_MIRROR", 600)
    if result["exit_code"] != 0: print(result["stdout"], result["stderr"]); raise SystemExit(1)
    raw = result["stdout"].splitlines(); payload = json.loads(next(line for line in raw if line.startswith('{')))
    destination = out / "allocation_artifacts/massive_polyformer_packet_20261012_v2"
    for row in payload["files"]:
        data = base64.b64decode(row["data"]); assert hashlib.sha256(data).hexdigest() == row["sha256"]
        path = destination / row["relative"]; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    print(json.dumps(payload["proof"], indent=2))


if __name__ == "__main__": main()
