"""Final practical policy/count/custody/source audit and local metadata mirror."""
import base64
import hashlib
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE = Path(__file__).resolve().parents[1]
VERIFY = GUARD + r'''
import gzip
from collections import Counter
stage=phase/'massive_polyformer_practical_packet_20261012_v1';old=phase/'massive_polyformer_packet_20261012_v2'
package=stage/'sources/training_pipeline_v2_20261012';sys.path.insert(0,str(package))
from compile_polyformer_boundary_matrix import receipt,verify_receipt
from compile_polyformer_practical_matrix import POLICY,practical_domain
from adapters.polyformer_endpoints import independent_contracts
from spec import fingerprint,validate
plan=json.loads((stage/'matrix/PLAN.json').read_text());summary=json.loads((stage/'matrix/SUMMARY.json').read_text())
policy=json.loads((stage/'POLICY.json').read_text())
assert policy==POLICY and fingerprint(policy)==summary['policy_hash']=='ffe27a417e8c1bbb89f6b175493613db682896d21ae668f1619888d9cad31ab1'
assert fingerprint(plan)==summary['plan_hash']=='ab48284f66c7933e082c2830583e8dcc1b27accb6d6ed45f3549580f359090c2'
assert not plan['scientific_launch_enabled'] and plan['human_GO_required'] and not summary['GO_created'] and not summary['scientific_training_started']
qa=json.loads((stage/'PRACTICAL_POLICY_CPU_QA_RESULT.json').read_text());assert qa['exit_code']==0 and 'Ran 2 tests' in qa['stderr'] and 'skipped' not in qa['stderr']
inherit=json.loads((stage/'NUMERICAL_QA_INHERITANCE.json').read_text());assert inherit['prior15_CPU_tests_passed']
for receipt_row in inherit['unchanged_numerical_and_common_controller_sources'].values():verify_receipt(phase,receipt_row)
oldplan=json.loads((old/'matrix/PLAN.json').read_text());oldsummary=json.loads((old/'matrix/SUMMARY.json').read_text());assert fingerprint(oldplan)==oldsummary['plan_hash']
old_specs={}
for row in oldplan['studies']:
 if 'citeseer' in row['path'] or 'pubmed' in row['path']:
  value=json.loads(gzip.decompress((phase/row['path']).read_bytes()));assert fingerprint(value)==row['spec_hash'];old_specs[value['dataset']]=value
matrix=json.loads((package/'docs/MATRIX_AXES_DRAFT.json').read_text());seen={};counts=Counter();stages=Counter();cohorts=[]
for binding in plan['studies']:
 value=json.loads(gzip.decompress((phase/binding['path']).read_bytes()));assert fingerprint(value)==binding['spec_hash'];validate(value)
 assert value['data']==old_specs[value['dataset']]['data'] and value['data']['evidence']['split_roster']==[0]
 for row in value['source']['files']:
  if row['path'] not in seen:verify_receipt(phase,row);seen[row['path']]=row
  else:assert seen[row['path']]==row
 recipe=next(r for r in matrix['baseline_recipe_catalogue']['rows'] if r['recipe_id']=='polyformer_mono_'+value['dataset'])
 selectors={r['run_id']:r for r in value['runs'] if r['kind']=='selection'};assert len(selectors)==36
 groups={};runmap={r['run_id']:r for r in value['runs']}
 for selector in selectors.values():
  actual=selector['config']['configurations'];expected=practical_domain(recipe,selector['method'],policy)
  assert [r['parameters'] for r in actual]==[r['parameters'] for r in expected] and len(actual)==9
  assert sum(r['source_anchor'] for r in actual)==1
  assert selector['parents']==[i for r in actual for i in r['run_ids']]
  for candidate in actual:
   assert len(candidate['run_ids'])==2
   for identifier in candidate['run_ids']:
    assert runmap[identifier]['seed'] in policy['search_seeds']
    assert all(runmap[identifier]['config'][key]==val for key,val in candidate['parameters'].items())
 for run in value['runs']:
  counts[run['kind']]+=1
  if run['kind']=='model_fit':
   stages[run['config']['stage']]+=1
   assert run['budget']=={'maximum':2000,'unit':'optimizer_updates','policy':'full_budget'}
   if run['config']['stage']=='comparative':assert run['config']['selected_config_from'] in selectors and run['seed'] in policy['comparative_seeds']
   if run['config'].get('independence_group'):groups.setdefault(run['config']['independence_group'],[]).append(run)
 for group in groups.values():independent_contracts(value,group)
 methods={r['method'] for r in value['runs'] if r['kind']=='ensemble_assembly'};pairs={(r['split'],r['seed']) for r in value['runs'] if r['kind']=='ensemble_assembly'}
 assert pairs=={(0,s) for s in policy['comparative_seeds']} and len(methods)==55
 for comparison in value['comparisons']:
  for name in comparison['references']+comparison['candidates']:
   assert {(r['split'],r['seed']) for r in value['runs'] if r['kind']=='ensemble_assembly' and r['method']==name}==pairs
 cohorts.append({'dataset':value['dataset'],'search_procedures':36,'effective_points_per_procedure':9,'source_anchors_in_domain':36,
  'comparative_split_seed_pairs':3,'actual_export_method_count':55,'genuine_independent_groups':len(groups),'all_references_matched':True})
assert counts=={'model_fit':1854,'selection':72,'ensemble_assembly':330,'calibration':330,'evaluation':330}
assert stages=={'search':1296,'comparative':558}
proof={'schema':'polyformer-practical-final-verification-v1','plan_hash':fingerprint(plan),'policy_hash':fingerprint(policy),
 'counts':dict(counts),'model_fit_stages':dict(stages),'formal_totals':summary['totals'],'actual_policy_and_reference_cohorts':cohorts,
 'unique_source_and_custody_files_rehashed':len(seen),'v2_numerical_QA_inherited_by_exact_source_hash':True,'practical_policy_tests_passed':2,
 'prior_unrestricted_and_author_packets_modified':False,'heldout_target_tensors_decoded':False,'scientific_fits':0,'benchmark_scores':0,'GO_created':False,
 'CAL_calls_separate_from_backbone':165000,'PLAN_file':receipt(phase,stage.relative_to(phase)/'matrix/PLAN.json'),
 'resource_status':'PROVISIONAL_REQUIRES_ROOT_QUALIFICATION'}
p=stage/'FINAL_VERIFICATION.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n')
paths=['FINAL_VERIFICATION.json','POLICY.json','NUMERICAL_QA_INHERITANCE.json','SOURCE_INVENTORY.json','PREPARATION_SUMMARY.json',
 'PRACTICAL_POLICY_CPU_QA_RESULT.json','DISABLED_POLYFORMER_PRACTICAL_COMPILER_RESULT.json','matrix/PLAN.json','matrix/SUMMARY.json']
paths += [str((phase/r['path']).relative_to(stage)) for r in plan['studies']]
print(json.dumps({'proof':proof,'files':[{'relative':r,'sha256':hashlib.sha256((stage/r).read_bytes()).hexdigest(),
 'data':base64.b64encode((stage/r).read_bytes()).decode()} for r in paths]}))
'''


def main():
    out = PACKAGE / "preparation/polyformer_practical_allocation_20261012_v1"
    result = invoke(VERIFY, [], out, "FINAL_VERIFY_AND_MIRROR", 600)
    if result["exit_code"] != 0: print(result["stdout"], result["stderr"]); raise SystemExit(1)
    payload = json.loads(next(line for line in result["stdout"].splitlines() if line.startswith('{')))
    destination = out / "allocation_artifacts/massive_polyformer_practical_packet_20261012_v1"
    for row in payload["files"]:
        raw = base64.b64decode(row["data"]); assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        path = destination / row["relative"]; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    print(json.dumps(payload["proof"], indent=2))


if __name__ == "__main__": main()
