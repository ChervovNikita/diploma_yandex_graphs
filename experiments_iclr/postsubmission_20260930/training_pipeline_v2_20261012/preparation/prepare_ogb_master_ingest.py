"""Make an immutable metadata successor compatible with the two-slot master."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE=Path(__file__).resolve().parents[1]
LABEL='ogb_master_ingest_root_20261012_v2'

REMOTE=GUARD+r'''
import importlib
runtime=phase/'master_controller_source_20261012_v5/runtime';sys.path.insert(0,str(runtime))
master=importlib.import_module('compile_master_plan')
old=phase/'ogb_batchnorm_author_cli_packet_20261012_v1'
plan=json.loads((old/'PLAN.json').read_text());summary=json.loads((old/'SUMMARY.json').read_text())
assert master.fingerprint(plan)=='29816f48fad269aae62fcc02bd63669a687e10eeed0629eb656ae17fb1aef23b'
assert summary['plan_hash']==master.fingerprint(plan) and plan['slots_per_gpu']==1
assert len(plan['groups'])==80 and summary['actual_author_fits']==0 and not summary['GO_created']
stage=phase/'ogb_author_master_ingest_packet_20261012_v2';assert not stage.exists();stage.mkdir()
plan['slots_per_gpu']=2
summary['plan_hash']=master.fingerprint(plan)
summary['root_two_slot_master_metadata_adaptation']={
 'original_plan':master.receipt(phase,str((old/'PLAN.json').relative_to(phase))),
 'only_changed_plan_field':'slots_per_gpu:1->2',
 'native_source_or_seed_or_study_or_recipe_or_data_changed':False,
 'actual_parallelism_requires_runtime_memory_admission':True}
for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]:
 path=stage/name;path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');path.chmod(0o444)
entry={'id':'ogb_batchnorm_native_author_separate_seed_policy',
 'plan':master.receipt(phase,str((stage/'PLAN.json').relative_to(phase))),
 'summary':master.receipt(phase,str((stage/'SUMMARY.json').relative_to(phase))),
 'plan_hash':master.fingerprint(plan),
 'coverage':[master.receipt(phase,str((old/name).relative_to(phase))) for name in (
  'NATIVE_SOURCE_AND_DISABLED_METADATA_PROOF.json','SEED_POLICY_REVISION_RECEIPT.json','ALLOCATION_ENGINEERING_CHECK_BINDING.json')]}
loaded=master.load_packet(phase,entry)
assert len(loaded['studies'])==1 and loaded['counts']['totals']['author_cli_groups']==80
assert loaded['counts']['totals']['model_fit_groups']==0 and loaded['counts']['totals']['actual_author_fits']==0
proof={'schema':'OGB-author-root-readonly-master-ingest-v1','entry':entry,'counts':loaded['counts'],
 'metadata_only_load_packet_passed':True,'source_fidelity_scoped_in_coverage_receipts':True,
 'source_algorithm_or_results_or_original_GNNM_scores_changed':False,'scientific_fits':0,'GO_created':False,
 'raw_or_processed_data_or_target_tensors_decoded':False,'root_complete_master_integration_pending':True,
 'resource_status':'SOURCE_PACKET_PROVISIONAL_NOT_ROOT_QUALIFIED;scientific_worker_refuses_execution'}
for name,value in [('MASTER_INPUT_ENTRY.json',entry),('ROOT_INGEST_PROOF.json',proof)]:
 path=stage/name;path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');path.chmod(0o444)
print(json.dumps(proof))
'''


def main():
    out=PACKAGE/'preparation'/LABEL;out.mkdir(exist_ok=False)
    result=invoke(REMOTE,[],out,'ROOT_INGEST',120)
    assert result['exit_code']==0,result['stdout']
    value=json.loads(result['stdout']);(out/'ROOT_INGEST_PROOF.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'author_groups':value['counts']['totals']['author_cli_groups'],
        'maximum_native_optimizer_calls':value['counts']['totals']['author_registered_maximum_optimizer_updates'],
        'source_study_unchanged':True,'scientific_fits':0,'GO_created':False}))


if __name__=='__main__':main()
