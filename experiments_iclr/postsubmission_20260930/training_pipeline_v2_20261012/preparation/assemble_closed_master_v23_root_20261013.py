"""Assemble all closed source packets onto the corrected cumulative controller."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
OUT=Path(__file__).resolve().parent/'practical_master_20261013_v23'
REMOTE=GUARD+r'''
import importlib
controller=phase/'master_controller_source_20261013_v21/runtime';sys.path.insert(0,str(controller));master=importlib.import_module('compile_master_plan')
oldindex_path=phase/'practical_master_inputs_20261012_v22/INPUT_INDEX.json';index=json.loads(oldindex_path.read_text())
entries=index['packets'];replaced=[e for e in entries if e['id']=='retained_link_completion_exclusive_successor'];assert len(replaced)==1
successor=json.loads((phase/'ncnc_shared_replay_successor_20261012_v3/MASTER_INPUT_ENTRY_WITH_COVERAGE.json').read_text());assert successor['id']==replaced[0]['id']
entries=[successor if e['id']==successor['id'] else e for e in entries]
for relative in ['nsd_cold_geometry_source_root_20261012_v5/MASTER_INPUT_ENTRY.json',
 'native_bsnn_bayesian_composition_bridge_20261012_v1/MASTER_INPUT_ENTRY_WITH_COMPOSITION.json',
 'unresolved_source_ports_20261012_v1/private_response_ready_v2/MASTER_INPUT_ENTRY.json',
 'unresolved_source_ports_20261012_v1/label_corrector_ready_v2/MASTER_INPUT_ENTRY.json',
 'unresolved_source_ports_20261012_v1/typed_context_ready_v2/MASTER_INPUT_ENTRY.json']:
 entries.append(json.loads((phase/relative).read_text()))
entries.extend(json.loads((phase/'native_physics_oipg_newroles_packet_20261012_v1/MASTER_INPUT_ENTRIES.json').read_text()))
assert len(entries)==48 and len({e['id'] for e in entries})==48
index.update(packets=entries,scope='Complete closed-packet union for prospective new-data study; canonical coverage and resource admission under final audit',remaining_all_method_migrations='Final exact-recipe coverage audit required; no implicit readiness assertion')
inputs=phase/'practical_master_inputs_20261013_v23';assert not inputs.exists();inputs.mkdir()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
save(inputs/'INPUT_INDEX.json',index)
packets,ready=master.readiness(phase,index);save(inputs/'DRY_RUN_READINESS.json',ready)
assert not ready['invalid_inputs'],ready['invalid_inputs']
resources=json.loads((phase/'practical_master_inputs_20261012_v22/RESOURCES.json').read_text())
names={s['study_id'] for p in packets for s in p['studies']}
assert len(names)==92
for name in names:
 if name not in resources['study_profiles']:
  resources['study_profiles'][name]={'memory':{'kind':'provisional','scope':'Source reservation pending final receipt/geometry join'},'time':{'kind':'unknown','scope':'No source-specific complete trajectory throughput available; not ETA'}}
assert not set(resources['study_profiles'])-names
resources.update(qualification_status='FINAL_SOURCE_UNION_RESOURCE_GEOMETRY_AND_STORAGE_AUDIT_PENDING',ETA_available=False)
save(inputs/'RESOURCES.json',resources)
frozen=json.loads((phase/'master_controller_source_20261013_v21/MANIFEST.json').read_text());save(inputs/'HOST_BINDINGS.json',frozen['host_bindings'])
bindings=[master.receipt(phase,str((inputs/n).relative_to(phase))) for n in ('INPUT_INDEX.json','RESOURCES.json','HOST_BINDINGS.json')]
for p in resources['study_profiles'].values():
 for kind in ('memory','time'):bindings.extend(p[kind].get('evidence',[]))
bindings=master.merge_bindings(bindings)
stage=phase/'massive_practical_master_20261013_v23';assert not stage.exists()
plan,studies,summary=master.assemble(phase,packets,ready,str(controller.relative_to(phase)),master.controller_bindings(phase,str(controller.relative_to(phase))),resources,str(stage.relative_to(phase)),bindings,frozen['host_bindings'])
assert summary['counts']['totals']['atomic_fit_run_groups']==79485
assert all('Roman' not in s['dataset'] for s in studies)
for s in studies:
 token=s['dataset'].lower().replace('_','-').replace(' ','-')
 assert not any(d in token for d in ('roman-empire','amazon-ratings','minesweeper','questions','tolokers')),s['dataset']
summary.update(original_paper_experiment_reruns=False,original_GNNM_new_task_adaptations_declared=True,all_method_inventory_migration_completed=False,ETA_available=False,resource_gate='Final exact source geometry/storage admission and 18.77 runtime remain separate gates',closed_source_union=True)
prior=json.loads((phase/'massive_practical_master_20261012_v22/SUMMARY.json').read_text());summary['additional_execution_work']=prior['additional_execution_work']
summary['shared_replay_storage_scope']={'source_entry':successor,'eligible_model_fits':249,'epochs':100,'source_training_and_selection_unchanged':True,'raw_arrays_once_per_admitted_physical_device':True,'engineering_epoch_is_not_scientific_store_donor':True,'physical_18_77_runtime_admission_pending':True}
summary['functional_private_update_accounting']={'scope':'Private-response source uses extra functional private commits and discarded virtual maps separately from Torch optimizer.step calls.', 'additional_committed_private_model_maps':131760,'discarded_virtual_model_maps':131760,'source_response_registered_Adam_calls':724680,'source_response_total_committed_model_maps':856440}
for row in summary['per_worker'].values():
 row['scheduling_weight_seconds_not_ETA']=row['estimated_gpu_seconds'];row['estimated_serial_gpu_days']=None;row['wall_time_ETA']=None;row['cost_qualification_pending']=True
assert plan['phase_rendezvous_capacity']['all_hosts_fit_fixed_limits']
digest=master.write_master(phase,str(stage.relative_to(phase)),plan,studies,summary)
compact={'plan_hash':digest,'studies':len(studies),'packets':len(packets),'counts':summary['counts'],'per_worker':summary['per_worker'],'per_study':[{k:r[k] for k in ('study_id','dataset','counts','worker','reservation_mib','references')} for r in summary['per_study']], 'phase_rendezvous_capacity':summary['phase_rendezvous_capacity'],'source_packet_entries':entries,'scientific_fits':0,'GO':False,'coverage_and_resource_audit_pending':True,'files':{n:master.receipt(phase,str((stage/n).relative_to(phase))) for n in ('PLAN.json','SUMMARY.json','READINESS.json','WORKER_COSTS.csv')}}
save(stage/'COMPACT_SUMMARY.json',compact)
print(json.dumps(compact))
'''
def main():
 OUT.mkdir();(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes());r=invoke(REMOTE,[],OUT,'ASSEMBLY',240)
 assert r['exit_code']==0,'Preserve assembly failures; do not rewrite sealed source'
 v=json.loads(r['stdout']);(OUT/'COMPACT_SUMMARY.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'plan_hash':v['plan_hash'],'studies':v['studies'],'packets':v['packets'],'fits':v['counts']['totals']['atomic_fit_run_groups'],'GO':False}))
if __name__=='__main__':main()
