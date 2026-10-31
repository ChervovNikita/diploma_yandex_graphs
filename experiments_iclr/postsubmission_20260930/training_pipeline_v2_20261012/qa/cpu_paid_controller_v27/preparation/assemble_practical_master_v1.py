"""Combine exact materialized finite packets; no approval or scientific run."""
import json
from pathlib import Path

from prepare_molecular_packets_allocation import GUARD, invoke

HERE = Path(__file__).resolve().parent / 'practical_master_20261012_v2'
REMOTE = GUARD + r'''
import importlib
controller=phase/'master_controller_source_20261012_v1/runtime'
sys.path.insert(0,str(controller));master=importlib.import_module('compile_master_plan')
stage=phase/'massive_practical_master_20261012_v2';assert not stage.exists()
inputs=phase/'practical_master_inputs_20261012_v2';assert not inputs.exists();inputs.mkdir()
def save(path,value):
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
entries=[]
for relative in ('massive_native_practical_packet_20261012_v1','massive_coordinate_practical_packet_20261012_v1',
 'context_keys_packet_20261012_v1','massive_context_exclusion_packet_20261012_v2',
 'massive_polyformer_practical_packet_20261012_v1','practical_molecular_packet_20261012_v1',
 'link_boundary_packet_20261012_v3'):
 entry=json.loads((phase/relative/'MASTER_INPUT_ENTRY.json').read_text())
 entries.append(entry)
author='massive_polyformer_author_schedule_packet_20261012_v1/matrix'
entries.append({'id':'polyformer_unchanged_author_cli',
 'plan':master.receipt(phase,author+'/PLAN.json'),'summary':master.receipt(phase,author+'/SUMMARY.json')})
index={'schema':'disabled-master-input-index-v1','packets':entries,
 'scope':'Finite A1 representative study, not all historical predictive approaches',
 'remaining_all_method_migrations':'Explicit independent coverage/port waves remain required',
 'original_paper_experiment_reruns':False,'new_dataset_original_GNNM_required':True}
save(inputs/'INPUT_INDEX.json',index)
packets,ready=master.readiness(phase,index)
save(inputs/'DRY_RUN_READINESS.json',ready)
assert not ready['invalid_inputs'],ready['invalid_inputs']
control_files=master.controller_bindings(phase,str(controller.relative_to(phase)))
frozen=json.loads((phase/'master_controller_source_20261012_v1/MANIFEST.json').read_text())
prior=json.loads((phase/'massive_native_packet_20261012_v2/resources_v1/RESOURCES.json').read_text())
profiles={}
for packet in packets:
 for study in packet['studies']:
  profiles[study['study_id']]={
   'memory':{'kind':'provisional','scope':'Original packet positive reservations; full actual factory/optimizer/serializer/concurrency qualification pending'},
   'time':{'kind':'unknown','scope':'Original packet scheduling weights include unmeasured one-second placeholders and broad molecular/source estimates; not an ETA'}}
resources={'worker_capacities_cuda_bytes':prior['worker_capacities_cuda_bytes'],
 'group_granularity':'dependency','study_profiles':profiles,'phase_rendezvous':frozen['phase_rendezvous'],
 'qualification_status':'DISABLED_ACTUAL_FACTORY_RESOURCE_QUALIFICATION_PENDING',
 'existing_216_case_max_shape_native_probe_is_not_practical_whole_runtime_qualification':True,
 'two_slots_requested_not_measured_throughput':True,'ETA_available':False}
save(inputs/'RESOURCES.json',resources)
save(inputs/'HOST_BINDINGS.json',frozen['host_bindings'])
bindings=[master.receipt(phase,str((inputs/name).relative_to(phase)))
 for name in ('INPUT_INDEX.json','RESOURCES.json','HOST_BINDINGS.json')]
plan,studies,summary=master.assemble(phase,packets,ready,str(controller.relative_to(phase)),control_files,
 resources,str(stage.relative_to(phase)),bindings,frozen['host_bindings'])
summary['scope']='Finite A1 stage; all historical mechanism migration remains unfinished'
summary['resource_gate']='ACTUAL_FACTORY_MEMORY_TIME_SERIALIZER_CONCURRENCY_UNQUALIFIED'
summary['ETA_available']=False
summary['all_method_inventory_migration_completed']=False
summary['original_paper_experiment_reruns']=False
summary['original_GNNM_new_task_adaptations_declared']=True
for row in summary['per_worker'].values():
 row['scheduling_weight_seconds_not_ETA']=row['estimated_gpu_seconds']
 row['estimated_serial_gpu_days']=None
 row['wall_time_ETA']=None
 row['cost_qualification_pending']=True
assert summary['counts']['totals']['atomic_fit_run_groups']==8441
assert summary['counts']['totals']['paid_optimizer_starts']==8591
assert summary['counts']['totals']['physical_slow_body_instances']==9029
assert summary['counts']['model_optimizer_step_call_total']==48797200
assert summary['counts']['totals']['run_records']==18746
assert summary['counts']['totals']['author_cli_groups']==8
assert summary['counts']['totals']['author_sequential_repeat_trajectories']==80
assert plan['phase_rendezvous_capacity']['all_hosts_fit_fixed_limits']
digest=master.write_master(phase,str(stage.relative_to(phase)),plan,studies,summary)
launcher=importlib.import_module('launcher')
# Real source and opaque byte fingerprint validation; no array deserialization.
actual=launcher.load_plan(phase,stage/'PLAN.json')
assert actual[1]==digest
verification={'plan_hash':digest,'loaded_studies':len(actual[2]),'assigned_records':len(actual[3]),
 'all_records_exactly_once':True,'same_host_dependencies':True,'human_GO_created':False,
 'source_bytes_rehashed':True,'data_arrays_deserialized':False,'scientific_fits':0}
save(stage/'ACTUAL_CONTROLLER_LOAD_VERIFICATION.json',verification)
compact={'plan_hash':digest,'status':summary['status'],'counts':summary['counts'],
 'per_worker':{name:{key:row[key] for key in ('counts','scheduling_weight_seconds_not_ETA','capacity_bytes',
  'maximum_run_reservation_bytes','worst_pair_fits_declared_capacity','cost_qualification_pending','wall_time_ETA')}
   for name,row in summary['per_worker'].items()},
 'phase_rendezvous_capacity':plan['phase_rendezvous_capacity'],'scope':summary['scope'],
 'source_packets':[{'id':p['id'],'plan_hash':p['plan_hash']} for p in packets],
 'studies':len(studies),'original_paper_experiment_reruns':False,'approval_receipt_created':False,
 'scientific_fits':0,'resource_gate':summary['resource_gate'],'host_bindings':frozen['host_bindings'],
 'source_packet_and_learning_contracts_unchanged':True,'controller_load_verification':verification,
 'files':{name:master.receipt(phase,str((stage/name).relative_to(phase)))
   for name in ('PLAN.json','SUMMARY.json','READINESS.json','WORKER_COSTS.csv')}}
save(stage/'COMPACT_SUMMARY.json',compact)
print(json.dumps(compact))
'''


def main():
    assert not HERE.exists()
    HERE.mkdir()
    transport = invoke(REMOTE, [], HERE, 'ASSEMBLY_TRANSPORT', 600)
    if transport['exit_code'] != 0:
        raise RuntimeError('Practical master assembly failed; transport retained')
    value = json.loads(transport['stdout'])
    (HERE/'COMPACT_SUMMARY.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'plan_hash': value['plan_hash'], 'model_fits': value['counts']['totals']['atomic_fit_run_groups'],
                      'updates': value['counts']['model_optimizer_step_call_total'],
                      'training': False, 'resource_qualification_pending': True}))


if __name__ == '__main__':
    main()
