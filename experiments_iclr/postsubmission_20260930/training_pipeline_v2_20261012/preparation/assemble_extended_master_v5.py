"""Integrate completed source packets into a new disabled master, preserving v3."""
import json
from pathlib import Path

from prepare_molecular_packets_allocation import GUARD, invoke

HERE = Path(__file__).resolve().parent / 'practical_master_20261012_v5'
REMOTE = GUARD + r'''
import importlib,math
controller=phase/'master_controller_source_20261012_v3/runtime'
sys.path.insert(0,str(controller));master=importlib.import_module('compile_master_plan')
paths=['massive_native_practical_packet_20261012_v1',
 'massive_coordinate_practical_packet_20261012_v1','context_keys_packet_20261012_v1',
 'massive_context_exclusion_packet_20261012_v2','massive_polyformer_practical_packet_20261012_v1',
 'practical_molecular_packet_20261012_v1','link_boundary_packet_20261012_v3',
 'retained_pt_packet_20261012_v2','gru_memory_packet_20261012_v3',
 'massive_continuous_affinity_packet_20261012_v2','landmark_memory_packet_20261012_v3']
paths += ['massive_private_source_packets_20261012_v3/'+name
          for name in ('private_cmcl','private_adam','masked_core')]
missing=[name for name in paths if not (phase/name/'MASTER_INPUT_ENTRY.json').is_file()]
assert not missing, 'Packets must finish before master materialization: '+str(missing)
entries=[json.loads((phase/name/'MASTER_INPUT_ENTRY.json').read_text()) for name in paths]
author='massive_polyformer_author_schedule_packet_20261012_v1/matrix'
entries.append({'id':'polyformer_unchanged_author_cli',
 'plan':master.receipt(phase,author+'/PLAN.json'),'summary':master.receipt(phase,author+'/SUMMARY.json')})
index={'schema':'disabled-master-input-index-v1','packets':entries,
 'scope':'Prospective development study, including distinct retained source mechanisms and new-data GNNM references',
 'original_paper_experiment_reruns':False,'new_dataset_original_GNNM_required':True,
 'remaining_all_method_migrations':'Every uncovered/blocked canonical record remains explicit; this does not close those migrations'}
packets,ready=master.readiness(phase,index)
assert not ready['invalid_inputs'],ready
stage=phase/'massive_practical_master_20261012_v5';assert not stage.exists()
inputs=phase/'practical_master_inputs_20261012_v5';assert not inputs.exists();inputs.mkdir()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
save(inputs/'INPUT_INDEX.json',index);save(inputs/'DRY_RUN_READINESS.json',ready)
frozen=json.loads((phase/'master_controller_source_20261012_v3/MANIFEST.json').read_text())
prior=json.loads((phase/'practical_master_inputs_20261012_v3/RESOURCES.json').read_text())
profiles={}
for packet in packets:
 for study in packet['studies']:
  profiles[study['study_id']]={
   'memory':{'kind':'provisional','scope':'Source packet reservations retained where no exact full-input proxy envelope is integrated'},
   'time':{'kind':'unknown','scope':'Source scheduling weights are not ETA; optimizer/selection/endpoint and concurrency remain unmeasured'}}
# Conservative study-level envelopes replace node placeholders only where the
# exact factories/source shapes were already measured. They do not assert
# trained-state or selector qualification, nor qualification of other studies.
aggregate_path=phase/'practical_resource_qualifier_remaining_20261012_v2/AGGREGATE_PROXY_SUMMARY.json'
aggregate=json.loads(aggregate_path.read_text())
evidence=master.receipt(phase,str(aggregate_path.relative_to(phase)))
for observed in aggregate['by_study']:
 name=observed['study_id']
 if name not in profiles:continue
 maximum=max(observed['CUDA_peak_allocated_bytes']['maximum'],observed['CUDA_peak_reserved_bytes']['maximum'])
 reservation=int(math.ceil((maximum*1.25+1024**3)/(256*1024**2))*256)
 elapsed=observed['warm_TRAIN_median_seconds']['maximum']
 constructor=observed['constructor_seconds']['maximum']
 studies=[s for p in packets for s in p['studies'] if s['study_id']==name]
 assert len(studies)==1
 overrides={}
 for run in studies[0]['runs']:
  value={'reservation_mib':reservation}
  if run['kind']=='model_fit':
   value['estimated_seconds']=constructor+run['budget']['maximum']*elapsed
  overrides[run['run_id']]=value
 profiles[name]={
  'memory':{'kind':'measured_proxy','scope':'Random full-input TRAIN/moment/serializer prototype study maximum, plus25% and1GiB; not trained selector/endpoint admission',
            'evidence':[evidence]},
  'time':{'kind':'measured_proxy','scope':'Conservative measured random TRAIN-pass scheduling weight only; excludes optimizer kernels, SELECT/CAL/evaluation and concurrency; not ETA',
          'evidence':[evidence]},'runs':overrides}
resources={'worker_capacities_cuda_bytes':prior['worker_capacities_cuda_bytes'],
 'group_granularity':'dependency','runtime_requirements':prior['runtime_requirements'],
 'study_profiles':profiles,'phase_rendezvous':frozen['phase_rendezvous'],
 'qualification_status':'DISABLED_PARTIAL_TRAIN_PROXY_RESOURCE_ENVELOPES_NOT_WHOLE_MASTER_ADMISSION',
 'two_slots_requested_not_measured_throughput':True,'ETA_available':False}
save(inputs/'RESOURCES.json',resources);save(inputs/'HOST_BINDINGS.json',frozen['host_bindings'])
bindings=[master.receipt(phase,str((inputs/name).relative_to(phase)))
 for name in ('INPUT_INDEX.json','RESOURCES.json','HOST_BINDINGS.json')]+[evidence]
plan,studies,summary=master.assemble(phase,packets,ready,str(controller.relative_to(phase)),
 master.controller_bindings(phase,str(controller.relative_to(phase))),resources,
 str(stage.relative_to(phase)),bindings,frozen['host_bindings'])
summary.update(scope=index['scope'],original_paper_experiment_reruns=False,
 original_GNNM_new_task_adaptations_declared=True,ETA_available=False,
 all_method_inventory_migration_completed=False,
 resource_gate='PARTIAL_TRAIN_PROXY_RESOURCES_REMAINING_FACTORIES_ENDPOINTS_AND_CONCURRENCY_PENDING')
for row in summary['per_worker'].values():
 row['scheduling_weight_seconds_not_ETA']=row['estimated_gpu_seconds']
 row['estimated_serial_gpu_days']=None;row['wall_time_ETA']=None;row['cost_qualification_pending']=True
# Reconcile independently per packet; no implied cross-packet fit reuse.
from collections import Counter
expected=Counter()
for packet in packets:expected.update(packet['counts']['totals'])
assert dict(expected)==summary['counts']['totals']
assert plan['phase_rendezvous_capacity']['all_hosts_fit_fixed_limits']
digest=master.write_master(phase,str(stage.relative_to(phase)),plan,studies,summary)
actual=importlib.import_module('launcher').load_plan(phase,stage/'PLAN.json')
assert actual[1]==digest
verification={'plan_hash':digest,'loaded_studies':len(actual[2]),'assigned_records':len(actual[3]),
 'all_records_exactly_once':True,'same_host_dependencies':True,'human_GO_created':False,
 'source_bytes_rehashed':True,'data_arrays_deserialized':False,'scientific_fits':0}
save(stage/'ACTUAL_CONTROLLER_LOAD_VERIFICATION.json',verification)
compact={'plan_hash':digest,'status':summary['status'],'counts':summary['counts'],
 'source_packets':[{'id':p['id'],'plan_hash':p['plan_hash'],'counts':p['counts']} for p in packets],
 'studies':len(studies),'per_worker':{name:{key:row[key] for key in ('counts',
  'scheduling_weight_seconds_not_ETA','capacity_bytes','maximum_run_reservation_bytes',
  'worst_pair_fits_declared_capacity','cost_qualification_pending','wall_time_ETA')}
  for name,row in summary['per_worker'].items()},
 'per_study':[{'study_id':s['study_id'],'dataset':s['dataset'],
  'counts':master.typed_counts([s]),'resource_scope':s['resources']['master_resource_scope']}
  for s in studies],
 'original_paper_experiment_reruns':False,'approval_receipt_created':False,
 'scientific_fits':0,'source_packet_and_learning_contracts_unchanged':True,
 'ETA_available':False,'remaining_migrations_and_resource_gates':True,
 'controller_load_verification':verification,'host_bindings':frozen['host_bindings'],
 'files':{name:master.receipt(phase,str((stage/name).relative_to(phase)))
          for name in ('PLAN.json','SUMMARY.json','READINESS.json','WORKER_COSTS.csv')}}
save(stage/'COMPACT_SUMMARY.json',compact)
print(json.dumps(compact))
'''


def main():
    assert not HERE.exists()
    HERE.mkdir()
    (HERE/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    result=invoke(REMOTE,[],HERE,'ASSEMBLY_TRANSPORT',180)
    assert result['exit_code']==0, 'Disabled master assembly failure retained'
    value=json.loads(result['stdout'])
    (HERE/'COMPACT_SUMMARY.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'plan_hash':value['plan_hash'],'studies':value['studies'],
        'model_fit_groups':value['counts']['totals']['atomic_fit_run_groups'],
        'model_optimizer_calls':value['counts']['model_optimizer_step_call_total'],
        'GO':False,'scientific_training_started':False}))


if __name__=='__main__':
    main()
