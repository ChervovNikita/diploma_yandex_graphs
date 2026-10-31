"""Exact current source composition + sole Single8 compiler delta;metadata only."""
import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(P/'controller_delta'),
 str(ROOT/'retained_prototype_batch_20261012_v1/curvature_controller_v11_composition_v1'),
 str(ROOT/'master_controller_source_20261012_v11/runtime')]
import compile_master_plan as controller
entry=json.loads((P/'MASTER_INPUT_ENTRY.json').read_text());packet=controller.load_packet(ROOT,entry)
counts=packet['counts'];t=counts['totals']
assert t['atomic_fit_run_groups']==2520 and t['run_records']==7304 and t['served_banks']==1584
assert counts['model_optimizer_step_call_total']==12600000 and t['reused_exact_member0_fit_references']==180
source=[r for s in packet['studies'] for r in s['runs'] if 'single8_control' in r['config']]
assert len(source)==378
assert sum(controller.fit_semantics(r)['known_source_TRAIN_forward_calls'] for r in source)==15120000
assert sum(controller.fit_semantics(r)['known_source_TRAIN_backward_calls'] for r in source)==1890000
assert all(controller.routes(r)==1 and controller.fit_semantics(r)['optimizer_objects']==1 for r in source)
rejects=0
for key,value in (('members',4),('source_optimizer_count',4),('physical_slow_bodies',8),('source_TRAIN_forwards_per_update',2),('single8_control','unregistered')):
 bad=copy.deepcopy(source[0]);bad['config'][key]=value
 try:controller.fit_semantics(bad)
 except ValueError:rejects+=1
 else:raise AssertionError('Incorrect Single8 body/view/owner semantics admitted')
value={'schema':'Single8-exact-source-compiler-delta-metadata-ingest-v1','status':'passed','plan_hash':packet['plan_hash'],
 'typed_counts':counts,'new_Single8_fit_groups':378,'new_Single8_TRAIN_forwards':15120000,'new_Single8_TRAIN_backwards':1890000,
 'one_ordinary_body_and_optimizer_and_served_route':True,'rejected_false_ownership_cases':rejects,
 'source_data_arrays_rehashed':False,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False,
 'root_merge_and_exclusive_master_replacement_pending':True}
(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':'passed','groups':2520,'new_Single8_groups':378,'new_forward_calls':15120000,'rejects':rejects,'scientific_fits':0}))
