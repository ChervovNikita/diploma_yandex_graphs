import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path[:0]=[str(P/'controller_delta'),str(R/'retained_prototype_batch_20261012_v1/curvature_controller_v11_composition_v1'),str(R/'master_controller_source_20261012_v11/runtime')]
import compile_master_plan as c
packet=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));v=packet['counts'];t=v['totals'];assert t['atomic_fit_run_groups']==150 and t['run_records']==420 and t['served_banks']==90 and t['reused_exact_member0_fit_references']==30 and v['model_optimizer_step_call_total']==75000
rows=[r for s in packet['studies'] for r in s['runs'] if r['kind']=='model_fit'];assert sum(c.fit_semantics(r)['known_source_TRAIN_forward_calls'] for r in rows)==120000 and sum(c.fit_semantics(r)['known_source_TRAIN_backward_calls'] for r in rows)==120000
reject=0
for key,value in [('members',4),('source_optimizer_count',4),('learning_rate',.001),('source_TRAIN_forwards_per_update',4),('initialization_seed',1)]:
 r=copy.deepcopy(rows[0]);r['config'][key]=value
 try:c.fit_semantics(r)
 except ValueError:reject+=1
 else:raise AssertionError('Changed source BSNN method law admitted')
report={'schema':'minimal-original-BSNN-d2-typed-controller-ingest-v1','status':'passed','plan_hash':packet['plan_hash'],'typed_counts':v,'fit_groups':150,'model_maps':75000,'exact_I4_member0_references':30,'TRAIN_source_forwards':120000,'TRAIN_source_backwards':120000,'rejected_semantic_cases':reject,'isolated_unchangedHouseholder_provider1_0_1_CPU_only':True,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False,'root_master_merge_pending':True}
(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fit_groups':150,'model_maps':75000,'rejects':reject,'GO':False}))
