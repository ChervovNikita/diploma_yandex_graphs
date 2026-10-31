import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path[:0]=[str(P/'controller_delta'),str(R/'retained_prototype_batch_20261012_v1/curvature_controller_v11_composition_v1'),str(R/'master_controller_source_20261012_v11/runtime')]
import compile_master_plan as c
packet=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));v=packet['counts'];t=v['totals'];assert t['atomic_fit_run_groups']==21 and t['run_records']==66 and t['served_banks']==15 and t['reused_exact_member0_fit_references']==3 and v['model_optimizer_step_call_total']==21000
rows=[r for s in packet['studies'] for r in s['runs'] if r['kind']=='model_fit'];assert sum(c.fit_semantics(r)['known_source_TRAIN_forward_calls'] for r in rows)==48000
reject=0
for key,value in [('members',1),('source_optimizer_count',4),('train_minibatches_per_epoch',9),('source_TRAIN_forwards_per_minibatch',1),('source_rank_candidate_wrapper_constructions',0)]:
 r=copy.deepcopy(next(r for r in rows if r['config']['source_GINE_arm']=='lowrank_private'));r['config'][key]=value
 try:
  sem=c.fit_semantics(r)
 except ValueError:reject+=1
 else:raise AssertionError('Changed source GINE semantics admitted')
report={'schema':'minimal-source-lowrank-GINE-typed-controller-ingest-v1','status':'passed','plan_hash':packet['plan_hash'],'typed_counts':v,'fit_groups':21,'model_maps':21000,'TRAIN_source_forwards':48000,'TRAIN_source_backwards':21000,'rejected_semantic_cases':reject,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False,'root_master_merge_pending':True}
(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fit_groups':21,'model_maps':21000,'rejects':reject,'GO':False}))
