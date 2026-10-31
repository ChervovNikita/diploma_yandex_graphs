import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(P/'controller_delta'),str(ROOT/'retained_prototype_batch_20261012_v1/curvature_controller_v11_composition_v1'),str(ROOT/'master_controller_source_20261012_v11/runtime')]
import compile_master_plan as c
packet=c.load_packet(ROOT,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));v=packet['counts'];t=v['totals'];assert t['atomic_fit_run_groups']==2766 and t['run_records']==7018 and t['served_banks']==1410
assert v['model_optimizer_step_call_total']==13410000 and t['reused_exact_member0_fit_references']==210
runs=[r for s in packet['studies'] for r in s['runs'] if r['kind']=='model_fit' and 'unimp_source_contract' in r['config']];assert len(runs)==120 and sum(c.fit_semantics(r)['known_source_TRAIN_forward_calls'] for r in runs)==180000
rejects=0
for key,value in [('members',4),('source_optimizer_count',4),('learning_rate',.002),('initialization_seed',1),('source_TRAIN_forwards_per_update',5)]:
 r=copy.deepcopy(runs[0]);r['config'][key]=value
 try:c.fit_semantics(r)
 except ValueError:rejects+=1
 else:raise AssertionError('Changed UniMP source owner/map law admitted')
report={'schema':'UniMP-exact-source-compiler-delta-metadata-ingest-v1','status':'passed','plan_hash':packet['plan_hash'],'typed_counts':v,'new_UniMP_fit_groups':120,'new_UniMP_maps':180000,'new_I4_member0_references':30,'fixed_source_recipe_no_new_HPO':True,'rejected_source_semantic_cases':rejects,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False,'root_merge_and_exclusive_master_replacement_pending':True}
(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','groups':2766,'new_maps':180000,'rejects':rejects,'scientific_fits':0}))
