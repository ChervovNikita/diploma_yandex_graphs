import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));import compile_master_plan as c
v=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));t=v['counts']['totals'];assert t['atomic_fit_run_groups']==36 and t['run_records']==108 and t['served_banks']==24 and t['reused_exact_member0_fit_references']==6 and t['native_optimizer_update_cap_sum']==36000 and t['native_optimizer_updates_minimum_sum']==10836
rows=[r for s in v['studies'] for r in s['runs'] if r['kind']=='model_fit'];reject=0
for key,value in [('members',4),('learning_rate',.002),('initialization_seed',1),('factor_coordinates_per_route',2870),('prior_std',1.),('KL_TRAIN_divisor',100),('posterior_law','point')]:
 r=copy.deepcopy(next(r for r in rows if r['config']['rank1_arm']=='q_M1'));r['config'][key]=value
 try:c.fit_semantics(r)
 except ValueError:reject+=1
 else:raise AssertionError('Changed literal Rank1 law admitted')
exports=[r for s in v['studies'] for r in s['runs'] if r['kind']=='ensemble_assembly'];secondary=[r for r in exports if r['config']['source_secondary_raw12']];assert len(secondary)==12 and sum(r['config']['source_secondary_selected_forwards'] for r in secondary)==144 and all(r['config']['source_secondary_seed']==r['seed']+6000011 for r in secondary)
report={'schema':'Rank1-literal-v16-Film-typed-ingest-v1','status':'passed','plan_hash':v['plan_hash'],'typed_counts':v['counts'],'rejected_semantic_cases':reject,'fit_groups':36,'banks':24,'secondary_raw12_banks':12,'secondary_selected_forward_calls':144,'native_maps_bounded_not_exact':True,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False};(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fit_groups':36,'banks':24,'rejects':reject,'GO':False}))
