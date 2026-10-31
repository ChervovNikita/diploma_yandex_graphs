import ast,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));import compile_master_plan as c
v=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));t=v['counts']['totals'];assert t['atomic_fit_run_groups']==18 and t['run_records']==54 and t['served_banks']==12 and t['reused_exact_member0_fit_references']==3 and t['native_optimizer_update_cap_sum']==18000 and t['native_optimizer_updates_minimum_sum']==5418
rows=[r for s in v['studies'] for r in s['runs'] if r['kind']=='model_fit'];reject=0
for key,value in [('members',4),('learning_rate',.002),('source_TRAIN_backwards_per_update',4),('initialization_seed',1),('source_complete_body_constructors',4)]:
 r=copy.deepcopy(rows[0]);r['config'][key]=value
 try:c.fit_semantics(r)
 except ValueError:reject+=1
 else:raise AssertionError('Changed literal FiLM law admitted')
for k,val in [('maximum',1001),('patience',50)]:
 r=copy.deepcopy(rows[0]);r['budget'][k]=val
 try:c.fit_semantics(r)
 except ValueError:reject+=1
 else:raise AssertionError('Changed stopping admitted')
report={'schema':'FiLM-literal-v16-typed-ingest-v1','status':'passed','plan_hash':v['plan_hash'],'typed_counts':v['counts'],'rejected_semantic_cases':reject,'source_cap1000_patience300_preserved':True,'fit_groups':18,'banks':12,'native_maps_bounded_not_exact':True,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False};(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fit_groups':18,'banks':12,'rejects':reject,'GO':False}))
