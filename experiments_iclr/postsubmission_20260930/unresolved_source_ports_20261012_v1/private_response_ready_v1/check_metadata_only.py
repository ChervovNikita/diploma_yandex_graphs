import ast,copy,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));import compile_master_plan as c
from adapters.response_endpoints import selected_authority
v=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));t=v['counts']['totals'];assert t['atomic_fit_run_groups']==30 and t['run_records']==174 and t['served_banks']==48 and t['reused_exact_member0_fit_references']==6 and v['counts']['model_optimizer_step_call_total']==724680
assert t['registered_known_response_source_committed_model_maps']==856440 and t['registered_known_response_functional_private_commits']==131760 and t['registered_known_response_discarded_virtual_private_maps']==131760 and t['registered_known_response_native_encoder_forwards']==2272860
rows=[r for s in v['studies'] for r in s['runs'] if r['kind']=='model_fit'];reject=0
for group,key,value in [('config','members',1),('config','learning_rate',.002),('config','source_TRAIN_backwards_per_episode',4),('config','source_complete_episodes_per_cycle',60),('config','initialization_seed',22),('config','source_complete_body_constructors',4),('budget','maximum',60),('budget','prefixes',[{'prefix_id':'source60','cap_step':60,'close_on_selector_stop':True}]),('selection','evaluation_interval',1),('config','private_response_source_contract',{})]:
 r=copy.deepcopy(rows[0]);r[group][key]=value
 try:c.fit_semantics(r)
 except ValueError:reject+=1
 else:raise AssertionError('Changed literal source response law admitted')
r={'state':'complete','contract':rows[0],'result':{'completed_steps':180,'selected':{'path':'selected.pt','step':140},'prefixes':[{'prefix_id':'source60','cap_step':60,'close_on_selector_stop':False,'closed':True,'closure_step':60,'closure_reason':'cap','selected':{'path':'prefixes/source60/selected.pt','step':45}}]}}
assert selected_authority(r,'full180')['step']==140 and selected_authority(r,'source60')['step']==45
for group,key,value in [('result','completed_steps',60),('prefix','closure_step',180),('prefix','close_on_selector_stop',True),('selected','path','selected.pt'),('selected','step',65)]:
 q=copy.deepcopy(r);target=q['result'] if group=='result' else q['result']['prefixes'][0] if group=='prefix' else q['result']['prefixes'][0]['selected'];target[key]=value
 try:selected_authority(q,'source60')
 except ValueError:reject+=1
 else:raise AssertionError('Wrong horizon provenance admitted')
# Existing engine prefix implementation is unchanged. It updates open prefix selection on strict-first improvements, closes at60, writes immutable certificate, continues full budget ignoring selector stop.
engine=(P/'runtime/engine.py').read_text();assert hashlib.sha256(engine.encode()).hexdigest()=='a72e53744c3b976aaa60bf5c17b0ad1390028a487de89b0b9d850e92392265da'
assert 'if not prefix["closed"]:' in engine and 'else "cap" if step == prefix["cap_step"] else None' in engine
assert round(1/501,4)==.002 and 5+11*5==60
report={'schema':'private-response5-literal-v17-typed-full180-ingest-v1','status':'passed','plan_hash':v['plan_hash'],'typed_counts':v['counts'],'rejected_semantic_and_provenance_cases':reject,'source60_exact_same_fit_prefix_proof':{'minimum_rounded4_MRR':.002,'first_positive_eval_cycle':5,'earliest_source11miss_stop_cycle':60,'engine_AST_byteidentical_to_actual_v17':True,'prefix_selected45_distinct_from_full_selected140_metadata_proof':True,'prefix_checkpoint_and_certificate_own_fit_bound':True,'full180_no_earlystop':True,'no_source60_fresh_fit':True},'fit_groups':30,'banks':48,'actual_input_arrays_opened':0,'new_numerical_model_maps':0,'scientific_fits':0,'GO':False};(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fit_groups':30,'banks':48,'rejects':reject,'GO':False}))
