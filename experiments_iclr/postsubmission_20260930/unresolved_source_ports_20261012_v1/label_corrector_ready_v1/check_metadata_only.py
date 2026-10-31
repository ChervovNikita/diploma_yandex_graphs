import ast,copy,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));import compile_master_plan as c
v=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));t=v['counts']['totals'];assert t['atomic_fit_run_groups']==27 and t['run_records']==90 and t['served_banks']==21 and t['reused_exact_member0_fit_references']==3 and v['counts']['model_optimizer_step_call_total']==52800
assert t['registered_known_label_corrector_native_Adam_maps']==29700 and t['registered_known_label_corrector_Adam_maps']==23100 and t['partial_registered_known_source_TRAIN_forward_calls']==122100 and t['partial_registered_known_source_TRAIN_backward_calls']==85800
rows=[r for s in v['studies'] for r in s['runs'] if r['kind']=='model_fit'];reject=0
for group,key,value in [('config','members',1),('config','learning_rate',.002),('config','source_corrector_TRAIN_backwards_per_epoch',1),('config','local_transition_epochs',200),('config','initialization_seed',22),('config','source_optimizer_count',1),('budget','maximum',5000),('budget','policy','native_selector_stop'),('selection','checkpoint0',True),('selection','evaluation_interval',5),('config','label_corrector_source_contract',{})]:
 r=copy.deepcopy(rows[0]);r[group][key]=value
 try:c.fit_semantics(r)
 except ValueError:reject+=1
 else:raise AssertionError('Changed literal sourcecorrector law admitted')
# Constructor/controller equality for plain source alias uses the literal exact public Session, no core.
native=[r for r in rows if r['config']['label_corrector_arm']=='native_single' and r['body']==0];assert len(native)==3 and all(r['method']=='label_only_corrector_native_single' for r in native)
code=(P/'runtime/adapters/label_corrector.py').read_text();assert "public.Session('wikics','single',seed,device,bound(root,B['native']))" in code and 'v=n.train_step(self.train_batch,self.labels)' in code
report={'schema':'label-corrector4-literal-v17-typed-ingest-v1','status':'passed','plan_hash':v['plan_hash'],'typed_counts':v['counts'],'rejected_semantic_cases':reject,'source1100full_local100_controller_preserved':True,'exact_duplicate_native_single_alias_join':{'canonical_method_id':'label_only_corrector_native_single','source_constructor':'literal publicV2 Session(wikics,single)','source_training_callback':'literal Session.train_step','source_local_restore':'literal selection.local_transition preserving liveRNG','fit_body0_records':3,'I4_exactmember0_references':3,'new_adaptation':'SELECT3505/CAL1769;sourceGRAPH/TRAIN unchanged;sourceepochs/ownviews/Adam preserved'},'fit_groups':27,'banks':21,'actual_input_arrays_opened':0,'new_numerical_model_maps':0,'scientific_fits':0,'GO':False};(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fit_groups':27,'banks':21,'rejects':reject,'GO':False}))
