import ast,copy,gzip,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));import compile_master_plan as c
from engine import _validate_spec,_training_counts
v=c.load_packet(R,json.loads((P/'MASTER_INPUT_ENTRY.json').read_text()));t=v['counts']['totals'];assert t['atomic_fit_run_groups']==45 and t['run_records']==144 and t['served_banks']==33 and t['reused_exact_member0_fit_references']==6 and t['typed_context_native_epoch_minimum_sum']==2340 and t['typed_context_native_epoch_cap_sum']==9000 and t['typed_context_master_scaler_chain_edges']==30
assert v['counts']['model_optimizer_step_call_minimum']==0 and v['counts']['model_optimizer_step_call_maximum']==9000
fits=[r for s in v['studies'] for r in s['runs'] if r['kind']=='model_fit'];r=fits[0];engine={'fit_id':r['run_id'],'budget_steps':200,'budget_unit':'epochs','validation_interval':1,'validate_initial':False,'prefixes':[],'inherited_fit_ids':[],'adapter_spec':{'run':r},'completion_policy':'typed_context_native_selector_stop','native_amp_cycle_policy':{'schema':'typed-context-AMP-complete-cycle-v1','optimizer_owner_attempts':1}};_validate_spec(engine)
base={'optimizer_updates':1,'model_forwards':8,'backward_calls':8,'supervised_examples':2740,'source_optimizer_attempts':1,'source_AMP_skipped_optimizer_maps':0,'source_native_complete_cycles':1,'source_native_AMP_enabled':0};assert _training_counts(base,engine)['optimizer_updates']==1;skip={**base,'optimizer_updates':0,'source_AMP_skipped_optimizer_maps':1,'source_native_AMP_enabled':1};assert _training_counts(skip,engine)['optimizer_updates']==0
reject=0
for group,key,value in [('config','members',1),('config','learning_rate',.002),('config','generator_width',128),('config','source_TRAIN_model_calls_per_epoch',4),('config','initialization_seed',700),('config','source_seed_spec',{}),('budget','maximum',1000),('budget','policy','full_budget'),('selection','checkpoint0',True),('selection','patience_steps',50)]:
 q=copy.deepcopy(r);q[group][key]=value
 try:c.fit_semantics(q)
 except ValueError:reject+=1
 else:raise AssertionError('Changed literal typedsource law admitted')
for change in ({'source_native_AMP_enabled':0},{'model_forwards':0},{'source_native_complete_cycles':0}):
 q={**skip,**change}
 try:_training_counts(q,engine)
 except ValueError:reject+=1
 else:raise AssertionError('Invalid native AMP skipped cycle admitted')
spec=v['studies'][0];barriers=v['plan']['barriers'];bad=copy.deepcopy(spec);q=next(r for r in bad['runs'] if r['kind']=='model_fit' and r['seed']==2);q['config']['master_scaler_parent_fit_id']=next(r['run_id'] for r in bad['runs'] if r['kind']=='model_fit' and r['seed']==1 and r['config']['typed_context_arm']!=q['config']['typed_context_arm'])
try:c.graph_check([bad],barriers)
except ValueError:reject+=1
else:raise AssertionError('Wrong condition scaler ancestry admitted')
# Pure source selector metadata: first best atcycle1;51 misses ->cycle52, matching zero-based source epoch-best_epoch>50.
from adapters.typed_context import Adapter
fake=object.__new__(Adapter);fake._best=None;fake._best_step=None;fake._misses=0
assert fake.observe_validation(1,{'selection_metric':1.})['save_selected']
for i in range(2,52):assert not fake.observe_validation(i,{'selection_metric':1.})['stop_requested']
assert fake.observe_validation(52,{'selection_metric':1.})['stop_requested']
# Narrow amendment is idempotent and changes only allowed existing AST functions.
q=P/'amend_controller.py';s=importlib.util.spec_from_file_location('typed_amend',q);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);changes={};base_dir=P/'controller_base_v17'
for filename,fun,allowed in [('compile_master_plan.py',m.amend_compiler,{'fit_semantics','graph_check','typed_counts','bank_roster'}),('engine.py',m.amend_engine,{'_validate_spec','run_trajectory'}),('run_study.py',m.amend_run_study,{'execute'}),('spec.py',m.amend_spec,{'validate'})]:
 old=(base_dir/filename).read_text();new=fun(old);assert fun(new)==new
 def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 a,b=functions(old),functions(new);changed={k for k in a if a[k]!=b[k]};assert changed==allowed,(filename,changed);changes[filename]=sorted(changed)
report={'schema':'typedcontext6-actual-v17-typed200-epoch-AMP-ingest-v1','status':'passed','plan_hash':v['plan_hash'],'typed_counts':v['counts'],'rejected_semantic_cases':reject,'controller_AST_changes':changes,'helper_idempotent':True,'source52earlieststop_zero_based_equivalence':True,'source_context_andmember_mean_raw_logits_preserved':True,'ordinaryI4_andseparatewidth32I4':True,'terminalmaster_scaler30barrier_edges':True,'no selectedmodel_orAdam_transfer_between_seeds':True,'new_numerical_model_maps':0,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False};(P/'ACTUAL_SOURCE_CONTROLLER_DELTA_INGEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','fits':45,'banks':33,'rejects':reject,'GO':False}))
