"""Standard-library ancestry/accounting test: no model, optimizer, tensor or target decode."""
from pathlib import Path
import copy,gzip,hashlib,json,sys,tempfile,types
HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
sys.path[:0]=[str(HERE/'runtime'),str(PHASE/'preaggregation_growth_new_data_packet_20261012_v2/runtime')]
import compile_master_plan as master
import run_study
study=json.loads(gzip.decompress(next((PHASE/'preaggregation_growth_new_data_packet_20261012_v2/matrix/studies').glob('*.json.gz')).read_bytes()))
checks=[]
def rejects(label,fn):
 try:fn()
 except ValueError:checks.append(label);return
 raise AssertionError('Unexpected admission: '+label)
records,deps=master.graph_check([study],{});counts=master.typed_counts([study])
t=counts['totals'];assert t['atomic_fit_run_groups']==147 and t['paid_optimizer_starts']==192 and t['physical_slow_body_instances']==192
assert counts['model_optimizer_step_call_total']==2206980 and t['growth_registered_group_outer_cycles']==9360 and t['growth_registered_owner_outer_cycles']==12060
assert t['growth_new_paid_warm_fit_groups']==15 and t['growth_paid_post_fit_groups']==75 and t['paid_continuation_segments']==75 and t['initial_training_segments']==72
assert t['genuine_separately_selected_I4_banks']==9
checks.append('whole370row_growth147fit_typed_ingestion_exact_counts')
runs={r['run_id']:r for r in study['runs']};post=next(r for r in study['runs'] if r['config'].get('growth_stage')=='post');warm=runs[post['parents'][0]]
run_study.require_growth_terminal20_parent(warm,post);checks.append('exact_new_terminal20_parent_contract')
for key in ('initialization_seed','factor_seed','learning_rate','link_inputs','growth_sources','calibration_draw'):
 bad=copy.deepcopy(warm);bad['config'][key]='bad-source';rejects('parent_'+key+'_rejected',lambda b=bad:run_study.require_growth_terminal20_parent(b,post))
for key,value in (('metric','MRR'),('direction','min'),('ties','last'),('evaluation_interval',5),('checkpoint0',True)):
 bad=copy.deepcopy(warm);bad['selection'][key]=value;rejects('parent_selection_'+key+'_rejected',lambda b=bad:run_study.require_growth_terminal20_parent(b,post))
for key in ('split','seed','body'):
 bad=copy.deepcopy(warm);bad[key]+=1;rejects('parent_'+key+'_rejected',lambda b=bad:run_study.require_growth_terminal20_parent(b,post))
bad=copy.deepcopy(study);p=next(r for r in bad['runs'] if r['run_id']==post['run_id']);p['parents']=[];p['config'].pop('lifecycle');rejects('unpaid_post_rejected',lambda:master.graph_check([bad],{}))
for r in [next(r for r in study['runs'] if r['config'].get('growth_stage')==phase) for phase in ('warm','post','cold_control')]:
 for key in ('source_optimizer_count','physical_slow_bodies','source_complete_episodes_per_cycle','source_Adam_passes_per_episode','train_minibatches_per_epoch'):
  bad=copy.deepcopy(r);bad['config'][key]+=1;rejects(r['config']['growth_stage']+'_'+key+'_rejected',lambda b=bad:master.fit_semantics(b))
quad=next(r for r in study['runs'] if r['config'].get('growth_arm')=='independent_graph_growth4');q=master.fit_semantics(quad);assert q['optimizer_objects']==q['physical_slow_bodies']==4 and q['known_optimizer_step_calls']==60*61*3*4
original=next(r for r in study['runs'] if r['config'].get('growth_arm')=='original_gnnm');assert master.fit_semantics(original)['known_source_TRAIN_backward_calls']==80*61*12
checks.extend(['four_independent_source_Adam_owners_43920transitions','original_four_backwards_each_native_Adam_pass'])
# Use only the engine's metadata receipt law; avoid importing torch.
def checkpoint_receipt(path,*,fit_id,step):
 raw=Path(path).read_bytes();return {'path':Path(path).name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'fit_id':fit_id,'step':step}
engine=types.ModuleType('engine');engine.checkpoint_receipt=checkpoint_receipt;sys.modules['engine']=engine
with tempfile.TemporaryDirectory() as temporary:
 root=Path(temporary);out=root/'execution';path=out/study['study_id']/warm['run_id']/'selected.pt';path.parent.mkdir(parents=True);path.write_bytes(b'opaque metadata authority only')
 bound=checkpoint_receipt(path,fit_id=warm['run_id'],step=20)
 record={'state':'complete','contract':warm,'result':{'status':'complete','completed_steps':20,'selected':{'step':20},'selected_checkpoint':bound}}
 class Registry:
  def run(self,study_id,identifier):assert study_id==study['study_id'] and identifier==warm['run_id'];return record
 registry=Registry();result=run_study.paid_parent_checkpoints(root,study,post,registry,out);assert result[warm['run_id']]['receipt']==bound
 checks.append('runtime_terminal20_hash_authority_admitted')
 record['result']['selected_checkpoint']['step']=19;record['result']['selected']['step']=19
 rejects('runtime_nonterminal_selected19_rejected',lambda:run_study.paid_parent_checkpoints(root,study,post,registry,out))
 record['result']['selected_checkpoint']['step']=20;record['result']['selected']['step']=20;record['result']['completed_steps']=19
 rejects('runtime_incomplete_warm19_rejected',lambda:run_study.paid_parent_checkpoints(root,study,post,registry,out))
 record['result']['completed_steps']=20;path.write_bytes(b'tampered opaque authority')
 rejects('runtime_selected_byte_hash_tamper_rejected',lambda:run_study.paid_parent_checkpoints(root,study,post,registry,out))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps({'schema':'growth-frozen-v9-metadata-QA-v1','checks':checks,'checks_passed':len(checks),'controller_counts':counts,'numerical_source_tests_repeated':0,'models_constructed':0,'optimizer_updates':0,'heldout_target_decodes':0,'CUDA':0,'GO':False}))
