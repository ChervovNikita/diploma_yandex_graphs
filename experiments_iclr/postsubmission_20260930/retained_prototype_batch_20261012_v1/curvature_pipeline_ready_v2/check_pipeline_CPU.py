"""New tiny stage serialization/recovery checks. No actual provider arrays."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(P/'controller_delta_v10'),str(ROOT/'master_controller_source_20261012_v10/runtime')]
from run_study import load_factory,require_curvature_paid_parent,paid_parent_ids
from engine import run_trajectory,restore_checkpoint
from spec import fingerprint

def equal(torch,a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(torch,a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(torch,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch,gzip
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 packet=json.loads((P/'MASTER_INPUT_ENTRY.json').read_text());plan=json.loads((ROOT/packet['plan']['path']).read_text())
 study=json.loads(gzip.decompress((ROOT/plan['studies'][0]['path']).read_bytes()));records={r['run_id']:r for r in study['runs']}
 trial=next(r for r in study['runs'] if r['config'].get('curvature_stage')=='trials')
 warm=records[paid_parent_ids(trial)[0]]
 cont=next(r for r in study['runs'] if r['config'].get('curvature_stage')=='continuation')
 require_curvature_paid_parent(warm,trial);require_curvature_paid_parent(trial,cont)
 rejects=0
 for field,value in (('seed',999),('split',9),('body',1)):
  bad=copy.deepcopy(warm);bad[field]=value
  try:require_curvature_paid_parent(bad,trial)
  except ValueError:rejects+=1
  else:raise AssertionError('Mismatched source cohort admitted')
 for key,value in (('dropout',.85),('curvature_fixed_last',False),('node_inputs',{}),('selected_config_from','other')):
  bad=copy.deepcopy(warm);bad['config'][key]=value
  try:require_curvature_paid_parent(bad,trial)
  except ValueError:rejects+=1
  else:raise AssertionError('Mismatched source custody admitted')
 factory=load_factory(ROOT,trial['adapter']);space=factory.__globals__;port=space['source'];Class=space['CurvatureAdapter']
 sources=port.modules(ROOT,json.loads((P/'SOURCE_BINDINGS.json').read_text()),{'learning_rate':.001,'dropout':.7})
 api=sources['native_api'];integration=sources['integration'];boundary=sources['backbone_boundary_adapter']
 n=12;torch.manual_seed(2701);x=torch.rand(n,745);ids=torch.arange(n)
 edge=torch.stack((ids.repeat(3),torch.cat((ids,(ids+1)%n,(ids-1)%n))))
 graph=SimpleNamespace(teacher_input=x,teacher_edge_index=edge,teacher_backbone='polynormer_r',classes=8)
 train=SimpleNamespace(nodes=ids[:10],labels=ids[:10]%8);select=SimpleNamespace(nodes=ids[10:],labels=ids[10:]%8)
 spec=api.specification('polynormer_r','single_author',0,88101)
 fixture=P/'CPU_lifecycle_fixture';fixture.mkdir(exist_ok=True)
 def construct_native():
  with port.aliases(sources):m=api.TeacherFamily(spec);o=api.optimizer_for(m)
  return m,o
 def engine_spec(name,steps,zero=False):
  return dict(fit_id=name,budget_steps=steps,budget_unit='epochs' if name=='toy-trials' else 'optimizer_updates',validation_interval=1,
   validate_initial=zero,prefixes=[],inherited_fit_ids=[],adapter_spec={'scope':'shortened synthetic CPU source lifecycle only','fixture_local2_global1_continuation2':True})
 def provenance(name):return {'run_id':name,'cohort_seed':88101,'actual_model_seed':88101,'paid_parent_receipt':{'scope':'own shortened synthetic fixture only'},'not_a_scientific_contract':True}
 m,o=construct_native();a=Class(torch,sources,m,o,graph,train,select,stage='warm',arm='joint_five_arm_source',provenance=provenance('toy-warm'),native_spec=spec,local_updates=2,maximum=3)
 warm_dir=fixture/'warm'
 if not (warm_dir/'result.json').exists() or json.loads((warm_dir/'result.json').read_text())['status']!='complete':
  run_trajectory(torch,a,engine_spec('toy-warm',3),warm_dir)
 warm_payload=torch.load(warm_dir/'selected.pt',weights_only=False);warm_state=warm_payload['adapter_state']['external']
 assert warm_payload['step']==3 and warm_state['native_warm']['metadata']['actual_updates']==3
 assert warm_state['native_warm']['metadata']['native_local_rng_restored'] is False
 assert json.loads((warm_dir/'result.json').read_text())['operation_counts']['Adam_constructor_objects']==2
 with port.aliases(sources):native,nativeopt=integration.restore_native(api,warm_state['native_warm'],'cpu')
 raw=integration.clone_boundary(native,boundary,4);opt,_=integration.transport_optimizer(native,warm_state['native_warm']['optimizer'],raw)
 constants=sources['selector'].FrozenConstants(**json.loads((P/'PROSPECTIVE_CONSTANTS.json').read_text())['values'])
 t=Class(torch,sources,raw,opt,graph,train,None,stage='trials',arm='joint_five_arm_source',provenance=provenance('toy-trials'),native_spec=spec,
  trial_parent=warm_state,constants=constants,constructor_objects=2)
 t.native_for_trial=native;integration.rng_restore(warm_state['native_warm']['rng'])
 trial_dir=fixture/'trials'
 if not (trial_dir/'result.json').exists() or json.loads((trial_dir/'result.json').read_text())['status']!='complete':
  run_trajectory(torch,t,engine_spec('toy-trials',1),trial_dir)
 trial_payload=torch.load(trial_dir/'selected.pt',weights_only=False);ts=trial_payload['adapter_state']['external']
 assert set(ts['trial_states'])==set(space['ARMS']) and ts['trial_receipt']['SELECT_targets_read'] is False
 assert 1<=ts['trial_receipt']['trial_maps_started']<=10
 tr=json.loads((trial_dir/'result.json').read_text());maps=tr['operation_counts']['optimizer_updates']
 assert maps==ts['trial_receipt']['trial_maps_completed']
 assert tr['operation_counts']['Adam_constructor_objects']==7+ts['trial_receipt']['trial_maps_started']
 chosen=ts['trial_states']['selected_graph_pair'];arm='selected_graph_pair'
 native,_=construct_native();native.set_global_stage(True);raw=integration.clone_boundary(native,boundary,4)
 raw.load_state_dict(chosen['model']);opt=integration.restore_named_optimizer(raw,chosen['optimizer']);integration.rng_restore(chosen['rng'])
 c=Class(torch,sources,raw,opt,graph,train,select,stage='continuation',arm=arm,provenance=provenance('toy-continuation'),native_spec=spec,maximum=2)
 cont_dir=fixture/'continuation'
 if not (cont_dir/'result.json').exists() or json.loads((cont_dir/'result.json').read_text())['status']!='complete':
  run_trajectory(torch,c,engine_spec('toy-continuation',2,True),cont_dir)
 result=json.loads((cont_dir/'result.json').read_text());assert result['completed_steps']==2
 assert result['validation_calls']==3 and result['operation_counts']['optimizer_updates']==2
 restore_checkpoint(torch,cont_dir/'final_training.pt',c,expected_fit_id='toy-continuation')
 before=c.capture_state();restore_checkpoint(torch,cont_dir/'final_training.pt',c,expected_fit_id='toy-continuation')
 assert equal(torch,before,c.capture_state())
 # Plain pretrial checkpoint material cannot contain modules/optimizers.
 def plain(v):
  if torch.is_tensor(v) or v is None or type(v) in (str,bool,int,float):return True
  if isinstance(v,dict):return all(plain(x) for x in v.values())
  if isinstance(v,(list,tuple)):return all(plain(x) for x in v)
  return False
 assert plain(ts['trial_states']) and not torch.cuda.is_initialized()
 counts={name:json.loads((fixture/name/'result.json').read_text())['operation_counts'] for name in ('warm','trials','continuation')}
 value={'schema':'curvature-paid-pipeline-new-lifecycle-CPU-QA-v2','status':'passed','cases':[
  'seven_mismatched_paid_source_contracts_rejected','source_local_named_Adam_live_RNG_fixed_last_warm',
  'literal_joint_five_arm_pretrial_serialization_and_bounded_actual_trial_costs','source_continuation_epoch0_and_repeat_restore_exact'],
  'metadata_rejection_cases':rejects,'successful_fixture_model_maps':sum(r['optimizer_updates'] for r in counts.values()),
  'stage_operation_counts':counts,'trial_maps_started':ts['trial_receipt']['trial_maps_started'],
  'fixture_stage_budgets':{'local':2,'global_warm':1,'continuation':2},
  'shortened_fixture_is_not_scientific_parent_authority':True,'SELECT_read_in_trials':False,
  'closed_Q12_Q21_audits_rerun':False,'actual_input_arrays_opened':0,'scientific_fits':0,'CUDA_compute':0,'GO':False}
 (P/'PIPELINE_CPU_QA.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
