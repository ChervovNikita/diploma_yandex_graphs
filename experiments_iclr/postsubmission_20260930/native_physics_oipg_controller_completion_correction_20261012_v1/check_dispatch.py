"""Invoke exact execute pre-fit path with actual metadata validators, no registry/fits."""
import copy,gzip,hashlib,importlib.util,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;PHASE=OUT.parent
sys.path[:0]=[str(OUT/'base'),str(PHASE/'native_physics_oipg_newroles_packet_20261012_v1/runtime')]
class DispatchPassed(RuntimeError):pass
def denied_registry(*a,**kw):raise DispatchPassed('Stopped at registry construction, before fit admission and numerical imports')
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def fixtures():
 result=[]
 for name,policy in [('native_physics_oipg_newroles_packet_20261012_v1/matrix/studies/native-Physics-joint-dropout-newroles-v1.json.gz','tabm_native_selector_stop'),
  ('native_hgt_mixed_newdata_packet_20261012_v2/matrix/studies/native-HGT-mixed-DBLP-newroles-v2.json.gz','hgt_native_selector_stop'),
  ('native_meansage_peer_tour_newdata_packet_20261012_v1/matrix/peer/studies/native-meansage-peer-photo-v1.json.gz','native_selector_stop')]:
  s=json.loads(gzip.decompress((PHASE/name).read_bytes()));seen=set()
  for r in s['runs']:
   if r['kind']!='model_fit' or r['budget']['policy']!=policy:continue
   key=r['config'].get('physics_arm',r['config'].get('hgt_arm',r['config'].get('meansage_arm')))
   if key in seen:continue
   seen.add(key);result.append((key,copy.deepcopy(r)))
 from bsnn_stopping import BUDGET,SELECTION,VARIANTS
 for variant,(layers,second,dropout,input_dropout) in VARIANTS.items():
  for arm in ('native_M1','original_GNNM'):
   members=4 if arm=='original_GNNM' else 1;seed=6101
   c={'bayesian_bsnn_variant':variant,'bayesian_bsnn_arm':arm,'architecture':'native_Bayesian_BSNN_typed','backbone':'original_BayesBundleSheafDiffusion_d2',
    'classes':2,'width':32,'depth':layers,'second_linear':second,'dropout':dropout,'input_dropout':input_dropout,'learning_rate':.01,'weight_decay':.0005,
    'members':members,'physical_slow_bodies':1,'source_optimizer_count':1,'source_complete_body_constructors':1,'source_TRAIN_forwards_per_update':members,
    'source_TRAIN_backwards_per_update':members,'source_SELECT_draws':4*members,'source_KL_rule':'native_incidence_mean_sigmoid40_no_layer_multiplier',
    'source_optimizer_rule':'source_grouped_Adam_lr.01_both_decay.0005','historical_state_inherited':False,'is_original_gnnm':arm=='original_GNNM','initialization_seed':seed,
    'fusion':'mean_logits' if arm=='original_GNNM' else 'mean_probabilities'}
   result.append((variant+'-'+arm,{'kind':'model_fit','run_id':variant+'-'+arm,'seed':seed,'body':0,'budget':BUDGET,'selection':SELECTION,'config':c,'adapter':'fixture/adapters/bsnn_bayesian.py:make_adapter'}))
 result.append(('full_budget',{'kind':'model_fit','run_id':'full','budget':{'policy':'full_budget'},'config':{}}))
 return result
def exercise(module,run):
 module.load=lambda *_:({'runs':[run]},'engineering_memory_only_hash')
 module.require_approval=lambda *_:{'engineering_approval_stub':True,'GO_receipt_created':False}
 module.require_phase_approval=lambda *_:None
 module.Registry=denied_registry
 try:module.execute(PHASE,'unused_memory_fixture',run['run_id'],'unused_registry','unused_output','unused_approval','cpu')
 except DispatchPassed:return {'dispatch':'passed','stopped_before_registry_and_model':True}
 except ValueError as error:return {'dispatch':'rejected','message':str(error)}
 raise AssertionError('Execute unexpectedly crossed pre-fit boundary')
def main():
 old=load_module(OUT/'base/run_study.py','frozen_v18_runner_dispatch');new=load_module(OUT/'run_study.py','corrected_runner_dispatch')
 import native_stopping,tabm_stopping,hgt_stopping,bsnn_stopping
 modules=[(native_stopping,'require_completion'),(tabm_stopping,'validate_tabm_run'),(hgt_stopping,'validate_hgt_run'),(bsnn_stopping,'validate_bsnn_run')];calls=[]
 for module,name in modules:
  function=getattr(module,name)
  def wrapped(run,f=function,n=name):calls.append(n);return f(run)
  setattr(module,name,wrapped)
 results=[];old_rejections=[]
 for label,run in fixtures():
  calls.clear();before=exercise(old,copy.deepcopy(run));before_calls=list(calls);calls.clear();after=exercise(new,copy.deepcopy(run));after_calls=list(calls)
  if after['dispatch']!='passed' or len(after_calls)!=1:raise AssertionError('Exactly one actual validator/fallback per supported policy required: '+label)
  affected=run['budget']['policy'] in ('tabm_native_selector_stop','bsnn_native_selector_stop')
  if affected:
   if before['dispatch']!='rejected' or before.get('message')!='Unsupported completion policy':raise AssertionError('Original pre-fit rejection not reproduced')
   old_rejections.append(label)
  elif before['dispatch']!='passed':raise AssertionError('Existing HGT/native/full path changed')
  results.append({'case':label,'policy':run['budget']['policy'],'before':before,'before_actual_functions':before_calls,'after':after,'after_actual_functions':after_calls})
 calls.clear();unsupported={'kind':'model_fit','run_id':'unsupported','budget':{'policy':'unknown'},'config':{}};denied=exercise(new,unsupported)
 assert denied['dispatch']=='rejected' and calls==['require_completion']
 assert not any(name in sys.modules for name in ('torch','numpy','dgl','ogb'))
 proof={'schema':'v18-exact-execute-completion-dispatch-correction-proof-v1','supported_cases':len(results),'cases':results,'reproduced_pre_fit_rejections':old_rejections,
  'unsupported_policy_still_rejected':denied,'actual_execute_function_invoked':True,'actual_metadata_validator_bodies_invoked':True,
  'memory_only_loader_and_approval_stubs_explicit':True,'registry_creation_denied':True,'numerical_imports':False,'models_constructed':0,'fits':0,'scores':0,'GO_receipt_created':False,
  'loader_only_audit_would_not_detect_this_execute_rejection':True,'corrected_runner_sha256':hashlib.sha256((OUT/'run_study.py').read_bytes()).hexdigest()}
 path=OUT/'DISPATCH_PROOF.json'
 if path.exists():raise FileExistsError('Preserve proof')
 path.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n');print(json.dumps({'cases':len(results),'reproduced_rejections':len(old_rejections),'exact_execute_dispatch_passed':True,'models':0,'fits':0,'GO':False}))
if __name__=='__main__':main()
