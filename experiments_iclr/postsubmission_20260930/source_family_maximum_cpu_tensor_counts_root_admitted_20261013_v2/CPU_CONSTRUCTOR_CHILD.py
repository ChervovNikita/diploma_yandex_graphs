import argparse,hashlib,json,gzip,os,sys,socket,time,traceback,inspect,resource,types
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--stage',required=True);p.add_argument('--case',required=True);a=p.parse_args()
assert socket.gethostname()=='anogena-2-0';assert os.environ['CUDA_VISIBLE_DEVICES']==''
root=Path(a.phase).resolve();stage=root/a.stage;requests=json.loads((stage/'REQUESTS.json').read_text());case=next(r for r in requests if r['case_id']==a.case);out=stage/case['case_id'];out.mkdir()
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def receipt(q):return {'path':str(q.relative_to(root)),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
plan=json.loads((root/'massive_practical_master_20261013_v24/PLAN.json').read_text());assert hashlib.sha256(canon(plan)).hexdigest()=='8f691a4d7b9d833595d531b75b7cb78a3cba0cdb06d00015a6e15355e1883957'
bound=next(s for s in plan['studies'] if Path(s['path']).name in (case['study_id']+'.json',case['study_id']+'.json.gz'));q=root/bound['path'];raw=q.read_bytes();study=json.loads(gzip.decompress(raw) if q.suffix=='.gz' else raw);assert hashlib.sha256(canon(study)).hexdigest()==bound['spec_hash'];run=next(r for r in study['runs'] if r['run_id']==case['run_id']);assert hashlib.sha256(canon(run['config'])).hexdigest()==case['config_sha256'];source=root/case['factory'].split(':',1)[0];assert receipt(source)==case['factory_receipt'];assert run['adapter']==case['factory']
sys.path.insert(0,str(root/'master_controller_source_20261013_v22/runtime'))
result={'schema':'actual-source-family-CPU-constructor-tensor-count-v1','case':case,'scientific_fits':0,'optimizer_maps':0,'TRAIN_callbacks':0,'SELECT_scoring':0,'heldout_scoring':0,'model_forward_calls':0,'fabricated_paid_parent_states':0,'device':'cpu','GO':False,'RSS_cap_bytes':34359738368,'timeout_seconds':600,'full_branch_or_trained_state_admission':False};start=time.monotonic()
try:
 import torch
 torch.set_num_threads(1)
 from run_study import load_factory
 import engine
 denies={'module_forward_attempts':0,'optimizer_step_attempts':0,'backward_attempts':0,'VJP_attempts':0}
 def no_forward(*args,**kw):denies['module_forward_attempts']+=1;raise RuntimeError('CPU constructor admission excludes every neural model forward')
 def no_step(*args,**kw):denies['optimizer_step_attempts']+=1;raise RuntimeError('CPU constructor admission excludes every optimizer map')
 def no_backward(*args,**kw):denies['backward_attempts']+=1;raise RuntimeError('CPU constructor admission excludes every backward')
 def no_grad(*args,**kw):denies['VJP_attempts']+=1;raise RuntimeError('CPU constructor admission excludes every autograd VJP')
 old_init=torch.optim.Optimizer.__init__
 def optimizer_init(self,*args,**kw):
  old_init(self,*args,**kw);self.step=types.MethodType(no_step,self)
 forbidden=set()
 def scan(v):
  if isinstance(v,dict):
   for k,x in v.items():
    if ('cal' in k.lower() or 'eval' in k.lower()) and 'target' in k.lower():
     if isinstance(x,str):forbidden.add((root/x).resolve())
     elif isinstance(x,dict) and isinstance(x.get('path'),str):forbidden.add((root/x['path']).resolve())
    scan(x)
  elif isinstance(v,list):
   for x in v:scan(x)
 scan(study)
 old_load=torch.load
 def guarded_load(path,*args,**kw):
  if isinstance(path,(str,Path)) and Path(path).resolve() in forbidden:raise RuntimeError('CPU constructor cannot read sealed endpoint target files')
  return old_load(path,*args,**kw)
 with patch.object(torch.nn.Module,'_call_impl',no_forward),patch.object(torch.optim.Optimizer,'__init__',optimizer_init),patch.object(torch.Tensor,'backward',no_backward),patch.object(torch.autograd,'grad',no_grad),patch.object(torch,'load',guarded_load):
  factory=load_factory(root,case['factory']);arguments={'torch':torch,'project_root':root,'study':study,'run':run,'device':'cpu'}
  if 'execution_context' in inspect.signature(factory).parameters:arguments['execution_context']={}
  adapter=factory(**arguments)
  # No train_step, predict, validate, selector, source model forward or paid
  # reconstruction is called. Only source-defined initialized capture state.
  capture=adapter.capture_state();model=adapter.model;optimizer=adapter.optimizer
  payload=engine.capture_checkpoint(torch,adapter,fit_id='CPU_ONLY_UNTRAINED_'+case['case_id'],step=0,validation={'objective':0.,'ce':0.,'regularization':0.},decision={'save_selected':False,'stop_requested':False})
  def tensor_bytes(v):
   memo=set();raw=0;count=0
   def walk(x):
    nonlocal raw,count
    if torch.is_tensor(x):
     if x.layout==torch.strided:
      key=(x.device,x.dtype,x.untyped_storage().data_ptr(),x.untyped_storage().nbytes(),x.storage_offset(),tuple(x.shape),tuple(x.stride()))
      if key in memo:return
      memo.add(key);raw+=x.numel()*x.element_size();count+=1
     elif x.layout==torch.sparse_coo:
      walk(x._indices());walk(x._values())
     else:
      walk(x.values())
      for fn in ('crow_indices','col_indices') if 'csr' in str(x.layout) else ('ccol_indices','row_indices'):walk(getattr(x,fn)())
    elif isinstance(x,dict):
     for z in x.values():walk(z)
    elif isinstance(x,(tuple,list)):
     for z in x:walk(z)
   walk(v);return {'unique_tensor_payload_bytes':raw,'distinct_tensor_views':count}
  P=sum(p.numel()*p.element_size() for p in model.parameters());owned=[p for group in optimizer.param_groups for p in group['params']];owned_bytes=sum(p.numel()*p.element_size() for p in owned)
  snapshot=out/'DISPOSABLE_CPU_ONLY_UNTRAINED_ENGINEERING_STATE.pt';torch.save(payload,snapshot);serialized=receipt(snapshot)
  result.update(status='PASS_CPU_INITIAL_SOURCE_STATE',model_parameter_bytes=P,model_parameter_objects=sum(1 for p in model.parameters()),optimizer_owned_parameter_bytes=owned_bytes,optimizer_class=type(optimizer).__module__+'.'+type(optimizer).__qualname__,model_state=tensor_bytes(model.state_dict()),adapter_initial_state=tensor_bytes(capture),optimizer_initial_state=tensor_bytes(optimizer.state_dict()),full_initial_payload=tensor_bytes(payload),initial_engine_payload_serialization=serialized,standard_Adam_moment_payload_upper_if_source_Adam_is_confirmed_bytes=2*owned_bytes+8*len(owned),initial_primitive_header_and_archive_bytes=serialized['bytes']-tensor_bytes(payload)['unique_tensor_payload_bytes'],constructor_denials=denies,source_factory_after=receipt(source))
  snapshot.unlink();result['disposable_fixture_removed']=True
except BaseException as ex:
 result.update(status='FINITE_CONSTRUCTOR_BLOCKER',exception_type=type(ex).__name__,exception=str(ex),traceback=traceback.format_exc(),constructor_denials=locals().get('denies'))
result['wall_seconds']=time.monotonic()-start;result['RSS_peak_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
(out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('case','traceback','source_factory_after')},sort_keys=True));sys.exit(0 if result['status']=='PASS_CPU_INITIAL_SOURCE_STATE' else 2)
