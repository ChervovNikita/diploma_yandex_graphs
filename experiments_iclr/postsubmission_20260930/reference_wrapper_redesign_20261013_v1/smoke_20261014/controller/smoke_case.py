"""Real-source short smoke and checkpoint replay; no synthetic data/model fallback.

Callbacks own actual source loading/model/loss/scheduler and TRAIN/validation.
The controller owns global RNG, costs, actual optimizer calls and output records.
"""
from __future__ import annotations
import argparse,copy,functools,gc,hashlib,importlib.util,json,math,os,platform,random,resource,socket,subprocess,sys,time,traceback
from pathlib import Path

ALLOCATION_UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REPLAY_CRITERION='exact-loaded-state_then_tensor-RMS-v2_20261014'

def require(value,message):
 if not value:raise ValueError(message)
def json_fallback(value):
 if isinstance(value,Path):return str(value)
 if hasattr(value,'item'):
  try:return value.item()
  except (ValueError,RuntimeError):pass
 return str(value)
def atomic_json(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()));temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=json_fallback)+'\n');temporary.replace(path)
def file_receipt(path):
 path=Path(path);h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def require_allocation():
 require(socket.gethostname()=='anogena-2-0','Smoke must use the authorized one-GPU allocation')
 value=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
 require(value==[ALLOCATION_UUID],'Wrong allocation GPU route; no model may be constructed')
def own_memory(torch):
 rss=None;peak=None
 p=Path('/proc/self/status')
 if p.exists():
  rows={r.split(':',1)[0]:r.split(':',1)[1].strip() for r in p.read_text().splitlines() if ':' in r}
  def kb(name):return int(rows[name].split()[0])*1024 if name in rows else None
  rss,peak=kb('VmRSS'),kb('VmHWM')
 if peak is None:peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=='Darwin' else 1024)
 return {'rss_bytes':rss,'peak_rss_bytes':peak,'cuda_allocated_bytes':torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,'cuda_reserved_bytes':torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
def synchronize(torch):
 if torch.cuda.is_available():torch.cuda.synchronize()
def global_rng(torch):
 import numpy as np
 return {'python':random.getstate(),'numpy':copy.deepcopy(np.random.get_state()),'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def rng_digest(torch,state):
 packed={'python':state['python'],'numpy':[state['numpy'][0],state['numpy'][1].tolist(),*state['numpy'][2:]],'torch_cpu':hashlib.sha256(state['torch_cpu'].cpu().numpy().tobytes()).hexdigest(),'torch_cuda':[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in state['torch_cuda']]}
 return hashlib.sha256(json.dumps(packed,sort_keys=True).encode()).hexdigest()
def restore_rng(torch,state):
 import numpy as np
 random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
 if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['torch_cuda'])
def cpu_clone(torch,value):
 if torch.is_tensor(value):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu_clone(torch,v) for k,v in value.items()}
 if isinstance(value,list):return [cpu_clone(torch,v) for v in value]
 if isinstance(value,tuple):return tuple(cpu_clone(torch,v) for v in value)
 return copy.deepcopy(value)
def leaves(torch,value,path='root'):
 if torch.is_tensor(value):
  if value.layout==torch.strided:yield path,value
  elif value.layout==torch.sparse_coo:
   v=value.coalesce();yield path+'/sparse_indices',v.indices();yield path+'/sparse_values',v.values()
  elif value.layout in (torch.sparse_csr,torch.sparse_bsr):
   yield path+'/crow_indices',value.crow_indices();yield path+'/col_indices',value.col_indices();yield path+'/sparse_values',value.values()
  elif value.layout in (torch.sparse_csc,torch.sparse_bsc):
   yield path+'/ccol_indices',value.ccol_indices();yield path+'/row_indices',value.row_indices();yield path+'/sparse_values',value.values()
  else:raise ValueError('Unsupported source-state tensor layout:'+str(value.layout))
 elif isinstance(value,dict):
  for k in sorted(value,key=str):yield from leaves(torch,value[k],path+'/'+str(k))
 elif isinstance(value,(list,tuple)):
  for i,v in enumerate(value):yield from leaves(torch,v,path+'/'+str(i))
def finite_tree(torch,value,label):
 count=0
 for name,tensor in leaves(torch,value):
  count+=1
  if tensor.is_floating_point() or tensor.is_complex():require(bool(torch.isfinite(tensor).all()),'Nonfinite '+label+':'+name)
 return count
def tensor_summary(torch,value):
 rows=[]
 for name,tensor in leaves(torch,value):
  t=tensor.detach().cpu().contiguous();raw=t.reshape(-1).view(torch.uint8).numpy().tobytes()
  rows.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'elements':t.numel(),'sha256':hashlib.sha256(raw).hexdigest()})
 return {'tensor_leaves':len(rows),'elements':sum(r['elements'] for r in rows),'tensors':rows[:24],'tensor_list_truncated':len(rows)>24,'all_tensor_metadata_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
def compare_trees(torch,left,right,rtol,atol,*,tensor_policy='coordinate'):
 """Compact tolerance comparison, never a bit-equality investigation."""
 require(tensor_policy in ('coordinate','RMS'),'Unknown engineering tensor criterion')
 a=dict(leaves(torch,left));b=dict(leaves(torch,right));mismatches=[];maximum=0.;maximum_rms=0.;compared=0;differences=[]
 for name in sorted(set(a)|set(b)):
  if name not in a or name not in b:mismatches.append({'name':name,'reason':'missing tensor'});continue
  x,y=a[name].detach().cpu(),b[name].detach().cpu()
  if x.shape!=y.shape or x.dtype!=y.dtype:mismatches.append({'name':name,'reason':'shape/dtype'});continue
  compared+=1
  if x.is_floating_point() or x.is_complex():
   # CPU float64/complex128 arithmetic prevents the criterion itself from
   # losing small differences. Per-tensor RMS equals L2/sqrt(number of elements).
   promoted=torch.complex128 if x.is_complex() else torch.float64;xd,yd=x.to(promoted),y.to(promoted);diff=(xd-yd).abs();delta=float(diff.max()) if x.numel() else 0.;rms=float(diff.square().mean().sqrt()) if x.numel() else 0.;reference_rms=float(yd.abs().square().mean().sqrt()) if y.numel() else 0.;threshold=atol+rtol*reference_rms;maximum=max(maximum,delta);maximum_rms=max(maximum_rms,rms)
   ok=(math.isfinite(rms) and math.isfinite(reference_rms) and rms<=threshold) if tensor_policy=='RMS' else torch.allclose(x,y,rtol=rtol,atol=atol,equal_nan=False)
   differences.append({'name':name,'max_abs_diff':delta,'rms_diff':rms,'reference_rms':reference_rms,'rms_threshold':threshold,'passed':bool(ok)})
  else:delta=0.;ok=torch.equal(x,y)
  if not ok:mismatches.append({'name':name,'reason':'outside declared engineering tolerance','max_abs_diff':delta})
 # Scalars such as source phase/counter flags are also compared when callbacks expose them.
 def scalar_rows(v,path='root'):
  if isinstance(v,dict):
   for k in sorted(v,key=str):yield from scalar_rows(v[k],path+'/'+str(k))
  elif isinstance(v,(list,tuple)):
   for i,x in enumerate(v):yield from scalar_rows(x,path+'/'+str(i))
  elif not torch.is_tensor(v) and isinstance(v,(int,float,bool,str,type(None))):yield path,v
 sa,sb=dict(scalar_rows(left)),dict(scalar_rows(right))
 for name in sorted(set(sa)|set(sb)):
  x,y=sa.get(name),sb.get(name)
  if isinstance(x,float) and isinstance(y,float):ok=math.isfinite(x) and math.isfinite(y) and abs(x-y)<=atol+rtol*abs(y)
  else:ok=type(x)==type(y) and x==y
  if not ok:mismatches.append({'name':name,'reason':'source scalar/clock differs'})
 differences.sort(key=lambda row:row['max_abs_diff'],reverse=True)
 return {'passed':not mismatches,'rtol':rtol,'atol':atol,'floating_tensor_policy':tensor_policy,'tensor_leaves_compared':compared,'maximum_tensor_absolute_difference':maximum,'maximum_tensor_RMS_difference':maximum_rms,'largest_tensor_differences':differences[:8],'tensor_difference_list_truncated':len(differences)>8,'mismatch_count':len(mismatches),'mismatches':mismatches[:8],'mismatch_list_truncated':len(mismatches)>8}
def compare_loaded_exact(torch,left,right):
 """Check serialization/restoration before any nondeterministic training work."""
 import numpy as np
 mismatches=[];tensor_count=0;scalar_count=0;array_count=0
 def walk(a,b,path='root'):
  nonlocal tensor_count,scalar_count,array_count
  if torch.is_tensor(a) or torch.is_tensor(b):
   if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.shape!=b.shape or a.dtype!=b.dtype or a.layout!=b.layout:mismatches.append({'name':path,'reason':'tensor type/shape/dtype/layout'});return
   aa=dict(leaves(torch,a,path));bb=dict(leaves(torch,b,path));tensor_count+=len(aa)
   if set(aa)!=set(bb) or any(not torch.equal(aa[k].detach().cpu(),bb[k].detach().cpu()) for k in aa):mismatches.append({'name':path,'reason':'loaded tensor values differ'})
  elif isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
   array_count+=1
   if not (isinstance(a,np.ndarray) and isinstance(b,np.ndarray)) or a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):mismatches.append({'name':path,'reason':'loaded numpy array differs'})
  elif isinstance(a,dict) or isinstance(b,dict):
   if not (isinstance(a,dict) and isinstance(b,dict)) or set(a)!=set(b):mismatches.append({'name':path,'reason':'mapping keys/type'});return
   for k in sorted(a,key=str):walk(a[k],b[k],path+'/'+str(k))
  elif isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
   if type(a)!=type(b) or len(a)!=len(b):mismatches.append({'name':path,'reason':'sequence type/length'});return
   for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
  else:
   scalar_count+=1
   try:ok=type(a)==type(b) and bool(a==b)
   except (ValueError,TypeError):ok=False
   if not ok:mismatches.append({'name':path,'reason':'loaded scalar/plain state differs'})
 walk(left,right)
 return {'passed':not mismatches,'criterion':'exact_values_before_resumed_update','tensor_leaves_compared':tensor_count,'numpy_arrays_compared':array_count,'plain_scalar_leaves_compared':scalar_count,'mismatch_count':len(mismatches),'mismatches':mismatches[:8],'mismatch_list_truncated':len(mismatches)>8}
def independent_bank(torch,session,raw,rtol,atol):
 if not getattr(session,'is_independent4',False):return None
 require(getattr(session,'bank_rule',None)=='mean_logits','I4 smoke requires declared raw-logit/raw-score mean')
 member=raw.get('member_outputs');pooled=raw['outputs']
 if isinstance(pooled,dict):member=pooled.get('members',pooled.get('member_logits',member));pooled=pooled.get('pooled',pooled.get('pooled_logits'))
 if isinstance(member,(list,tuple)):member=torch.stack(member)
 require(torch.is_tensor(member) and member.shape[0]==4 and torch.is_tensor(pooled),'I4 validate requires four real member outputs plus actual pooled output')
 require(pooled.shape==member.shape[1:],'I4 pooled shape differs')
 report=compare_trees(torch,{'pooled':member.mean(0)},{'pooled':pooled},rtol,atol);require(report['passed'],'I4 output is not the mean of its four actual selected native members');return report
def optimizers(session):
 value=getattr(session,'optimizers',None)
 if value is None:
  value=getattr(session,'optimizer',None);value=getattr(value,'optimizers',value)
 if not isinstance(value,(list,tuple)):value=[value]
 require(value and all(x is not None and hasattr(x,'step') and hasattr(x,'state_dict') for x in value),'Real optimizer owner(s) required')
 require(len({id(x) for x in value})==len(value),'Repeated optimizer object')
 return list(value)
def identity(torch,session,owners):
 params=dict(session.model.named_parameters());ids={id(p) for p in params.values()};anchors=getattr(session,'source_parameter_references',None)
 require(isinstance(anchors,dict) and anchors,'Actual original source Parameter anchors required')
 replaced=set(getattr(session,'replaced_source_parameter_names',[]));require(not replaced,'Reference-wrapper campaign requires empty replaced_source_parameter_names; native Parameter identities must remain')
 missing=[name for name,p in anchors.items() if name not in replaced and id(p) not in ids]
 require(not missing,'Original native Parameters lost outside declared replacement sites: '+str(missing[:8]))
 groups=[{id(p) for group in o.param_groups for p in group['params']} for o in owners]
 require(all(group<=ids for group in groups),'Optimizer owns a foreign Parameter')
 if len(owners)>1:
  require(all(not groups[i]&groups[j] for i in range(len(groups)) for j in range(i)),'I4 optimizer owners share Parameters')
 if getattr(session,'is_independent4',False):
  require(len(owners)==4 and isinstance(session.model,torch.nn.ModuleList) and len(session.model)==4,'I4 requires four native modules and four own optimizers')
  storage=[]
  for model in session.model:
   storage.append({(str(p.device),p.untyped_storage().data_ptr()) for p in model.parameters() if p.numel()})
  require(all(not storage[i]&storage[j] for i in range(4) for j in range(i)),'I4 members share parameter storage')
 return {'current_parameter_objects':len(params),'source_anchor_count':len(anchors),'preserved_source_anchors':len(anchors)-len(replaced),'declared_replaced_source_parameters':sorted(replaced),'optimizer_owners':len(owners),'independent4':getattr(session,'is_independent4',False),'passed':True}
class GradientScopeError(ValueError):
 def __init__(self,message,report):super().__init__(message);self.report=report
def gradient_report(torch,session,owners):
 named=dict(session.model.named_parameters());allowed=getattr(session,'allow_missing_grad_names',{})
 if isinstance(allowed,(list,tuple,set)):allowed={n:'source inactive/delayed branch; exact name declared by callback' for n in allowed}
 require(isinstance(allowed,dict) and '*' not in allowed and set(allowed)<=set(named),'Missing-gradient exceptions must name actual source parameters')
 missing=[];nonzero=0;finite=0
 for name,p in named.items():
  if not p.requires_grad:continue
  if p.grad is None:missing.append(name);continue
  require(bool(torch.isfinite(p.grad).all()),'Nonfinite gradient:'+name);finite+=1;nonzero+=int(bool((p.grad!=0).any()))
 unknown=[n for n in missing if n not in allowed];diagnostic={'finite_gradient_tensors':finite,'nonzero_gradient_tensors':nonzero,'missing_gradient_names':missing,'unexplained_missing_gradient_names':unknown,'declared_missing_gradient_reasons':{n:allowed[n] for n in missing if n in allowed}}
 if nonzero==0:raise GradientScopeError('No actual nonzero trainable gradient',diagnostic)
 if unknown:raise GradientScopeError('Unexplained missing source gradients:'+str(unknown[:12]),diagnostic)
 each=[]
 for o in owners:
  ps=[p for group in o.param_groups for p in group['params']]
  each.append({'parameters_with_gradient':sum(p.grad is not None for p in ps),'parameters_with_nonzero_gradient':sum(p.grad is not None and bool((p.grad!=0).any()) for p in ps)})
 require(all(r['parameters_with_nonzero_gradient']>0 for r in each),'An optimizer owner received no nonzero gradient')
 return {'finite_gradient_tensors':finite,'nonzero_gradient_tensors':nonzero,'missing_gradient_names':missing,'declared_missing_gradient_reasons':{n:allowed[n] for n in missing},'per_optimizer':each,'passed':True}
def optimizer_state_clocks(owners):
 rows=[]
 for index,o in enumerate(owners):
  clocks=[]
  for s in o.state.values():
   if isinstance(s,dict) and 'step' in s:
    v=s['step'];clocks.append(float(v.item() if hasattr(v,'item') else v))
  rows.append({'owner':index,'step_state_count':len(clocks),'minimum_parameter_step':min(clocks) if clocks else None,'maximum_parameter_step':max(clocks) if clocks else None,'distinct_parameter_steps':sorted(set(clocks))})
 return rows
class StepObserver:
 def __init__(self,owners):
  self.owners=owners;self.original=[];self.attempts=[0]*len(owners);self.completed=[0]*len(owners)
  for i,o in enumerate(owners):
   fn=o.step;self.original.append(fn)
   def make(index,original):
    @functools.wraps(original)
    def step(*args,**kwargs):
     self.attempts[index]+=1;v=original(*args,**kwargs);self.completed[index]+=1;return v
    return step
   o.step=make(i,fn)
 def close(self):
  for o,fn in zip(self.owners,self.original):o.step=fn
 def snapshot(self):return {'attempts':list(self.attempts),'completed':list(self.completed)}
def loss_value(raw):
 value=next((raw[k] for k in ('loss','objective','ce') if k in raw),None);require(value is not None,'Callback must return a finite loss/objective')
 value=float(value.detach().item() if hasattr(value,'detach') else value);require(math.isfinite(value),'Nonfinite source loss');return value
def callback_metadata(session):
 value=getattr(session,'source_metadata',None)
 if value is None:value=getattr(session,'metadata',None)
 return value() if callable(value) else value
def callback_work(raw):
 # This is receipt normalization only; it does not infer missing native work.
 if raw.get('operation_counts') is not None:return raw['operation_counts']
 if raw.get('work') is not None:
  result=copy.deepcopy(raw['work'])
  if isinstance(result,dict):
   extra={k:raw[k] for k in ('optimizer_updates','logical_update','update','native_route_forwards','native_route_forward_views','activation_checkpointing','checkpoint_recomputation_work_not_counted','source_bn_policy','native_skipped_batches','native_batch_graphs','train_records','graph_nodes','whole_native_train') if k in raw}
   if extra:result['source_callback_fields']=extra
  return result
 if raw.get('member_updates') is not None:return {'member_updates':raw['member_updates'],'native_work_remains_nested':True}
 extra={k:v for k,v in raw.items() if k not in ('loss','objective','ce','outputs','member_outputs','metrics')}
 return {'unnormalized_source_callback_fields':extra} if extra else None
def validation_evidence(raw):
 return {k:raw[k] for k in ('role','VALID_scope','VALID_graphs','VALID_nodes','complete_VALID_positive_records','complete_VALID_negative_records','native_skipped_batches','TEST_accessed','TEST_opened','TEST_scored','test_scoring','TEST_accessed','work','operation_counts','member_validation_evidence') if k in raw}

def load_callback(case,project_root):
 path=Path(case['callback_module']);path=path if path.is_absolute() else Path(project_root)/path;path=path.resolve();require(path.is_relative_to(Path(project_root).resolve()),'Callback must stay inside project')
 receipt=file_receipt(path)
 if case.get('callback_sha256'):require(receipt['sha256']==case['callback_sha256'],'Callback source changed')
 spec=importlib.util.spec_from_file_location('source_smoke_'+receipt['sha256'][:16],path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return getattr(module,case.get('callback_name','create_session')),receipt

def run_case(case,project_root,output,*,barrier=None,slot=None,barrier_timeout=300):
 import torch
 output=Path(output);require(not output.exists(),'Preserve previous smoke evidence; choose fresh output');output.mkdir(parents=True)
 start=time.perf_counter();phase='construction';session=None;observer=None;rtol=float(case.get('rtol',1e-4));atol=float(case.get('atol',1e-6));require(rtol>=0 and atol>=0,'Invalid declared tolerance')
 result={'schema':'real-source-short-smoke-v1','case_id':case['id'],'procedure':case['procedure'],'recipe':case['recipe'],'scientific_accuracy_result':False,'training_callbacks_requested':4,'logical_trajectory_updates':3,'resume_replayed_third_callback':True,'rtol':rtol,'atol':atol,'status':'running','full_horizon_stopping_or_convergence_qualified':False,'source_data_required':True,'dummy_data_fallback':False,'steps':[],'resource_observations':[],'replay_criterion_version':REPLAY_CRITERION,'controller_source':file_receipt(Path(__file__)),'replay_criterion':{'immediate_loaded_session_tensors_scalars_modes_clocks_and_RNG':'exact values before resumed update','post_update_floating_model_optimizer_tensors':'per-tensor RMS(delta) <= atol + rtol * RMS(resumed reference)','post_update_shapes_dtypes_integer_tensors_and_nonnumeric_state':'exact','predictions_and_loss':'unchanged coordinate rtol/atol','rtol':rtol,'atol':atol}}
 def observe(name):synchronize(torch);result['resource_observations'].append({'phase':name,'elapsed_seconds':time.perf_counter()-start,**own_memory(torch)})
 def validate():
  t=time.perf_counter();raw=session.validate();require(isinstance(raw,dict) and 'outputs' in raw,'Validation must return actual source output tensor(s)');require(finite_tree(torch,raw['outputs'],'validation outputs')>0,'No source validation outputs');loss=loss_value(raw)
  return raw,{'seconds':time.perf_counter()-t,'loss':loss,'outputs':tensor_summary(torch,raw['outputs']),'independent4_mean_aggregation':independent_bank(torch,session,raw,rtol,atol),'source_VALID_evidence':validation_evidence(raw)}
 def comparison_state():
  extra=session.comparison_state() if hasattr(session,'comparison_state') else {}
  return cpu_clone(torch,{'model':session.model.state_dict(),'optimizers':[o.state_dict() for o in owners],'source_resume_critical':extra})
 def update(label):
  nonlocal phase
  phase=label;t=time.perf_counter();before=observer.snapshot();raw=session.train_update();synchronize(torch);require(isinstance(raw,dict),'train_update must return a dict');loss=loss_value(raw);finite_tree(torch,session.model.state_dict(),'trained source model')
  for o in owners:finite_tree(torch,o.state_dict(),'optimizer state')
  report=gradient_report(torch,session,owners);current=observer.snapshot();calls={'attempts':[a-b for a,b in zip(current['attempts'],before['attempts'])],'completed':[a-b for a,b in zip(current['completed'],before['completed'])]};require(all(v>0 for v in calls['completed']),'Each optimizer owner must perform real update work')
  row={'label':label,'seconds':time.perf_counter()-t,'loss':loss,'actual_optimizer_step_calls':calls,'optimizer_state_clocks':optimizer_state_clocks(owners),'gradients':report,'source_operation_counts':callback_work(raw),'parameter_identity':identity(torch,session,owners)};result['steps'].append(row);observe(label);atomic_json(output/'PROGRESS.json',result);return raw
 try:
  torch.set_num_threads(int(case.get('torch_threads',2)))
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
  data_root=Path(case['data_root']);data_root=data_root if data_root.is_absolute() else Path(project_root)/data_root;data_root=data_root.resolve();require(data_root.is_relative_to(Path(project_root).resolve()),'Deliberate source data must stay inside project')
  factory,receipt=load_callback(case,project_root);result['callback_source']=receipt
  recipe=case['recipe'];seed_source='recipe._seed' if '_seed' in recipe else 'recipe.seed' if 'seed' in recipe else 'case.seed';seed=recipe.get('_seed',recipe.get('seed',case.get('seed')));require(type(seed) is int and seed>=0,'Explicit frozen nonnegative pairing seed required before source factory')
  import numpy as np
  random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
  if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
  result['controller_seed']={'value':seed,'source':seed_source,'python_seed':seed,'numpy_seed':seed%(2**32),'torch_CPU_seed':seed,'torch_CUDA_seed':seed if torch.cuda.is_available() else None,'bound_before_factory':True}
  before=time.perf_counter();session=factory(recipe,case['procedure'],data_root,case.get('device','cuda:0'));result['construction_seconds']=time.perf_counter()-before;result['source_metadata']=callback_metadata(session);result['metadata_and_work_receipt_version']='callable-metadata_and_native-work-fallback-v1_20261014';owners=optimizers(session);require(len(owners)==case.get('expected_optimizer_owners',4 if case['procedure']=='independent4' else 1),'Wrong real optimizer-owner count');require(case['procedure']!='independent4' or getattr(session,'is_independent4',False),'Independent4 descriptor needs independent4 session');result['initial_parameter_identity']=identity(torch,session,owners);result['source_data_provenance']=getattr(session,'data_provenance',{'bound_data_root':str(data_root),'callback_source':receipt,'additional_source_tensor_provenance_not_supplied':True});observer=StepObserver(owners);observe('constructed')
  if barrier is not None:
   barrier=Path(barrier);barrier.mkdir(parents=True,exist_ok=True);atomic_json(barrier/('ready'+str(slot)+'.json'),{'pid':os.getpid(),'case_id':case['id'],'constructed':True,'memory':own_memory(torch)});tick=time.monotonic()
   while not (barrier/'RELEASE.json').exists():
    require(time.monotonic()-tick<barrier_timeout,'Pair construction barrier timeout');time.sleep(.1)
  update('first_update');update('second_update');phase='validation_after_second';_,second=validate();result['validation_after_second']=second;observe(phase)
  phase='checkpoint_save';before=time.perf_counter();payload={'schema':'real-source-smoke-resume-v1','case_id':case['id'],'session_state':cpu_clone(torch,session.state()),'global_rng':global_rng(torch),'module_modes':{n:m.training for n,m in session.model.named_modules()},'requires_grad':{n:p.requires_grad for n,p in session.model.named_parameters()}};checkpoint=output/'after_two_updates.pt';torch.save(payload,checkpoint);result['checkpoint']={**file_receipt(checkpoint),'write_seconds':time.perf_counter()-before,'full_source_state':True,'global_and_callback_private_RNG_required':True};finite_tree(torch,payload['session_state'],'saved session state');observe(phase)
  update('third_uninterrupted');phase='validation_uninterrupted_third';a,va=validate();state_a=comparison_state();outputs_a=cpu_clone(torch,a['outputs']);rng_a=global_rng(torch);result['validation_uninterrupted_third']=va;observe(phase)
  phase='checkpoint_reload';before=time.perf_counter();saved=torch.load(checkpoint,map_location='cpu',weights_only=False);require(saved['case_id']==case['id'],'Wrong source checkpoint');saved_session_reference=cpu_clone(torch,saved['session_state']);session.load_state(saved['session_state']);modules=dict(session.model.named_modules());parameters=dict(session.model.named_parameters());require(set(modules)==set(saved['module_modes']) and set(parameters)==set(saved['requires_grad']),'Model structure changed on checkpoint restore')
  for n,m in modules.items():m.training=saved['module_modes'][n]
  for n,p in parameters.items():p.requires_grad_(saved['requires_grad'][n])
  restore_rng(torch,saved['global_rng']);result['checkpoint_reload_seconds']=time.perf_counter()-before;result['reloaded_parameter_identity']=identity(torch,session,owners)
  result['immediate_loaded_state_checks']={'full_session_state':compare_loaded_exact(torch,saved_session_reference,cpu_clone(torch,session.state())),'module_modes':compare_loaded_exact(torch,saved['module_modes'],{n:m.training for n,m in session.model.named_modules()}),'requires_grad':compare_loaded_exact(torch,saved['requires_grad'],{n:p.requires_grad for n,p in session.model.named_parameters()}),'global_RNG':{'passed':rng_digest(torch,saved['global_rng'])==rng_digest(torch,global_rng(torch)),'criterion':'exact restored global RNG digest'}}
  require(all(v['passed'] for v in result['immediate_loaded_state_checks'].values()),'Checkpoint immediate loaded full state differs before resumed update');observe(phase)
  update('third_resumed');phase='validation_resumed_third';b,vb=validate();state_b=comparison_state();rng_b=global_rng(torch);result['validation_resumed_third']=vb;observe(phase)
  result['resume_checks']={'outputs':compare_trees(torch,outputs_a,cpu_clone(torch,b['outputs']),rtol,atol),'model_optimizer_and_declared_private_state':compare_trees(torch,state_a,state_b,rtol,atol,tensor_policy='RMS'),'loss':{'passed':abs(va['loss']-vb['loss'])<=atol+rtol*abs(va['loss']),'uninterrupted':va['loss'],'resumed':vb['loss'],'rtol':rtol,'atol':atol},'global_RNG':{'passed':rng_digest(torch,rng_a)==rng_digest(torch,rng_b),'uninterrupted_sha256':rng_digest(torch,rng_a),'resumed_sha256':rng_digest(torch,rng_b),'model_bit_equality_required':False}}
  require(all(v['passed'] for v in result['resume_checks'].values()),'Checkpoint third-update replay outside declared engineering tolerance')
  result['status']='passed';result['actual_optimizer_step_calls_total']=observer.snapshot();result['final_optimizer_state_clocks']=optimizer_state_clocks(owners);result['independent_bank_rule']=getattr(session,'bank_rule',None)
 except Exception as error:
  result.update(status='failed',failure={'phase':phase,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=12)})
  if isinstance(error,GradientScopeError):result['failure']['gradient_diagnostics']=error.report
  if observer is not None:result['actual_optimizer_step_calls_total']=observer.snapshot()
 finally:
  result['wall_seconds']=time.perf_counter()-start
  try:observe('finished')
  except Exception as error:result['final_resource_capture_error']=str(error)
  if observer is not None:observer.close()
  atomic_json(output/'RESULT.json',result)
  if session is not None:
   try:
    if hasattr(session,'close'):session.close()
   except Exception as error:atomic_json(output/'CLEANUP_ERROR.json',{'message':str(error)})
  del session;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--case',required=True);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);p.add_argument('--barrier');p.add_argument('--slot',type=int);p.add_argument('--barrier-timeout',type=int,default=300);a=p.parse_args();require_allocation();root=Path(a.project_root).resolve()
 for value in (a.manifest,a.output,a.barrier):
  if value:require(Path(value).resolve().is_relative_to(root),'Smoke evidence/manifest must stay inside project')
 d=json.loads(Path(a.manifest).read_text());rows=[c for c in d['cases'] if c['id']==a.case];require(len(rows)==1,'One exact smoke case required');r=run_case(rows[0],root,a.output,barrier=a.barrier,slot=a.slot,barrier_timeout=a.barrier_timeout);print(json.dumps({'case_id':a.case,'status':r['status'],'wall_seconds':r['wall_seconds'],'result':str(Path(a.output)/'RESULT.json')}));sys.exit(0 if r['status']=='passed' else 1)
if __name__=='__main__':main()
