"""Paid Photo curvature stages; exact source bodies with plain checkpoint custody.

No launcher or GO. The controller supplies verified, newly paid direct ancestors.
Trials read GRAPH/TRAIN only and return five independent pretrial states.
"""
import copy
from contextlib import contextmanager
from types import SimpleNamespace
from . import curvature_lifecycle as source
from .classifier import ClassifierAdapter
from ..engine import AdapterStepError

ARMS=('common_only','fixed_first_graph_pair','selected_graph_pair',
      'selected_permuted_span','selected_random_span')
COLD=('native_M1','original_GNNM')
CONTRACT={
 'schema':'paid-Photo-curvature-pipeline-v2','dataset':'Photo10 only',
 'native':'Polynormer-r seven local/two global; heads8*64; input dropout.2; coupled Adam weight_decay5e-5; source PyG NormalizeFeatures once after provider',
 'warm':'200 local strict-first min mean-logit SELECT NLL; named model/Adam handoff with live end-local RNG;50 global fixed last',
 'trials':'one common plus at most nine finite discarded coupled Adam maps; GRAPH/TRAIN only; five independent pretrial model/named Adam clones and original warm continuation RNG',
 'continuation':'4750 global own-member CE maps; pooled mean raw-logit SELECT NLL; strict-first tie; pretrial epoch0 eligible',
 'cold':'5000 maps;200 local and4800 global; local model/named Adam handoff/live RNG; global-only selection',
 'I4':'four native M1 fits; actual seed=cohort+100003*body; exact paired M1 body0 reference',
 'constants':'root prospectively accepted for new-data preparation; no historical exactness or GO',
 'dedup':'joint18 search warm/trial blocks proven by literal five-arm selector; comparative150 blocks priced separately'}

def canonical_topology(torch,edges,nodes):
 if edges.dtype!=torch.int64 or edges.ndim!=2 or edges.shape[0]!=2:
  raise ValueError('GRAPH must provide int64 edge pairs')
 u,v=edges.cpu();mask=u!=v;u,v=u[mask],v[mask]
 key=torch.minimum(u,v)*nodes+torch.maximum(u,v)
 key=torch.unique(key,sorted=True)
 return torch.stack((key//nodes,key%nodes))

def _parent(torch,run,context,stage):
 lifecycle=run['config'].get('lifecycle',{})
 ids=lifecycle.get('parent_fit_ids',[])
 rows=context.get('paid_parent_checkpoints',{})
 if len(ids)!=1 or set(rows)!=set(ids):
  raise ValueError('One exact verified direct curvature paid parent required')
 row=rows[ids[0]]
 if row['contract']['config'].get('curvature_stage')!=stage:
  raise ValueError('Wrong typed paid curvature parent stage')
 payload=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
 if payload['fit_id']!=ids[0] or payload['step']!=(250 if stage=='warm' else 1):
  raise ValueError('Curvature uses fixed-last own warm/trial checkpoint authority')
 state=payload['adapter_state']['external']
 return state,row['receipt']

class CurvatureAdapter(ClassifierAdapter):
 def __init__(self,torch,sources,model,opt,graph,train,select,*,stage,arm,provenance,
              native_spec,local_updates=200,maximum=None,trial_parent=None,constants=None,
              constructor_objects=1):
  self.sources=sources;self.integration=sources['integration'];self.graph=graph
  self.train_role=train;self.select_role=select;self.stage=stage;self.arm=arm
  self.native_spec=copy.deepcopy(native_spec);self.raw=stage in ('trials','continuation')
  self.local_updates=local_updates;self.maximum=maximum or {'warm':250,'cold':5000,'trials':1,'continuation':4750}[stage]
  self.local_state=None;self.local_best=None;self.transition_completed=self.raw
  self.clock=0;self.trace=[];self.trial_states={};self.trial_receipt=None
  self.trial_parent=copy.deepcopy(trial_parent);self.constants=constants
  self.constructor_objects=constructor_objects;self.constructor_objects_reported=0
  def logits(m,training,step):
   return m(*self.integration.raw_arguments(graph,'polynormer_r')) if self.raw else m(graph)
  def capture():return self._capture_external()
  def restore(value):self._restore_external(value)
  role=select if select is not None else train
  super().__init__(torch,torch.nn.functional,model,opt,logits_fn=logits,
    train_ids=train.nodes,train_y=train.labels,select_ids=role.nodes,select_y=role.labels,
    regularizer=lambda m:graph.teacher_input.new_zeros(()),fusion='mean_logits',
    selector={'metric':'pooled_nll','direction':'min','ties':'first','patience_steps':5001,'save_every':1},
    provenance=provenance,external_capture=capture,external_restore=restore)

 def _capture_external(self):
  value={'schema':'curvature-paid-stage-state-v2','stage':self.stage,'arm':self.arm,
   'native_spec':copy.deepcopy(self.native_spec),'global':bool(self.model.core._global) if self.raw else self.model.global_stage,
   'rng':self.integration.rng_snapshot(),'clock':self.clock,
   'local_state':copy.deepcopy(self.local_state),'local_best':self.local_best,
   'transition_completed':self.transition_completed,'trace':copy.deepcopy(self.trace),
   'trial_states':copy.deepcopy(self.trial_states),'trial_receipt':copy.deepcopy(self.trial_receipt),
   'constructor_objects':self.constructor_objects,'constructor_objects_reported':self.constructor_objects_reported}
  if self.stage=='warm' and self.clock==self.maximum:
   value['native_warm']=self.integration.native_checkpoint(self.model,self.optimizer,
    {'actual_updates':self.maximum,'fixed_last_warm':True,'local_model_and_adam_restored':True,
     'native_local_rng_restored':False,'local_selected_metadata':self.local_state['metadata']})
  return value

 def _restore_external(self,value):
  if value['schema']!='curvature-paid-stage-state-v2' or value['stage']!=self.stage or value['arm']!=self.arm:
   raise ValueError('Typed curvature checkpoint stage/arm differs')
  if value['native_spec']!=self.native_spec:raise ValueError('Curvature source specification changed')
  self.model.set_global_stage(value['global'])
  for name in ('clock','local_state','local_best','transition_completed','trace','trial_states','trial_receipt','constructor_objects','constructor_objects_reported'):
   setattr(self,name,copy.deepcopy(value[name]))
  self.integration.rng_restore(value['rng'])

 def _new_constructors(self,counts):
  delta=self.constructor_objects-self.constructor_objects_reported
  counts['Adam_constructor_objects']=delta;self.constructor_objects_reported=self.constructor_objects

 def _transition(self):
  if self.local_state is None:raise ValueError('No strict-first finite local source state')
  rng=self.integration.rng_snapshot()
  self.model.load_state_dict(self.local_state['model'])
  self.optimizer=self.integration.restore_named_optimizer(self.model,self.local_state['optimizer'])
  self.constructor_objects+=1;self.model.set_global_stage(True);self.transition_completed=True
  self.integration.rng_restore(rng)
  # Cold native source global selection starts fresh; warm selects fixed-last.
  self._best=None;self._best_step=None;self._stale=0

 def train_step(self,step):
  if step!=self.clock+1:raise ValueError('Curvature paid stage clock skipped or repeated')
  if self.stage=='trials':return self._trial_step(step)
  counts={'optimizer_updates':0,'optimizer_step_attempts':0,'model_forward_attempts':0,
          'backward_attempts':0,'completed_source_TRAIN_maps':0,
          'supervised_examples':len(self.train_y)*self.model.members}
  try:
   if not self.raw and step==self.local_updates+1:self._transition()
   original=self.optimizer.step
   def tracked(*args,**kwargs):
    counts['optimizer_step_attempts']+=1
    result=original(*args,**kwargs);counts['optimizer_updates']+=1;return result
   self.optimizer.step=tracked
   def pre_hook(module,args):counts['model_forward_attempts']+=1
   def forward_hook(module,args,output):
    if output.requires_grad:
     def backward_hook(gradient):counts['backward_attempts']+=1;return gradient
     output.register_hook(backward_hook)
   before=self.model.register_forward_pre_hook(pre_hook);after=self.model.register_forward_hook(forward_hook)
   try:loss,nonzero=self.integration.train_update(self.model,self.optimizer,self.graph,self.train_role,raw=self.raw)
   finally:
    self.optimizer.step=original;before.remove();after.remove()
   counts['completed_source_TRAIN_maps']=1
   self.clock=step;self._new_constructors(counts)
   return {'objective':loss,'ce':loss,'regularization':0.,'nonzero_gradient_tensors':nonzero,
           'operation_counts':counts}
  except BaseException as error:
   self._new_constructors(counts)
   raise AdapterStepError(str(error),operation_counts=counts) from error

 def _trial_step(self,step):
  selector=self.sources['selector'];initializer=self.sources['graph_band_route_initializer']
  integration=self.integration;boundary=self.sources['backbone_boundary_adapter']
  rng=copy.deepcopy(self.trial_parent['native_warm']['rng'])
  counts={'optimizer_updates':0,'trial_maps_started':0,'trial_maps_completed':0,
          'optimizer_step_attempts':0,'partial_trial_optimizer_step_possible':0}
  # The exact selector function is compiled in its own namespace. Trace wrappers
  # replace only accounting calls temporarily; source values/maps are unchanged.
  namespace=selector.select_initializations.__globals__;original=namespace['one_adam_trial']
  original_restore=integration.restore_named_optimizer
  def tracked_restore(*args,**kwargs):
   value=original_restore(*args,**kwargs);self.constructor_objects+=1;return value
  def trial(*args,**kwargs):
   counts['trial_maps_started']+=1
   try:receipt=original(*args,**kwargs)
   except BaseException as error:
    self.trace.append({'kind':'discarded_trial','index':counts['trial_maps_started'],
     'status':'constructor_or_pretrial_failure','updates':0,'optimizer_step_attempted':False,'error':str(error)})
    raise
   self.trace.append({'kind':'discarded_trial','index':counts['trial_maps_started'],**copy.deepcopy(receipt)})
   counts['trial_maps_completed']+=receipt['updates'];counts['optimizer_updates']+=receipt['updates']
   counts['optimizer_step_attempts']+=int(receipt['optimizer_step_attempted'])
   counts['partial_trial_optimizer_step_possible']+=int(receipt.get('partial_optimizer_step_possible',False))
   return receipt
  namespace['one_adam_trial']=trial;integration.restore_named_optimizer=tracked_restore
  try:
   native=self.native_for_trial
   k1=integration.clone_boundary(native,boundary,1);k1.eval()
   identity=integration.identity_logits_audit(native,k1,self.model,self.graph)
   theta0,fn,binding=selector.bind_head_only(k1,'polynormer_r',integration.raw_arguments(self.graph,'polynormer_r'))
   canonical=canonical_topology(self.t,self.graph.teacher_edge_index,len(self.graph.teacher_input))
   S=initializer.symmetric_normalized_adjacency(len(self.graph.teacher_input),canonical,theta0.dtype,theta0.device)
   frozen=integration.named_optimizer_snapshot(self.model,self.optimizer)
   states,receipt=selector.select_initializations(self.model,frozen,fn,theta0,S,
    self.t.arange(len(self.graph.teacher_input),device=theta0.device),self.train_ids,self.train_y,
    'polynormer_r',integration.raw_arguments(self.graph,'polynormer_r'),rng,
    self.provenance['cohort_seed'],self.constants,initializer,integration)
   for arm,row in states.items():
    self.trial_states[arm]={'model':integration.cpu_copy(row['model'].state_dict()),
     'optimizer':integration.named_optimizer_snapshot(row['model'],row['optimizer']),
     'rng':copy.deepcopy(row['rng']),'native_spec':copy.deepcopy(self.native_spec),'global':True}
   receipt.update(identity_logits=identity,head_binding=binding,
    typed_paid_warm=self.provenance['paid_parent_receipt'],SELECT_targets_read=False,
    serialized_plain_pretrial_states=True,discarded_trial_trace=copy.deepcopy(self.trace))
   if counts['trial_maps_completed']!=receipt['trial_maps_completed'] or not 1<=counts['trial_maps_started']<=10:
    raise ValueError('Literal selector and runtime paid trial ledger disagree')
   self.trial_receipt=receipt;self.clock=step;self._new_constructors(counts)
   return {'objective':receipt['common_trial']['post_trial_pooled_train_ce'],
    'ce':receipt['common_trial']['post_trial_pooled_train_ce'],'regularization':0.,'operation_counts':counts}
  except BaseException as error:
   self.trial_receipt={'status':'failed','discarded_trial_trace':copy.deepcopy(self.trace),
     'known_partial_counts':copy.deepcopy(counts),'trial_states_returned':False}
   self._new_constructors(counts)
   raise AdapterStepError(str(error),operation_counts=counts) from error
  finally:
   namespace['one_adam_trial']=original;integration.restore_named_optimizer=original_restore
   integration.rng_restore(rng)

 def validate(self,step):
  role=self.train_role if self.stage=='trials' else self.select_role
  logits,nll=self.integration.evaluate(self.model,self.graph,role,raw=self.raw)
  correct=int(logits.mean(0)[role.nodes].argmax(-1).eq(role.labels).sum())
  return {'objective':nll,'ce':nll,'regularization':0.,'pooled_nll':nll,
   'selection_metric':nll,'correct':correct,'examples':len(role.labels),
   'accuracy':correct/len(role.labels),'fusion':'mean_logits',
   'label_scope':'TRAIN_only_finite_trial_receipt' if self.stage=='trials' else 'SELECT',
   'operation_counts':{'optimizer_updates':0,'model_forwards':1,'route_predictions':self.model.members}}

 def observe_validation(self,step,metrics):
  if self.stage in ('warm','cold') and step<=self.local_updates:
   if self.local_best is None or metrics['pooled_nll']<self.local_best:
    self.local_best=metrics['pooled_nll']
    self.local_state=self.integration.native_checkpoint(self.model,self.optimizer,
     {'stage':'local','stage_epoch':step,'actual_update':step,'primary_validation_nll':self.local_best})
  if self.stage in ('warm','trials'):
   save=step==self.maximum
   return {'save_selected':save,'stop_requested':False,'checkpoint_eligible':save,
    'fixed_last_paid_stage':True,'best_saved_metric':metrics['pooled_nll'] if save else None}
  if self.stage=='cold' and step<=self.local_updates:
   return {'save_selected':False,'stop_requested':False,'checkpoint_eligible':False,'global_only_final':True}
  decision=super().observe_validation(step,metrics);decision['stop_requested']=False
  return decision

def make_adapter(*,torch,project_root,study,run,device,execution_context):
 from pathlib import Path
 import json
 c=run['config'];stage=c['curvature_stage'];arm=c['curvature_arm']
 if study['dataset']!='amazon-photo' or stage not in ('warm','trials','continuation','cold'):
  raise ValueError('Curvature v2 admits prospective Photo10 only')
 if stage=='continuation' and arm not in ARMS or stage=='cold' and arm not in COLD:
  raise ValueError('Unregistered curvature arm')
 point={k:c[k] for k in ('learning_rate','dropout')};sources=source.modules(project_root,c['curvature_sources'],point)
 integration=sources['integration'];api=sources['native_api'];boundary=sources['backbone_boundary_adapter']
 def read(row):return torch.load(source.bound(project_root,row),map_location='cpu',weights_only=True)
 graph_data=read(c['node_inputs']['graph']);train_data=read(c['node_inputs']['train'])
 select_data=None if stage=='trials' else read(c['node_inputs']['select'])
 x,edges=graph_data['x'].to(device),graph_data['edge_index'].to(device)
 if x.shape[1]!=745 or c['classes']!=8:raise ValueError('Exact Photo source dimensions required')
 canonical=canonical_topology(torch,edges,len(x))
 with source.aliases(sources):
  graph=api.prepare_graph(x,canonical,'polynormer_r',
   {'torch':str(torch.__version__),'PyG':__import__('torch_geometric').__version__,
    'runtime_pin_admission':'root_owned'},c['node_inputs']['graph'])
 train=SimpleNamespace(nodes=train_data['ids'].to(device),labels=train_data['y'].to(device))
 select=None if select_data is None else SimpleNamespace(nodes=select_data['ids'].to(device),labels=select_data['y'].to(device))
 if select is not None and torch.isin(train.nodes,select.nodes).any():raise ValueError('TRAIN/SELECT overlap')
 actual_seed=run['seed']+100003*run['body'];constants=None;parent=None;parent_receipt=None
 with source.aliases(sources):
  if stage in ('warm','cold'):
   spec=api.specification('polynormer_r','gnnm_boundary_4' if arm=='original_GNNM' else 'single_author',0,actual_seed)
   model=api.TeacherFamily(spec).to(device);opt=api.optimizer_for(model);objects=1
  else:
   parent,parent_receipt=_parent(torch,run,execution_context,'warm' if stage=='trials' else 'trials')
   if stage=='trials':
    warm=parent['native_warm'];spec=warm['specification'];native,nativeopt=integration.restore_native(api,warm,device)
    model=integration.clone_boundary(native,boundary,4)
    opt,_=integration.transport_optimizer(native,warm['optimizer'],model);objects=2
    constants=sources['selector'].FrozenConstants(**json.loads(source.bound(project_root,c['curvature_constants']).read_text())['values'])
    integration.rng_restore(warm['rng'])
   else:
    state=parent['trial_states'][arm];spec=state['native_spec'];native=api.TeacherFamily(spec).to(device)
    native.set_global_stage(True);model=integration.clone_boundary(native,boundary,4)
    model.load_state_dict(state['model']);opt=integration.restore_named_optimizer(model,state['optimizer']);objects=1
    model.set_global_stage(state['global']);integration.rng_restore(state['rng'])
 provenance={'run_id':run['run_id'],'config':c,'actual_model_seed':actual_seed,'cohort_seed':run['seed'],
   'paid_parent_receipt':parent_receipt,'source_contract':CONTRACT,'binding_receipt':sources['binding_receipt']}
 result=CurvatureAdapter(torch,sources,model,opt,graph,train,select,stage=stage,arm=arm,
   provenance=provenance,native_spec=spec,trial_parent=parent,constants=constants,constructor_objects=objects)
 if stage=='trials':result.native_for_trial=native
 return result
