"""Literal WikiCS Session plus detached old-view-B corrector lifecycle."""
import ast,copy,hashlib,random,types
from types import SimpleNamespace
from .common_native_wrapper import bound,load_module
CONTRACT={'schema':'literal-label-corrector4-WikiCS-Session-v1','native':'exact publicV2 Session(wikics,single),Polynormer512,L7local/L2global,own2views/1backward/1Adam','core':'unchanged sealed labelonlyBEcore orfixedsameoperationcontrols;TRAIN580/commonquery290;allnonselfdenominator;labelonlyvalues;Q/Ksigns;zerooutput','order':'commonQdraw,native2ownviews/CE/Adam,detachedOLDviewB H/logits correctorownCE/Adam,nativeSELECTforwardonce reusedforcorrectedserving','selector':'1100full epochs/no stop;strictfirstnativeownlocalSELECTfirst100;at101restoreownlocalnative+sameepochcorrectorparameters/optimizers;liveendlocalRNG/mask/counters retained;strictfirst coherent correctedfinal','seeds':'source6101/6203/6307;actualseed=base+1009body fornewI4;initializeractual+1700001,maskactual+1900001;controlhead+1009head;nativeprivateactual+300001','baseline':'exact Session singleM1/genuinecoldI4;originalsourceBEBlockinput/local/globalhead with2ownviews4routes8backwards1Adam;nointernalfactors','adaptations':['12Oct graph/TRAIN byteidenticalsource;SELECT3505/CAL1769 queryroles replaceolddevelopment5274','CAL/evaluate targets isolated afterwards;after-sourcepool fixedscalarCAL500','explicit genuineI4 body seeds andsourceoriginalboundary comparator;authorCLI/numericaltrajectory equality notclaimed'],'historical_state_inherited':False}
WORK={'label_only4_detached':(4,4,4,1),'multihead_single4':(1,4,1,1),'shared_backbone_untied_correctors4':(4,4,4,4),'one_path':(1,1,1,1),'native_single':(1,0,0,0),'original_GNNM':(4,0,0,0)}

def definitions(root,row,names,env):
 nodes=[copy.deepcopy(n) for n in ast.parse(bound(root,row).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Literal source primitive inventory required')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),row['path']+'::literal_AST','exec'),env);return SimpleNamespace(**env)

class OptimizerFacade:
 def __init__(self,native,core):
  self.native,self.core=native,core;self.owners=list(native.optimizers)+([] if core is None else list(getattr(core,'optimizers',[core.optimizer])));self.param_groups=[g for owner in self.owners for g in owner.param_groups]
 @property
 def state(self):return {p:s for owner in self.owners for p,s in owner.state.items()}
 def state_dict(self):return {'native':[o.state_dict() for o in self.native.optimizers],'corrector':None if self.core is None else self.core.optimizer.state_dict()}
 def load_state_dict(self,v):
  if len(v['native'])!=len(self.native.optimizers) or (v['corrector'] is None)!=(self.core is None):raise ValueError('Exact coherent native/core optimizer bank')
  for o,s in zip(self.native.optimizers,v['native']):o.load_state_dict(s)
  if self.core is not None:self.core.optimizer.load_state_dict(v['corrector'])
  self.param_groups=[g for owner in self.owners for g in owner.param_groups]

def construct(torch,root,B,condition,seed,device,graph,train):
 for r in B.values():bound(root,r)
 common=load_module(root,B['integration_common']);integration=load_module(root,B['integration']);capture=load_module(root,B['capture']);public,data,driver,core_module,public_root=common.dependencies()
 native=public.Session('wikics','single',seed,device,bound(root,B['native']));core=None;boundary=None
 if condition=='original_GNNM':
  helpers=definitions(root,B['original_helpers'],('original_block','attach_boundary'),{'ast':ast,'hashlib':hashlib,'types':types,'bound':bound})
  _,boundary=helpers.attach_boundary(root,{'original_boundary':B['original_boundary']},torch,native.model);native.model.to(native.device);native.optimizers=[torch.optim.Adam(native.model.parameters(),lr=.001,eps=1e-8,weight_decay=0.)];native.streams=[]
  for member in range(4):
   with torch.random.fork_rng(devices=native.cuda_devices):
    torch.manual_seed(seed+1009*member+300001);row={'cpu':torch.get_rng_state().clone()}
    if native.cuda_index is not None:torch.cuda.manual_seed(seed+1009*member+300001);row['cuda']=torch.cuda.get_rng_state(native.cuda_index).clone()
   native.streams.append(row)
  step=definitions(root,B['original_step'],('require','_streamed_f_step'),{});native.train_step=types.MethodType(step._streamed_f_step,native);native._operator_failed=False;native.operator_work=dict(update_attempts=0,completed_updates=0,member_view_forwards=0,member_view_backwards=0,Adam_steps=0)
 elif condition!='native_single':
  td={**graph,**train};binding=integration._control_binding(condition,None);core=integration._new_corrector(native,td,core_module,public_root,common,condition,binding)
 def model_type():
  class Coherent(torch.nn.Module):
   def __init__(self):
    super().__init__();self.native=native.model
    if core is not None:self.corrector=core
    self.role_buffers=[] if core is None else ['corrector.'+n for n,_ in core.named_buffers()]
   def state_dict(self,*args,**kwargs):
    v=super().state_dict(*args,**kwargs)
    for n in self.role_buffers:v.pop(n,None)
    return v
   def load_state_dict(self,v,strict=True,assign=False):
    if set(v)!=set(self.state_dict()):raise ValueError('Exact learned topology;immutableGRAPH/TRAIN buffers are checked,never loaded')
    if core is not None:core._check_roles()
    out=super().load_state_dict(v,strict=False,assign=assign)
    if set(out.missing_keys)!=set(self.role_buffers) or out.unexpected_keys:raise ValueError('Only fixed role buffers excluded from learnedstate restore')
    if core is not None:core._check_roles()
    return out
  return Coherent
 model=model_type()();cap=None if core is None else capture.NativeSingleCapture(native,len(graph['x']),512,10)
 return SimpleNamespace(common=common,integration=integration,driver=driver,native=native,core=core,boundary=boundary,capture=cap,model=model,optimizer=OptimizerFacade(native,core))

class Adapter:
 def __init__(self,torch,state,graph,train,select,condition,provenance):
  self.t,self.s=torch,state;self.model,self.optimizer=state.model,state.optimizer;self.native,self.core=state.native,state.core;self.condition=condition;self.provenance=copy.deepcopy(provenance);self.device=self.native.device;self.graph=graph;self.train=train;self.select=select;self.full_batch={k:v.to(self.device) for k,v in graph.items()};self.full_batch['ids']=torch.arange(len(graph['x']),device=self.device);self.train_batch={**self.full_batch,'ids':train['ids'].to(self.device)};self.labels=train['y'].to(self.device);self._best=None;self._best_step=None;self.local={'best':None,'step':None,'snapshot':None,'transition_completed':False};self.parameter_epoch=0;self.counters={};self.role_identity=state.common.context_identity(self.native,{**graph,**train});self.initialization={'source':'coldliteralSession','trained_donor':False,'boundary':state.boundary};self.last_native_rows=None
 def _corrector_metadata(self):return self.s.integration._capture_metadata(self.native,self.parameter_epoch,self.s.common)
 def local_transition(self):
  n,c=self.native,self.core;state=self.local['snapshot']
  if state is None or self.local['transition_completed'] or n.model.models[0].body._global:raise ValueError('One strictfirst nativeownlocal coherent restore required')
  if c is not None:self.parameter_epoch=self.s.integration._restore_native_own_local(n,c,state,self.provenance,self.s.common)
  else:
   old=self.s.common.native_rng_state(n);n.core['selection'].local_transition(n.model,n.optimizers,lambda filename:state,ordinary_independent=False);self.s.common.require_same_native_rng(n,old,'literal ownlocal native restore');self.parameter_epoch=self.local['step']
  self.local['transition_completed']=True
 def train_step(self,step):
  n,c,t=self.native,self.core,self.t
  if step!=n.steps+1:raise ValueError('Complete source epoch order required')
  transition=0
  if step==101:self.local_transition();transition=1
  if c is not None:
   old=self.s.common.native_rng_state(n);mask=c.draw_common_query_mask();self.s.common.require_same_native_rng(n,old,'source commonQ');self.s.capture.begin('TRAIN',2);metadata=self._corrector_metadata()
  v=n.train_step(self.train_batch,self.labels)
  if c is not None:
   H,base=self.s.capture.finish();old=self.s.common.native_rng_state(n);cv=self.s.integration._train_core(c,H,base,mask,self.condition,metadata);self.s.common.require_same_native_rng(n,old,'detached OLDviewB core update')
  self.parameter_epoch=step;members,heads,back,adam=WORK[self.condition];original=self.condition=='original_GNNM';counts={'optimizer_updates':1+adam,'model_forwards':(8 if original else 2)+heads,'route_predictions':(8 if original else 2)+members if c is not None else (8 if original else 2),'backward_calls':(8 if original else 1)+back,'source_native_TRAIN_forwards':8 if original else 2,'source_native_TRAIN_backwards':8 if original else 1,'source_corrector_TRAIN_head_forwards':heads,'source_corrector_TRAIN_backwards':back,'source_corrector_Adam_maps':adam,'source_common_query_masks':int(c is not None),'source_coherent_local_restores':transition,'supervised_examples':len(self.labels)*(8 if original else 2)+(290*members if c is not None else 0)}
  for k,val in counts.items():self.counters[k]=self.counters.get(k,0)+val
  return {'objective':float(v['loss'])+(float(cv.get('own_corrected_CE_mean',cv.get('own_CE_mean',0))) if c is not None else 0),'ce':float(v['own_mean']),'regularization':0.,'operation_counts':counts}
 def predict_rows(self,ids,step,mode='SERVE'):
  t,n,c=self.t,self.native,self.core;ids=ids.to(self.device);n.model.eval();old=self.s.common.native_rng_state(n)
  with t.no_grad():
   if c is None:logits,_=n.forward({**self.full_batch,'ids':ids});self.last_native_rows=logits.detach()
   else:
    self.s.capture.begin(mode,1);native,_=n.forward(self.full_batch);H,base=self.s.capture.finish();self.last_native_rows=base[ids][None].detach();p=self.s.integration._serve_core(c,H,base,ids,self.condition,self._corrector_metadata());logits=p['member_logits']
  self.s.common.require_same_native_rng(n,old,'source selected serving');return logits
 def validate(self,step):
  ids=self.select['ids'];bank=self.predict_rows(ids,step,mode='VALID');truth=self.select['y'].to(bank.device);pool=bank.mean(0).softmax(-1) if self.condition=='original_GNNM' else bank.softmax(-1).mean(0);native_pool=self.last_native_rows.mean(0).softmax(-1) if self.condition=='original_GNNM' else self.last_native_rows.softmax(-1).mean(0);native=float((native_pool.argmax(-1)==truth).float().mean());metric=float((pool.argmax(-1)==truth).float().mean())
  if step<=100 and (self.local['best'] is None or native>self.local['best']):
   self.local['best'],self.local['step']=native,step
   if self.core is None:self.local['snapshot']={'model':self.native._cpu_tree(self.native.model.state_dict()),'optimizers':[self.native._cpu_tree(o.state_dict()) for o in self.native.optimizers]}
   else:
    state=self.s.integration._snapshot(self.native,self.core,self.s.driver,step,native,metric,[],self.provenance,'native_own_local');state['native'].pop('numpy_rng',None);self.local['snapshot']=state
  import math
  lp=bank.mean(0).log_softmax(-1) if self.condition=='original_GNNM' else self.t.logsumexp(bank.log_softmax(-1),dim=0)-math.log(len(bank));nll=float(-lp.gather(1,truth[:,None]).mean())
  return {'objective':nll,'ce':nll,'regularization':0.,'selection_metric':metric,'accuracy':metric,'native_own_accuracy':native,'queries':len(ids),'local_native_best_epoch':self.local['step']}
 def observe_validation(self,step,v):
  better=self._best is None or v['selection_metric']>self._best
  if better:self._best,self._best_step=v['selection_metric'],step
  return {'save_selected':better,'stop_requested':False,'checkpoint_eligible':True,'best_saved_metric':self._best,'best_saved_step':self._best_step,'stale_steps':0}
 def capture_state(self):
  n,c,t=self.native,self.core,self.t;nr=n.np.random.get_state();return {'provenance':copy.deepcopy(self.provenance),'role_identity':self.role_identity,'native_steps':n.steps,'native_streams':self.s.common.clone_streams(n.streams),'global_mode':bool(n.model.models[0].body._global),'core_state':None if c is None else {'steps':c.steps,'counters':copy.deepcopy(c.counters),'mask_rng':c.mask_generator.get_state().clone(),'failed':c._failed_update},'local':copy.deepcopy(self.local),'parameter_epoch':self.parameter_epoch,'counters':copy.deepcopy(self.counters),'selector':{'best':self._best,'step':self._best_step},'operator_work':copy.deepcopy(getattr(n,'operator_work',None)),'torch_cpu':t.get_rng_state().clone(),'torch_cuda':None if n.cuda_index is None else t.cuda.get_rng_state(n.cuda_index).clone(),'python':random.getstate(),'numpy':(nr[0],nr[1].tolist(),nr[2],nr[3],nr[4])}
 def restore_state(self,v):
  n,c,t=self.native,self.core,self.t
  if v['provenance']!=self.provenance or v['role_identity']!=self.role_identity or (v['core_state'] is None)!=(c is None):raise ValueError('Exact source coherent model andimmutablegraph/TRAIN custody required')
  n.steps=v['native_steps'];n.streams=self.s.common.clone_streams(v['native_streams']);n.model.set_global(v['global_mode']);self.local=copy.deepcopy(v['local']);self.parameter_epoch=v['parameter_epoch'];self.counters=copy.deepcopy(v['counters']);self._best,self._best_step=v['selector']['best'],v['selector']['step']
  if c is not None:c.steps=v['core_state']['steps'];c.counters=copy.deepcopy(v['core_state']['counters']);c.mask_generator.set_state(v['core_state']['mask_rng']);c._failed_update=v['core_state']['failed'];c._pending_mask=None;c._check_roles();c._finite_state()
  if v['operator_work'] is not None:n.operator_work=copy.deepcopy(v['operator_work']);n._operator_failed=False
  t.set_rng_state(v['torch_cpu']);random.setstate(v['python']);nr=v['numpy'];n.np.random.set_state((nr[0],n.np.asarray(nr[1],dtype='uint32'),nr[2],nr[3],nr[4]))
  if v['torch_cuda'] is not None:t.cuda.set_rng_state(v['torch_cuda'],n.cuda_index)

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config']
 if study['dataset']!='wikics' or c['label_corrector_source_contract']!=CONTRACT:raise ValueError('Exact sourceWikiCS labelcorrector contract required')
 graph,train,select=[torch.load(bound(project_root,c['node_inputs'][role]),map_location='cpu',weights_only=True) for role in ('graph','train','select')]
 if graph['x'].shape!=(11701,300) or train['ids'].shape!=(580,) or select['ids'].shape!=(3505,):raise ValueError('ExactGRAPH/TRAIN andadoptedSELECT role required')
 seed=c['initialization_seed'];state=construct(torch,project_root,c['label_corrector_sources'],c['label_corrector_arm'],seed,device,graph,train);return Adapter(torch,state,graph,train,select,c['label_corrector_arm'],{'study':study['study_id'],'run_id':run['run_id'],'actual_model_seed':seed,'config':c,'contract':CONTRACT})
