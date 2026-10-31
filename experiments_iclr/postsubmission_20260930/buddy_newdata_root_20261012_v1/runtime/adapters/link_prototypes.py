"""Literal endpoint frames and pooled/count support learning on native collab.

Lazy imports; complete source TRAIN replay and membership teacher. No historical
fit, state, eligibility, release, source-runtime assertion or score is imported.
"""
import ast,copy,hashlib,json,random,sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from . import common_native_wrapper as common
from . import link as bridge

PROTO='graph_ncNC_member_completion_qualification_preparation_20261003_v2/'
PATTERN='graph_ncNC_structural_pattern_pilot_preparation_20261004_v5/'
POOL='pooled_joint_quality_source_preparation_20261004_v1/'
REPLAY='graph_ncNC_train_paired_replay_provider_preparation_20261004_v1/'
PATHS={'graph_ops':PROTO+'graph_ops.py','prototype':PROTO+'prototype.py','endpoint':'native_endpoint_private_factor_mechanism_assessment_20261005_v1/adapter.py','controls':'endpoint_frame_same_operation_controls_20261005_v1/controls.py','design_spec':'graph_ncNC_collab_predictive_pilot_design_20261003_v1/design_spec.py'}
PATHS.update({k:PATTERN+k+'.py' for k in ('pilot_common','pilot_model','pilot_state','pilot_data','pattern_model','pattern_teacher','pattern_objective')})
PATHS.update({k:POOL+k+'.py' for k in ('pooled_model','count_density','count_model','learning')})
PATHS.update({k:REPLAY+k+'.py' for k in ('replay_common','replay_provider')})
PATHS.update(native='graph_link_competing_family_source_20261003_v1/source/ncn__model.py',utils='graph_link_competing_family_source_20261003_v1/source/ncn__utils.py')
ARMS=('endpoint_private4','endpoint_shared_frame4','same_four_frames_native_M1','one_frame_native_M1','P0','J_P','F_P','C_mu')
CONTRACT={'schema':'retained-portable-collab-endpoint-count-source-port-v1','source_exact_recipe':False,
 'task':'Exact ogbl-collab link GRAPH/TRAIN/SELECT authorities; portable source NCNC depth1, F128/H64, native query masking. CiteSeer labels/NCN aliases are not node classification scopes.',
 'endpoint':'literal four learned Householder axes e_m versus one learned shared e0 frame; exact symmetric outer endpoint product; no epsilon/clipping; recursive scorers no_grad, outer feature gradients live. Same-four-frames native M1 and separately selected one-frame I4 source controls retained.',
 'count':'literal P0/J_P/F_P pooled4 allmap fast signs,43790 params; C_mu native single plus195->64->64 slots,530->64->1 symmetric count head,88900 params; full two-sided FP64 DP, detached main unary marginals, live count NLL, checkpoint recomputes all pairs; exact per-query denominator including empty zero rows',
 'teacher':'ObservationTeacher.from_train is membership of complete TRAIN pairs, not a trained teacher. Never stored in predictor inputs; actual graph/query/common/residual context checked before any teacher lookup.',
 'replay':'source master2026100401, native PyG default full negative draw then native PermIterator, complete permutation/negatives/tail/context receipts written and authenticated before updates; model RNG restored exactly; source taildrop17 batches65536 per epoch',
 'controller':'100 full epochs =1700 complete Adam maps; source balanced positive/negative logistic sum; source SELECT official60084 positives/100000 shared negatives strict Hits50, every epoch, first tie; no CAL authority so raw-only sealed evaluate',
 'HPO':'nine finite joint encoder/decoder LR multipliers and decoder dropout points around source anchors. Encoder native input/edge/forward dropout fixed; fixed H64, source initialization/domain seed formula bound prospectively to new seed roster.',
 'source_compatibility':'Torch2.1 CPU autocast query uses is_autocast_cpu_enabled; generator Xavier helper uses exact Tensor.uniform native fan-in/fan-out bound; source bytes unchanged',
 'costs':'C_mu preserves discarded unstepped native Adam constructor before final two-group Adam. Replay preparation and every support count/DP/query pass/failed attempt charged, resources unmeasured.'}

@contextmanager
def aliases(values):
 old={k:sys.modules.get(k) for k in values};sys.modules.update(values)
 try:yield
 finally:
  for k,v in old.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v

def source_modules(root,binding):
 values={}
 for key in ('graph_ops','prototype','pilot_common','pilot_data','pilot_state','pilot_model','design_spec','pattern_model','pattern_teacher','pattern_objective','count_density','pooled_model','count_model','learning','replay_common','replay_provider','endpoint','controls'):
  with aliases(values):values[key]=common.load_module(root,binding[key])
 return values

def compile_bound(module,path,extra,transform):
 tree=copy.deepcopy(ast.parse(path.read_text()));tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or n.module not in {'prototype','pattern_model','pattern_objective','graph_ops','pilot_common','pilot_model','count_density','pooled_model','replay_common'}]
 tree=transform(tree);code=ast.fix_missing_locations(tree);namespace=dict(vars(module),**extra);exec(compile(code,'<explicit_link_source_binding>','exec'),namespace)
 return SimpleNamespace(**namespace),common.ast_hash(code)

def construct(torch,root,bindings,*,arm,seed,body,lr_multiplier,dropout,device):
 if arm not in ARMS or lr_multiplier not in (.5,1.,2.) or dropout not in (.15,.3,.6):raise ValueError('Explicit literal link arm/HPO point required')
 modules=source_modules(root,bindings);p=modules['prototype'];actual_seed=seed+1000003*body
 def recipe(**kw):return p.Recipe(**{'decoder_dropout':dropout,'encoder_lr':.0082*lr_multiplier,'decoder_lr':.0037*lr_multiplier,**kw})
 if arm.startswith('endpoint_') or arm in ('same_four_frames_native_M1','one_frame_native_M1'):
  _,private=modules['endpoint'].make_adapter_classes(p,modules['graph_ops']);controls=modules['controls'].make_control_classes(p,modules['graph_ops'],modules['endpoint']);modules['pilot_model'].seed_all(actual_seed)
  if arm=='endpoint_private4':model=private(recipe())
  elif arm=='endpoint_shared_frame4':model=controls['shared_frame_f4'](recipe())
  elif arm=='same_four_frames_native_M1':model=controls['same_four_frames_native_single'](recipe(member_count=1))
  else:model=controls['independent_native_single_one_frame'](recipe(member_count=1),axis_index=body)
  model.to(device);optimizer=p.native_optimizer(model);return model,optimizer,modules,{'actual_initialization_seed':actual_seed,'source_constructor':'literal endpoint/private or same-operation control','source_frame_axis_index':body if arm=='one_frame_native_M1' else None,'unstepped_discarded_optimizer_objects':0}
 def pooled_bind(tree):
  for node in tree.body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODEL_SEED' for t in node.targets):node.value=ast.Constant(actual_seed)
  for node in ast.walk(tree):
   if isinstance(node,ast.Constant) and node.value=='pooled-joint-quality-v1|seed=610041|':node.value=f'pooled-joint-quality-v1|seed={actual_seed}|'
   if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.is_autocast_enabled' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value=='cpu':node.func=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='is_autocast_cpu_enabled',ctx=ast.Load());node.args=[]
  return tree
 pooled,pooled_AST=compile_bound(modules['pooled_model'],common.bound(root,bindings['pooled_model']),{'Recipe':recipe},pooled_bind)
 if arm!='C_mu':model,optimizer=pooled.make_pooled(device);return model,optimizer,modules,{'actual_initialization_seed':actual_seed,'bound_pooled_AST_sha256':pooled_AST,'unstepped_discarded_optimizer_objects':0}
 native,utils=bridge.native_modules(root,{'model':bindings['native'],'utils':bindings['utils']});mods={'native':native,'design':modules['design_spec']}
 def native_bind(tree):
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and ast.unparse(n.func)=='native.IncompleteCN1Predictor':assert n.args[4].value==.3;n.args[4].value=dropout
   if isinstance(n,ast.Dict):
    for key,value in zip(n.keys,n.values):
     if isinstance(key,ast.Constant) and key.value=='lr' and isinstance(value,ast.Constant) and value.value in (.0082,.0037):value.value*=lr_multiplier
  return tree
 pilot,pilot_AST=compile_bound(modules['pilot_model'],common.bound(root,bindings['pilot_model']),{},native_bind)
 def count_bind(tree):
  import math
  count=0
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.is_autocast_enabled' and len(n.args)==1 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='cpu':n.func=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='is_autocast_cpu_enabled',ctx=ast.Load());n.args=[]
   if isinstance(n,ast.Call) and ast.unparse(n.func)=='nn.init.xavier_uniform_':
    assert ast.unparse(n)=='nn.init.xavier_uniform_(module.weight, generator=gen)';replacement=ast.parse('_generator_Xavier_primitive(module.weight, gen)').body[0].value;n.func,n.args,n.keywords=replacement.func,replacement.args,replacement.keywords;count+=1
   if isinstance(n,ast.Dict):
    for key,value in zip(n.keys,n.values):
     if isinstance(key,ast.Constant) and key.value=='lr' and isinstance(value,ast.Constant) and value.value in (.0082,.0037):value.value*=lr_multiplier
  assert count==1;return tree
 def xavier_primitive(value,generator):
  import math
  bound=math.sqrt(6.0/(value.shape[0]+value.shape[1]))
  with torch.no_grad():return value.uniform_(-bound,bound,generator=generator)
 count,count_AST=compile_bound(modules['count_model'],common.bound(root,bindings['count_model']),{'MODEL_SEED':actual_seed,'domain_seed':pooled.domain_seed,'make_native':pilot.make_native,'_generator_Xavier_primitive':xavier_primitive},count_bind)
 model,optimizer=count.make_count(mods,device);return model,optimizer,modules,{'actual_initialization_seed':actual_seed,'bound_pooled_AST_sha256':pooled_AST,'bound_native_constructor_AST_sha256':pilot_AST,'bound_count_AST_sha256':count_AST,'unstepped_discarded_optimizer_objects':1}

def runtime_replay(torch,modules,utils,sampler):
 # CPU synthetic fixtures omit only a nonexistent selected CUDA-device fence.
 rt={'torch':torch,'state':modules['pilot_state'],'seed_all':modules['pilot_model'].seed_all,'graphs':modules['graph_ops'],'tensor_sha':modules['pilot_data'].tensor_sha,'PermIterator':utils.PermIterator,'sampler':sampler}
 original=modules['replay_provider'];path=Path(original.__file__)
 if not torch.cuda.is_available():
  def adapt(tree):
   for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name=='isolated_rng':
     class CPUFence(ast.NodeTransformer):
      def visit_Expr(self,n):return ast.Pass() if ast.unparse(n)=='torch.cuda.synchronize(0)' else self.generic_visit(n)
     CPUFence().visit(node)
   return tree
  provider,digest=compile_bound(original,path,{},adapt)
 else:provider=original;digest=None
 return rt,provider,digest

class LinkPrototypeAdapter:
 def __init__(self,torch,numpy,model,opt,modules,details,*,root,run,study,graph,train,select,replay_directory,utils,sampler):
  self.t,self.np,self.model,self.optimizer=torch,numpy,model,opt;self.modules=modules;self.device=next(model.parameters()).device;self.graph,self.train,self.select=graph,train,select;self.run,self.config=run,run['config'];self.root=Path(root);self.replay_directory=Path(replay_directory);self.utils,self.sampler=utils,sampler;self.arm=self.config['link_prototype_arm'];self._best=None;self._best_step=None;self._epoch=0
  self.provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':self.config,'source_bindings':details,'source_contract':CONTRACT};self._rng=bridge.rng_state(torch,numpy)
  self.rt,self.provider,self.CPU_fence_AST=runtime_replay(torch,modules,utils,sampler)
  pairs,raw=train['positive'],graph['raw_edge_index'];TRAIN={'nodes':len(graph['x']),'records':len(pairs),'ordered_raw_entries':raw.shape[1],'batch_size':self.config['training']['batch_size'],'pairs_sha256':self.rt['tensor_sha'](pairs),'raw_sha256':self.rt['tensor_sha'](raw)}
  self.data={'pairs':pairs,'raw':raw,'nodes':len(graph['x']),'batch_size':self.config['training']['batch_size'],'identity':{'TRAIN':TRAIN,'source_replay_schema':'unchanged_source_integer_context_v1','input_receipts':self.config['link_inputs'],'source_receipts':self.config['prototype_sources']}}
  self.full_graph=modules['graph_ops'].Graph.from_pairs(pairs,len(graph['x']));self.teacher=modules['pattern_teacher'].ObservationTeacher.from_train(pairs,len(graph['x'])) if self.arm in ('P0','J_P','F_P','C_mu') else None
 @contextmanager
 def _scope(self,training):
  outer=bridge.rng_state(self.t,self.np);bridge.restore_rng(self.t,self.np,self._rng)
  try:yield
  finally:
   if training:self._rng=bridge.rng_state(self.t,self.np)
   bridge.restore_rng(self.t,self.np,outer)
 def replay_for_epoch(self,epoch):
  directory=self.replay_directory/('epoch_'+str(epoch));permutation,negative=self.provider.draw_epoch(self.rt,self.data,2026100401,epoch)
  receipt=self.provider.write_epoch(self.rt,self.data,permutation,negative,2026100401,epoch,directory)
  digest=hashlib.sha256((directory/'EPOCH.json').read_bytes()).hexdigest();return self.provider.EpochReplay(self.rt,self.data,directory,digest)
 def train_step(self,epoch):
  if epoch!=self._epoch+1:raise ValueError('Complete source epoch order required')
  self.model.train();values=[];mains=[];auxiliaries=[];counts={'optimizer_updates':0,'backward_calls':0,'source_replay_epochs':1,'support_context_checks':0,'encoder_passes':0,'positive_queries':0,'negative_queries':0}
  with self._scope(True):
   replay=self.replay_for_epoch(epoch)
   if replay.receipt['full_batches']!=self.config['train_minibatches_per_epoch']:raise ValueError('Source full-batch/tail geometry changed')
   for batch in range(replay.receipt['full_batches']):
    if self.teacher is not None:
     value=self.modules['learning'].train_replay_batch(self.model,self.optimizer,self.graph['x'],replay,batch,self.teacher,self.arm);values.append(value['total']);mains.append(value['main']);auxiliaries.append(value['auxiliary']);counts['support_context_checks']+=1
    else:
     records,negative,_=replay.query_batch(batch);context=replay.regenerate(batch);self.optimizer.zero_grad(set_to_none=True)
     loss=self.modules['prototype'].training_batch_loss(self.model,self.graph['x'],self.train['positive'],records,negative[records],'private');loss.backward();self.optimizer.step();values.append(float(loss.detach()));mains.append(float(loss.detach()));auxiliaries.append(0.);counts['support_context_checks']+=1
    counts['optimizer_updates']+=1;counts['backward_calls']+=1;counts['encoder_passes']+=1;counts['positive_queries']+=self.config['training']['batch_size'];counts['negative_queries']+=self.config['training']['batch_size']
  self._epoch=epoch;mean=sum(values)/len(values);return {'objective':mean,'ce':sum(mains)/len(mains),'regularization':sum(auxiliaries)/len(auxiliaries),'operation_counts':counts}
 def predict_pairs(self,pairs):
  self.model.eval();parts=[]
  with self._scope(False),self.t.no_grad():
   h=self.model.encode(self.graph['x'],self.full_graph) if hasattr(self.model,'encode') else self.model.encoder(self.graph['x'],self.full_graph)
   for start in range(0,len(pairs),self.config['training']['eval_batch_size']):
    q=pairs[start:start+self.config['training']['eval_batch_size']].to(self.device)
    z=self.model.query_forward(h,self.full_graph,q,auxiliary_grad=False)[0] if hasattr(self.model,'query_forward') else self.model.decoder(h,self.full_graph,q,'private')
    if z.ndim!=2 or z.shape[0]!=len(q) or not self.t.isfinite(z).all():raise FloatingPointError('Incomplete source raw link bank')
    parts.append(z.T.cpu())
  return self.t.cat(parts,1)
 def validate(self,epoch):
  from ogb.linkproppred import Evaluator
  pos=self.predict_pairs(self.select['positive']).mean(0);neg=self.predict_pairs(self.select['negative']).mean(0);evaluator=Evaluator.__new__(Evaluator);evaluator.name='ogbl-collab';evaluator.K=50;evaluator.eval_metric='hits@50';hits=float(evaluator.eval({'y_pred_pos':pos,'y_pred_neg':neg})['hits@50']);explicit=int((pos>neg.topk(50).values[-1]).sum())/len(pos)
  if hits!=explicit:raise ValueError('Source official strict shared-negative Hits50 differs')
  ce=float(-self.t.nn.functional.logsigmoid(pos).mean()-self.t.nn.functional.logsigmoid(-neg).mean())
  return {'selection_metric':hits,'hits@50':hits,'ce':ce,'objective':ce,'regularization':0.,'positive_count':len(pos),'negative_count':len(neg),'fusion':'mean_raw_scores'}
 def observe_validation(self,epoch,metrics):
  better=self._best is None or metrics['selection_metric']>self._best
  if better:self._best=metrics['selection_metric'];self._best_step=epoch
  return {'save_selected':better,'stop_requested':False,'best_saved_metric':self._best,'best_saved_step':self._best_step,'full_budget':True}
 def capture_state(self):return {'provenance':copy.deepcopy(self.provenance),'fit_rng':copy.deepcopy(self._rng),'outer_rng':bridge.rng_state(self.t,self.np),'epoch':self._epoch,'selector':{'best':self._best,'best_step':self._best_step}}
 def restore_state(self,state):
  if state['provenance']!=self.provenance:raise ValueError('Source/model/role/protocol ancestry changed')
  self._rng=copy.deepcopy(state['fit_rng']);self._epoch=state['epoch'];self._best=state['selector']['best'];self._best_step=state['selector']['best_step'];bridge.restore_rng(self.t,self.np,state['outer_rng'])

def make_adapter(*,torch,project_root,study,run,device):
 import numpy
 from torch_geometric.utils import negative_sampling
 c=run['config'];assert study['dataset']=='ogbl-collab' and c['provider_nodes']==235868 and c['train_minibatches_per_epoch']==17 and c['training']['batch_size']==65536
 inputs={k:torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True) for k,row in c['link_inputs'].items()};graph,train,select=inputs['graph'],inputs['train'],inputs['select']
 if set(graph)!={'x','raw_edge_index'} or set(train)!={'positive'} or set(select)!={'positive','negative'}:raise ValueError('Exact collab source authorities required')
 assert graph['x'].shape==(235868,128) and len(train['positive'])==1179052 and len(select['positive'])==60084 and len(select['negative'])==100000
 target=torch.device(device);model,opt,modules,details=construct(torch,project_root,c['prototype_sources'],arm=c['link_prototype_arm'],seed=run['seed'],body=run['body'],lr_multiplier=c['lr_multiplier'],dropout=c['dropout'],device=device);native,utils=bridge.native_modules(project_root,{'model':c['prototype_sources']['native'],'utils':c['prototype_sources']['utils']})
 graph={k:v.to(target) for k,v in graph.items()};train={k:v.to(target) for k,v in train.items()};select={k:v.to(target) for k,v in select.items()};directory=Path(project_root)/c['future_replay_output_root']/study['study_id']/run['run_id']
 return LinkPrototypeAdapter(torch,numpy,model,opt,modules,details,root=project_root,run=run,study=study,graph=graph,train=train,select=select,replay_directory=directory,utils=utils,sampler=negative_sampling)
