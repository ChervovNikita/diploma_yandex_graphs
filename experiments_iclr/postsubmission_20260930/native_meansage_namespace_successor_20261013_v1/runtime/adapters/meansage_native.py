"""Exact native peer/tour equations, source TF1Adam and persistent route RNG.

Imports are inert. New-data geometry, paid seeds and algebraic counts are
explicit namespace bindings. Source bodies, maps, factors, scans and dropout
equations retain their pinned bytes. Historical states/results are never read.
"""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import random
import types
from .classifier import ClassifierAdapter
from engine import AdapterStepError
from spec import confined, normalized_id, ORIGINAL_PAPER_DATASETS

PINS={
 'native':'242396b38ab43a99ac9f0b00f63424569688a223c0fc3838fa5aeb94e3b5a301',
 'tour':'c6086cbfb0b06fdc79a12edd1ee42720218345a4197d9237e95b51694c8b2ac5',
 'scan':'5d68ff927221a03d5da4dc4b4a506cf729cd61866190f8bd3fae8e9b90537e05',
 'tours':'86f6504de408ec07e10fa109946e6bd0b1acb22753e1dc1b8c7b8947ba3461ea',
 'author':'dfd19cf6da0460a4de49697e2c5afe185e85e8192db4bb51e548868b9a2d1401',
 'original':'07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
 'original_driver':'e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc',
}
PEER_ARMS=('P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4')
TOUR_ARMS=('P1','F1','I4','S4','PV1','WI4','V1','SW4','C4','R4')
INDEPENDENT_ARMS=('I4','IA4','WI4')
SELECTION={'metric':'objective','direction':'min','ties':'last','evaluation_interval':1,'checkpoint0':False,'patience_steps':50}
BUDGET={'unit':'optimizer_updates','policy':'native_selector_stop','maximum':100000,'minimum':51,'patience':50}
CONTRACT={'schema':'native-meansage-peer-tour-newdata-v1','native_width':32,'layers':2,'dropout':.4,
 'native_optimizer':'source_TF1Adam_float32_beta_powers_eps_before_bias_correction',
 'native_lr':.001,'native_penalty':'0.05 first-layer ordinary/effective matrix square; mean own routes once',
 'selector':'post-update latest tied SELECT mean-own CE+native penalty; stop50 strictly worse or100000 cap; no checkpoint0',
 'native_serving':'mean_class_probabilities','calibration':'member_logprob_temperature_before_probability_pool; CPUfloat64Adam lr.05 clamp[-5,5]; latest min steps0..500; all500 updates',
 'independent_factory':'source constructs allfour; choose declared run.body; discarded constructors charged; exactlyone optimizer per fit',
 'source_semantics_preserved':True,'geometry_seed_count_bindings_declared':True,'TensorFlow_bitwise_replication':False,
 'historical_states_scores_authorization_inherited':False,
 'original_GNNM':'literal full TABMModel SAGE2 H128 four source members, LayerNorm; four ownCE/4 backwards then one AdamW WD0 lr.001; latest mean-own SELECTCE/native stop is declared adaptation; raw-logit-mean serving',
 'original_GNNM_architecture_identical_to_H32_clipped_native':False,
 'source_legacy_Physics_data_or_fit_reproduction':False,'higher_order_scan_supported':False}

def receipt(root,path):
 raw=Path(path).read_bytes()
 return {'path':str(Path(path).resolve().relative_to(Path(root).resolve())),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def bound(root,row):
 path=confined(root,row['path']);got=receipt(root,path)
 if any(got[k]!=row[k] for k in ('path','bytes','sha256')):raise ValueError('Bound file changed: '+row['path'])
 return path

def source(root,row):
 path=bound(root,row);ns={'__name__':'meansage_isolated_source','__file__':str(path)}
 exec(compile(path.read_text(),str(path),'exec'),ns)
 return types.SimpleNamespace(**ns)

def counts(features,classes):
 h=32;p=2*features*h+h+2*h*classes+classes;f=2*(features+h)+2*(h+classes)
 ordinary={'P1':p,'F1':p+f,'I4':4*(p+f),'S4':p+3*(h+classes)+4*f}
 peer={**ordinary,'PA1':p+1024,'IA4':4*(p+f+1024),
       **{a:ordinary['S4']+1024 for a in ('SA4','SM4','SX4')},'UX4':4*(p+f+1024)}
 tour={**ordinary,'PV1':p+3136,'WI4':4*(p+f+3136),'V1':p+f+3136,
       **{a:ordinary['S4']+3136 for a in ('SW4','C4','R4')}}
 return peer,tour

def native_source(root,row,*,nodes,features,classes,seed):
 m=source(root,row);m.build_arm.__globals__.update(NODES=nodes,FEATURES=features,HIDDEN=32,CLASSES=classes,BLOCKS=(seed,),EXPECTED_ARM_PARAMETERS=counts(features,classes)[0])
 return m

def tour_source(root,row,*,nodes,features,classes,seed):
 m=source(root,row);spec=copy.deepcopy(m.SPEC)
 for a,v in spec.items():v['parameters']=counts(features,classes)[1][a]
 m.build_arm.__globals__.update(NODES=nodes,FEATURES=features,HIDDEN=32,CLASSES=classes,BLOCKS=(seed,),SPEC=spec)
 return m

def tour_helper(root,row,*,nodes,edges,seed):
 path=bound(root,row);tree=ast.parse(path.read_text())
 class Shapes(ast.NodeTransformer):
  def visit_Constant(self,node):
   names={34493:'BOUND_NODES',530417:'BOUND_EDGES',68985:'BOUND_MAX_TOKENS'}
   return ast.copy_location(ast.Name(id=names[node.value],ctx=ast.Load()),node) if type(node.value)==int and node.value in names else node
 ns={'__name__':'meansage_bound_tours','__file__':str(path),'BOUND_NODES':nodes,'BOUND_EDGES':edges,'BOUND_MAX_TOKENS':2*nodes-1}
 exec(compile(ast.fix_missing_locations(Shapes().visit(tree)),str(path)+'::explicit_geometry','exec'),ns);ns['BLOCKS']=(seed,)
 return types.SimpleNamespace(**ns)

def tensor_hash(t):
 if t.is_sparse:
  return hashlib.sha256((tensor_hash(t.indices())+tensor_hash(t.values())+str(tuple(t.shape))).encode()).hexdigest()
 raw=t.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes()
 return hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+raw).hexdigest()

def tree_hash(t,value):
 if t.is_tensor(value):return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':tensor_hash(value)}
 if isinstance(value,dict):return {str(k):tree_hash(t,v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [tree_hash(t,v) for v in value]
 return value

def graph_check(t,x,edge):
 n=len(x);values=x.values() if x.is_sparse else x
 if x.dtype!=t.float32 or x.ndim!=2 or x.requires_grad or (x.is_sparse and not x.is_coalesced()) or not bool(t.isfinite(values).all()) or not bool(((values==0)|(values==1)).all()):raise ValueError('Full raw binary float32 X required; no normalization')
 if edge.dtype!=t.long or edge.shape[0]!=2 or edge.ndim!=2 or edge.numel()==0 or int(edge.min())<0 or int(edge.max())>=n:raise ValueError('Complete int64 graph required')
 support=t.sparse_coo_tensor(edge,t.ones(edge.shape[1]),(n,n)).coalesce()
 if support._nnz()!=edge.shape[1] or not t.equal(support.values(),t.ones_like(support.values())):raise ValueError('Exact binary coalesced support')
 ids=support.indices();diagonal=ids[0]==ids[1]
 if not t.equal(ids[0][diagonal],t.arange(n)):raise ValueError('Exactly one native self loop per node')
 if not t.equal(support.indices(),t.sparse_coo_tensor(edge.flip(0),t.ones(edge.shape[1]),(n,n)).coalesce().indices()):raise ValueError('Reciprocal native support')

def components(edge,n):
 neighbors=[[] for _ in range(n)]
 for u,v in zip(*edge.tolist()):
  if u!=v:neighbors[u].append(v)
 visited=set();count=0
 for root in range(n):
  if root in visited:continue
  count+=1;visited.add(root);pending=[root]
  while pending:
   for v in neighbors[pending.pop()]:
    if v not in visited:visited.add(v);pending.append(v)
 return count

def prepare_graph(t,root,graph_binding,sources,seeds,directory):
 """Actual GRAPH-only preparation; creates no model, optimizer or target read."""
 import json
 directory=Path(directory);directory.mkdir(parents=True)
 graph=t.load(bound(root,graph_binding),map_location='cpu',weights_only=True)
 if set(graph)!={'x','edge_index'}:raise ValueError('Target-free GRAPH schema required')
 x,edge=graph['x'],graph['edge_index'];graph_check(t,x,edge);n,f=x.shape
 sparse=x if x.is_sparse else x.to_sparse_coo().coalesce();degree=t.bincount(edge[1],minlength=n).float()[:,None]
 native=native_source(root,sources['native'],nodes=n,features=f,classes=8,seed=seeds[0])
 C,D=native.incidence_from_raw_binary_features(t,sparse)
 inc={'schema':'raw-binary-native-incidence-v1','graph':graph_binding,'C':C,'D':D,'X_hash':tensor_hash(sparse),'labels_used':False}
 path=directory/'incidence.pt';t.save(inc,path);inc_binding=receipt(root,path)
 cc=components(edge,n);tour_bindings={};outer=t.get_rng_state().clone()
 for seed in seeds:
  helper=tour_helper(root,sources['tours'],nodes=n,edges=edge.shape[1],seed=seed)
  tours=helper.build_tours(t,edge,seed,expected_components=cc)
  if not t.equal(outer,t.get_rng_state()):raise ValueError('Private tour preparation changed global RNG')
  capsule={'schema':'native-meansage-fixed-tours-v1','graph':graph_binding,'source':sources['tours'],'seed':seed,'nodes':n,'components':cc,'tours':tours,'labels_used':False,'model_constructors':0,'optimizer_constructors':0}
  path=directory/f'tours_{seed}.pt';t.save(capsule,path);tour_bindings[str(seed)]=receipt(root,path)
 value={'schema':'native-meansage-photo-graph-preparation-v1','graph':graph_binding,'nodes':n,'features':f,'classes':8,'edges':edge.shape[1],'self_edges':n,'components':cc,
  'raw_binary':True,'reciprocal_coalesced':True,'degree_hash':tensor_hash(degree),'X_hash':tensor_hash(sparse),'incidence':inc_binding,'tours':tour_bindings,
  'tour_length':2*n-cc,'tour_sampling':'private randomized DFS spanning forest arrival/return; not uniform spanning tree',
  'target_tensors_opened':False,'models_constructed':0,'optimizers_constructed':0,'scientific_fits':0,'scores':0}
 (directory/'CUSTODY.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return value

class SourceStreams:
 def __init__(self,t,device,*,block,members,features,classes,start_index=0):
  import numpy as np
  self.t,self.np,self.device=t,np,t.device(device);self.states=[]
  for m in range(start_index,start_index+members):
   seed=block+1000003*m;g=t.Generator(device='cpu').manual_seed(seed)
   for shape in ((features,32),(features,32),(32,classes),(32,classes)):
    b=math.sqrt(6./sum(shape));t.empty(shape).uniform_(-b,b,generator=g)
   ng=np.random.RandomState(seed).get_state()
   self.states.append({'python':random.Random(seed).getstate(),'numpy':(ng[0],ng[1].tolist(),ng[2],ng[3],ng[4]),'cpu':g.get_state(),
    'cuda':t.Generator(device=self.device).manual_seed(seed).get_state() if self.device.type=='cuda' else None})
 def capture(self):
  ng=self.np.random.get_state()
  return {'python':random.getstate(),'numpy':(ng[0],ng[1].tolist(),ng[2],ng[3],ng[4]),'cpu':self.t.get_rng_state(),
    'cuda':self.t.cuda.get_rng_state(self.device) if self.device.type=='cuda' else None}
 def restore(self,state):
  random.setstate(state['python']);ng=state['numpy'];self.np.random.set_state((ng[0],self.np.asarray(ng[1],dtype=self.np.uint32),ng[2],ng[3],ng[4]))
  self.t.set_rng_state(state['cpu'].cpu())
  if self.device.type=='cuda':self.t.cuda.set_rng_state(state['cuda'].cpu(),self.device)
 @contextmanager
 def scope(self,index):
  outer=self.capture();self.restore(self.states[index])
  try:yield
  finally:self.states[index]=self.capture();self.restore(outer)

def validate_config(study,run):
 c=copy.deepcopy(run['config']);arm=c['meansage_arm'];family=c['meansage_family'];original=arm=='original_gnnm'
 if family not in ('peer','tour') or arm not in (PEER_ARMS if family=='peer' else TOUR_ARMS)+('original_gnnm',):raise ValueError('Exact source family/arm required')
 if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS:raise ValueError('Original datasets excluded')
 if (c.get('learning_rate')!=.001 or c.get('dropout')!=.4 or c.get('weight_decay')!=0 or c.get('depth')!=2 or c.get('width')!=(128 if original else 32) or c.get('classes')!=8 or c.get('source_optimizer_count')!=1):raise ValueError('Native fixed recipe, actual geometry and one own optimizer required')
 if study['stage']!='engineering' and (run['budget']!=BUDGET or run['selection']!=SELECTION):raise ValueError('Exact source stop100k/p50/latest-own-total/no0 required')
 if set(c['meansage_sources'])!=set(PINS) or any(c['meansage_sources'][k]['sha256']!=v for k,v in PINS.items()):raise ValueError('Exact source pins required')
 if set(c['node_inputs'])!={'graph','train','select'}:raise ValueError('GRAPH/TRAIN/SELECT only')
 for role in ('train','select'):
  if c['node_inputs'][role]!=study['data']['roles'][role][str(run['split'])]:raise ValueError('Role authority changed')
 if c['initialization_seed']!=run['seed']+1000003*run['body']:raise ValueError('Actual body seed changed')
 expected_members=4 if original or arm in ('S4','SA4','SM4','SX4','UX4','PV1','V1','SW4','C4','R4') else 1
 if c['members']!=expected_members or c['physical_slow_bodies']!=(4 if arm=='UX4' else 1) or c['is_original_gnnm']!=original:raise ValueError('Actual logical view/body ownership differs')
 if run['body'] not in (range(4) if arm in INDEPENDENT_ARMS or arm in ('P1','PA1') else (0,)):raise ValueError('Undeclared body index')
 return c

class NativeAdapter(ClassifierAdapter):
 def train_step(self,step):
  counts={'optimizer_updates':0,'optimizer_step_attempts':0,'backward_calls':0,'backward_attempts':0,'model_forwards':0,'forward_attempts':0,'route_predictions':0,'supervised_examples':0}
  try:
   self.model.train();self.optimizer.zero_grad();losses=[]
   if self.model.original:
    for view in range(4):
     counts['forward_attempts']+=1;z=self.model.view(view);counts['model_forwards']+=1;counts['route_predictions']+=1
     loss=self.F.cross_entropy(z[self.train_ids],self.train_y);losses.append(self._scalar(loss));counts['backward_attempts']+=1;(loss/4).backward();counts['backward_calls']+=1
    ce=sum(losses)/4;penalty=0.;total=ce
   else:
    counts['forward_attempts']+=1;logits=self.model();counts.update(model_forwards=1,route_predictions=len(logits))
    ce_tensor=self.t.stack([self.F.cross_entropy(z[self.train_ids],self.train_y) for z in logits]).mean();penalty_tensor=self.model.regularizer();loss=ce_tensor+penalty_tensor
    total,ce,penalty=map(self._scalar,(loss,ce_tensor,penalty_tensor));counts['backward_attempts']+=1;loss.backward();counts['backward_calls']+=1
   counts['supervised_examples']=counts['route_predictions']*len(self.train_ids);counts.update(self.model.operations(True))
   if any(p.grad is None or not bool(self.t.isfinite(p.grad).all()) for p in self.model.parameters()):raise FloatingPointError('All source parameter gradients must be finite and present')
   counts['optimizer_step_attempts']+=1;self.optimizer.step();counts['optimizer_updates']+=1
   if not self.preparation_charged:counts.update(self.preparation_counts);self.preparation_charged=True
   return {'objective':total,'ce':ce,'regularization':penalty,'operation_counts':counts}
  except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error
 def validate(self,step):
  self.model.eval()
  with self.t.no_grad():
   logits=self.model();ce=self.t.stack([self.F.cross_entropy(z[self.select_ids],self.select_y) for z in logits]).mean();penalty=self.model.regularizer();total=ce+penalty
   if not bool(self.t.isfinite(logits).all()):raise FloatingPointError('Nonfinite full SELECT logits')
  value={'objective':self._scalar(total),'ce':self._scalar(ce),'regularization':self._scalar(penalty),'selection_metric':self._scalar(total),
   'operation_counts':{'optimizer_updates':0,'backward_calls':0,'model_forwards':4 if self.model.original else 1,'route_predictions':len(logits),'supervised_examples':0,**self.model.operations(False)}}
  return value
 def capture_state(self):
  state=super().capture_state();state['native_meansage']={'streams':copy.deepcopy(self.streams.states),'global_rng':self.streams.capture(),
   'custody':copy.deepcopy(self.custody),'preparation_charged':self.preparation_charged};return state
 def restore_state(self,state):
  extra=state['native_meansage']
  if extra['custody']!=self.custody:raise ValueError('Source/GRAPH/tour/incidence custody changed')
  super().restore_state(state);self.streams.states=copy.deepcopy(extra['streams']);self.streams.restore(extra['global_rng']);self.preparation_charged=extra['preparation_charged']

def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
 from torch.nn import functional as F
 root=Path(project_root).resolve();c=validate_config(study,run);arm=c['meansage_arm'];family=c['meansage_family'];bindings=c['meansage_sources']
 for row in bindings.values():bound(root,row)
 packs={k:torch.load(bound(root,v),map_location='cpu',weights_only=True) for k,v in c['node_inputs'].items()};graph=packs['graph']
 if set(graph)!={'x','edge_index'} or any(set(packs[r])!={'ids','y'} for r in ('train','select')):raise ValueError('Separated source role pack schemas')
 graph_check(torch,graph['x'],graph['edge_index']);n,f=graph['x'].shape
 if study['stage']!='engineering' and (n,f,c['classes'])!=(7650,745,8):raise ValueError('Bound actual Photo geometry changed')
 for role in ('train','select'):
  p=packs[role]
  if p['ids'].dtype!=torch.long or p['y'].dtype!=torch.long or p['ids'].ndim!=1 or p['y'].shape!=p['ids'].shape or len(p['ids'])<1 or len(torch.unique(p['ids']))!=len(p['ids']) or int(p['ids'].min())<0 or int(p['ids'].max())>=n or int(p['y'].min())<0 or int(p['y'].max())>=c['classes']:raise ValueError('Unique in-range native role pack required')
 if torch.isin(packs['train']['ids'],packs['select']['ids']).any():raise ValueError('TRAIN/SELECT overlap')
 x=graph['x'];x=(x if x.is_sparse else x.to_sparse_coo().coalesce()).to(device);edge=graph['edge_index'].to(device)
 A=torch.sparse_coo_tensor(edge.flip(0),torch.ones(edge.shape[1],device=device),(n,n)).coalesce();degree=torch.bincount(A.indices()[0],minlength=n).float()[:,None]
 original=arm=='original_gnnm';independent=arm in INDEPENDENT_ARMS;block=run['seed'] if independent else c['initialization_seed']
 native=native_source(root,bindings['native'],nodes=n,features=f,classes=c['classes'],seed=block)
 tours=None;C=D=None
 if family=='peer':
  capsule=torch.load(bound(root,c['meansage_incidence']),map_location='cpu',weights_only=True)
  if capsule['graph']!=c['node_inputs']['graph'] or capsule['X_hash']!=tensor_hash(x):raise ValueError('Raw incidence authority differs')
  C,D=capsule['C'].to(device),capsule['D'].to(device)
 if family=='tour' and arm not in ('P1','F1','I4','S4','original_gnnm'):
  capsule=torch.load(bound(root,c['meansage_tours']),map_location='cpu',weights_only=True)
  if capsule['graph']!=c['node_inputs']['graph'] or capsule['seed']!=run['seed'] or capsule['labels_used'] is not False:raise ValueError('Fixed source tour authority differs')
  tours=tuple({k:t[k].to(device) for k in ('tokens','reverse_index','starts','occurrences')} for t in capsule['tours'])
 if original:
  original_source=source(root,bindings['original'])
  with torch.random.fork_rng(devices=[]):
   torch.default_generator.manual_seed(c['initialization_seed']);selected=original_source.TABMModel('SAGE',2,f,128,c['classes'],1,8,'LayerNorm',.4,tabm_inits=4,device='cpu')
  constructor_bodies=1;constructor_scanners=constructor_feedback=0
 elif family=='peer':
  core=native.build_arm(torch,F,arm,block);native.inspect_parameter_ownership(core);selected=core['models'][run['body'] if independent else 0]
  constructor_bodies=core['physical_backbones'];constructor_scanners=0;constructor_feedback=sum(len(m.feedback_maps) for m in core['models'])
 else:
  compiler=tour_source(root,bindings['tour'],nodes=n,features=f,classes=c['classes'],seed=block);backend=source(root,bindings['scan'])
  core=compiler.build_arm(torch,F,arm,block,native_core=native,scan_backend=backend);compiler.inspect_parameter_ownership(core);selected=core['models'][run['body'] if independent else 0]
  constructor_bodies=core['native_bodies'];constructor_scanners=sum(m.scanner is not None for m in core['models']);constructor_feedback=0
 selected=selected.to(device)
 dense_original_x=x.to_dense() if original else None
 streams=SourceStreams(torch,device,block=block,members=c['members'],features=f,classes=c['classes'],start_index=run['body'] if independent else 0)
 class Bank(torch.nn.Module):
  def __init__(self):
   super().__init__();self.native=selected;self.original=original;self.fixed=(x,A,degree)+tuple(v for v in (C,D) if v is not None)+tuple(v for tour in tours or () for v in tour.values());self.versions=tuple(v._version for v in self.fixed)
  def guard(self):
   if self.versions!=tuple(v._version for v in self.fixed):raise ValueError('Fixed full input/incidence/tour mutated')
  def view(self,v):
   with streams.scope(v):return self.native(types.SimpleNamespace(edge_index=edge),dense_original_x,tabm_seed=v)
  def forward(self):
   self.guard()
   if original:out=torch.stack([self.view(v) for v in range(4)])
   elif family=='peer':out=self.native(x,A,degree,C,D,streams.scope)
   else:out=self.native(x,A,degree,tours,streams.scope)
   self.guard();return out
  def regularizer(self):return degree.new_zeros(()) if original else self.native.regularizer()
  def operations(self,training):
   m=c['members'];scans=family=='tour' and arm not in ('P1','F1','I4','S4','original_gnnm');feedback=family=='peer' and arm in ('PA1','IA4','SA4','SM4','SX4','UX4')
   return {'native_meanSAGE_layer_calls':0 if original else 2*m,'original_SAGEConv_calls':2*m if original else 0,'native_sparse_X_dropout_calls':m if training and not original else 0,
    'native_hidden_dropout_calls':m if training and not original else 0,'native_hub_C_products':m if feedback else 0,'native_received_D_products':m if feedback else 0,
    'source_scan_direction_calls':2*m if scans and arm!='R4' else 0,'source_scan_live_backward_directions':2*m if training and scans and arm!='R4' else 0,
    'source_scan_occurrence_means':m if scans else 0}
 model=Bank();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8) if original else source(root,bindings['author']).optimizer_class(torch)(model.parameters())
 custody={'inputs':c['node_inputs'],'sources':bindings,'X_hash':tensor_hash(x),'A_hash':tensor_hash(A),'degree_hash':tensor_hash(degree),'incidence':c.get('meansage_incidence'),'tours':c.get('meansage_tours')}
 adapter=NativeAdapter(torch,F,model,optimizer,logits_fn=lambda m,training,step:m(),train_ids=packs['train']['ids'].to(device),train_y=packs['train']['y'].to(device),
  select_ids=packs['select']['ids'].to(device),select_y=packs['select']['y'].to(device),regularizer=lambda m:m.regularizer(),fusion=c['fusion'],
  selector={'metric':'objective','direction':'min','ties':'last','patience_steps':50,'save_every':1},provenance={'run_id':run['run_id'],'source':study['source'],'data':study['data'],'config':c,'actual_model_seed':c['initialization_seed'],'contract':CONTRACT,'actual_parent':None,'actual_H0_hash':None})
 adapter.streams=streams;adapter.custody=custody;adapter.preparation_charged=False
 adapter.preparation_counts={'source_complete_body_constructors':constructor_bodies,'source_scanner_constructors':constructor_scanners,'source_feedback_map_constructors':constructor_feedback,
  'source_discarded_body_constructors':constructor_bodies-c['physical_slow_bodies'],'source_optimizer_constructors':1,'source_prepared_incidence_loads':int(family=='peer'),'source_prepared_tour_loads':int(tours is not None)}
 return adapter
