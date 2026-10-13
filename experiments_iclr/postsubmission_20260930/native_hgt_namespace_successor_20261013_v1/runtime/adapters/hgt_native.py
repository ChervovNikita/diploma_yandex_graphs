"""Pinned native HGB HGT and exact module-semantic mixed backward policies.

No model, numerical import, data access, optimizer or fit runs at import.
New roles/seeds and original boundary typed adaptation are explicit. The
released feature2/raw directed relation schema and native inner equations stay.
"""
import ast,copy,hashlib,importlib.util,math,sys,types
from pathlib import Path
from .classifier import ClassifierAdapter
from engine import AdapterStepError
from spec import confined,normalized_id,ORIGINAL_PAPER_DATASETS

PINS={'implementation':'97a401fd289bbea9a21ae98d3b488ecea91a69547a2ec71d9d7392adb0195042',
 'families':'8f55059b017d91739e961b66dce5e489a5e24cd5e300da61343597bdbe81abec',
 'driver':'642c3478c2772d9b146fdea7e9b0cfbd97e2d319b38e428281d0441b1b8f77e8',
 'onecycle':'2798f6c810b20180dedf1a3ec5e22c7092c7c258298afbff5af326cc40b8dc7c',
 'inputs':'2f29f5839f53f711c9823c4f2671f4950e9c0e8e655000a0e5a5704944a9f866',
 'block_policy':'8f2cec2940f67e825d8b2532fc7623dfded0359be8a639d8bdfb00e2f932b6c2',
 'mixed_driver':'1b6402f07168a207ef941ff14509381b090e36ca011c3a672bf5f98782724fab',
 'mixed_common':'55bcdca2ca7feb586e9f428058e1ae70a48853c478903eb81244d0540dec5f1f',
 'original':'07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
 'original_driver':'e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc'}
ARMS=('native_M1','factor_M1','global_BE','shared_relation','CP','unrestricted','untied_HGT','wider_BE','original_gnnm',
      'own/own','pool/pool','pool/own','own/pool')
POLICIES=('own/own','pool/pool','pool/own','own/pool')
BUDGET={'unit':'optimizer_updates','policy':'hgt_native_selector_stop','maximum':300,'minimum':31,'patience':30}
SELECTION={'metric':'pooled_nll','direction':'min','ties':'last','evaluation_interval':1,'checkpoint0':False,'patience_steps':30}
CONTRACT={'schema':'native-HGT-mixed-DBLP-newroles-v2','width':64,'layers':3,'heads':8,'dropout':.2,'use_norm':True,'feature_type':2,
 'optimizer':'native AdamW constructor defaults lr.001/betas(.9,.999)/eps1e-8; WD1e-4, including untouched None gradients',
 'scheduler':'native OneCycleLR maxlr.001/total300/pct_start.05/cosine; step(update) after every model optimizer update; full data-only state saved/restored',
 'selector':'post-update mean raw logits SELECT CE; latest exact tie resets patience; stop30 strictly worse or300 cap; no0',
 'mixed':'exact source block_policy.partition/backward; one complete four-route forward; shared/model.core versus private/be a,b; mixed uses2 autograd.grad calls; no pooled hidden state',
 'native_M1':'fresh exact source NativeMember; explicit private route0 stream realizes native seed+700001 dropout law',
 'factor_M1':'NEW exact source be-mode row0 one-member slice; native core unchanged; four own fits produce genuine I4',
 'untied_HGT':'source four complete native bodies; member0 paired core; >0seed+200003member; one coupled ownCE optimizer/selector; not genuine I4',
 'wider_BE':'exact source minimal head-divisible width72 reaches CP count; helper CP count draw precedes wider slow initialization; discarded constructors charged',
 'original_GNNM':'pinned actual BEBlock R/S/B at each typed input adapter and target head; one shared complete NativeHGT inner body; source RSB initialization and mean logits; explicit compatible typed architecture and common native HGT optimizer/selector adaptation',
 'original_preserves_native_backbone_nonlinearities':True,'extra_homogeneous_GELU_or_output_LN':False,
 'roles':'fresh custom20TRAIN/30SELECT/30CAL perclass, remaining released development pool evaluate; official test label member never opened',
 'serving':'mean raw logits','CAL':'declared common post-mean-logit scalar500 Adam lr.01; separate from source training',
 'historical_fit_states_outcomes_or_authorization_inherited':False,'published_score_replication':False,'novelty_claim':False}

def receipt(root,path):
 raw=Path(path).read_bytes();return {'path':str(Path(path).resolve().relative_to(Path(root).resolve())),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def bound(root,row):
 p=confined(root,row['path']);r=receipt(root,p)
 if any(r[k]!=row[k] for k in ('path','bytes','sha256')):raise ValueError('Source/input byte custody changed: '+row['path'])
 return p

def source(root,row,name):
 p=bound(root,row);identifier='HGT_source_'+name+'_'+row['sha256'][:12]
 sp=importlib.util.spec_from_file_location(identifier,p);m=importlib.util.module_from_spec(sp);sys.modules[identifier]=m;sp.loader.exec_module(m);return m

def be_class(root,row,t):
 """Execute exactly the pinned BEBlock class AST, without unrelated PyG imports."""
 p=bound(root,row);node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BEBlock')
 ns={'__name__':'literal_GNNM_BEBlock','torch':t,'nn':t.nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p)+'::BEBlock_exact_AST','exec'),ns)
 return ns['BEBlock'],hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()

def expected_parameters(input_dims,classes,width=64,types_count=4,relations=6,layers=3):
 h=width;heads=8;d=h//heads
 native=sum(f*h+h for f in input_dims.values())+layers*(4*types_count*(h*h+h)+2*h*types_count+relations*heads+2*relations*heads*d*d+types_count)+h*classes+classes
 be=native+layers*8*h
 return {'native_M1':native,'factor_M1':native+layers*2*h,'global_BE':be,
  'own/own':be,'pool/pool':be,'pool/own':be,'own/pool':be,'CP':be+layers*(4+relations+h),
  'shared_relation':be+layers*(1+relations+h),'unrestricted':native+layers*(4*h+4*relations*h),'untied_HGT':4*native,
  'original_gnnm':native-(types_count*h+classes)+4*sum(input_dims.values())+8*types_count*h+4*(h+2*classes)}

def validate_config(study,run):
 c=copy.deepcopy(run['config']);arm=c['hgt_arm']
 if arm not in ARMS or normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS:raise ValueError('Exact supported new HGT source arm')
 if set(c['hgt_sources'])!=set(PINS) or any(c['hgt_sources'][k]['sha256']!=v for k,v in PINS.items()):raise ValueError('Complete immutable HGT/mixed/original source pins')
 if c['source_optimizer_count']!=1 or c['physical_slow_bodies']!=(4 if arm=='untied_HGT' else 1) or c['members']!=(1 if arm in ('native_M1','factor_M1') else 4):raise ValueError('Exact source slow-body/member/one optimizer ownership')
 if c['width']!=(72 if arm=='wider_BE' else 64) or c['depth']!=3 or c['heads']!=8 or c['dropout']!=.2 or c['weight_decay']!=.0001 or c['learning_rate']!=.001 or c['classes']!=4 or c['fusion']!='mean_logits':raise ValueError('Native fixed HGT64or72/3layers/8heads/.2/OneCycle maxlr.001/WD1e-4')
 if c['initialization_seed']!=run['seed']+1000003*run['body'] or run['body'] not in (range(4) if arm in ('native_M1','factor_M1') else (0,)):raise ValueError('Fresh distinct own-body initialization seed')
 if study['stage']!='engineering' and (run['budget']!=BUDGET or run['selection']!=SELECTION):raise ValueError('Native post-update latest SELECTCE/p30/cap300/no0 stopping required')
 if set(c['node_inputs'])!={'graph','train','select'}:raise ValueError('Only GRAPH/TRAIN/SELECT fit authorities')
 for role in ('train','select'):
  if c['node_inputs'][role]!=study['data']['roles'][role][str(run['split'])]:raise ValueError('Role byte authority differs')
 return c

class HGTAdapter(ClassifierAdapter):
 def train_step(self,step):
  counts={'optimizer_updates':0,'optimizer_step_attempts':0,'model_forwards':0,'forward_attempts':0,'backward_calls':0,'backward_attempts':0,'source_autograd_grad_calls':0,'source_scheduler_steps':0}
  try:
   self.model.train();self.optimizer.zero_grad();counts['forward_attempts']+=1;z=self.model();counts.update(model_forwards=1,route_predictions=len(z),supervised_examples=len(z)*len(self.train_ids))
   own=self.implementation.mean_member_ce(z,self.train_ids,self.train_y);pool=self.F.cross_entropy(z.mean(0)[self.train_ids],self.train_y)
   if not self.t.isfinite(own) or not self.t.isfinite(pool):raise FloatingPointError('Nonfinite complete HGT own/pooled TRAIN CE')
   policy=self.provenance['config']['hgt_arm']
   if policy in POLICIES:
    counts['backward_attempts']+=2 if policy in ('pool/own','own/pool') else 1
    self.block_policy.backward(self.t,policy,self.blocks,z,self.train_ids,self.train_y,own)
    counts['backward_calls']+=2 if policy in ('pool/own','own/pool') else 1;counts['source_autograd_grad_calls']=2 if policy in ('pool/own','own/pool') else 0
   else:counts['backward_attempts']+=1;own.backward();counts['backward_calls']+=1
   # Preserve all unused None leaves: native AdamW omits decay/moments for them.
   if any(p.grad is not None and not self.t.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite HGT source parameter gradient')
   counts['source_None_gradient_leaves']=sum(p.grad is None for p in self.model.parameters())
   counts['optimizer_step_attempts']+=1;self.optimizer.step();counts['optimizer_updates']+=1;self.scheduler.step(step);counts['source_scheduler_steps']+=1
   counts.update(self.model.operations(True))
   if not self.preparation_charged:counts.update(self.preparation_counts);self.preparation_charged=True
   shared=pool if policy in ('pool/pool','pool/own') else own;private=pool if policy in ('pool/pool','own/pool') else own
   return {'objective':self._scalar(own),'ce':self._scalar(own),'regularization':0.,'pooled_ce':self._scalar(pool),
    'objective_role':'source own-CE trace; block assignments explicitly recorded; mixed does not assert a single scalar potential',
    'shared_block_CE':self._scalar(shared),'private_block_CE':self._scalar(private),'learning_rate':[g['lr'] for g in self.optimizer.param_groups],
    'operation_counts':counts}
  except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error
 def validate(self,step):
  self.model.eval()
  with self.t.no_grad():
   z=self.model();value=self.F.cross_entropy(z.mean(0)[self.select_ids],self.select_y);own=self.implementation.mean_member_ce(z,self.select_ids,self.select_y)
   if not self.t.isfinite(z).all():raise FloatingPointError('Nonfinite full HGT SELECT logits')
  return {'objective':self._scalar(value),'pooled_nll':self._scalar(value),'selection_metric':self._scalar(value),'ce':self._scalar(value),'regularization':0.,'own_ce':self._scalar(own),
   'operation_counts':{'optimizer_updates':0,'backward_calls':0,'model_forwards':1,'route_predictions':len(z),'supervised_examples':0,**self.model.operations(False)}}
 def capture_state(self):
  state=super().capture_state();state['native_HGT']={'scheduler':self.onecycle.portable_state(self.scheduler),'streams':self.streams.state_dict(),
   'custody':copy.deepcopy(self.custody),'semantic_blocks':self.blocks['receipt'] if self.blocks is not None else None,'preparation_charged':self.preparation_charged};return state
 def restore_state(self,state):
  extra=state['native_HGT']
  if extra['custody']!=self.custody or extra['semantic_blocks']!=(self.blocks['receipt'] if self.blocks is not None else None):raise ValueError('HGT complete graph/source/block custody differs')
  super().restore_state(state);self.onecycle.restore_state(self.scheduler,extra['scheduler']);self.streams.load_state_dict(extra['streams']);self.preparation_charged=extra['preparation_charged']

def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
 from torch.nn import functional as F
 if torch.get_default_dtype()!=torch.float32 or torch.empty(0).device.type!='cpu':raise ValueError('Source native float32 CPU construction before device transfer')
 root=Path(project_root).resolve();c=validate_config(study,run);arm=c['hgt_arm'];s=c['hgt_sources'];seed=c['initialization_seed']
 for row in s.values():bound(root,row)
 impl=source(root,s['implementation'],'implementation');families=source(root,s['families'],'families');onecycle=source(root,s['onecycle'],'onecycle');policy=source(root,s['block_policy'],'block_policy')
 pack={k:torch.load(bound(root,row),map_location='cpu',weights_only=True) for k,row in c['node_inputs'].items()};g=pack['graph']
 if set(g)!={'schema','features','relations'} or g['schema']['target']!='0' or g['schema']['feature_type']!=2:raise ValueError('Exact target-free feature2 typed GRAPH pack')
 schema=g['schema'];graph=impl.HeteroGraph(schema['node_counts'],[impl.Relation(r['raw_id'],r['source'],r['target'],r['src'].to(device),r['dst'].to(device)) for r in g['relations']],schema['relation_row_order'])
 features={k:v.to(device) for k,v in g['features'].items()}
 if set(features)!=set(graph.ntypes) or any(v.dtype!=torch.float32 or v.shape!=(graph.node_counts[k],schema['input_dims'][k]) or v.requires_grad or not torch.isfinite(v.values() if v.is_sparse else v).all() for k,v in features.items()):raise ValueError('Exact complete finite immutable native typed feature2 tensors')
 if study['stage']!='engineering' and (schema['node_counts']!={'0':4057,'1':14328,'2':7723,'3':20} or schema['input_dims']!={'0':334,'1':14328,'2':7723,'3':20} or sum(len(r.src) for r in graph.relations)!=239566):raise ValueError('Actual complete DBLP typed geometry changed')
 for role in ('train','select'):
  p=pack[role]
  if set(p)!={'ids','y'} or p['ids'].dtype!=torch.long or p['y'].dtype!=torch.long or p['ids'].ndim!=1 or p['y'].shape!=p['ids'].shape or len(p['ids'])<1 or len(torch.unique(p['ids']))!=len(p['ids']) or int(p['ids'].min())<0 or int(p['ids'].max())>=graph.node_counts['0'] or int(p['y'].min())<0 or int(p['y'].max())>=4:raise ValueError('Compact distinct in-range TRAIN/SELECT target0 packs')
 if torch.isin(pack['train']['ids'],pack['select']['ids']).any():raise ValueError('TRAIN/SELECT overlap')
 be_ast=None;factor_slice=arm=='factor_M1';native_M1=arm=='native_M1';source_arm='native_HGT' if native_M1 else 'global_BE' if arm in POLICIES or factor_slice else arm
 devices=[] if torch.device(device).type=='cpu' else [torch.device(device).index or 0]
 with torch.random.fork_rng(devices=devices):
  if arm=='original_gnnm':
   torch.manual_seed(seed);inner=impl.NativeHGT(graph,schema['input_dims'],64,4,3,8,True);BE,be_ast=be_class(root,s['original'],torch)
   class RoutedBE(BE):
    def __init__(self,in_shape,out_shape,is_first):super().__init__(in_shape,out_shape,is_first,'cpu',4);self.active_route=0;self.out_features=out_shape
    def forward(self,x):
     # Sparse feature2 identities use the same stored-entry multiply and W.
     if x.is_sparse:
      x=x.coalesce();scaled=torch.sparse_coo_tensor(x.indices(),x.values()*self.R[self.active_route][x.indices()[1]],x.shape,device=x.device,is_coalesced=True)
      return self.W(scaled)*self.S[self.active_route]+self.B[self.active_route]
     return super().forward(x,self.active_route)
   inner.adapt_ws=torch.nn.ModuleList(RoutedBE(schema['input_dims'][t],64,True) for t in graph.ntypes);inner.out=RoutedBE(64,4,False)
   module=inner.to(device);constructor_bodies=1
  else:
   module=families.build(impl,graph,schema['input_dims'],4,seed,seed+900001,'cpu',[source_arm])[source_arm]
   if factor_slice:
    core=module.core;factor=torch.nn.ModuleList(impl.PrivateFactors(v.a[:1],v.b[:1],torch.zeros(1),torch.zeros(len(graph.edge_dict)),torch.zeros(64),'be') for v in module.factors)
    module=types.SimpleNamespace(core=core,factors=factor)
   if native_M1:module=module.core
   if arm not in ('native_M1','factor_M1'):module=module.to(device)
   constructor_bodies=5 if arm=='untied_HGT' else 3 if arm=='wider_BE' else 2
 streams=impl.MemberStreams([seed+700001+1009*m for m in range(4)],device)
 if factor_slice:
  core,factor=module.core.to(device),module.factors.to(device)
  class FactorSingle(torch.nn.Module):
   def __init__(self):super().__init__();self.core=core;self.factors=factor
  module=FactorSingle()
 elif native_M1:module=module.to(device)
 blocks=policy.partition(impl,module) if arm in POLICIES else None
 class Bank(torch.nn.Module):
  def __init__(self):
   super().__init__();self.source_model=module;self.fixed=tuple(features.values())+tuple(v for r in graph.relations for v in (r.src,r.dst));self.versions=tuple(v._version for v in self.fixed)
  def forward(self):
   if self.versions!=tuple(v._version for v in self.fixed):raise ValueError('Typed GRAPH/features mutated')
   if arm in ('native_M1','factor_M1'):
    callback=(lambda:module.core(graph,features,'0',module.factors,0)) if factor_slice else lambda:module(graph,features,'0')
    out=(streams.run(0,callback) if self.training else callback()).unsqueeze(0)
   elif arm=='original_gnnm':
    values=[]
    for m in range(4):
     for adapter in module.adapt_ws:adapter.active_route=m
     module.out.active_route=m;callback=lambda:module(graph,features,'0');values.append(streams.run(m,callback) if self.training else callback())
    for adapter in module.adapt_ws:adapter.active_route=0
    module.out.active_route=0;out=torch.stack(values)
   else:out=module(graph,features,'0',streams if self.training else None)
   if self.versions!=tuple(v._version for v in self.fixed):raise ValueError('Typed graph changed during native forward')
   return out
  def operations(self,training):return {'source_native_HGT_complete_paths':c['members'],'source_native_HGT_layer_paths':3*c['members'],
    'source_canonical_relation_attention_paths':3*c['members']*len(graph.edge_dict),'source_HGT_dropout_draws':3*c['members']*len(graph.ntypes) if training else 0}
 model=Bank();expected=expected_parameters(schema['input_dims'],4,width=c['width'],types_count=len(graph.ntypes),relations=len(graph.edge_dict))
 if arm=='wider_BE':expected_count=expected['global_BE']
 else:expected_count=expected[arm]
 actual=sum(p.numel() for p in model.parameters())
 if actual!=expected_count:raise ValueError('Exact algebraic native/fast/body parameter count differs')
 optimizer=torch.optim.AdamW(model.parameters(),weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=300,max_lr=1e-3,pct_start=.05)
 adapter=HGTAdapter(torch,F,model,optimizer,logits_fn=lambda m,training,step:m(),train_ids=pack['train']['ids'].to(device),train_y=pack['train']['y'].to(device),select_ids=pack['select']['ids'].to(device),select_y=pack['select']['y'].to(device),
  regularizer=lambda m:next(m.parameters()).new_zeros(()),fusion='mean_logits',selector={'metric':'pooled_nll','direction':'min','ties':'last','patience_steps':30,'save_every':1},
  provenance={'run_id':run['run_id'],'source':study['source'],'data':study['data'],'config':c,'actual_model_seed':seed,'contract':CONTRACT,'actual_parent':None,'actual_H0_hash':None})
 adapter.implementation=impl;adapter.block_policy=policy;adapter.blocks=blocks;adapter.streams=streams;adapter.scheduler=scheduler;adapter.onecycle=onecycle;adapter.preparation_charged=False
 adapter.custody={'inputs':c['node_inputs'],'sources':s,'graph_schema':schema,'source_BEBlock_AST_sha256':be_ast,'factor_row0_slice':factor_slice,'actual_parameters':actual}
 adapter.preparation_counts={'source_complete_body_constructors':constructor_bodies,'source_discarded_body_constructors':constructor_bodies-c['physical_slow_bodies'],
  'source_optimizer_constructors':1,'source_scheduler_constructors':1,'source_base_factor_modules':0 if arm=='original_gnnm' else 9,
  'source_factor_module_constructors':0 if arm=='original_gnnm' else 27 if arm=='wider_BE' else 9 if arm in ('native_M1','untied_HGT') else 15 if arm=='factor_M1' else 12,
  'source_original_typed_BEBlock_constructors':5 if arm=='original_gnnm' else 0,'source_native_Model_copy_constructors':1 if arm not in ('original_gnnm','wider_BE') else 0}
 return adapter
