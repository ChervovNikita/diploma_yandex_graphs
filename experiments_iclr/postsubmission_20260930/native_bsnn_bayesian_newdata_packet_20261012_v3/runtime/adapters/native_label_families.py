"""Independent SAGE neighborhood/sparse-feature ports; literal learned laws.

Extract complete source constructors and TRAIN units. Rebind only provider
dimensions and declared prospective roles/controller. Historical fit owners,
outcomes, checkpoints and authorization are never imported or inherited.
"""
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import random
from types import SimpleNamespace
import time

try:
 from .classifier import ClassifierAdapter
 from .common_native_wrapper import make_adapter as common_adapter,CONTRACT as COMMON_CONTRACT,render_sources,load_module
 from ..spec import confined
except ImportError:
 from adapters.classifier import ClassifierAdapter
 from adapters.common_native_wrapper import make_adapter as common_adapter,CONTRACT as COMMON_CONTRACT,render_sources,load_module
 from spec import confined

NEIGHBOR_ARMS=('shared4_private_quartile','shared4_private_moments','shared4_common_direction_quartile',
 'factorized_M1_all_signature_quartile','factorized_genuine_I4_all_signature_quartile')
SPARSE_ARMS=('shared4_private_kernel','shared4_common_kernel','shared4_fixed_feature_union',
 'ordinary_M1_feature_kernel','factorized_M1_feature_kernel','genuine_factorized_I4_feature_kernel')


def receipt(root,path):
 path=confined(root,str(path));raw=path.read_bytes()
 return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
 if receipt(root,row['path'])!=row:raise ValueError('Exact frozen source/provider changed')
 return confined(root,row['path'])
def method(cls,name):return copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
def compile_class(name,methods,env,bases=()):
 tree=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name=name,bases=[ast.Name(id=b,ctx=ast.Load()) for b in bases],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
 exec(compile(tree,'<independent_exact_source_provider>','exec'),env);return env[name]


def source_methods(path,features,classes):
 tree=ast.parse(Path(path).read_bytes());family=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Family')
 nodes=[method(family,n) for n in ('seeds','make','scope','logits')];changed=[]
 class Dimensions(ast.NodeTransformer):
  def visit_Call(self,n):
   self.generic_visit(n)
   if ast.unparse(n.func)=='self.Model':
    if len(n.args)!=9 or ast.literal_eval(n.args[2])!=300 or ast.literal_eval(n.args[4])!=10:raise ValueError('Native provider anchors differ')
    n.args[2]=ast.Constant(features);n.args[4]=ast.Constant(classes);changed.extend(['native_features','native_classes'])
   if ast.unparse(n.func)=='torch.randint':
    if ast.literal_eval(n.args[2])!=(1,300):raise ValueError('Source sign shape differs')
    n.args[2]=ast.Tuple(elts=[ast.Constant(1),ast.Constant(features)],ctx=ast.Load());changed.append('first_sign_features')
   return n
 nodes=[Dimensions().visit(n) for n in nodes]
 if changed!=['native_features','native_classes','first_sign_features']:raise ValueError('Three exact provider sites required')
 return family,nodes,changed


def initialize(source,torch,root,c,x,edges,ids,y,device):
 import numpy as np
 b=c['wrapper_sources'];source.torch=torch;source.F=torch.nn.functional;source.device=torch.device(device);source.cuda=source.device.index or 0
 source.cfg={'backbone':'SAGE','depth':2,'hidden':128,'dropout':c['dropout'],'learning_rate':c['learning_rate'],'rank':16}
 source.Model=load_module(root,b['native']).Model;source.routes=load_module(root,b['routes']);source.factors=load_module(root,b['factors'])
 seed_tree=ast.parse(bound(root,b['driver']).read_bytes());seed=next(n for n in seed_tree.body if isinstance(n,ast.FunctionDef) and n.name=='set_seed')
 seed_env={'torch':torch,'np':np,'random':random};exec(compile(ast.Module(body=[seed],type_ignores=[]),'set_seed::source','exec'),seed_env);source.set_seed=seed_env['set_seed']
 train_tree=ast.parse(bound(root,b['base_train']).read_bytes());fn=next(n for n in train_tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_step')
 env={'F':torch.nn.functional};exec(compile(ast.Module(body=[fn],type_ignores=[]),'train_step::source','exec'),env);source.train_step=env['train_step']
 labels=torch.zeros(len(x),dtype=torch.long,device=device);mask=torch.zeros(len(x),dtype=torch.bool,device=device);labels[ids]=y;mask[ids]=True
 source.train_ids=ids;source.data=SimpleNamespace(x=x,edge_index=edges,y=labels,train_mask=mask)
 source.batch={'x':x,'edge_index':edges,'graph':SimpleNamespace(edge_index=edges)}
 source.synchronize=lambda:torch.cuda.synchronize(device) if source.device.type=='cuda' else None


def neighborhood_source(root,c,features,classes):
 path=bound(root,c['family_sources']['extension']);tree=ast.parse(path.read_bytes())
 nested=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Family')
 basepath=bound(root,c['wrapper_sources']['family']);_,base,changes=source_methods(basepath,features,classes)
 make=next(n for n in base if n.name=='make');make.name='construct_native'
 stop=next(i for i,n in enumerate(make.body) if ast.unparse(n)=='model = wrapper if members > 1 else body')
 make.body=make.body[:stop]+ast.parse('return body,wrapper').body
 env={'time':time,'ARMS':NEIGHBOR_ARMS,'descriptor':load_module(root,c['family_sources']['quartiles'])}
 compile_class('NativeBuild',[n for n in base if n.name in ('seeds','scope','construct_native')],env)
 nodes=[method(nested,n) for n in ('make','predictions','train_update','logits')];head_edits=[]
 class Heads(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==10:head_edits.append('correction_classes');return ast.Constant(classes)
   return n
 nodes[0]=Heads().visit(nodes[0])
 if len(head_edits)!=2:raise ValueError('Exactly two correction class provider sites')
 init=method(nested,'__init__');anchor=ast.parse('torch,edges,nodes=self.torch,self.data.edge_index,len(self.data.x)').body[0]
 start=next(i for i,n in enumerate(init.body) if ast.dump(n)==ast.dump(anchor))
 stop=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Expr) and ast.unparse(n.value).startswith('self.source_hashes.update'))
 prep=ast.parse('def prepare_support(self):\n    pass').body[0];prep.body=init.body[start:stop]
 nodes.append(prep);cls=compile_class('NeighborhoodFamily',nodes,env,('NativeBuild',))
 return cls(),{'provider_sites':changes+head_edits,'source_extension_SHA':c['family_sources']['extension']['sha256'],'whole_constructor_and_train_methods_extracted':True}


def feature_graph(root,c,features,nodes):
 path=bound(root,c['family_sources']['feature_graph']);tree=ast.parse(path.read_bytes());edits=[]
 class Provider(ast.NodeTransformer):
  def visit_Compare(self,n):
   self.generic_visit(n)
   if ast.unparse(n.left)=='raw_x.shape' and ast.literal_eval(n.comparators[0])==(11701,300):
    n.comparators[0]=ast.Tuple(elts=[ast.Constant(nodes),ast.Constant(features)],ctx=ast.Load());edits.append('support_expected_N_F')
   if ast.unparse(n.left)=='hidden.shape[0]' and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value==11701:
    n.comparators[0]=ast.Constant(nodes);edits.append('hidden_expected_N')
   return n
  def visit_BinOp(self,n):
   self.generic_visit(n)
   if ast.unparse(n)=='n * 300':
    n.right=ast.Constant(features);edits.append('projection_MAC_count_features')
   return n
  def visit_Call(self,n):
   self.generic_visit(n)
   if ast.unparse(n.func)=='torch.empty' and len(n.args)==2 and isinstance(n.args[0],ast.Constant) and n.args[0].value==300:
    n.args[0]=ast.Constant(features);edits.append('projection_features')
   if ast.unparse(n.func)=='math.sqrt' and len(n.args)==1 and isinstance(n.args[0],ast.Constant) and n.args[0].value==300:
    n.args[0]=ast.Constant(features);edits.append('projection_initialization_features')
   return n
 tree=ast.fix_missing_locations(Provider().visit(tree))
 if edits!=['support_expected_N_F','projection_features','projection_initialization_features','projection_initialization_features','projection_MAC_count_features','hidden_expected_N']:raise ValueError('Six exact support/projection/accounting provider anchors')
 env={'__file__':str(path),'__name__':'independent_sparse_feature_provider'};exec(compile(tree,str(path)+'::provider-only','exec'),env)
 return SimpleNamespace(**env),edits


def sparse_source(root,c,features,classes,nodes):
 family,methods,changes=source_methods(bound(root,c['family_sources']['family']),features,classes)
 graph,extra=feature_graph(root,c,features,nodes);env={'math':math}
 cls=compile_class('SparseFamily',methods,env);source=cls();source.feature_graph=graph
 fit=method(family,'fit');loop=next(n for n in ast.walk(fit) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
 if not isinstance(loop.body[0],ast.If) or ast.unparse(loop.body[0].test)!='bank':raise ValueError('Complete source sparse TRAIN branch required')
 fn=ast.parse('def source_train(self,model,optimizer,streams,members,bank):\n    torch=self.torch\n    return train_loss').body[0]
 fn.body=[fn.body[0],copy.deepcopy(loop.body[0]),fn.body[-1]];env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'sparse_TRAIN::source','exec'),env)
 source.source_train=fn=env['source_train'].__get__(source,cls)
 return source,{'provider_sites':changes+extra,'source_TRAIN_branch_AST_SHA':hashlib.sha256(ast.dump(loop.body[0]).encode()).hexdigest(),'whole_constructor_and_TRAIN_branch_extracted':True}


def make_adapter(*,torch,project_root,study,run,device):
 import numpy as np
 root=Path(project_root).resolve();c=run['config'];family=c['ported_family'];arm=c['family_arm']
 if study['dataset'] not in ('wikics','amazon-photo') or c['backbone']!='SAGE' or c['width']!=128 or c['depth']!=2 or c['weight_decay']!=0:
  raise ValueError('Only declared native SAGE new homogeneous provider/domain')
 if arm in ('ordinary_M1','factorized_plain_M1','original_gnnm'):
  return common_adapter(torch=torch,project_root=root,study=study,run=run,device=device)
 data={k:torch.load(bound(root,v),map_location='cpu',weights_only=True) for k,v in c['node_inputs'].items()}
 graph,train,select=[data[k] for k in ('graph','train','select')]
 if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'}:raise ValueError('Exact independent role authorities')
 x=graph['x'].to(device);edges=graph['edge_index'].to(device);ids=train['ids'].to(device);y=train['y'].to(device)
 if x.dtype!=torch.float32 or x.ndim!=2 or not torch.isfinite(x).all() or edges.dtype!=torch.long or edges.ndim!=2 or edges.shape[0]!=2:raise ValueError('Native full factual feature/COO shape')
 for r in (train,select):
  if r['ids'].dtype!=torch.long or r['y'].dtype!=torch.long or r['ids'].ndim!=1 or r['y'].shape!=r['ids'].shape or len(r['ids'])<1 or len(torch.unique(r['ids']))!=len(r['ids']):raise ValueError('Unique categorical role IDs')
 if torch.isin(train['ids'],select['ids']).any():raise ValueError('TRAIN/SELECT overlap')
 members=c['members'];offset=run['body'];kernel=None
 if family=='neighborhood_distribution':
  source,proof=neighborhood_source(root,c,x.shape[1],c['classes']);initialize(source,torch,root,c,x,edges,ids,y,device)
  # Body0 singleton and source I4-body0 have identical constructor fields.
  source.current={'arm':NEIGHBOR_ARMS[4] if arm==NEIGHBOR_ARMS[3] and offset>0 else arm}
  source.prepare_support();model,optimizer,streams=source.make(run['seed'],offset,'baseline',True,members)
  def unit():return source.train_update(model,optimizer,streams,members>1,members)
 elif family=='sparse_feature_kernel':
  source,proof=sparse_source(root,c,x.shape[1],c['classes'],len(x))
  payload=torch.load(bound(root,c['sparse_support']),map_location='cpu',weights_only=True)
  if payload['source_graph']!=c['node_inputs']['graph'] or payload['feature_graph_source']!=c['family_sources']['feature_graph'] or payload['provider_sites']!=proof['provider_sites'][3:]:raise ValueError('Exact prepared support/source/provider ancestry')
  edges=payload['edge_index'].to(device);initialize(source,torch,root,c,x,edges,ids,y,device)
  mode='private' if arm==SPARSE_ARMS[0] else 'fixed' if arm==SPARSE_ARMS[2] else 'common'
  model,optimizer,streams=source.make(run['seed'],offset,'baseline',arm!=SPARSE_ARMS[3],members,mode);kernel=model._feature_kernel
  def unit():return source.source_train(model,optimizer,streams,members,members>1)
 else:raise ValueError('Exact neighborhood or sparse-feature family')
 def logits_fn(current,training,step):return source.logits(current,streams,members>1,torch.arange(len(x),device=device))
 def custom_train(adapter,step):
  prior=copy.deepcopy(kernel.counts) if kernel is not None else {};loss=unit()
  extra={k:kernel.counts[k]-v for k,v in prior.items()} if kernel is not None else {}
  return {'objective':loss,'ce':loss,'regularization':0.,'operation_counts':{'optimizer_updates':1,'model_forwards':members,'route_predictions':members,
   'backward_calls':1,'autograd_grad_VJP_calls':0,'native_graph_block_calls':2*members,'supervised_examples':len(ids)*members,
   **{('feature_kernel_'+k):v for k,v in extra.items()}}}
 def capture():
  state=np.random.get_state()
  return {'streams':copy.deepcopy(streams),'numpy_rng':[state[0],state[1].tolist(),*state[2:]],
   'factor_members':{n:m.member for n,m in model.named_modules() if isinstance(m,source.factors.FactorLinear)},
   'training_modes':{n:m.training for n,m in model.named_modules()},'kernel_counts':copy.deepcopy(kernel.counts) if kernel is not None else {}}
 def restore(v):
  streams[:]=copy.deepcopy(v['streams']);state=v['numpy_rng'];np.random.set_state((state[0],np.array(state[1],dtype=np.uint32),*state[2:]));modules=dict(model.named_modules())
  for n,m in v['factor_members'].items():modules[n].member=m
  for n,m in v['training_modes'].items():modules[n].training=m
  if kernel is not None:kernel.counts.clear();kernel.counts.update(v['kernel_counts'])
 adapter=ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits_fn,train_ids=ids,train_y=y,
  select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda current:x.new_zeros(()),fusion='mean_probabilities',
  selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5001,'save_every':1},custom_train=custom_train,external_capture=capture,external_restore=restore,
  provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'provider_only_source_proof':proof,
   'actual_model_seed':source.seeds(run['seed'],offset)[0],'historical_outcomes_or_authorization_inherited':False})
 adapter.source_family=source;adapter.source_streams=streams
 return adapter
