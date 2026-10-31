"""Literal public WikiPolynormer unit losses and private relation-credit replay.

Original modules stay unchanged. Fresh constructor namespaces bind only new
provider dimensions, finite HPO, member seed ancestry, and TRAIN cardinality.
"""
import ast,copy,hashlib,json,sys
from functools import partial
from types import SimpleNamespace
from . import common_native_wrapper as common
from .classifier import ClassifierAdapter

PUBLIC='portable_internal_be_public_interface_20261007_v2'
PATHS={'public':PUBLIC+'/portable.py','native':'amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py',
 'unit':'internal_BE_WikiCS_unit_mechanism_ablation_source_20261007_v1/run_cell.py',
 'attention':'portable_private_local_attention_20261008_v1/private_local_attention.py',
 'initializer':'graph_relation_independent_native_local_scorer_source_20261010_v1/initializer.py',
 'permissions':'graph_relation_private_credit_source_20261008_v2/permissions.py',
 'relation_replay':'graph_relation_private_credit_source_20261008_v2/replay_adapter.py',
 'public_replay':'portable_context_steering_public_interface_20261008_v1/context_recompute.py',
 'pool':'public_internal_be_private_steering_adapter_20261007_v1/adapter.py',
 'profile':'public_internal_be_private_steering_adapter_20261007_v1/BACKBONE_AUDIT.json'}
ARMS=('native_M1','unit_shared4','alignment_only','residual_only','independent_scorer_alphaF','independent_scorer_relationJ')
SEEDS=(88101,88103,88201,88203,88207)
CONTRACT={'schema':'retained-WikiPolynormer-mechanisms-source-port-v1','source_exact_recipe':False,
 'native':'literal portable WikiBackbone native CPU construction/reset;7 local/2 global/H512/heads1; native tied q=k global attention; allmap unit dense factors; private dropout seed+1009*member+300001',
 'unit':'literal two full own CE views; source .05 alignment-only or residual-only facade, temperature.2,512 evenly spaced TRAIN auxiliary objects; inactive original loss differentiable zero',
 'private_credit':'14 local native scorer banks initialized before original Adam by CPU independent native Glorot row formula100000*seed+770000000+10000*layer+100*side+member; nonscorer/default RNG unchanged; alphaF or relationJ; no embedding auxiliary; two output F/J collections and16 disjoint member-view VJPs then one native Adam; inactive gradients remain None',
 'provider_binding':'WikiCS20/Photo10 homogeneous feature/class dimensions and complete TRAIN cardinality only; source task dispatch remains wikics because it denotes the native node architecture',
 'HPO':'nine finite LR/dropout neighbors around.001/.5; dropout binds native input/local/global fields together, architecture7/2/H512/heads1 fixed; exact source anchor retained',
 'controller':'full5000 with local100/global4900 adaptation; source strict-first pooled accuracy after every update; local selected model+Adam handoff and live end-local RNG/steps retained; selected serving may be local or global',
 'I4':'four genuine native M1 trajectories with actual initialization seed=cohort_seed+1009*body; M1 body0 exact prospective reference; full component HPO retained',
 'original_GNNM':'required common-v2 literal original R/S/B baseline on these same new tasks, exact reuse candidates root-owned; no source-source constructor equivalence asserted'}

def compiled_function(module,path,names,extra=None,transform=None):
 tree=ast.parse(path.read_text());nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert len(nodes)==len(names)
 if transform:nodes=[transform(n) for n in nodes]
 code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));space=dict(vars(module),**(extra or {}));exec(compile(code,'<explicit_prospective_source_binding>','exec'),space)
 return space,common.ast_hash(code)

def bound_public(root,bindings,*,features,classes,dropout,learning_rate,train_count,before_optimizer=None):
 public=common.load_module(root,bindings['public']);recipe=copy.deepcopy(public.recipe('wikics'))
 recipe['model'].update(in_channels=features,out_channels=classes,dropout=dropout,in_dropout=dropout,global_dropout=dropout)
 recipe['training'].update(lr=learning_rate,epochs=5000,local_epochs=100,batch_size=train_count)
 def bound_recipe(task):
  if task!='wikics':raise ValueError('Exact native node task constructor required')
  return copy.deepcopy(recipe)
 tree=ast.parse(common.bound(root,bindings['public']).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session');init=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
 positions=[i for i,n in enumerate(init.body) if isinstance(n,ast.FunctionDef) and n.name=='optimizer'];assert len(positions)==1
 if before_optimizer is not None:init.body.insert(positions[0],ast.parse('_before_optimizer(self)').body[0])
 code=ast.fix_missing_locations(ast.Module(body=[init],type_ignores=[]));space=dict(vars(public),recipe=bound_recipe,_before_optimizer=before_optimizer)
 exec(compile(code,'<source_Session_provider_HPO_namespace>','exec'),space)
 cls=type('ProspectiveSourceSession',(public.Session,),{'__init__':space['__init__']})
 return cls,recipe,{'portable_source_sha256':bindings['public']['sha256'],'constructor_AST_sha256':common.ast_hash(code),'bindings':['features','classes','native_input_local_global_dropout','native_Adam_LR','complete_TRAIN_count','local100_global4900'],'before_original_Adam_hook':before_optimizer is not None}

def bound_permissions(root,bindings,session,policy):
 original=common.load_module(root,bindings['permissions']);profile=json.loads(common.bound(root,bindings['profile']).read_text());spec=profile['tasks']['wikics'];spec['model_config']=copy.deepcopy(session.config['model'])
 spec['factor_maps']['models.0.body.lin_in']['weight_shape']=[512,session.config['model']['in_channels']]
 for leaf in ('pred_local','pred_global'):spec['factor_maps']['models.0.body.'+leaf]['weight_shape']=[session.config['model']['out_channels'],512]
 def adapt(node):
  edits=0
  for n in ast.walk(node):
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='profile' for t in n.targets):
    assert ast.unparse(n.value)=='json.loads(profile_path.read_text())';n.value=ast.parse('copy.deepcopy(profile_path)').body[0].value;edits+=1
   if isinstance(n,ast.Compare) and ast.unparse(n) == "group['lr'] == 0.001":n.comparators[0]=ast.parse("session.config['training']['lr']").body[0].value;edits+=1
  assert edits==2;return node
 space,digest=compiled_function(original,common.bound(root,bindings['permissions']),['partition'],{'copy':copy},adapt)
 return space['partition'](session,profile,policy),{'source_sha256':bindings['permissions']['sha256'],'bound_AST_sha256':digest,'profile_source_sha256':bindings['profile']['sha256'],'provider_profile':profile,'changed_fields':'model input/classes/dropout, stem and both heads shapes, native HPO LR assertion; roles/counts unchanged'}

def bind_replay_count(root,bindings,session,train_count):
 permissions=common.load_module(root,bindings['permissions']);prior=sys.modules.get('permissions');sys.modules['permissions']=permissions
 try:relation=common.load_module(root,bindings['relation_replay'])
 finally:
  if prior is None:sys.modules.pop('permissions',None)
  else:sys.modules['permissions']=prior
 original=common.load_module(root,bindings['public_replay'])
 def count_bind(function):
  found=0
  for n in ast.walk(function):
   if isinstance(n,ast.Compare) and ast.unparse(n)=='len(labels) != 580':n.comparators[0].value=train_count;found+=1
  assert found==1;return function
 def adapt(node):
  if node.name=='compiled_replay':
   positions=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='function' for t in n.targets) and ast.unparse(n.value)=='copy.deepcopy(functions[0])'];assert len(positions)==1
   node.body.insert(positions[0]+1,ast.parse('function = _bind_TRAIN_count(function)').body[0])
  return node
 space,digest=compiled_function(relation,common.bound(root,bindings['relation_replay']),['compiled_replay','install'],{'_bind_TRAIN_count':count_bind},adapt)
 space['install'](session,original,common.bound(root,bindings['public_replay']))
 return {'source_sha256':bindings['relation_replay']['sha256'],'provider_replay_AST_sha256':session.relation_replay_AST_sha256,'binding_driver_AST_sha256':digest,'TRAIN_count580_bound':train_count}

def construct(torch,root,bindings,*,arm,features,classes,seed,body,dropout,learning_rate,train_count,device):
 if arm not in ARMS or dropout not in (.25,.5,.75) or learning_rate not in (.0005,.001,.002):raise ValueError('Explicit source arm/HPO required')
 actual_seed=seed+1009*body;private=arm.startswith('independent_scorer_')
 hook=common.load_module(root,bindings['attention']) if private else None
 init=common.load_module(root,bindings['initializer']) if private else None
 if private:
  # Only prospective integer seed roster changes; row formula and draw law are literal.
  namespace,init_digest=compiled_function(init,common.bound(root,bindings['initializer']),['row_seed','initialize'],{'SEEDS':tuple(SEEDS)})
  def install(session):
   assert not hasattr(session,'optimizers') and session.model.members==4 and not session.model.independent
   controller=hook.install_private_local_attention(session.model.models[0].body,4)
   controller.independent_native_start=namespace['initialize'](torch,controller,actual_seed)
   original=session.model.member_forward;session.model.member_forward=partial(controller.forward_member,original)
   session.private_local_attention=controller;session.independent_native_start=controller.independent_native_start
 else:install=None
 cls,recipe,binding=bound_public(root,bindings,features=features,classes=classes,dropout=dropout,learning_rate=learning_rate,train_count=train_count,before_optimizer=install)
 underlying='single' if arm=='native_M1' else 'be_unit' if arm=='unit_shared4' else 'be_unit_contrastive'
 session=cls('wikics',underlying,actual_seed,device,str(common.bound(root,bindings['native'])))
 details={'constructor':binding,'actual_initialization_seed':actual_seed,'source_recipe':recipe}
 if arm in ('alignment_only','residual_only'):
  unit=common.load_module(root,bindings['unit']);facade=unit.ObjectivesFacade(session.core['objectives'],arm)
  session.core=dict(session.core,objectives=facade);session.config['contrastive'].update(alignment_weight=.05 if arm=='alignment_only' else 0.,residual_weight=.05 if arm=='residual_only' else 0.)
  session.unit_facade=facade
 if private:
  session.model.contrastive=False;session.config['contrastive'].update(alignment_weight=0.,residual_weight=0.)
  session.relation_partition,permission_binding=bound_permissions(root,bindings,session,arm.removeprefix('independent_scorer_'))
  pool_path=common.bound(root,bindings['pool']);pool_tree=ast.parse(pool_path.read_text());pool_fn=next(n for n in pool_tree.body if isinstance(n,ast.FunctionDef) and n.name=='served_pool_supervision')
  pool_space={'math':__import__('math')};exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(pool_fn)],type_ignores=[])),'<literal_source_pool_function>','exec'),pool_space);session.relation_pool=pool_space['served_pool_supervision']
  details.update(permissions=permission_binding,replay=bind_replay_count(root,bindings,session,train_count),independent_initializer_AST_sha256=init_digest,independent_native_start=init.stable_receipt(session.independent_native_start))
 assert len(session.optimizers)==1 and not session.optimizers[0].state
 return session,details

class StagedAdapter(ClassifierAdapter):
 def observe_validation(self,step,metrics):
  decision=super().observe_validation(step,metrics)
  if step<=self.local_updates and (self.local_best is None or metrics['selection_metric']>self.local_best):
   self.local_best=float(metrics['selection_metric']);self.local_state=self.session._cpu_tree({'model':self.model.state_dict(),'optimizers':[self.optimizer.state_dict()]})
  return decision
 def train_step(self,step):
  if step==self.local_updates+1:
   if self.local_state is None:raise ValueError('Source local SELECT state missing')
   streams=copy.deepcopy(self.session.streams);live_steps=self.session.steps
   self.session.core['selection'].local_transition(self.model,[self.optimizer],lambda name:self.local_state,False)
   assert self.session.steps==live_steps and all(self.t.equal(a['cpu'],b['cpu']) for a,b in zip(streams,self.session.streams))
   self.transition_completed=True
  return super().train_step(step)

def attach(torch,session,details,*,x,edges,train_ids,train_y,select_ids,select_y,arm,provenance,local_updates=100):
 adapter=None
 def predictions(ids):return session.forward({'x':x,'edge_index':edges,'ids':ids})[0]
 def logits(model,training,step):return predictions(torch.arange(len(x),device=x.device))
 def train(adapter,step):
  before=copy.deepcopy(getattr(session,'execution_totals',{}));value=session.train_step({'x':x,'edge_index':edges,'ids':train_ids},train_y);members=session.model.members
  counts={'optimizer_updates':1,'fullgraph_native_member_trajectories':members*2,'backward_calls':1,'source_own_CE_views':2,'supervised_examples':len(train_y)*members*2}
  if before:counts.update({k:session.execution_totals[k]-v for k,v in before.items()});counts['fullgraph_native_member_trajectories']=counts['shadow_member_forwards']+counts['replay_member_forwards'];counts['backward_calls']=0
  return {'objective':float(value['loss']),'ce':float(value['own_mean']),'regularization':float(value['auxiliary']),'operation_counts':counts}
 def capture():
  value={'streams':copy.deepcopy(session.streams),'source_steps':session.steps,'global':bool(session.model.models[0].body._global),'local_best':adapter.local_best,'local_state':copy.deepcopy(adapter.local_state),'transition_completed':adapter.transition_completed,'numpy_rng':copy.deepcopy(session.np.random.get_state()),'source_work':copy.deepcopy(getattr(session,'execution_totals',{})),'source_replay_diagnostics':copy.deepcopy(getattr(session,'last_replay_diagnostics',None))}
  if hasattr(session,'unit_facade'):value['unit_loss_calls']=[session.unit_facade.alignment_source_calls,session.unit_facade.residual_source_calls]
  return value
 def restore(value):
  session.streams=copy.deepcopy(value['streams']);session.steps=value['source_steps'];session.model.set_global(value['global']);adapter.local_best=value['local_best'];adapter.local_state=copy.deepcopy(value['local_state']);adapter.transition_completed=value['transition_completed'];session.np.random.set_state(value['numpy_rng'])
  if value['source_work']:session.execution_totals=copy.deepcopy(value['source_work']);session.last_replay_diagnostics=copy.deepcopy(value['source_replay_diagnostics'])
  if 'unit_loss_calls' in value:session.unit_facade.alignment_source_calls,session.unit_facade.residual_source_calls=value['unit_loss_calls']
 adapter=StagedAdapter(torch,torch.nn.functional,session.model,session.optimizers[0],logits_fn=logits,train_ids=train_ids,train_y=train_y,select_ids=select_ids,select_y=select_y,regularizer=lambda m:x.new_zeros(()),fusion='mean_probabilities',selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5000,'save_every':1},provenance={**provenance,'source_contract':CONTRACT,'bindings':details},external_capture=capture,external_restore=restore,custom_train=train)
 adapter.session=session;adapter.local_updates=local_updates;adapter.local_state=None;adapter.local_best=None;adapter.transition_completed=False;adapter.predict_rows=lambda ids,step:predictions(ids.to(x.device));return adapter

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config'];assert study['dataset'] in common.DATASETS and c['backbone']=='native_WikiPolynormer'
 def read(row):return torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=(read(c['node_inputs'][k]) for k in ('graph','train','select'));x,edges=graph['x'].to(device),graph['edge_index'].to(device)
 assert tuple(x.shape)==(c['provider_graph']['nodes'],c['provider_graph']['features']) and tuple(edges.shape)==(2,c['provider_graph']['edges'])
 assert not torch.isin(train['ids'],select['ids']).any()
 session,details=construct(torch,project_root,c['mechanism_sources'],arm=c['mechanism_arm'],features=x.shape[1],classes=c['classes'],seed=run['seed'],body=run['body'],dropout=c['dropout'],learning_rate=c['learning_rate'],train_count=len(train['ids']),device=device)
 return attach(torch,session,details,x=x,edges=edges,train_ids=train['ids'].to(device),train_y=train['y'].to(device),select_ids=select['ids'].to(device),select_y=select['y'].to(device),arm=c['mechanism_arm'],provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c},local_updates=c['local_transition_updates'])
