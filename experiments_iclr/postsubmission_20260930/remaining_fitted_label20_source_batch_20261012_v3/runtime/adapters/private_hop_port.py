"""Exact private-hop V3 PubMed/PolyFormer source; no universal bridge claim."""
import ast
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace
import time

try:
 from .classifier import ClassifierAdapter
 from .native_label_families import bound,load_module
 from . import polyformer as p
 from . import polyformer_boundary as boundary
except ImportError:
 from adapters.classifier import ClassifierAdapter
 from adapters.native_label_families import bound,load_module
 from adapters import polyformer as p
 from adapters import polyformer_boundary as boundary


def source_environment(torch,root,c,nodes=19717):
 import numpy as np
 for v in c['hop_sources'].values():bound(root,v)
 p.SOURCE_FILES=c['polyformer_files'];boundary.BOUNDARY_SOURCE=c['boundary_source']
 plan_tree=ast.parse(bound(root,c['hop_sources']['plan']).read_bytes());plan={}
 definitions=[n for n in plan_tree.body if isinstance(n,(ast.Assign,ast.FunctionDef)) and (isinstance(n,ast.FunctionDef) and n.name=='condition'
   or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CONDITIONS','NATIVE_MONO','RULE') for t in n.targets))]
 exec(compile(ast.Module(body=definitions,type_ignores=[]),'private_hop_plan::source','exec'),plan)
 recipe=copy.deepcopy(plan['NATIVE_MONO']);recipe['dropout']=c['dropout'];recipe['lr']=c['learning_rate'];recipe['attn_lr']=.1*c['learning_rate']
 def construct(recipe,seed,device):
  with p._constructor_seed(torch,seed,device):return p.native_class(torch,root)(None,SimpleNamespace(**recipe)).to(device)
 factors_env=p.factor_definitions(torch,root);factors=SimpleNamespace(**factors_env)
 def preprocess(x,edges,K=2):return p.mono_tokens(torch,root,x,edges,K=K,preprocessing='pinned_mono_base')
 env={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'hashlib':hashlib,'random':random,'time':time,'SimpleNamespace':SimpleNamespace,
  'contextmanager':contextmanager,'seeded':lambda seed,dev:p._constructor_seed(torch,seed,dev),'construct':construct,
  'factors_module':lambda:factors,'native_preprocess':preprocess,'condition':plan['condition'],'NATIVE_MONO':recipe,'RULE':plan['RULE']}
 tree=ast.parse(bound(root,c['hop_sources']['session']).read_bytes());nodes_code=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
 changes=[]
 class Provider(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==19717:changes.append('node_shape');return ast.Constant(nodes)
   if isinstance(n.value,float) and n.value in (.0005,.005):changes.append('source_joint_LR');return ast.Constant(.1*c['learning_rate'] if n.value==.0005 else c['learning_rate'])
   return n
 nodes_code=[Provider().visit(n) for n in nodes_code];exec(compile(ast.fix_missing_locations(ast.Module(body=nodes_code,type_ignores=[])),'V3_Session::provider_shape_LR','exec'),env)
 hop_tree=ast.parse(bound(root,c['hop_sources']['hop']).read_bytes());hop_env=dict(env,ENABLED=True)
 hop_nodes=[copy.deepcopy(n) for n in hop_tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.Assign))]
 hop_nodes=[n for n in hop_nodes if not isinstance(n,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='ENABLED' for t in n.targets)]
 hop_nodes=[Provider().visit(n) for n in hop_nodes];exec(compile(ast.fix_missing_locations(ast.Module(body=hop_nodes,type_ignores=[])),'private_hop::independent_source','exec'),hop_env)
 return env,hop_env,recipe,{'provider_only_guard_and_LR_sites':changes,'source_factual_auxiliary_gather_commit_code_retained':True,
   'exact_PubMed_F500_C3_H256_23sites':True,'historical_disabled_flag_not_scientific_authorization':True}


def make_adapter(*,torch,project_root,study,run,device):
 import numpy as np
 root=Path(project_root).resolve();c=run['config'];arm=c['family_arm']
 if study['dataset']!='pubmed' or c['classes']!=3 or c['width']!=256 or c['depth']!=2 or c['backbone']!='PolyFormer_mono_K2':raise ValueError('Exact PubMed native PolyFormer K2 source geometry only')
 inputs={k:torch.load(bound(root,r),map_location='cpu',weights_only=True) for k,r in c['node_inputs'].items()};graph,train,select=[inputs[k] for k in ('graph','train','select')]
 if set(graph)!={'x','edge_index'} or any(set(v)!={'ids','y'} for v in (train,select)) or torch.isin(train['ids'],select['ids']).any():raise ValueError('Independent factual GRAPH/TRAIN/SELECT')
 if graph['x'].shape!=(19717,500):raise ValueError('Source body/factor guards require complete19717x500 PubMed')
 seed=run['seed']+1009*run['body'];env,hop_env,recipe,proof=source_environment(torch,root,c)
 ids=train['ids'].to(device);y=train['y'].to(device);x=graph['x'].to(device);edges=graph['edge_index'].to(device);session=None;credit=None
 if arm=='original_gnnm':
  tokens=p.mono_tokens(torch,root,graph['x'],graph['edge_index'],K=2,preprocessing='pinned_mono_base');tokens=[t.to(device) for t in tokens]
  native={k:recipe[k] for k in p.NATIVE_FIELDS};model=boundary.build_bank(torch,root,tokens,native=native,classes=3,dataset='pubmed',members=4,
   initialization_seed=seed,dropout_seeds=[seed+300001+1009*m for m in range(4)])
  optimizer=boundary.grouped_optimizer(torch,model,{'learning_rate':c['learning_rate'],'attn_lr':.1*c['learning_rate']})
  logits_fn=lambda current,training,step:current();fusion='mean_logits'
  def unit(adapter,step):return boundary.own_training(adapter,step)
  def capture():
   return {'boundary_runtime':model.capture_runtime_state(),'training_modes':{n:m.training for n,m in model.named_modules()}}
  def restore(v):
   model.restore_runtime_state(v['boundary_runtime']);modules=dict(model.named_modules())
   for n,mode in v['training_modes'].items():modules[n].training=mode
 else:
  name='shared4_own' if arm in ('private_missinghop','native_shared4_own') else 'single_native'
  session=env['Session'](name,seed,x,edges,ids,y,device=device)
  if arm=='factor1_native':hop_env['prepare_factor1'](session)
  if arm in ('private_missinghop','factor1_native'):credit=hop_env['HopCredit'](session,arm)
  model=session.bodies[0];optimizer=session.optimizers[0];fusion='mean_probabilities'
  def logits_fn(current,training,step):return torch.stack([session._forward(m,'factual')[0] for m in range(session.spec['members'])])
  def unit(adapter,step):
   before=copy.deepcopy(session.counters)
   if credit is not None:values=credit.step();own=sum(values['factual_ce'])/len(values['factual_ce'])
   else:values=session.train_step();own=values['scalar_objective']
   d={k:session.counters[k]-before[k] for k in before};paths=d['factual_forwards']+d['masked_forwards']
   return {'objective':own,'ce':own,'regularization':0.,'operation_counts':{'optimizer_updates':1,'model_forwards':paths,'route_predictions':session.spec['members'],
    'factual_fullgraph_forwards':d['factual_forwards'],'auxiliary_fullgraph_forwards':d['masked_forwards'],'backward_calls':0 if credit is not None else d['backwards'],
    'autograd_grad_VJP_calls':d['backwards'] if credit is not None else 0,'supervised_examples':len(ids)*paths}}
  def capture():
   nr=np.random.get_state()
   return {'streams':copy.deepcopy(session.streams.states),'session_counters':copy.deepcopy(session.counters),'credit_counters':copy.deepcopy(credit.counters) if credit else {},
    'numpy_rng':[nr[0],nr[1].tolist(),*nr[2:]],'factor_members':{n:m.member for n,m in model.named_modules() if isinstance(m,session.factors.FactorLinear)},
    'poly_attn_bias':{n:m.bias.detach().cpu().clone() for n,m in model.named_modules() if type(m).__name__=='PolyAttn'},'training_modes':{n:m.training for n,m in model.named_modules()}}
  def restore(v):
   session.streams.states=copy.deepcopy(v['streams']);session.counters.clear();session.counters.update(v['session_counters']);modules=dict(model.named_modules())
   if credit:credit.counters.clear();credit.counters.update(v['credit_counters']);credit._pending=None
   nr=v['numpy_rng'];np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),*nr[2:]))
   for n,m in v['factor_members'].items():modules[n].member=m
   for n,value in v['poly_attn_bias'].items():modules[n].bias=value.to(device).clone()
   for n,m in v['training_modes'].items():modules[n].training=m
 adapter=ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits_fn,train_ids=ids,train_y=y,select_ids=select['ids'].to(device),select_y=select['y'].to(device),
  regularizer=lambda current:x.new_zeros(()),fusion=fusion,selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5001,'save_every':1},custom_train=unit,external_capture=capture,external_restore=restore,
  provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'actual_model_seed':seed,'source_recipe':recipe,'source_bindings':proof,
   'source_native_optimizer':'Adam per-named-parameter attention/baseLR ratio.1,weightdecays1e-8/.001;boundaryGNNM actual sourceBEBlock with root zero-decay AdamW',
   'historical_alias_meanSAGE_corrected_to_actual_PolyFormer_K2':True,'historical_authorization_or_outcomes_inherited':False})
 adapter.source_session=session;adapter.source_credit=credit
 return adapter
