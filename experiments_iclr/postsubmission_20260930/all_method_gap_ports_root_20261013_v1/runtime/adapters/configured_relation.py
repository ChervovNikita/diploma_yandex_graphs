"""Copied-native relation-credit policies on the literal WikiCS source recipe.

Uses the sealed replay and permission partition. No independent-scorer
initializer, historical checkpoint or numerical outcome is used.
"""
import ast,copy,math
from functools import partial
from types import SimpleNamespace

POLICIES=('alphaF','allJ','phiJ','relationJ')
CONTRACT={'schema':'copied-native-relation-credit-source1100-v1',
 'operator':'native WikiPolynormer512/local7/global2/tiedqk;copied native local scorer rows before original Adam;all dense factorsunit',
 'objective':'two full TRAIN own-CE views;F/J=.5F+.5pooledprobabilityCE;two disjoint exhaustive per-member-view VJP groups,one old-state Adam',
 'policies':list(POLICIES),'training':'source1100/local100;own joint first pooled probability accuracy;local selected model+Adam restore with live end-local RNG;no early stop',
 'adaptations':['same originalGRAPH/TRAIN580, SELECT3505/CAL1769 replaces originalVALID5274','fresh current-role native controls;no historical bank or score reuse'],
 'auxiliary':False,'historical_state_inherited':False}

def make_adapter(*,torch,project_root,study,run,device):
 from run_study import load_factory
 c=run['config'];policy=c['configured_relation_policy']
 if (study['dataset']!='wikics' or c['configured_relation_source_contract']!=CONTRACT or policy not in POLICIES
     or c['initialization_seed']!=run['seed'] or run['body']!=0):
  raise ValueError('Exact copied-native WikiCS relation policy/seed contract required')
 constructor=load_factory(project_root,c['relation_base_adapter']+':construct')
 base=SimpleNamespace(**constructor.__globals__);B=c['mechanism_sources']
 for row in B.values():base.common.bound(project_root,row)
 graph,train,select=[torch.load(base.common.bound(project_root,c['node_inputs'][k]),map_location='cpu',weights_only=True) for k in ('graph','train','select')]
 if graph['x'].shape!=(11701,300) or len(train['ids'])!=580 or len(select['ids'])!=3505:
  raise ValueError('Complete literal GRAPH/TRAIN and adopted SELECT geometry required')
 if torch.isin(train['ids'],select['ids']).any():raise ValueError('Independent TRAIN/SELECT roles required')
 hook=base.common.load_module(project_root,B['attention'])
 def install(session):
  if hasattr(session,'optimizers') or session.model.members!=4 or session.model.independent:
   raise ValueError('Copied scorers must precede original fresh Adam')
  controller=hook.install_private_local_attention(session.model.models[0].body,4)
  original=session.model.member_forward
  session.model.member_forward=partial(controller.forward_member,original)
  session.private_local_attention=controller
 cls,recipe,binding=base.bound_public(project_root,B,features=300,classes=10,dropout=.5,learning_rate=.001,train_count=580,before_optimizer=install)
 session=cls('wikics','be_unit_contrastive',run['seed'],device,str(base.common.bound(project_root,B['native'])))
 session.config=copy.deepcopy(session.config);session.config['training']['epochs']=1100
 session.model.contrastive=False;session.config['contrastive'].update(alignment_weight=0.,residual_weight=0.)
 session.relation_partition,permissions=base.bound_permissions(project_root,B,session,policy)
 pool_path=base.common.bound(project_root,B['pool']);pool_tree=ast.parse(pool_path.read_text())
 pool_function=next(n for n in pool_tree.body if isinstance(n,ast.FunctionDef) and n.name=='served_pool_supervision')
 pool_namespace={'math':math}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(pool_function)],type_ignores=[])),str(pool_path)+':literal_pool','exec'),pool_namespace)
 session.relation_pool=pool_namespace['served_pool_supervision']
 session.relation_allocator_cap_bytes=None;session.last_relation_losses=None
 replay=base.bind_replay_count(project_root,B,session,580)
 factor=session.core['factors'].FactorLinear
 for m in session.model.modules():
  if isinstance(m,factor) and (not torch.equal(m.r,torch.ones_like(m.r)) or not torch.equal(m.s,torch.ones_like(m.s))):
   raise ValueError('Original dense factors must start at unit')
 for name,p in session.relation_partition['all']:
  if session.relation_partition['roles'][name]=='private_local_scorer' and any(not torch.equal(p[0],p[i]) for i in range(1,4)):
   raise ValueError('Private scorer rows must copy native initialization')
 x,edges=graph['x'].to(device),graph['edge_index'].to(device)
 adapter=base.attach(torch,session,{'constructor':binding,'permissions':permissions,'replay':replay,'source_recipe':session.config,
  'initialization':'copiednative;no independentinitializer called'},x=x,edges=edges,train_ids=train['ids'].to(device),train_y=train['y'].to(device),
  select_ids=select['ids'].to(device),select_y=select['y'].to(device),arm=policy,
  provenance={'run_id':run['run_id'],'source_contract':CONTRACT,'config':c,'source':study['source'],'data':study['data']},local_updates=100)
 return adapter
