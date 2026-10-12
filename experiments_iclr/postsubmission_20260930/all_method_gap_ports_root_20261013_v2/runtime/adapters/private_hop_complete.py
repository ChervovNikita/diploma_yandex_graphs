"""Literal six-condition private-hop update with source2000/p250 selection."""
import copy
from types import SimpleNamespace
CONDITIONS=('shared4_own','private_missinghop','full_aux','common_nonfull','allblock_missinghop','factor1_allview','native_M1','original_GNNM')
CONTRACT={'schema':'complete-private-hop-six-condition-source2000-v1','native':'PubMed19717/F500/C3/PolyFormer-mono-K2/H256/L2;23unit-factor sites;groupedAdam .005/.0005',
 'conditions':list(CONDITIONS),'source_loop':'2000cap/strictfirst pooled accuracy/everyepoch/no0/patience250',
 'credit':'unchanged HopCredit gather/commit;lambda.5;factualcommon gradients;four exact auxiliary/private/all-block controls;no collapsed substitute',
 'adaptations':['current disjoint PubMed TRAIN/SELECT/CAL/evaluate custody','freshnative/genuineI4/originalboundary andsame-information allviewI4 additions'],
 'historical_state_inherited':False}

def make_adapter(*,torch,project_root,study,run,device):
 from run_study import load_factory
 c=run['config'];arm=c['complete_hop_arm']
 if arm not in CONDITIONS or c['complete_hop_source_contract']!=CONTRACT:raise ValueError('Literal complete sourcehop condition required')
 source=load_factory(project_root,c['hop_base_adapter']+':source_environment');base=SimpleNamespace(**source.__globals__)
 # Preserve original boundary/native implementation; complete HopCredit
 # conditions below use the same source environment and tensor authorities.
 if arm=='original_GNNM':
  actual=copy.deepcopy(run);actual['config']['family_arm']='original_gnnm'
  adapter=base.make_adapter(torch=torch,project_root=project_root,study=study,run=actual,device=device)
  adapter.selector['patience_steps']=250
  return adapter
 inputs={k:torch.load(base.bound(project_root,r),map_location='cpu',weights_only=True) for k,r in c['node_inputs'].items()}
 graph,train,select=[inputs[k] for k in ('graph','train','select')]
 if study['dataset']!='pubmed' or graph['x'].shape!=(19717,500) or torch.isin(train['ids'],select['ids']).any():raise ValueError('Complete source PubMed and disjoint roles required')
 seed=c['initialization_seed'];env,hop,recipe,proof=source(torch,project_root,c)
 x,edges=graph['x'].to(device),graph['edge_index'].to(device);ids,y=train['ids'].to(device),train['y'].to(device)
 members=1 if arm in ('native_M1','factor1_allview') else 4
 session=env['Session']('single_native' if members==1 else 'shared4_own',seed,x,edges,ids,y,device=device)
 credit=None
 if arm=='factor1_allview':hop['prepare_factor1'](session)
 if arm in hop['M4']+hop['M1']:credit=hop['HopCredit'](session,arm)
 model,optimizer=session.bodies[0],session.optimizers[0]
 def logits(current,training,step):return torch.stack([session._forward(m,'factual')[0] for m in range(members)])
 def update(adapter,step):
  before=copy.deepcopy(session.counters)
  if credit is None:v=session.train_step();loss=v['scalar_objective']
  else:v=credit.step();loss=sum(v['factual_ce'])/len(v['factual_ce'])
  work={k:session.counters[k]-before[k] for k in before};paths=work['factual_forwards']+work['masked_forwards']
  return {'objective':loss,'ce':loss,'regularization':0.,'operation_counts':{'optimizer_updates':1,'model_forwards':paths,'route_predictions':members,
   'factual_fullgraph_forwards':work['factual_forwards'],'auxiliary_fullgraph_forwards':work['masked_forwards'],
   'backward_calls':work['backwards'] if credit is None else 0,'autograd_grad_VJP_calls':work['backwards'] if credit is not None else 0,
   'supervised_examples':len(ids)*paths}}
 def capture():
  import numpy as np
  nr=np.random.get_state()
  return {'streams':copy.deepcopy(session.streams.states),'source_counters':copy.deepcopy(session.counters),
   'negative_generators':[g.getstate() for g in session.negative_generators],
   'credit_counters':copy.deepcopy(credit.counters) if credit else None,
   'factor_members':{n:m.member for n,m in model.named_modules() if isinstance(m,session.factors.FactorLinear)},
   'plain_PolyAttn_bias':{n:m.bias.detach().cpu().clone() for n,m in model.named_modules() if type(m).__name__=='PolyAttn'},
   'numpy':[nr[0],nr[1].tolist(),*nr[2:]],'module_modes':{n:m.training for n,m in model.named_modules()}}
 def restore(v):
  import numpy as np
  session.streams.states=copy.deepcopy(v['streams']);session.counters.clear();session.counters.update(v['source_counters'])
  for g,state in zip(session.negative_generators,v['negative_generators']):g.setstate(state)
  if credit is not None:credit.counters.clear();credit.counters.update(v['credit_counters']);credit._pending=None
  modules=dict(model.named_modules())
  for n,value in v['factor_members'].items():modules[n].member=value
  for n,value in v['plain_PolyAttn_bias'].items():modules[n].bias=value.to(device).clone()
  for n,value in v['module_modes'].items():modules[n].training=value
  nr=v['numpy'];np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),*nr[2:]))
 adapter=base.ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits,train_ids=ids,train_y=y,select_ids=select['ids'].to(device),select_y=select['y'].to(device),
  regularizer=lambda current:x.new_zeros(()),fusion='mean_probabilities',selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':250,'save_every':1},
  custom_train=update,external_capture=capture,external_restore=restore,provenance={'config':c,'source_contract':CONTRACT,'source_recipe':recipe,'binding':proof,'actual_seed':seed})
 adapter.source_session=session;adapter.source_credit=credit
 return adapter
