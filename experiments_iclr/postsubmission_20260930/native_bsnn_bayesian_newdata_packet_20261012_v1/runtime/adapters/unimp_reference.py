"""Literal Torch UniMPv2, one65/35 TRAIN-label episode per source epoch.

The source operator remains exact Torch adaptation, with no claim of original
Paddle int64 mask semantics. Only compact TRAIN labels enter label context.
"""
import copy
import random
from . import common_native_wrapper as common
from .classifier import ClassifierAdapter
try:
 from ..engine import AdapterStepError
except ImportError:
 from engine import AdapterStepError

CONTRACT={'schema':'literal-Torch-UniMPv2-new-node-source-port-v1','canonical_method_id':'unimpv2_torch_reference',
 'source':'Pinned PGL-derived Torch model;3 virtual nodes;two300-hidden Transformer layers,final4head class layer;10attention APPNP passes',
 'declared_mask':'float_0p1_pre_softmax_explicit_adaptation;original Paddle int64 feed semantics unresolved',
 'source_training':'one in-place NumPy shuffle/epoch;floor(.65*TRAIN) visible labels;remaining35percent queried;one CE/backward/coupled Adam;1500 full epochs',
 'source_serving':'all and only TRAIN label context;SELECT/CAL/evaluation labels never enter model;factual rows only',
 'source_optimizer':'Adam betas.9/.999 eps1e-8,no clipping;source anchor LR.001/WD.0005',
 'HPO':'literal fixed source anchor LR.001/coupled WD.0005;no new HPO knobs;all source drop/mask settings unchanged',
 'provider_binding':'existing standardized factual features/edges and class schema;source vnode/edge/normalization construction exact;no hidden580TRAIN assumption',
 'selector':'strict-first full SELECT accuracy,every source epoch;checkpoint0 excluded;no early termination',
 'I4':'four own source trajectories;prospective actual init/dropout/episode seed=cohort+1009*body;M1body0 exactreference',
 'compatible_original_GNNM':'preserved original common/SAGE GNNM source controls on same tasks;no UniMP constructor equivalence claim'}

def construct(torch,root,bindings,*,x,edges,classes,seed,learning_rate,weight_decay,device):
 if learning_rate != .001 or weight_decay != .0005:
  raise ValueError('Literal fixed source Adam setting required')
 implementation=common.load_module(root,bindings['model'])
 torch.manual_seed(seed)
 if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
 topology=implementation.author_topology(torch,x,edges)
 model=implementation.build_model(torch,x.shape[1],classes,attention_mask_policy=implementation.ATTENTION_MASK_POLICY).to(device)
 model.members=1
 optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay,betas=(.9,.999),eps=1e-8)
 return model,optimizer,topology

def attach(torch,model,opt,topology,*,train_ids,train_y,select_ids,select_y,seed,provenance):
 import numpy as np
 if len(train_ids)<2:raise ValueError('Nonempty complementary source context/query required')
 masks=np.random.RandomState(seed+1900001);order=np.arange(len(train_ids),dtype=np.int64)
 visible_count=int(.65*len(train_ids))
 if not 0<visible_count<len(train_ids):raise ValueError('TRAIN count cannot realize literal65/35 episode')
 adapter=None;clock=0;last_episode=None
 def predictions(ids):return model(topology,train_ids,train_y)[ids].unsqueeze(0)
 def logits(owner,training,step):return predictions(torch.arange(topology.original_nodes,device=train_ids.device))
 def capture():
  rng=masks.get_state()
  return {'mask_rng':[rng[0],rng[1].tolist(),*rng[2:]],'mask_order':order.tolist(),'clock':clock,
   'last_episode':copy.deepcopy(last_episode),'visible_count':visible_count,
   'source_model':CONTRACT,'TRAIN_context_ids':train_ids.detach().cpu().clone()}
 def restore(value):
  nonlocal clock,last_episode
  if value['source_model']!=CONTRACT or value['visible_count']!=visible_count or not torch.equal(value['TRAIN_context_ids'],train_ids.cpu()):
   raise ValueError('Source context/operator episode ancestry changed')
  state=value['mask_rng'];masks.set_state((state[0],np.asarray(state[1],dtype=np.uint32),*state[2:]))
  order[:]=np.asarray(value['mask_order'],dtype=np.int64);clock=value['clock'];last_episode=copy.deepcopy(value['last_episode'])
 def train(owner,step):
  nonlocal clock,last_episode
  counts={'optimizer_updates':0,'optimizer_step_attempts':0,'model_forward_attempts':0,'backward_attempts':0}
  try:
   if step!=clock+1:raise ValueError('Source epoch skipped or repeated')
   masks.shuffle(order)
   visible=torch.from_numpy(order[:visible_count].copy()).to(train_ids.device)
   query=torch.from_numpy(order[visible_count:].copy()).to(train_ids.device)
   if torch.isin(train_ids[visible],train_ids[query]).any():raise ValueError('TRAIN query label entered its visible context')
   model.train();opt.zero_grad(set_to_none=True);counts['model_forward_attempts']=1
   raw=model(topology,train_ids[visible],train_y[visible])
   loss=torch.nn.functional.cross_entropy(raw[train_ids[query]],train_y[query])
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source complementary TRAIN CE')
   counts['backward_attempts']=1;loss.backward()
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
    raise FloatingPointError('Nonfinite source UniMP gradient')
   counts['optimizer_step_attempts']=1;opt.step();counts['optimizer_updates']=1
   clock=step;last_episode={'epoch':step,'visible_positions':visible.cpu().tolist(),'query_positions':query.cpu().tolist(),
    'visible_count':len(visible),'query_count':len(query),'query_context_disjoint':True,'SELECT_or_heldout_labels_in_context':False}
   counts.update(source_label_mask_episodes=1,visible_TRAIN_context_labels=len(visible),supervised_query_examples=len(query))
   return {'objective':float(loss.detach()),'ce':float(loss.detach()),'regularization':0.,'operation_counts':counts}
  except BaseException as error:
   counts['partial_optimizer_step_possible']=int(counts['optimizer_step_attempts']>counts['optimizer_updates'])
   raise AdapterStepError(str(error),operation_counts=counts) from error
 adapter=ClassifierAdapter(torch,torch.nn.functional,model,opt,logits_fn=logits,train_ids=train_ids,train_y=train_y,
  select_ids=select_ids,select_y=select_y,regularizer=lambda m:topology.x.new_zeros(()),fusion='mean_probabilities',
  selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':1501,'save_every':1},
  provenance={**provenance,'actual_model_seed':seed,'source_contract':CONTRACT},external_capture=capture,external_restore=restore,custom_train=train)
 adapter.predict_rows=lambda ids,step:predictions(ids.to(train_ids.device))
 def validate(step):
  model.eval()
  with torch.no_grad():
   raw=predictions(select_ids)[0]
   if not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite full SELECT logits')
   correct=int(raw.argmax(-1).eq(select_y).sum());nll=float(torch.nn.functional.cross_entropy(raw.double(),select_y))
  return {'objective':nll,'ce':nll,'regularization':0.,'correct':correct,'examples':len(select_y),
   'accuracy':correct/len(select_y),'pooled_nll':nll,'fusion':'mean_probabilities','selection_metric':correct/len(select_y)}
 adapter.validate=validate
 return adapter

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config']
 if study['dataset'] not in common.DATASETS or c['backbone']!='native_Torch_UniMPv2':
  raise ValueError('Declared new homogeneous node task/source body required')
 def read(row):return torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=(read(c['node_inputs'][key]) for key in ('graph','train','select'))
 if torch.isin(train['ids'],select['ids']).any():raise ValueError('TRAIN/SELECT overlap')
 actual_seed=c['initialization_seed']
 if actual_seed!=run['seed']+1009*run['body']:raise ValueError('Explicit UniMP source seed ancestry changed')
 x,edges=graph['x'].to(device),graph['edge_index'].to(device)
 model,opt,topology=construct(torch,project_root,c['unimp_sources'],x=x,edges=edges,classes=c['classes'],seed=actual_seed,
  learning_rate=c['learning_rate'],weight_decay=c['weight_decay'],device=device)
 return attach(torch,model,opt,topology,train_ids=train['ids'].to(device),train_y=train['y'].to(device),
  select_ids=select['ids'].to(device),select_y=select['y'].to(device),seed=actual_seed,
  provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c})
