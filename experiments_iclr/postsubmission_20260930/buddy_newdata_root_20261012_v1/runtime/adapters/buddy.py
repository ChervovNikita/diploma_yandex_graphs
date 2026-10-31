"""Native BUDDY sign_k=0 and shared-map ensembles on one training-only cache.

No scientific fit occurs at import. Genuine I4 is four separate adapter owners.
The original GNNM boundary adaptation is explicit for BUDDY's two input streams.
"""
import ast
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from . import common_native_wrapper as common
from . import link as bridge

ARMS=('native1024','single256','factorized4','matched_single','original_gnnm')
CONTRACT={
 'schema':'native-BUDDY-training-cache-new-data-port-v1',
 'native':'Pinned author BUDDY class, sign_k0, all native nonlinearities and BatchNorm; training-only weighted collab topology, unchanged HLL/MinHash source; no trainable node embeddings, RA, SIGN propagation or validation-edge augmentation.',
 'candidate':'Exact retained FactorizedBUDDY; all four learned linear maps shared with private trainable R/S, R CPU signs/S ones, private BN affines and running buffers. Endpoint products occur within routes; final raw logits averaged.',
 'original_gnnm':'Exact pinned BEBlock at both input affines label_lin_layer and lin_feat and final lin; each first R has source Rademacher initialization, terminal R/S ones, B zero, W Xavier. Native lin_out and all other BUDDY operations/buffers shared. Four sequential own BCE/4 backwards then one Adam; shared sequential dropout/BN stream. Explicit multi-input graph-task adaptation, not original paper rerun.',
 'controls':'Native1024 source anchor, native256 M1, fresh separately selected native256 I4, label-free nearest parameter-matched native width; body0 is the exact M1 fit reference.',
 'optimizer':'Native Adam .02, defaults betas(.9,.999), epsilon1e-8, WD0; source100 full epochs, batch1024 fixed one training negative per positive, native shuffle including tail.',
 'HPO':'LR(.005,.02,.08) x joint feature/label dropout multiplier(.5,1,2), around source(.05,.1). No unseen method-specific knobs.',
 'selection':'Official SELECT Hits50 strict first, every full epoch, mean raw logits; SELECT BCE sampled positive/100000 negative pool diagnostic only. No CAL role or SELECT reuse.',
 'provenance':'Source implementation/recipe baseline with deliberate training-only support at evaluation; not a claim to reproduce published metrics under full author test-time graph policy. New seeds, source-cache construction costs preserved. No old predictions/states/locks reused.',
 'resource':'Unmeasured full-input optimizer/SELECT/concurrency/IO; disabled until resource admission and explicit scientific GO.'}


@contextmanager
def aliases(values):
 import sys
 old={k:sys.modules.get(k) for k in values};sys.modules.update(values)
 try:yield
 finally:
  for k,v in old.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v


def modules(root,bindings):
 values={}
 for name in ('guards','models','cache_builder'):
  with aliases(values):values[name]=common.load_module(root,bindings[name])
 return values


def original_model(torch,root,source,native,device):
 tree=ast.parse(common.bound(root,source).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BEBlock')
 code=ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]));ns={'torch':torch,'nn':torch.nn}
 exec(compile(code,'<pinned_original_BEBlock>','exec'),ns);BE=ns['BEBlock']
 class Route(torch.nn.Module):
  def __init__(self,layer,first):
   super().__init__();self.block=BE(layer.in_features,layer.out_features,first,device,4);self.route=0
  def forward(self,x):return self.block(x,self.route)
 class Boundary(torch.nn.Module):
  def __init__(self):
   super().__init__();self.native=native;self.members=4
   for name,first in (('label_lin_layer',True),('lin_feat',True),('lin',False)):
    setattr(self.native,name,Route(getattr(self.native,name),first))
  def one(self,inputs,member):
   for name in ('label_lin_layer','lin_feat','lin'):getattr(self.native,name).route=member
   return self.native(*inputs).reshape(-1)
  def forward(self,*inputs):return torch.stack([self.one(inputs,m) for m in range(4)],1)
 return Boundary(),common.ast_hash(code)


def construct(torch,root,config,seed,body,device):
 arm=config['buddy_arm']
 if arm not in ARMS:raise ValueError('Explicit BUDDY arm required')
 mods=modules(root,config['buddy_sources']);native_config=copy.deepcopy(config['native_config']);native_config.update(lr=config['learning_rate'],feature_dropout=.05*config['dropout_multiplier'],label_dropout=.1*config['dropout_multiplier'])
 actual=seed+10000*body;torch.manual_seed(actual)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(actual)
 if arm=='original_gnnm':model,digest=original_model(torch,root,config['original_BE_source'],mods['models'].single(native_config,256),device)
 else:model=mods['models'].make_model(native_config,arm,actual);digest=None
 model=model.to(device);optimizer=torch.optim.Adam(model.parameters(),lr=native_config['lr'],weight_decay=0.)
 return model,optimizer,mods,{'actual_initialization_seed':actual,'factor_seed':actual+90000 if arm=='factorized4' else None,'original_BE_AST_sha256':digest,'source_contract':CONTRACT}


def batch_rows(torch,length,size,generator=None):
 indices=torch.randperm(length,generator=generator) if generator is not None else torch.arange(length)
 stop=length-1 if length%size==1 and length>size else length
 for start in range(0,stop,size):
  end=min(start+size,stop)
  if end==stop and stop<length:end=length
  yield indices[start:end]


class BUDDYAdapter:
 def __init__(self,torch,numpy,model,optimizer,mods,details,*,root,study,run,common_input,train,select):
  self.t,self.np,self.model,self.optimizer=torch,numpy,model,optimizer;self.mods=mods;self.details=details
  self.root=Path(root);self.study,self.run,self.config=study,run,run['config'];self.device=next(model.parameters()).device
  self.common,self.train,self.select=common_input,train,select;self._rng=bridge.rng_state(torch,numpy);self._epoch=0;self._best=None;self._best_step=None
  self.provenance={'run_id':run['run_id'],'source':study['source'],'data':study['data'],'config':self.config,'bindings':details}
 def _inputs(self,data,rows):
  links=data['links'][rows];degrees=self.common['degrees'][links]
  return (data['sf'][rows].to(self.device),self.common['x'][links].to(self.device),degrees[:,0].to(self.device),degrees[:,1].to(self.device))
 @contextmanager
 def _scope(self,training):
  outer=bridge.rng_state(self.t,self.np);bridge.restore_rng(self.t,self.np,self._rng)
  try:yield
  finally:
   if training:self._rng=bridge.rng_state(self.t,self.np)
   bridge.restore_rng(self.t,self.np,outer)
 def train_step(self,epoch):
  from engine import AdapterStepError
  if epoch!=self._epoch+1:raise ValueError('Complete native epoch order required')
  self.model.train();counts={'optimizer_updates':0,'backward_calls':0,'TRAIN_forward_calls':0,'records':0};total=0.
  try:
   with self._scope(True):
    seed=self.details['actual_initialization_seed'];gen=self.t.Generator().manual_seed(seed+100000*epoch);self.t.manual_seed(seed+1000000+epoch)
    if self.device.type=='cuda':self.t.cuda.manual_seed_all(seed+1000000+epoch)
    for rows in batch_rows(self.t,len(self.train['links']),1024,gen):
     inputs=self._inputs(self.train,rows);labels=(rows<self.train['n_positive']).float().to(self.device);self.optimizer.zero_grad(set_to_none=True)
     if self.config['buddy_arm']=='original_gnnm':
      value=0.
      for member in range(4):
       logits=self.model.one(inputs,member);loss=self.t.nn.functional.binary_cross_entropy_with_logits(logits,labels)/4
       if not self.t.isfinite(loss):raise FloatingPointError('Nonfinite own-route BUDDY BCE')
       loss.backward();value+=float(loss.detach());counts['backward_calls']+=1;counts['TRAIN_forward_calls']+=1
     else:
      logits=self.model(*inputs).reshape(len(rows),-1);loss=self.t.nn.functional.binary_cross_entropy_with_logits(logits,labels[:,None].expand_as(logits))
      if not self.t.isfinite(loss):raise FloatingPointError('Nonfinite BUDDY BCE')
      loss.backward();value=float(loss.detach());counts['backward_calls']+=1;counts['TRAIN_forward_calls']+=1
     if any(p.grad is not None and not self.t.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite BUDDY gradient')
     self.optimizer.step();counts['optimizer_updates']+=1;counts['records']+=len(rows);total+=value*len(rows)
   if counts['optimizer_updates']!=self.config['train_minibatches_per_epoch'] or counts['records']!=len(self.train['links']):raise ValueError('Native complete batch/tail law changed')
   self._epoch=epoch;value=total/counts['records'];return {'objective':value,'ce':value,'regularization':0.,'operation_counts':counts}
  except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error
 def predict_cache(self,data):
  self.model.eval();parts=[]
  with self._scope(False),self.t.no_grad():
   for rows in batch_rows(self.t,len(data['links']),65536):
    z=self.model(*self._inputs(data,rows)).reshape(len(rows),-1)
    if not self.t.isfinite(z).all():raise FloatingPointError('Nonfinite raw BUDDY bank')
    parts.append(z.T.cpu())
  return self.t.cat(parts,1)
 def predict_pairs(self,pairs):
  data=self.t.load(common.bound(self.root,self.config['buddy_inputs']['query']),map_location='cpu',weights_only=True)
  if set(data)!={'ids','links','sf'} or not self.t.equal(data['links'],pairs.cpu()):raise ValueError('Complete source label-free BUDDY query authority required')
  return self.predict_cache(data)
 def validate(self,epoch):
  scores=self.predict_cache(self.select).mean(0);n=self.select['n_positive'];pos,neg=scores[:n],scores[n:]
  if len(neg)!=100000:raise ValueError('Unchanged official negative SELECT pool required')
  hits=int((pos>neg.topk(50).values[-1]).sum())/n;labels=self.t.cat((self.t.ones(n),self.t.zeros(len(neg))))
  ce=float(self.t.nn.functional.binary_cross_entropy_with_logits(scores,labels))
  return {'selection_metric':hits,'hits@50':hits,'ce':ce,'objective':ce,'regularization':0.,'positive_count':n,'negative_count':len(neg),'fusion':'mean_raw_scores'}
 def observe_validation(self,epoch,metrics):
  better=self._best is None or metrics['selection_metric']>self._best
  if better:self._best,self._best_step=metrics['selection_metric'],epoch
  return {'save_selected':better,'stop_requested':False,'best_saved_metric':self._best,'best_saved_step':self._best_step,'full_budget':True}
 def capture_state(self):return {'provenance':copy.deepcopy(self.provenance),'fit_rng':copy.deepcopy(self._rng),'outer_rng':bridge.rng_state(self.t,self.np),'epoch':self._epoch,'best':self._best,'best_step':self._best_step}
 def restore_state(self,state):
  if state['provenance']!=self.provenance:raise ValueError('BUDDY source/input/procedure ancestry changed')
  self._rng=copy.deepcopy(state['fit_rng']);self._epoch=state['epoch'];self._best=state['best'];self._best_step=state['best_step'];bridge.restore_rng(self.t,self.np,state['outer_rng'])


def make_adapter(*,torch,project_root,study,run,device):
 import numpy
 c=run['config'];assert study['dataset']=='ogbl-collab' and c['train_minibatches_per_epoch']==2303 and c['provider_nodes']==235868
 loaded={k:torch.load(common.bound(project_root,v),map_location='cpu',weights_only=True) for k,v in c['buddy_inputs'].items() if k in ('common','train','select')}
 cm=loaded['common'];cm={k:v for k,v in cm.items() if k in ('x','degrees')}
 assert cm['x'].shape==(235868,128) and cm['degrees'].shape==(235868,)
 for name,n,negative in (('train',1179052,1179052),('select',60084,100000)):
  d=loaded[name];assert set(d)=={'links','sf','n_positive'} and d['n_positive']==n and d['links'].shape==(n+negative,2) and d['sf'].shape==(n+negative,8)
 model,opt,mods,details=construct(torch,project_root,c,run['seed'],run['body'],device)
 return BUDDYAdapter(torch,numpy,model,opt,mods,details,root=project_root,study=study,run=run,common_input=cm,train=loaded['train'],select=loaded['select'])
