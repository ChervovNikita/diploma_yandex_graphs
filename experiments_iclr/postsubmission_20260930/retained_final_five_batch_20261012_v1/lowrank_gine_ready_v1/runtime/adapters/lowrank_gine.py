"""Exact width128/L3 GINE low-rank source; one finite graph epoch per call."""
import ast,copy,json,math,random
from pathlib import Path
from . import common_native_wrapper as common
from .molecular import MolecularAdapter,validate_pack,validate_prediction_pack
try:
 from ..engine import AdapterStepError
except ImportError:
 from engine import AdapterStepError

ARMS=('native_M1','original_GNNM','middle_private','lowrank_private')
CONTRACT={'schema':'literal-lowrank-GINE-minimal-binary-graph-port-v1','canonical_method_id':'lowrank_gine_private_message_factors','model':'literal MolecularProjectorEnsemble H128/L3/FFNmult2/drop.1/M4 and retained original BE boundaries','lowrank':'actual factory enumerates ranks0..128;closest actual registered count to private middle block1;lower rank ties;random Kaiming A/zeroB/scaling1;all shared and private parameters remain trainable','optimizer':'literal inherited AdamW LR.001/WD0/eps1e-8/betas.9,.999;100full epochs;batch128;one joint mean-member equal-graph finite-targetBCE/backward/map','shuffle':'literal PyG DataLoader shuffle=True generator seed19709+1000*actual_optimizer_seed+epoch,num_workers0,include finaltail','selector':'max pooled SELECT ROCauc,then min pooled BCE,then earlier sourceepoch;checkpoint0excluded;no earlytermination','serving':'complete own member graph paths;mean raw logits after readout;source sequential shared Torch dropout RNG','new_task':'ogbg-molbace full official TRAIN and custodied SELECT/CAL/evaluate;no new HPO knobs','native_controls':'newly paid exact source MolecularSingle native M1/genuineI4,source MolecularProjectorEnsemble original boundary GNNM,and exact private-middle target'}

def construct(torch,root,binding,arm,seed,device):
 if arm not in ARMS:raise ValueError('Exact source GINE arm required')
 import numpy as np
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
 source=common.load_module(root,binding['model']);g=source.GINE
 original=g.MolecularSingle(source.FIXED_OPTIONS) if arm=='native_M1' else g.MolecularProjectorEnsemble(source.FIXED_OPTIONS,members=4)
 definition={'arm':arm,'source_options':source.FIXED_OPTIONS,'members':original.members,'counts':source.counts(original),'registered_constructor_objects':1}
 if arm=='middle_private':original.privatize([1]);definition.update(counts=source.counts(original),private_blocks=[1])
 if arm=='lowrank_private':
  original,low=source.make_model(original,init_seed=29001+1000*seed);definition.update(lowrank=low,counts=low['counts'],registered_constructor_objects=132,rank_candidate_wrapper_objects=129,chosen_wrapper_objects=1,copied_private_target_objects=1)
 model=original.to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,eps=1e-8,betas=(.9,.999))
 # The inherited exact live loss is AST-loaded without dataset imports/CLI.
 row=next(r for r in binding['phase_dependencies'] if r['path'].endswith('new_tasks/molhiv_pilot.py'));tree=ast.parse(common.bound(root,row).read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='graph_member_bce');env={'torch':torch,'F':torch.nn.functional};exec(compile(ast.Module(body=[node],type_ignores=[]),row['path']+'::graph_member_bce','exec'),env)
 return model,opt,definition,env['graph_member_bce']

class GINEAdapter(MolecularAdapter):
 def __init__(self,torch,model,opt,definition,loss_fn,train,select,config,provenance):
  import numpy as np
  from sklearn.metrics import roc_auc_score
  from torch_geometric.data import Data,Batch
  self.t,self.model,self.optimizer=torch,model,opt;self.train,self.select,self.config=train,select,config;self.provenance=copy.deepcopy(provenance)
  self.Data,self.Batch,self.auc_function,self.np=Data,Batch,roc_auc_score,np;self.device=next(model.parameters()).device;self.members=model.members;self.definition=definition;self.loss_fn=loss_fn;self._epoch=0;self._best_key=None;self._best_step=None
 def _graphs(self,pack):
  result=[]
  for index in range(len(pack['ids'])):
   ns,ne=map(int,pack['node_ptr'][index:index+2]);es,ee=map(int,pack['edge_ptr'][index:index+2]);kwargs={'x':pack['x'][ns:ne],'edge_index':pack['edge_index'][:,es:ee],'edge_attr':pack['edge_attr'][es:ee]}
   if 'y' in pack:kwargs['y']=pack['y'][index:index+1]
   result.append(self.Data(**kwargs))
  return result
 def train_step(self,epoch):
  from torch_geometric.loader import DataLoader
  counts={'optimizer_updates':0,'optimizer_step_attempts':0,'model_forward_attempts':0,'model_forwards':0,'backward_attempts':0,'backward_calls':0,'supervised_examples':0}
  try:
   if epoch!=self._epoch+1:raise ValueError('Exact source complete epoch clock required')
   generator=self.t.Generator().manual_seed(19709+1000*self.config['initialization_seed']+epoch)
   loader=DataLoader(self._graphs(self.train),batch_size=128,shuffle=True,generator=generator,num_workers=0)
   self.model.train();weighted=0.;seen=0
   for batch in loader:
    batch=batch.to(self.device);self.optimizer.zero_grad(set_to_none=True);counts['model_forward_attempts']+=self.members
    z=self.model(batch,batch.x);counts['model_forwards']+=self.members;loss=self.loss_fn(z,batch.y.float())
    if not self.t.isfinite(loss):raise FloatingPointError('Nonfinite source equal-graph BCE')
    counts['backward_attempts']+=1;loss.backward();counts['backward_calls']+=1
    if any(p.grad is not None and not self.t.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite source GINE gradient')
    counts['optimizer_step_attempts']+=1;self.optimizer.step();counts['optimizer_updates']+=1;n=batch.num_graphs;weighted+=float(loss.detach())*n;seen+=n;counts['supervised_examples']+=n
   if seen!=len(self.train['ids']) or counts['optimizer_updates']!=math.ceil(seen/128):raise ValueError('Source full-graph epoch/tail ownership changed')
   self._epoch=epoch
   return {'objective':weighted/seen,'ce':weighted/seen,'regularization':0.,'sample_count':seen,'operation_counts':counts,'source_graph_order_seed':19709+1000*self.config['initialization_seed']+epoch}
  except BaseException as e:raise AdapterStepError(str(e),operation_counts=counts) from e
 def predict_pack(self,pack):
  from torch_geometric.loader import DataLoader
  if 'y' not in pack:pack=validate_prediction_pack(self.t,pack)
  self.model.eval();rows=[]
  with self.t.no_grad():
   for batch in DataLoader(self._graphs(pack),batch_size=128,shuffle=False,num_workers=0):
    batch=batch.to(self.device);z=self.model(batch,batch.x)
    if not self.t.isfinite(z).all():raise FloatingPointError('Nonfinite source GINE serving')
    rows.append(z.detach().cpu())
  return self.t.cat(rows,dim=1)
 def validate(self,epoch):
  z=self.predict_pack(self.select);y=self.select['y'];pooled=z.mean(0);ce=float(self.loss_fn(pooled[None],y));auc=float(self.auc_function(y[:,0].numpy(),pooled[:,0].numpy()))
  return {'objective':ce,'ce':ce,'pooled_bce':ce,'rocauc':auc,'selection_metric':auc,'regularization':0.,'fusion':'mean_logits','sample_count':len(y)}
 def observe_validation(self,epoch,metrics):
  key=(metrics['rocauc'],-metrics['pooled_bce'],-epoch);improve=self._best_key is None or key>self._best_key
  if improve:self._best_key=key;self._best_step=epoch
  return {'save_selected':improve,'stop_requested':False,'best_saved_metric':self._best_key[0],'best_saved_step':self._best_step,'full_budget':True}
 def capture_state(self):
  name,keys,pos,gauss,cached=self.np.random.get_state()
  return {'provenance':copy.deepcopy(self.provenance),'epoch':self._epoch,'best_key':self._best_key,'best_step':self._best_step,'torch_cpu':self.t.get_rng_state().clone(),'torch_cuda':[x.clone() for x in self.t.cuda.get_rng_state_all()] if self.t.cuda.is_initialized() else [],'python':random.getstate(),'numpy':(name,keys.tolist(),pos,gauss,cached),'source_plain_active_members':{n:m.active_member for n,m in self.model.named_modules() if hasattr(m,'active_member')}}
 def restore_state(self,v):
  if v['provenance']!=self.provenance:raise ValueError('Exact source/rank/task ancestry changed')
  self._epoch=v['epoch'];self._best_key=v['best_key'];self._best_step=v['best_step'];self.t.set_rng_state(v['torch_cpu']);random.setstate(v['python'])
  if v['torch_cuda']:self.t.cuda.set_rng_state_all(v['torch_cuda'])
  name,keys,pos,gauss,cached=v['numpy'];self.np.random.set_state((name,self.np.asarray(keys,dtype='uint32'),pos,gauss,cached));modules=dict(self.model.named_modules())
  for n,member in v['source_plain_active_members'].items():modules[n].active_member=member

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config']
 if study['dataset']!='ogbg-molbace' or c['source_GINE_arm'] not in ARMS or c['optimizer']!={'name':'AdamW','learning_rate':.001,'weight_decay':0.,'eps':1e-8,'betas':[.9,.999]}:raise ValueError('Literal one-task source GINE recipe required')
 packs={k:validate_pack(torch,torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True),1,'complete_binary') for k,row in c['molecular_inputs'].items()}
 if torch.isin(packs['train']['ids'],packs['select']['ids']).any():raise ValueError('TRAIN/SELECT graph IDs overlap')
 if len(packs['train']['ids'])!=1210:raise ValueError('Full official MolBACE TRAIN1210 required')
 model,opt,definition,loss=construct(torch,project_root,c['lowrank_sources'],c['source_GINE_arm'],c['initialization_seed'],device)
 return GINEAdapter(torch,model,opt,definition,loss,packs['train'],packs['select'],c,{'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'definition':definition})
