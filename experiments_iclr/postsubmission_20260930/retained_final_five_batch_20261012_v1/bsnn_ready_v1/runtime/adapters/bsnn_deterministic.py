"""Original pinned deterministic d2 bundle; exact isolated Householder provider."""
import ast,copy,hashlib,importlib,json,os,random,sys
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace,ModuleType
from . import common_native_wrapper as common
from .classifier import ClassifierAdapter
from .nsd_sources import extract
try:
 from ..engine import AdapterStepError
except ImportError:
 from engine import AdapterStepError
CACHE={}
CONTRACT={'schema':'exact-original-deterministic-BSNN-d2-new-binary-node-port-v1','canonical_method_id':'bsnn_deterministic_d2','native':'author DiscreteBundleSheafDiffusion d2/hidden32/L2/householder/separate perlayer sheaf+edgeweight learners/normalised/ELU','provider':'unchanged torch-householder1.0.1 original Python/C++ source;isolated project local setuptools68.2.2+ninja1.11.1.4 and own extension build;sharedvenv untouched;no Cayley/generalNSD substitute','source_optimizer':'original grouped_parameters coupled Adam LR.01/bothdecay5e-4/no clip/no scheduler;500full sourceepochs;no newHPO','selector':'full SELECT maximum AUROC,lower NLL then earlier sourceepoch;cadence1;checkpoint0excluded;sourcepatience200 not earlytermination in prospective fullbudget','rng':'exact original V2 Python/NumPy/Torch CPU/CUDA;initialized streamcontinuesTRAIN;evalsnapshot/seedbase+2000003+epoch/restore','new_task':'fresh public TwitchDE MUSAE maturebinary/raw3170features;10custodied rolemasks x3sourcecomparativeseeds;no Tolokers','I4':'four separately initialized source trajectories;explicit seed+1009body;nativeM1 exactbody0reference','original_GNNM':'exact source BEBlock replacesnative lin1/lin2 only;all native interior identities/Householder geometry retained;source4CE/4backward/oneAdamWstepWD0;meanrawlogitsserving','placement':'original CPU construction;exact V2 static placement plus literal original perlayer weightlearner full_left_right_idx transfer'}

def native(torch,root,B):
 root=Path(root);D=root/B['householder_provider_relative_root'];key=str(root/B['author_directory'])
 if key in CACHE:return CACHE[key]
 for row in B['native_files']:common.bound(root,row)
 for row in B['provider']['files']:
  q=root/row['path']
  if q.suffix in ('.py','.cpp','.so') and hashlib.sha256(q.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Exact isolated source or compiled Householder provider changed')
 previous={n:m for n,m in sys.modules.items() if n in ('models','lib') or n.startswith(('models.','lib.'))};path=list(sys.path);oldext=os.environ.get('TORCH_EXTENSIONS_DIR');oldpath=os.environ.get('PATH','')
 for n in previous:sys.modules.pop(n)
 try:
  sys.path[:0]=[str(D/'toolchain_setuptools68'),str(D),key];os.environ['TORCH_EXTENSIONS_DIR']=str(D/'extension_build');os.environ['MAX_JOBS']='1';os.environ['PATH']=str(D/'ninja/data/bin')+':'+oldpath
  module=importlib.import_module('models.disc_models');house=importlib.import_module('torch_householder')
  if not Path(house.__file__).resolve().is_relative_to(D.resolve()):raise ValueError('Exact own Householder Python/C++ provider required')
  for n,m in sys.modules.items():
   if n in ('models','lib') or n.startswith(('models.','lib.')):
    if not Path(m.__file__).resolve().is_relative_to(Path(key).resolve()):raise ValueError('Original author models/lib only')
  CACHE[key]=module.DiscreteBundleSheafDiffusion;return CACHE[key]
 finally:
  for n in list(sys.modules):
   if n in ('models','lib') or n.startswith(('models.','lib.')):sys.modules.pop(n)
  sys.modules.update(previous);sys.path[:]=path;os.environ['PATH']=oldpath
  if oldext is None:os.environ.pop('TORCH_EXTENSIONS_DIR',None)
  else:os.environ['TORCH_EXTENSIONS_DIR']=oldext

def helpers(torch,root,B):return extract(root,B['baseline'],('seed_all','capture_rng','restore_rng','evaluation_rng','metrics'),{'random':random,'contextmanager':contextmanager})

def construct(torch,root,B,graph,arm,seed,device):
 import numpy as np
 if arm not in ('native_M1','original_GNNM'):raise ValueError('Original deterministic bundle source and boundary control only')
 h=helpers(torch,root,B);h.seed_all(np,torch,seed);cls=native(torch,root,B);protocol=json.loads(common.bound(root,B['protocol']).read_bytes());args={**protocol['configs'][0]['native_args'],'input_dim':graph['x'].shape[1],'output_dim':2,'graph_size':len(graph['x']),'device':str(device)}
 placement=common.load_module(root,B['placement'])
 class Wrapper:
  def make_native_factory(self,cpu_edges,cpu_args):
   def create():
    m=cls(cpu_edges,dict(cpu_args))
    if len(m.sheaf_learners)!=2 or len(m.weight_learners)!=2:raise ValueError('Separate original perlayerlearners required')
    for learner in m.weight_learners:learner.full_left_right_idx=learner.full_left_right_idx.to(device)
    return m
   return create
 f=placement.make_native_placed_factory(torch,Wrapper(),graph['edge_index'],graph['edge_index'].to(device),args)
 if arm=='native_M1':
  model=f();a,b=model.grouped_parameters();opt=torch.optim.Adam([{'params':a,'weight_decay':.0005},{'params':b,'weight_decay':.0005}],lr=.01)
  model.members=1
 else:
  original=extract(root,B['source_GNNM_boundary_factory'],('original_bank',),{'extract':extract});model=original.original_bank(torch,root,B,None,f,seed);model.members=4;opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.)
 return model,opt,h,args

class BSNNAdapter(ClassifierAdapter):
 def probabilities(self,z):return z.exp().mean(0) if self.fusion=='mean_probabilities' else super().probabilities(z)
 def validate(self,step):
  import numpy as np
  from sklearn.metrics import roc_auc_score
  self.model.eval()
  with self.h.evaluation_rng(np,self.t,self.seed,step),self.t.no_grad():
   z=self.logits_fn(self.model,False,step);p=self.probabilities(z)[self.select_ids];v=self.h.metrics(self.t,roc_auc_score,p.log(),self.select_y)
  return {**v,'objective':v['nll'],'ce':v['nll'],'regularization':0.,'selection_metric':v['auroc'],'source_epoch':step,'fusion':self.fusion}
 def observe_validation(self,step,v):
  key=(v['auroc'],-v['nll'],-step);improve=self._key is None or key>self._key
  if improve:self._key=key;self._best_step=step
  return {'save_selected':improve,'stop_requested':False,'best_saved_metric':self._key[0],'best_saved_step':self._best_step,'stale_steps':0,'checkpoint_eligible':True}

def attach(torch,model,opt,h,args,graph,train,select,arm,seed,provenance):
 import numpy as np
 from . import nsd_sources
 x=graph['x'].to(next(model.parameters()).device);ids,y=train['ids'].to(x.device),train['y'].to(x.device);adapter=None;clock=0
 def logits(m,training,step):
  z=m(x);return z.unsqueeze(0) if z.ndim==2 else z
 def unit(a,step):
  nonlocal clock
  counts={'optimizer_updates':0,'optimizer_step_attempts':0,'model_forward_attempts':0,'backward_attempts':0,'backward_calls':0}
  try:
   if step!=clock+1:raise ValueError('Source full epoch clock changed')
   model.train();opt.zero_grad(set_to_none=True);losses=[]
   if arm=='original_GNNM':
    for member in range(4):
     counts['model_forward_attempts']+=1;z=model.forward_route(member,x);v=torch.nn.functional.cross_entropy(z[ids],y);counts['backward_attempts']+=1;(v/4).backward();counts['backward_calls']+=1;losses.append(float(v.detach()))
   else:
    counts['model_forward_attempts']+=1;v=torch.nn.functional.nll_loss(model(x)[ids],y);counts['backward_attempts']+=1;v.backward();counts['backward_calls']+=1;losses.append(float(v.detach()))
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite exact bundle gradient')
   counts['optimizer_step_attempts']=1;opt.step();counts['optimizer_updates']=1;clock=step
   return {'objective':sum(losses)/len(losses),'ce':sum(losses)/len(losses),'regularization':0.,'operation_counts':counts}
  except BaseException as e:raise AdapterStepError(str(e),operation_counts=counts) from e
 def plain():return {n:m.L.detach().cpu().clone() if torch.is_tensor(m.L) else None for n,m in model.named_modules() if hasattr(m,'L')}
 def capture():return {'source_rng':h.capture_rng(np,torch),'source_epoch':clock,'source_plain_L':plain(),'selector_key':adapter._key,'native_args':args}
 def restore(v):
  nonlocal clock
  if v['native_args']!=args:raise ValueError('Original deterministic bundle source ancestry changed')
  h.restore_rng(np,torch,v['source_rng']);clock=v['source_epoch'];adapter._key=v['selector_key'];modules=dict(model.named_modules())
  for n,L in v['source_plain_L'].items():modules[n].L=None if L is None else L.to(x.device).clone()
 adapter=BSNNAdapter(torch,torch.nn.functional,model,opt,logits_fn=logits,train_ids=ids,train_y=y,select_ids=select['ids'].to(x.device),select_y=select['y'].to(x.device),regularizer=lambda m:x.new_zeros(()),fusion='mean_logits' if arm=='original_GNNM' else 'mean_probabilities',selector={'metric':'auroc','direction':'max','ties':'first','patience_steps':501,'save_every':1},custom_train=unit,external_capture=capture,external_restore=restore,provenance=provenance);adapter.h,adapter.seed,adapter._key=h,seed,None
 adapter.predict_rows=lambda rows,step:logits(model,False,step)[:,rows.to(x.device)];return adapter

def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
 c=run['config'];B=c['bsnn_sources']
 if study['dataset']!='twitch-de-musae-raw' or c['classes']!=2:raise ValueError('New public binary source task required')
 read=lambda row:torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True);graph,train,select=(read(c['node_inputs'][k]) for k in ('graph','train','select'))
 if torch.isin(train['ids'],select['ids']).any():raise ValueError('TRAIN/SELECT role overlap')
 seed=c['initialization_seed']
 if seed!=run['seed']+1009*run['body']:raise ValueError('Exact source prospective I4 ancestry')
 model,opt,h,args=construct(torch,project_root,B,graph,c['bsnn_arm'],seed,device)
 return attach(torch,model,opt,h,args,graph,train,select,c['bsnn_arm'],seed,{'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'native_args':args})
