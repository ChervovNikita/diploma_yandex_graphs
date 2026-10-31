"""Literal source native FiLM constructors; lazy imports and custodied roles."""
import ast,copy,hashlib,inspect,types
from types import SimpleNamespace
from pathlib import Path
from .common_native_wrapper import bound,load_module,construct_source
from .classifier import ClassifierAdapter
CONTRACT={'schema':'literal-source-FiLM-WikiCS-minimal-v1','capacity':'two unmodified PyG2.7.0 FiLMConv on exact source Model scaffold;all14 affine maps exact FactorLinear even M1','objective':'mean own CE;one backward/AdamW WD0;four full native paths for shared4','selector':'strict-first max probability-mean SELECT accuracy,postupdate cadence1,cap1000,patience300,no checkpoint0','seeds':[7301,7403,7507],'source_graph':'literal source stored COO and float32 features retained;current TRAIN byte-identical labels','adaptations':['fresh prospective SELECT3505/CAL1769 rather than merged historical VALID5274','one current source-compatible public mask0','exact M1 body0 reused in genuine I4;three fresh other bodies','new original GNNM SAGE comparison,capacity is not FiLM-equivalent','after-serving-pool scalar CAL500;raw results also retained;source old fivefold prepool calibration not claimed'],'historical_state_inherited':False}
METHODS=('seeds','native_body','make','scope','logits','training_outputs')

def construct(torch,root,B,graph,arm,seed,body,device):
 if arm=='original_GNNM':
  source,model,opt,streams,primitives,rendered=construct_source(torch,root,B['common'],features=300,classes=10,seed=seed,body=body,arm='original_gnnm',backbone='SAGE',dropout=.2,learning_rate=.001,device=device)
  return source,model,opt,streams,primitives,rendered
 import torch_geometric
 from torch_geometric.nn.conv import FiLMConv,MessagePassing
 if torch_geometric.__version__!='2.7.0':raise ValueError('Pinned PyG2.7.0 required')
 for key,obj in [('film_conv',FiLMConv),('message_passing',MessagePassing)]:
  raw=Path(inspect.getfile(obj)).read_bytes()
  if len(raw)!=B[key]['bytes'] or hashlib.sha256(raw).hexdigest()!=B[key]['sha256']:raise ValueError('Installed PyG source differs')
 tree=ast.parse(bound(root,B['family']).read_text());family=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Family');methods=[copy.deepcopy(next(n for n in family.body if isinstance(n,ast.FunctionDef) and n.name==k)) for k in METHODS]
 mod=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='SourceBuild',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]));ns={};exec(compile(mod,'<literal_FiLM_family_AST>','exec'),ns);source=ns['SourceBuild']();source.torch=torch;source.F=torch.nn.functional;source.device=torch.device(device);source.cuda=source.device.index or 0
 source.cfg={'depth':2,'hidden':128,'dropout':.2,'rank':16,'learning_rate':.001,'block_seed_offset':4000037};source.set_seed=load_module(root,B['common']['driver']).set_seed;source.Model=load_module(root,B['common']['native']).Model;source.routes=load_module(root,B['common']['routes']);source.factors=load_module(root,B['common']['factors']);source.FiLMConv=FiLMConv;source.film_adapter_source=load_module(root,B['film_adapter']);source.edge_metadata={'nodes':11701,'edges':graph['edge_index'].shape[1],'self_edges':int(graph['edge_index'][0].eq(graph['edge_index'][1]).sum())};source._single_adapter=None;source._film_adapter=None
 members=4 if arm=='shared4' else 1;model,opt,streams=source.make(seed,body,'baseline',True,members,'shared' if members==4 else 'single')
 rendered={'source_method_AST_hashes':{m.name:hashlib.sha256(ast.dump(m,include_attributes=False).encode()).hexdigest() for m in methods},'AST_method_changes':[],'provider_dimension_changes':[]}
 return source,model,opt,streams,{},rendered

def attach(torch,source,model,opt,streams,primitives,rendered,graph,train,select,arm,seed,body,provenance):
 device=next(model.parameters()).device;x,e=graph['x'].to(device),graph['edge_index'].to(device);ids,y=train['ids'].to(device),train['y'].to(device);sid,sy=select['ids'].to(device),select['y'].to(device)
 labels=torch.zeros(len(x),dtype=torch.long,device=device);mask=torch.zeros(len(x),dtype=torch.bool,device=device);labels[ids],mask[ids]=y,True;source.data=SimpleNamespace(x=x,edge_index=e,y=labels,train_mask=mask);source.train_ids=ids;source.batch={'x':x,'edge_index':e,'graph':source.data};bank=arm=='shared4';members=4 if arm in ('shared4','original_GNNM') else 1
 def predict(current,training,step):
  if arm=='original_GNNM':
   with source.scope(streams,0):return torch.stack([current(source.data,x,tabm_seed=m) for m in range(4)])
  return source.logits(current,streams,bank,torch.arange(len(x),device=device))
 def update(adapter,step):
  model.train();opt.zero_grad()
  if arm=='original_GNNM':
   with source.scope(streams,0):loss=primitives['train_step_tabm'](model,source.data,opt,False)
  else:
   logits,_=source.training_outputs(model,streams,bank);targets=labels[ids] if bank else labels[mask];loss=source.F.cross_entropy(logits.flatten(0,1),targets.repeat(members));loss.backward()
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite source gradient')
   opt.step();loss=float(loss.detach())
  return {'objective':float(loss),'ce':float(loss),'regularization':0.,'operation_counts':{'optimizer_updates':1,'backward_calls':4 if arm=='original_GNNM' else 1,'model_forwards':members,'route_predictions':members,'supervised_examples':len(ids),'native_graph_block_calls':2*members}}
 def capture():return {'streams':copy.deepcopy(streams),'training_modes':{k:m.training for k,m in model.named_modules()},'factor_members':{k:m.member for k,m in model.named_modules() if isinstance(m,source.factors.FactorLinear)}}
 def restore(state):
  streams[:]=copy.deepcopy(state['streams']);mods=dict(model.named_modules())
  for k,v in state['training_modes'].items():mods[k].training=v
  for k,v in state['factor_members'].items():mods[k].member=v
 fusion='mean_logits' if arm=='original_GNNM' else 'mean_probabilities';adapter=ClassifierAdapter(torch,torch.nn.functional,model,opt,logits_fn=predict,train_ids=ids,train_y=y,select_ids=sid,select_y=sy,regularizer=lambda _:torch.zeros((),device=device),fusion=fusion,selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':300,'save_every':1},provenance=provenance,external_capture=capture,external_restore=restore,custom_train=update)
 adapter.source=source;adapter.source_rendered=rendered
 # Exact source full role probability mean, FP32 strict accuracy; stable double diagnostic NLL.
 def validate(step):
  model.eval()
  with torch.no_grad():
   z=predict(model,False,step)[:,sid];p=z.softmax(-1).mean(0) if fusion=='mean_probabilities' else z.mean(0).softmax(-1);correct=int(p.argmax(-1).eq(sy).sum());lp=torch.logsumexp(z.double().log_softmax(-1),0)-__import__('math').log(members) if fusion=='mean_probabilities' else z.double().mean(0).log_softmax(-1);nll=float(-lp.gather(1,sy[:,None]).mean());return {'accuracy':float(p.argmax(-1).eq(sy).float().mean()),'correct':correct,'examples':len(sid),'pooled_nll':nll,'objective':nll,'ce':nll,'regularization':0.,'selection_metric':float(p.argmax(-1).eq(sy).float().mean()),'fusion':fusion}
 adapter.validate=validate
 return adapter

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config'];arm=c['film_arm']
 if study['dataset']!='wikics' or c['film_source_contract']!=CONTRACT or arm not in ('native_M1','shared4','original_GNNM'):raise ValueError('Exact fixed FiLM source cell required')
 from ..native_stopping import validate_native_run
 validate_native_run(run)
 def load(row):return torch.load(bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=[load(c['node_inputs'][k]) for k in ('graph','train','select')]
 if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'} or graph['x'].shape!=(11701,300) or graph['x'].dtype!=torch.float32 or graph['edge_index'].shape!=(2,c['provider_graph']['edges']) or train['ids'].shape!=(580,):raise ValueError('Custodied literal source GRAPH/TRAIN and new SELECT required')
 if torch.isin(train['ids'],select['ids']).any():raise ValueError('Role overlap')
 state=construct(torch,project_root,c['film_sources'],graph,arm,run['seed'],run['body'],device)
 return attach(torch,*state,graph,train,select,arm,run['seed'],run['body'],{'actual_model_seed':c['initialization_seed'],'study':study['study_id'],'arm':arm,'source_bindings':c['film_sources'],'node_inputs':c['node_inputs'],'contract':CONTRACT})
