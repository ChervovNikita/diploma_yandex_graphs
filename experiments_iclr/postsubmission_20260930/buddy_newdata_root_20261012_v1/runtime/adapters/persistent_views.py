"""Literal eight-pass native Polynormer graph-view bank with source staging."""
import ast,copy,gzip,json,random,sys
from types import SimpleNamespace
from . import common_native_wrapper as common
from .classifier import ClassifierAdapter

BASE='accuracy_first_graph_view_source_preparation_20261004_v2/'
PATHS={k:BASE+k+'.py' for k in ('bank','schedule','views','driver')}
PATHS.update(native='amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py',boundary='amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py')
ARMS=('tied_persistent','untied_persistent','tied_shuffled','tied_random_null','native_M1','original_gnnm')
CONTRACT={'schema':'persistent-graph-views-literal-source-port-v1','source_exact_recipe':False,
 'native':'native Polynormer10 local/1 global,256 per head/2 heads=512; CPU constructor then device reset then literal R/S/B boundary family with native stem signs; untied ownership deepcopies the tied initialized image without reset or new draw',
 'TRAIN':'literal native members0..3 then assigned-view members0..3;8 complete TRAIN CE/4 backwards; one original Adam after full old-state accumulation; objective2*mean own CE; native graph mean probabilities serving',
 'views':'literal TRAIN-only floor10% equal/different pair deletion; loops/multiplicity/order unchanged; fixed degree-pair-stratum null masks; source coverage hashes bound; Photo masks4/5/9 retain failed precondition for all5 seeds',
 'schedule':'persistent rows0/1 equal and2/3 different; shuffled complementary hash2of4 assignments restarted at stage; exact source seed/hash law',
 'controller':'local200/global4800 full5000 adaptation; strict-first pooled native SELECT; source selected-local model+Adam, live actual clock/RNG retained; overall selected serving may be local winner',
 'HPO':'nine LR/local-global-dropout points around.001/.3; native inputdrop.2 fixed,10/1/H256/heads2 fixed; source source-anchor preserved',
 'native_controls':'native M1 and genuine separately selectedI4 retain native constructor/reset, source stage and two full native CE views/update; literal original GNNM R/S/B control has same eight native-graph ownCE opportunities and original mean-logit serving declared',
 'eligibility':'No retry, fraction change, category omission or stratum borrowing. Source-ineligible cells fail before model construction; disabled roster retains all masks/seeds/costs.'}

def sources(root,bindings,*,features,classes,dropout,learning_rate,local=200,global_=4800):
 views=common.load_module(root,bindings['views']);schedule_path=common.bound(root,bindings['schedule']);tree=ast.parse(schedule_path.read_text());tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or n.module!='views']
 edits=0
 for n in tree.body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name) and t.id in ('LOCAL_UPDATES','GLOBAL_UPDATES'):n.value=ast.Constant(local if t.id=='LOCAL_UPDATES' else global_);edits+=1
 assert edits==2;namespace={'__name__':'bound_persistent_schedule_'+str(id(tree)),'digest':views.digest};module=type(sys)(namespace['__name__']);sys.modules[module.__name__]=module;module.__dict__.update(namespace);exec(compile(ast.fix_missing_locations(tree),'<source_schedule5000_binding>','exec'),module.__dict__);schedule=module
 bank_path=common.bound(root,bindings['bank']);tree=ast.parse(bank_path.read_text());ctor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_bank');ctor.body=[n for n in ctor.body if not isinstance(n,ast.ImportFrom) or n.module!='schedule'];edits=0
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and ast.unparse(n.func)=='rt.torch.optim.Adam':
   lr=next(k for k in n.keywords if k.arg=='lr');assert lr.value.value==.001;lr.value=ast.Name(id='_bound_LR',ctx=ast.Load());edits+=1
  if isinstance(n,ast.Tuple) and ast.unparse(n)=='(0.001, 0.0, (0.9, 0.999), 1e-08, False)':n.elts[0]=ast.Name(id='_bound_LR',ctx=ast.Load());edits+=1
 assert edits==2;space={'CONDITIONS':schedule.CONDITIONS,'_bound_LR':learning_rate};exec(compile(ast.fix_missing_locations(tree),'<native_source_Adam_HPO_binding>','exec'),space);bank=SimpleNamespace(**space)
 recipe=copy.deepcopy(bank.NATIVE_RECIPE);recipe.update(in_channels=features,out_channels=classes,dropout=dropout,global_dropout=dropout)
 driver_path=common.bound(root,bindings['driver']);tree=ast.parse(driver_path.read_text());nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('train_step','eval_logits')];edits=0
 for n in ast.walk(ast.Module(body=nodes,type_ignores=[])):
  if isinstance(n,ast.Tuple) and ast.unparse(n) in ('(x.shape[0], 5)','(4, x.shape[0], 5)'):n.elts[-1].value=classes;edits+=1
 assert edits==2;code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));space={'stage':bank.stage,'stage_at':schedule.stage_at,'pass_plan':schedule.pass_plan,'check_gradients':bank.check_gradients,'optimizer_names':bank.optimizer_names};exec(compile(code,'<literal_source_CLASS_shape_binding>','exec'),space)
 return bank,schedule,SimpleNamespace(**space),recipe,{'bank_source_sha256':bindings['bank']['sha256'],'schedule_source_sha256':bindings['schedule']['sha256'],'driver_source_sha256':bindings['driver']['sha256'],'bound_driver_AST_sha256':common.ast_hash(code),'provider_HPO_edits':['features','classes','local/global_dropout','Adam_LR','global2500->4800'],'native_input_dropout':.2}

def construct(torch,root,bindings,*,arm,features,classes,seed,body,dropout,learning_rate,device,local=200,global_=4800):
 import numpy
 if arm not in ARMS:raise ValueError('Literal persistent/native-control arm required')
 bank,schedule,driver,recipe,details=sources(root,bindings,features=features,classes=classes,dropout=dropout,learning_rate=learning_rate,local=local,global_=global_)
 rt=SimpleNamespace(torch=torch,numpy=numpy,native=common.load_module(root,bindings['native']),boundary=common.load_module(root,bindings['boundary']))
 actual_seed=seed+1000003*body
 if arm=='native_M1':
  bank.seed_all(rt,actual_seed,device);model=rt.native.Polynormer(**recipe);model.to(device);model.reset_parameters();model._global=False;optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.,betas=(.9,.999),eps=1e-8,amsgrad=False);construction={'binding':'same source constructor/device/reset without boundary wrapper; native M1 capable control'}
 else:model,optimizer,construction=bank.build_bank(rt,'tied_persistent' if arm=='original_gnnm' else arm,actual_seed,device,synthetic_recipe=recipe)
 def metadata(value):
  if torch.is_tensor(value):
   import hashlib
   return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
  if isinstance(value,dict):return {k:metadata(v) for k,v in value.items()}
  if isinstance(value,(tuple,list)):return [metadata(v) for v in value]
  return value
 return rt,bank,schedule,driver,model,optimizer,{**details,'construction':metadata(construction),'actual_initialization_seed':actual_seed}

class ViewAdapter(ClassifierAdapter):
 def observe_validation(self,step,metrics):
  decision=super().observe_validation(step,metrics)
  if step<=self.local_updates and (self.local_best is None or metrics['selection_metric']>self.local_best):
   self.local_best=float(metrics['selection_metric']);self.local_image={'model':self.bank.cpu_tree(self.rt,self.model.state_dict()),'optimizer':self.bank.cpu_tree(self.rt,self.optimizer.state_dict())}
   if self.arm!='native_M1':self.local_image.update(optimizer_parameter_names=self.bank.optimizer_names(self.model,self.optimizer),primitives=self.bank.primitives(self.model),selection={'global':False,'actual_update':step})
  return decision
 def train_step(self,step):
  if step!=self.clock.actual_update+1:raise ValueError('Actual source clock differs from controller update')
  if step==self.local_updates+1:
   if self.local_image is None:raise ValueError('Selected local source image missing')
   if self.arm=='native_M1':
    live=self.bank.cpu_tree(self.rt,self.bank.rng_state(self.rt,self.device));self.model.load_state_dict(self.local_image['model']);self.optimizer.load_state_dict(self.local_image['optimizer']);self.model._global=True;self.bank.restore_rng(self.rt,self.device,live)
   else:self.bank.transition(self.rt,self.model,self.optimizer,self.device,self.local_image,self.clock)
  self.clock.advance();return super().train_step(step)

def attach(torch,rt,bank,schedule,driver,model,optimizer,details,*,x,graphs,train_ids,train_y,select_ids,select_y,arm,seed,provenance):
 adapter=None;members=1 if arm=='native_M1' else 4;source_ids=tuple(train_ids.cpu().tolist());source_labels=tuple(train_y.cpu().tolist())
 def logits(current,training,step):return current(x,graphs['native']).unsqueeze(0) if arm=='native_M1' else driver.eval_logits(rt,current,x,graphs['native'])
 def train(adapter,step):
  if arm=='native_M1':
   model.train();optimizer.zero_grad(set_to_none=True);loss=0.
   for view in range(2):
    z=model(x,graphs['native']);value=torch.nn.functional.cross_entropy(z[train_ids],train_y);value.backward();loss+=float(value.detach())
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Native two-view gradient')
   optimizer.step()
  else:
   active_graphs={k:graphs['native'] for k in graphs} if arm=='original_gnnm' else graphs
   loss=driver.train_step(rt,model,optimizer,x,active_graphs,source_ids,source_labels,'tied_persistent' if arm=='original_gnnm' else arm,seed,step)
  return {'objective':loss,'ce':loss,'regularization':0.,'operation_counts':{'optimizer_updates':1,'backward_calls':2*members,'fullgraph_native_member_trajectories':2*members,'supervised_examples':2*members*len(train_y),'physical_slow_bodies':4 if arm=='untied_persistent' else 1}}
 def capture():
  return {'source_rng':bank.cpu_tree(rt,bank.rng_state(rt,x.device)),'source_clock':adapter.clock.actual_update,'global':bool(model._global) if arm=='native_M1' else bank.stage(model),'local_image':copy.deepcopy(adapter.local_image),'local_best':adapter.local_best}
 def restore(value):
  adapter.clock.actual_update=value['source_clock'];adapter.local_image=copy.deepcopy(value['local_image']);adapter.local_best=value['local_best']
  if arm=='native_M1':model._global=value['global']
  else:bank.set_stage(model,value['global'])
  bank.restore_rng(rt,x.device,value['source_rng'])
 adapter=ViewAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits,train_ids=train_ids,train_y=train_y,select_ids=select_ids,select_y=select_y,regularizer=lambda m:x.new_zeros(()),fusion='mean_logits' if arm=='original_gnnm' else 'mean_probabilities',selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5000,'save_every':1},provenance={**provenance,'source_contract':CONTRACT,'source_bindings':details},external_capture=capture,external_restore=restore,custom_train=train)
 adapter.rt=rt;adapter.bank=bank;adapter.schedule=schedule;adapter.clock=schedule.BankClock();adapter.arm=arm;adapter.local_updates=schedule.LOCAL_UPDATES;adapter.local_image=None;adapter.local_best=None;adapter.device=x.device;return adapter

def prepared_graphs(torch,root,edges,coverage,deleted,device):
 record=json.loads(common.bound(root,coverage).read_text());report=record['coverage']
 if not report['coverage_eligible']:raise ValueError('Source persistent-view coverage precondition failed: preserve recorded zero semantic deletion/shortfall, no resampling')
 pairs=json.loads(gzip.decompress(common.bound(root,deleted).read_bytes()));native=tuple(map(tuple,edges.T.tolist()))
 graphs={'native':edges.to(device)}
 for name,removed in pairs.items():
  deleted_set={tuple(p) for p in removed};retained=[(u,v) for u,v in native if (min(u,v),max(u,v)) not in deleted_set or u==v]
  # Literal source digest serializes ordered pair records as canonical JSON.
  import hashlib
  observed=hashlib.sha256(json.dumps(retained,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  assert observed==report['views'][name]['edges_sha256'];graphs[name]=torch.tensor(retained,dtype=torch.long,device=device).T.contiguous()
 return graphs

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config'];assert study['dataset'] in common.DATASETS and c['backbone']=='native_view_Polynormer'
 if c.get('source_view_coverage_eligible') is False:raise ValueError('Recorded source-view precondition failure retained in disabled roster; no model/optimizer constructed')
 def read(row):return torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=(read(c['node_inputs'][k]) for k in ('graph','train','select'));x=graph['x'].to(device);edges=graph['edge_index']
 assert tuple(x.shape)==(c['provider_graph']['nodes'],c['provider_graph']['features']) and tuple(edges.shape)==(2,c['provider_graph']['edges']) and not torch.isin(train['ids'],select['ids']).any()
 graphs=prepared_graphs(torch,project_root,edges,c['view_coverage'],c['deleted_pairs'],device) if c['view_arm'] not in ('native_M1','original_gnnm') else {k:edges.to(device) for k in ('native','equal','different','null_equal','null_different')}
 rt,bank,schedule,driver,model,opt,details=construct(torch,project_root,c['view_sources'],arm=c['view_arm'],features=x.shape[1],classes=c['classes'],seed=run['seed'],body=run['body'],dropout=c['dropout'],learning_rate=c['learning_rate'],device=device)
 return attach(torch,rt,bank,schedule,driver,model,opt,details,x=x,graphs=graphs,train_ids=train['ids'].to(device),train_y=train['y'].to(device),select_ids=select['ids'].to(device),select_y=select['y'].to(device),arm=c['view_arm'],seed=run['seed'],provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c})
