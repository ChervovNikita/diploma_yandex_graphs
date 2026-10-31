"""Exact cold NCN private response and episode ownership;lazy numerical imports."""
import ast,copy,hashlib,importlib,importlib.util,random,sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace,ModuleType
from .common_native_wrapper import bound,load_module
CONTRACT={'schema':'literal-private-response-CiteSeer-Heart5-minimal-v1','native':'exact source NCN3703x256,L1puregcn,full nonlinear NCN or unframedF4head;sourceprivate roles','source5':'E_end_live,S_end_live,U_end_live,J4_end_joint,F1_end_live;fixed endpoint256inner/64outer;noHPO','gradient':'literal virtual live functionalprivateAdam,outer dropoutoff/sharedAdam,recomputedprivateAdamcommit;no outer private label update;discardedvirtualmoments;row0four own streams averaged','Adam':'source .001,betas(.9,.999),eps1e-8,WD0,foreachFalse,fusedFalse;functionalprivate stablezerohistory unclipped','controller':'full180cycles/no early stopping/eval5;exactsource60prefix ofsamefit;firstrounded500negativeMRRpositive implies earliestsource11missstop60;strict-firstselected;no checkpoint0','geometry':'every source paired endpoint+matched_random draw built before updates;common unionmask;61episodes/tail complete;no fallback','baseline':'sourcecapableSingleordinary_jointM1/freshownI4;ordinarynative4 isjointwholecohort;original literal boundaryBEBlock source3pass/4CEbackwardsperpass;separatevirtual/committed counts','adaptations':['prospective horizon extension60to180;sourceoptimizer/objective/sampler/eval5 unchanged;source60 selectedprefix notnewfit','12Oct custodied encounteredpublicHeARTroles;source500negative SELECT retained;CAL/evaluate isolated afterwards','fresh own I4 (+5body modelseed);M1 member0 exactreuse','new originalGNNM sourcecontext/exposure adapted comparator','after-serving-pool CAL500 andraw output retained;sourceoldraw-only protocol notclaimed identical'],'historical_state_inherited':False}

@contextmanager
def aliases(values):
 missing=object();old={k:sys.modules.get(k,missing) for k in values};sys.modules.update(values)
 try:yield
 finally:
  for k,v in old.items():
   if v is missing:sys.modules.pop(k,None)
   else:sys.modules[k]=v

def definitions(root,row,names,env):
 tree=ast.parse(bound(root,row).read_text());selected=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if {n.name for n in selected}!=set(names):raise ValueError('Exact source primitive roster required')
 exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),row['path']+'::literal_AST','exec'),env)
 return SimpleNamespace(**env)

def sources(root,B,row0=False):
 for row in B.values():bound(root,row)
 # Own package avoids another packet's baseline_models globals.
 q=bound(root,B['native']);ns='private_response_native_'+B['native']['sha256'][:12];pkg=ns+'.NCN'
 if ns not in sys.modules:
  m=ModuleType(ns);m.__path__=[str(q.parent.parent)];m.__package__=ns;sys.modules[ns]=m
 if pkg not in sys.modules:
  m=ModuleType(pkg);m.__path__=[str(q.parent)];m.__package__=pkg;sys.modules[pkg]=m
 name=pkg+'.model'
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,q);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 native=sys.modules[name];heads=load_module(root,B['heads']);adam=load_module(root,B['row0_private_adam'] if row0 else B['private_adam'])
 with aliases({'private_adam':adam}):steps=load_module(root,B['row0_steps'] if row0 else B['steps'])
 values={k:load_module(root,B[k]) for k in ('geometry','cycle','adjoint')};models=load_module(root,B['row0_models'] if row0 else B['models']);custody=definitions(root,B['custody_functions'],('make_pair','queries','support_tensor'),{})
 s=SimpleNamespace(native=native,heads=heads,adam=adam,steps=steps,models=models,custody=custody,bindings=B,post=SimpleNamespace(own_loss=steps.own_loss),**values)
 s.original=definitions(root,B['original_helpers'],('original_boundary','boundary_episode'),{'torch':native.torch,'definitions':definitions})
 return s

@contextmanager
def adjoint_scope(torch,s):
 saved=s.native.spmm_add;wrapped=s.adjoint.make_constant_adjacency_spmm(torch,saved);s.native.spmm_add=wrapped
 try:yield
 finally:s.native.spmm_add=saved

class HybridAdam:
 """Serializable oneTorchAdam andonefunctionalprivate stateowner,eachparameter once."""
 def __init__(self,shared,model,private,previous):
  self.shared=shared;self.model=model;self.private=list(private);self.previous=previous;named=dict(model.named_parameters());self.param_groups=list(shared.param_groups)+[{'params':[named[k] for k in private],'lr':.001,'functional_private_Adam':True}]
 @property
 def state(self):return {**self.shared.state,**{dict(self.model.named_parameters())[k]:v for k,v in self.previous.items()}}
 def state_dict(self):return {'shared_TorchAdam':self.shared.state_dict(),'private_functional_Adam':copy.deepcopy(self.previous),'private_names':self.private}
 def load_state_dict(self,v):
  if v['private_names']!=self.private:raise ValueError('Exact private role clock required')
  self.shared.load_state_dict(v['shared_TorchAdam']);named=dict(self.model.named_parameters());self.previous={k:{j:x.to(named[k].device) if hasattr(x,'to') else x for j,x in state.items()} for k,state in v['private_functional_Adam'].items()};self.param_groups=list(self.shared.param_groups)+[{'params':[named[k] for k in self.private],'lr':.001,'functional_private_Adam':True}]

def construct(torch,root,B,arm,seed,factor_seed,device):
 s=sources(root,B,arm=='row0_single');dev=torch.device(device)
 if arm=='original_GNNM':model,partition=s.original.original_boundary(torch,root,s,seed,factor_seed,dev);shared=tuple(n for n,p in model.named_parameters());private=()
 else:model,shared,private,partition=s.models.build(s.native,s.heads,arm=arm if arm!='native_M1' else 'capable_single',seed=seed,factor_seed=factor_seed,device=dev)
 live=arm in ('shared_f4','capable_single','untied4','row0_single');params=dict(model.named_parameters());opt=torch.optim.Adam([params[k] for k in shared] if live else list(params.values()),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
 if live:opt=HybridAdam(opt,model,private,s.adam.initial_state(torch,{k:params[k] for k in private}))
 return s,model,opt,shared,private,partition

class Adapter:
 def __init__(self,torch,s,model,opt,shared,private,partition,graph,train,select,arm,seed,geometry_seed,provenance):
  import numpy as np
  self.t,self.np,self.s=torch,np,s;self.model,self.optimizer=model,opt;self.shared,self.private,self.partition=shared,private,partition;self.x=graph['x'].to(next(model.parameters()).device);self.device=self.x.device;self.train=train['positive'];self.select=select;self.arm=arm;self.seed=seed;self.geometry_seed=geometry_seed;self.provenance=copy.deepcopy(provenance);self.initialization={'source':'coldliteralconstructor','trained_donor':False};self.streams=s.steps.DropoutStreams(torch,self.device,seed);self.endpoint_streams=s.geometry.route_streams(geometry_seed,4);self.random_streams=s.geometry.route_streams(geometry_seed,4,control=True);self.full_support=s.custody.support_tensor(torch,self.train,range(len(self.train)),len(self.x),self.device);self._best=None;self._best_step=None;self._misses=0;self._best_for_stop=0.;self.counters={};self.last_draw_receipt=None
 def episode(self,support,inner,outer):
  t,s=self.t,self.s;live=isinstance(self.optimizer,HybridAdam)
  with adjoint_scope(t,s):
   if live:self.optimizer.previous,value=s.steps.episode_step(t,self.model,self.shared,self.private,self.optimizer.shared,self.optimizer.previous,self.x,support,inner,outer,self.streams,live=True,commit='recomputed')
   elif self.arm=='original_GNNM':loss=s.original.boundary_episode(t,s,self.model,self.optimizer,self.x,support,inner,outer,self.streams);value={'task_losses':loss,'native_joint_Adam_updates':3,'functional_forward_calls':12,'model_encoder_forward_count':12}
   else:value=s.steps.ordinary_episode_step(t,self.model,self.optimizer,self.x,support,inner,outer,self.streams)
  commits=2 if live else 3;backward=9 if self.arm=='original_GNNM' else value['functional_forward_calls'];return value,{'optimizer_updates':1 if live else commits,'source_committed_model_maps':commits,'source_committed_shared_TorchAdam_maps':1 if live else commits,'source_committed_private_functional_Adam_maps':int(live),'source_discarded_virtual_private_Adam_maps':int(live),'model_forwards':value['functional_forward_calls'],'source_native_encoder_forwards':value['model_encoder_forward_count'],'route_predictions':value['model_encoder_forward_count'],'backward_calls':backward if live else 12 if self.arm=='original_GNNM' else 3,'source_complete_episodes':1,'supervised_examples':sum(p[0].shape[1]+p[1].shape[1] for p in inner)*2+(outer[0].shape[1]+outer[1].shape[1])*self.model.member_count}
 def train_step(self,step):
  t,s=self.t,self.s;negative,order,episodes=s.custody.make_pair(t,s.geometry,s.cycle,self.train,len(self.x),self.geometry_seed,step-1,64,256,self.endpoint_streams,self.random_streams)
  if len(self.train)!=3870 or len(episodes)!=61:raise ValueError('Source complete3870TRAIN/tail61episodes required')
  draw={'cycle':step,'negative':negative.tolist(),'outer_order':order,'paired_episodes':episodes};self.last_draw_receipt=hashlib.sha256(__import__('json').dumps(draw,sort_keys=True).encode()).hexdigest()
  if hasattr(self,'draw_output'):
   path=Path(self.draw_output)/('cycle_'+str(step)+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True);raw=__import__('gzip').compress(__import__('json').dumps(draw,sort_keys=True).encode(),mtime=0)
   if path.exists() and path.read_bytes()!=raw:raise ValueError('Retain conflicting source integer draw attempt')
   path.write_bytes(raw)
  tick={};losses=[]
  for endpoint,random_episode,kept,description in episodes:
   support=s.custody.support_tensor(t,self.train,kept,len(self.x),self.device);inner,outer=s.custody.queries(t,self.train,negative,endpoint,self.device);v,counts=self.episode(support,inner,outer)
   for k,c in counts.items():tick[k]=tick.get(k,0)+c
   losses.append(v['virtual_outer_objective'] if 'virtual_outer_objective' in v else sum(v['task_losses'])/len(v['task_losses']))
  for k,c in tick.items():self.counters[k]=self.counters.get(k,0)+c
  loss=sum(losses)/len(losses);return {'objective':loss,'ce':loss,'regularization':0.,'operation_counts':tick,'source_draw_sha256':self.last_draw_receipt}
 def score(self,positive,negative,chunk=512):
  t=self.t;outp=[];outn=[]
  with t.no_grad():
   for start in range(0,len(positive),chunk):
    p=positive[start:start+chunk].t().to(self.device);pool=negative[start:start+chunk];n=pool.permute(2,0,1).reshape(2,-1).to(self.device);a,b=self.s.steps.functional_forward(t,self.model,dict(self.model.named_parameters()),self.x,self.full_support,(p,n),training=False);outp.append(a.cpu());outn.append(b.reshape(len(p[0]),pool.shape[1],self.model.member_count).cpu())
  return t.cat(outp),t.cat(outn)
 def validate(self,step):
  p,n=self.score(self.select['positive'],self.select['negative']);r=1+.5*((n.mean(2)>=p.mean(1)[:,None]).sum(1)+(n.mean(2)>p.mean(1)[:,None]).sum(1));mrr=round(float(r.float().reciprocal().mean()),4);return {'selection_metric':mrr,'MRR':mrr,'queries':len(p),'negative_pool_width':n.shape[1],'source_metric_round4':True}
 def observe_validation(self,step,v):
  score=v['selection_metric'];above=self._best is None or score>self._best
  if above:self._best,self._best_step=score,step
  if score>self._best_for_stop:self._best_for_stop,self._misses=score,0
  else:self._misses+=1
  return {'save_selected':above,'stop_requested':self._misses>=11,'best_saved_metric':self._best,'best_saved_step':self._best_step,'stale_steps':self._misses,'checkpoint_eligible':True}
 def capture_state(self):
  t=self.t;return {'provenance':copy.deepcopy(self.provenance),'partition':copy.deepcopy(self.partition),'source_private_dropout':self.streams.state_dict(),'endpoint_sampling':[{k:v.getstate() for k,v in row.items()} for row in self.endpoint_streams],'matched_random_sampling':[{k:v.getstate() for k,v in row.items()} for row in self.random_streams],'torch_cpu':t.get_rng_state().clone(),'torch_cuda':t.cuda.get_rng_state(self.device).clone() if self.device.type=='cuda' else None,'python':random.getstate(),'selector':{'best':self._best,'best_step':self._best_step,'best_for_stop':self._best_for_stop,'misses':self._misses},'counters':copy.deepcopy(self.counters),'last_draw_receipt':self.last_draw_receipt}
 def restore_state(self,v):
  if v['provenance']!=self.provenance or v['partition']!=self.partition:raise ValueError('Exact source role/model ancestry required')
  for i,row in enumerate(self.streams.rows):row.update(copy.deepcopy(v['source_private_dropout'][str(i)]))
  for streams,key in [(self.endpoint_streams,'endpoint_sampling'),(self.random_streams,'matched_random_sampling')]:
   for row,state in zip(streams,v[key]):
    for k,r in row.items():r.setstate(state[k])
  self.t.set_rng_state(v['torch_cpu']);random.setstate(v['python'])
  if v['torch_cuda'] is not None:self.t.cuda.set_rng_state(v['torch_cuda'],self.device)
  self._best,self._best_step,self._best_for_stop,self._misses=[v['selector'][k] for k in ('best','best_step','best_for_stop','misses')];self.counters=copy.deepcopy(v['counters']);self.last_draw_receipt=v['last_draw_receipt']

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config'];arm=c['response_arm']
 if study['dataset']!='citeseer-heart' or c['private_response_source_contract']!=CONTRACT:raise ValueError('Exact adopted literal response source cell required')
 def load(row):return torch.load(bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=[load(c['link_inputs'][k]) for k in ('graph','train','select')]
 if set(graph)!={'x','train_support_edge_index'} or set(train)!={'positive'} or graph['x'].shape!=(3327,3703) or train['positive'].shape!=(3870,2) or select['negative'].shape[1]!=500:raise ValueError('Source compatible custodiedGRAPH/TRAIN/SELECT required')
 state=construct(torch,project_root,c['response_sources'],arm,c['initialization_seed'],c['factor_seed'],device);a=Adapter(torch,*state,graph,train,select,arm,c['initialization_seed'],run['seed'],{'study':study['study_id'],'run_id':run['run_id'],'config':c,'actual_model_seed':c['initialization_seed'],'contract':CONTRACT});a.draw_output=Path(project_root)/c['source_draw_output']/run['run_id'];return a
