"""Literal v3 typed-label source composition; numerical imports are lazy."""
import ast,copy,hashlib,importlib,importlib.util,json,random,sys,types
from pathlib import Path
from types import SimpleNamespace,ModuleType
try:from ..source_utils import bound,load,native_runtime
except ImportError:from source_utils import bound,load,native_runtime
CONTRACT={'schema':'literal-typed-label-context6-SeHGNN-source200-v1','native':'exact IMDB SeHGNN512/512/classes5;25feature+12labelchannels;sixfactor sites;no graph/learned-width shrink','context':'sourceTrainOnly/twofrozen complement halves;12diagonalremovednative positive channels;3masses+15conditional signed balances;local/global18D;each query label excluded','operator':'sourceContextGrouped feature_projection.0/.4;biasoutsidefactor;Uzero;HdedicatedhashedXavier;shared/untiedwidth32,singlewidth128;localadd base+delta,mul base+delta*centered','training':'literal driver.Session.train_epoch;oneshuffledTRAIN order eachtarget once/member;streamedhalf ownBCE;sharedmean/untiedsum;oneAdam.001 WD0,200cap strictfirst rawpooledBCEmin stale>50;no checkpoint0','AMP':'native enabledCUDA masterGradScaler percondition acrossordered3source role seed records;separatefresh selected reconstruction scaler;terminalmaster state iscarried,selectedstate isnottrainingdonor','serving':'FP32/twoexcludedtargetcontexts meanrawcontextlogits thenmeanrawmemberlogits;TRAINdiagnostics complementonly;all sourcecosts retained','reference':'ordinaryownselectedI4 plusseparatewidth32same-information ownselectedI4;fourdisjoint bodyinitializations;M1body0reuse;sourceuntiedmatchedinit4 andsingle128 arenottheseI4bodies;originalactualtypedBEBlock comparison','adaptations':['currentcustodiedIMDBTRAIN685/SELECT274/CAL137/evaluate275 split0/1/2;sourceoptimizer/context/generator/member seedspecs1/2/3 retained;oldnative80percent role assertion replacedwithnewrolecustody,explicitroleadaptation','source200/p50 stopping unchanged;newCAL500/raw endpoints isolated','sameinformationI4width32 explicitreferenceconstructor,fromsourceREFERENCE_REQUIREMENTS;additionalsourceoriginaltypedboundary comparator','engineeringCPU constructor/disabledGradScaler guard only;nohistoricalsource runtime/release/quality admission'],'historical_state_inherited':False}
CONDITIONS=('shared_own_pair4','shared_local_mul4','shared_local_add4','shared_global_mul4','single_local_mul1','untied_local_mul4','native_M1','same_information_local_mul1_width32','original_GNNM')
SEEDS=({'role_seed':1,'optimizer_seed':1,'context_seed':104729,'generator_seed':190001,'member_rng_seeds':[210001,210002,210003,210004]}, {'role_seed':2,'optimizer_seed':2,'context_seed':130363,'generator_seed':190002,'member_rng_seeds':[220001,220002,220003,220004]}, {'role_seed':3,'optimizer_seed':3,'context_seed':155921,'generator_seed':190003,'member_rng_seeds':[230001,230002,230003,230004]})

def source_package(root,B,engineering=False,width32=False):
 for row in B['typed_source_files'].values():bound(root,row)
 folder=Path(root)/B['typed_package'];name='typed_context_v3_'+B['typed_source_files']['model.py']['sha256'][:12]
 if name not in sys.modules:
  p=ModuleType(name);p.__path__=[str(folder)];p.__package__=name;sys.modules[name]=p
 modules={k:importlib.import_module(name+'.'+k) for k in ('caps','context','provider','model','objective','driver')}
 if engineering:
  driver=modules['driver'];tree=ast.parse(bound(root,B['typed_source_files']['driver.py']).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session');changes=[]
  for n in ast.walk(cls):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='Exact native CUDA/master AMP interface':
    n.args[0]=ast.BoolOp(op=ast.Or(),values=[n.args[0],ast.parse("rt['device'].type=='cpu' and isinstance(scalar,torch.cuda.amp.GradScaler) and not scalar.is_enabled()",mode='eval').body]);changes.append('syntheticCPU_constructor_disabled_GradScaler_guard')
  if changes!=['syntheticCPU_constructor_disabled_GradScaler_guard']:raise ValueError('Exact isolated CPU constructor guard anchor required')
  env=dict(driver.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(folder/'driver.py')+'::syntheticCPU','exec'),env);modules['Session']=env['Session'];modules['CPU_AST_changes']=changes
 else:modules['Session']=modules['driver'].Session
 if width32:
  model=modules['model'];tree=ast.parse(bound(root,B['typed_source_files']['model.py']).read_text());fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_bank');hits=0
  for n in ast.walk(fun):
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='hidden' for t in n.targets) and isinstance(n.value,ast.IfExp) and isinstance(n.value.body,ast.Constant) and n.value.body.value==128:n.value.body.value=32;hits+=1
  if hits!=1:raise ValueError('Only explicit source reference single generator128to32 may change')
  env=dict(model.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[])),str(folder/'model.py')+'::REFERENCE_REQUIREMENTS_width32','exec'),env);modules['reference_build_bank']=env['build_bank']
 return SimpleNamespace(**modules)

class SourceCosts:
 def __init__(self,rt):self.rt=rt;self.rows=[]
 def measure(self,scope,gpu=False):
  from contextlib import contextmanager
  import time
  @contextmanager
  def manager():
   start=time.monotonic();row={'scope':scope}
   if gpu and self.rt['device'].type=='cuda':self.rt['torch'].cuda.synchronize(self.rt['device'])
   try:yield row;row['status']='complete'
   except BaseException as e:row.update(status='failed',error=str(e));raise
   finally:
    if gpu and self.rt['device'].type=='cuda':self.rt['torch'].cuda.synchronize(self.rt['device'])
    row['seconds']=time.monotonic()-start;self.rows.append(row)
  return manager()

def plain_bank(torch,models,shared,condition):
 class Bank(torch.nn.Module):
  def __init__(self):
   super().__init__();self.members=models;self.shared=shared;self.contextual=False;self.condition=condition
  def forward_member(self,m,batch,features,labels,context_field,mask=None):return self.members[m](batch,features,labels,mask=mask)
  def slow_parameters(self):return tuple(self.parameters()) if shared else ()
  def private_parameters(self,m):return () if shared else tuple(self.members[m].parameters())
  def verify_ownership(self):
   if len({id(p) for p in self.parameters()})!=len(tuple(self.parameters())):raise ValueError('One deduplicated native/original bank')
   return {'members':len(self.members),'shared_native_weights':shared}
  def verify_shared_state_dict(self,state):
   if shared:
    params=dict(self.members[0].named_parameters())
    for key in params:
     vals=[state['members.'+str(m)+'.'+key] for m in range(len(self.members))]
     if any(not torch.equal(v,vals[0]) for v in vals[1:]):raise ValueError('Shared original learned aliases must agree')
 return Bank()

def definition(root,row,names,env):
 nodes=[copy.deepcopy(n) for n in ast.parse(bound(root,row).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Exact source helper roster')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),row['path']+'::literalAST','exec'),env);return SimpleNamespace(**env)

def construct(rt,engine,mods,inputs,scalar,costs,condition,spec,identity,caps,root,B):
 t=rt['torch'];driver=mods.driver
 if condition not in ('native_M1','original_GNNM'):
  constructor=mods.Session
  if condition=='same_information_local_mul1_width32':
   # Change only the constructor function looked up by the exact source Session.
   env=dict(constructor.__init__.__globals__);env['build_bank']=mods.reference_build_bank;original=constructor.__init__;init=types.FunctionType(original.__code__,env,original.__name__,original.__defaults__,original.__closure__)
   class ReferenceSession(constructor):pass
   ReferenceSession.__init__=init;constructor=ReferenceSession;condition_for_source='single_local_mul1'
  else:condition_for_source=condition
  session=constructor(rt,engine,inputs,scalar,costs,condition_for_source,spec,identity,caps);session.condition=condition;return session
 # Source training/evaluation methods remain literal; only the two required
 # comparator factories differ from source six-site conditional constructors.
 s=object.__new__(mods.Session);s.rt,s.engine,s.inputs,s.scalar,s.costs=rt,engine,inputs,scalar,costs;s.condition,s.spec,s.identity,s.caps=condition,dict(spec),identity,caps;s.counters={k:0 for k in ('epochs','TRAIN_native_calls','TRAIN_backward_calls','TRAIN_query_rows','actual_Adam_steps','serving_native_calls','serving_query_rows')};s.initial_checks=[];s.member_rng=[];s.cache_guard=driver.cache_stamp(inputs);prototype=engine.make_model(rt,inputs,costs);s.native_initial_digest=driver.state_digest(t,prototype)
 if condition=='original_GNNM':
  helper=definition(root,B['original_helper'],('original_boundary',),{'ast':ast,'copy':copy,'hashlib':hashlib,'types':types,'bound':bound});models,s.boundary_binding=helper.original_boundary(rt,B,root,prototype);s.bank=plain_bank(t,models,True,condition)
 else:s.bank=plain_bank(t,t.nn.ModuleList([prototype]),False,condition)
 s.bank.to(rt['device']);s.all_parameters=tuple(s.bank.parameters());s.optimizer=t.optim.Adam(s.all_parameters,lr=.001,weight_decay=0.);s.verify_optimizer();caller=engine.capture_rng(rt['numpy'],t)
 try:
  for seed in spec['member_rng_seeds'][:len(s.bank.members)]:rt['native'].set_random_seed(seed);s.member_rng.append(engine.capture_rng(rt['numpy'],t))
 finally:engine.restore_rng(rt['numpy'],t,caller)
 return s

def pack_rng(value):
 if isinstance(value,dict):return {k:pack_rng(v) for k,v in value.items()}
 if isinstance(value,list):return [pack_rng(v) for v in value]
 if isinstance(value,tuple):return tuple(pack_rng(v) for v in value)
 if type(value).__module__=='numpy' and hasattr(value,'tolist'):return {'numpy_uint32_array':value.tolist()}
 return value

def unpack_rng(np,value):
 if isinstance(value,dict):
  if set(value)=={'numpy_uint32_array'}:return np.asarray(value['numpy_uint32_array'],dtype='uint32')
  return {k:unpack_rng(np,v) for k,v in value.items()}
 if isinstance(value,list):return [unpack_rng(np,v) for v in value]
 if isinstance(value,tuple):return tuple(unpack_rng(np,v) for v in value)
 return value

def seed_inputs(rt,mods,record,root,fixture=None):
 t=rt['torch']
 if fixture is None:
  train=t.load(bound(root,record['train']),map_location='cpu',weights_only=True);select=t.load(bound(root,record['select']),map_location='cpu',weights_only=True);features=t.load(bound(root,record['features']),map_location='cpu',weights_only=True);cache=t.load(bound(root,record['contexts']),map_location='cpu',weights_only=True)
 else:train,select,features,cache=[fixture[k] for k in ('train','select','features','contexts')]
 plan=mods.context.HalfPlan(**cache['plan']);views=tuple(mods.context.HalfView(**v) for v in cache['views']);paired=mods.context.PairedViews(plan,views,tuple(cache['support_audit']))
 if tuple(train['ids'].tolist())!=plan.train_ids or set(plan.halves[0])&set(plan.halves[1]) or set(plan.halves[0])|set(plan.halves[1])!=set(plan.train_ids) or t.isin(train['ids'],select['ids']).any():raise ValueError('Custodied TRAIN,excluded-query halves anddisjointSELECT required')
 if sorted(features)!=rt['protocol']['feature_paths'] or set(views[0].label_features)!=set(mods.context.LABEL_PATHS):raise ValueError('Literal full25/12 channels')
 tr=mods.context.TrainOnly(tuple(train['ids'].tolist()),train['y'],plan.custody_sha256);valid=mods.provider.ValidationOnly(tuple(select['ids'].tolist()),select['y'],hashlib.sha256(json.dumps(record.get('select',{'synthetic':True}),sort_keys=True).encode()).hexdigest())
 # Preserve the native preparation shuffle consumption; the new role memberships
 # are independently custodied instead of asserted to be the historical80% split.
 ids=rt['numpy'].asarray(sorted(tr.ids+valid.ids),dtype='int64');rt['numpy'].random.shuffle(ids)
 loader=t.utils.data.DataLoader(rt['numpy'].asarray(tr.ids,dtype='int64'),batch_size=10000,shuffle=True,drop_last=False)
 return mods.provider.NativeSeedInputs(record.get('source_seed_spec',{'optimizer_seed':1})['optimizer_seed'],features,{k:None for k in mods.context.LABEL_PATHS},{k:v.shape[-1] for k,v in features.items()},tr,valid,paired,loader,{'current_role_receipts':{k:record.get(k) for k in ('train','select','contexts')},'context_identity':plan.identity_sha256},{k:list(v.shape) for k,v in features.items()})

class Adapter:
 def __init__(self,rt,engine,mods,session,provenance,cursor_output=None):
  self.rt,self.engine,self.mods,self.session=rt,engine,mods,session;self.t=rt['torch'];self.model,self.optimizer=session.bank,session.optimizer;self.provenance=copy.deepcopy(provenance);self.initialization={'source':'coldliteralSeHGNN','trained_donor':False,'native_initial_digest':session.native_initial_digest,'source_binding':provenance};self.cursor_output=cursor_output;self._best=None;self._best_step=None;self._misses=0;self.row_logits_fn=self.predict_rows
  old=self.model.load_state_dict
  def checked(state,*args,**kwargs):self.model.verify_shared_state_dict(state);return old(state,*args,**kwargs)
  self.model.load_state_dict=checked
 def train_step(self,step):
  s=self.session
  if step!=s.counters['epochs']+1:raise ValueError('Exact complete source epoch order required')
  before=copy.deepcopy(s.counters);own=s.train_epoch();delta={k:v-before[k] for k,v in s.counters.items()};maps=delta['actual_Adam_steps'];members=len(s.bank.members)
  objective=own['mean_member_own_BCE'] if s.bank.shared else sum(own['member_own_BCE'])
  return {'objective':objective,'ce':own['mean_member_own_BCE'],'regularization':0.,'source_own_BCE':own,'operation_counts':{'optimizer_updates':maps,'model_forwards':delta['TRAIN_native_calls'],'backward_calls':delta['TRAIN_backward_calls'],'route_predictions':members,'supervised_examples':delta['TRAIN_query_rows'],'source_optimizer_attempts':1,'source_AMP_skipped_optimizer_maps':1-maps,'source_native_complete_cycles':1,'source_native_AMP_enabled':int(s.scalar.is_enabled())}}
 def validate(self,step):
  s=self.session;before=copy.deepcopy(s.counters);scores,outputs=s.evaluate();loss=scores['VALID']['BCE'];members=len(s.bank.members)
  return {'objective':loss,'ce':loss,'regularization':0.,'selection_metric':loss,'pooled_BCE':loss,'source_scores':scores,'examples':len(s.inputs.validation.ids),'label_events':5*len(s.inputs.validation.ids),'operation_counts':{'optimizer_updates':0,'model_forwards':s.counters['serving_native_calls']-before['serving_native_calls'],'backward_calls':0,'route_predictions':members,'supervised_examples':0,'source_SELECT_and_TRAIN_diagnostic_query_rows':s.counters['serving_query_rows']-before['serving_query_rows']}}
 def observe_validation(self,step,v):
  better=self._best is None or v['selection_metric']<self._best
  if better:self._best,self._best_step,self._misses=v['selection_metric'],step,0
  else:self._misses+=1
  return {'save_selected':better,'stop_requested':self._misses>50,'checkpoint_eligible':True,'best_saved_metric':self._best,'best_saved_step':self._best_step,'stale_steps':self._misses}
 def predict_rows(self,ids):
  t,s=self.t,self.session;ids=tuple(ids.detach().cpu().tolist());inputs=s.inputs
  if not ids or len(set(ids))!=len(ids) or set(ids)&set(inputs.train.ids):raise ValueError('Unique heldoutquery IDs disjoint fromTRAIN')
  caller=self.engine.capture_rng(self.rt['numpy'],t);member_before=self.engine.cpu_tree(t,s.member_rng);modes=[(m,m.training) for m in self.model.modules()];out=[]
  try:
   self.model.eval()
   with t.no_grad(),t.cuda.amp.autocast(enabled=False):
    for m in range(len(s.bank.members)):
     contexts=[]
     for view in inputs.paired.views:
      batch,features,labels,field=self.mods.context.half_inputs(t,view,ids,inputs.feats,s.condition=='shared_global_mul4',s.caps);z=s.bank.forward_member(m,batch.to(self.rt['device']),{k:v.to(self.rt['device']) for k,v in features.items()},{k:v.to(self.rt['device']) for k,v in labels.items()},field);contexts.append(z.float())
     out.append(t.stack(contexts).mean(0))
   return t.stack(out)
  finally:
   for m,mode in modes:m.training=mode
   self.engine.restore_rng(self.rt['numpy'],t,caller)
   if not self.engine.exact(t,s.member_rng,member_before):raise ValueError('Serving cannot change acquisition memberRNG')
 def export_terminal_master_scaler(self,step):
  if self.cursor_output is None:raise ValueError('Only own registered sourcefit may export terminalmaster scaler')
  from ..source_utils import receipt
  q=Path(self.cursor_output)/'terminal_master_scaler.json';value={'schema':'typed-context-condition-master-scaler-terminal-v1','fit_id':self.provenance['run_id'],'completed_source_cycles':step,'condition':self.session.condition,'source_seed':self.provenance['source_seed'],'body':self.provenance['body'],'scalar_state':self.session.scalar.state_dict(),'source_contract':CONTRACT,'cold_model_or_Adam_state_included':False};q.parent.mkdir(parents=True,exist_ok=True);temp=q.with_suffix('.json.tmp');temp.write_text(json.dumps(value,sort_keys=True)+'\n');temp.replace(q);return receipt(self.provenance['project_root'],q)
 def capture_state(self):
  s=self.session;return {'provenance':copy.deepcopy(self.provenance),'context_identity':s.inputs.paired.plan.identity_sha256,'role_binding':s.inputs.role_binding,'member_rng':pack_rng(self.engine.cpu_tree(self.t,s.member_rng)),'master_rng':pack_rng(self.engine.capture_rng(self.rt['numpy'],self.t)),'scaler':self.engine.cpu_tree(self.t,s.scalar.state_dict()),'counters':copy.deepcopy(s.counters),'initial_checks':copy.deepcopy(s.initial_checks),'selector':{'best':self._best,'step':self._best_step,'misses':self._misses}}
 def restore_state(self,v):
  s=self.session
  if v['provenance']!=self.provenance or v['context_identity']!=s.inputs.paired.plan.identity_sha256 or v['role_binding']!=s.inputs.role_binding:raise ValueError('Exact typedsource model/context/ownfit authority required')
  s.member_rng=unpack_rng(self.rt['numpy'],copy.deepcopy(v['member_rng']));s.scalar.load_state_dict(v['scaler']);s.counters=copy.deepcopy(v['counters']);s.initial_checks=copy.deepcopy(v['initial_checks']);self._best,self._best_step,self._misses=[v['selector'][k] for k in ('best','step','misses')];self.engine.restore_rng(self.rt['numpy'],self.t,unpack_rng(self.rt['numpy'],v['master_rng']));s.verify_optimizer()

def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
 c=run['config'];root=Path(project_root).resolve();B=c['typed_context_sources'];arm=c['typed_context_arm'];engineering=bool(execution_context and execution_context.get('engineering_synthetic'));endpoint=bool(execution_context and execution_context.get('endpoint_reconstruction'))
 if arm not in CONDITIONS or c['typed_context_source_contract']!=CONTRACT:raise ValueError('Exact declared typedcontext source condition')
 if engineering:
  if study['stage']!='engineering' or str(device)!='cpu':raise ValueError('Only explicitly syntheticCPU source composition')
 elif not execution_context or execution_context['registry'].run(study['study_id'],run['run_id'])['state']!=('complete' if endpoint else 'running'):raise ValueError('Current registered sourcefit orselectedreconstruction required')
 rt,engine=native_runtime(root,B,device);mods=source_package(root,B,engineering,arm=='same_information_local_mul1_width32');spec=copy.deepcopy(c['source_seed_spec']);body=run['body'];actual_seed=c['initialization_seed'];spec['optimizer_seed']=actual_seed
 if arm in ('native_M1','same_information_local_mul1_width32'):spec['generator_seed']+=1009*body;spec['member_rng_seeds']=[spec['member_rng_seeds'][0]+1009*body]
 rt['native'].set_random_seed(actual_seed);inputs=seed_inputs(rt,mods,c['typed_context_inputs'],root,execution_context.get('fixture') if engineering else None);scalar=torch.cuda.amp.GradScaler(enabled=rt['device'].type=='cuda')
 previous=c.get('master_scaler_parent_fit_id')
 if previous and not engineering and not endpoint:
  row=execution_context['registry'].run(study['study_id'],previous);authority=(row['result'] or {}).get('terminal_master_scaler');parent=row['contract']
  if row['state']!='complete' or parent['config']['typed_context_arm']!=arm or parent['seed']!=run['seed']-1 or parent['body']!=body or authority is None:raise ValueError('Exact preceding condition/body terminalmaster scaler custody')
  payload=json.loads(bound(root,authority).read_text())
  if payload['fit_id']!=previous or payload['completed_source_cycles']!=row['result']['completed_steps'] or payload['source_contract']!=CONTRACT or payload['cold_model_or_Adam_state_included'] is not False:raise ValueError('Terminalmaster-only state;no selectedmodel/Adam transfer')
  scalar.load_state_dict(payload['scalar_state'])
 costs=SourceCosts(rt);identity=hashlib.sha256(json.dumps({'run_id':run['run_id'],'config':c},sort_keys=True).encode()).hexdigest();caps=mods.caps.Caps(source_bound=True,model=True,data=True,runtime=True,scientific=not engineering and not endpoint,root_review_sha256=identity);s=construct(rt,engine,mods,inputs,scalar,costs,arm,spec,identity,caps,root,B);cursor=None if engineering or endpoint else execution_context['output_root']/study['study_id']/run['run_id'];return Adapter(rt,engine,mods,s,{'run_id':run['run_id'],'study':study['study_id'],'config':c,'source_seed':run['seed'],'body':body,'actual_model_seed':actual_seed,'actual_seed_spec':spec,'project_root':str(root)},cursor)
