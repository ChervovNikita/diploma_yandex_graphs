"""Literal C64 train_epoch/complete VALID/D4 selection with fixed-bank M4."""
import ast,copy,json,math,sys
from types import ModuleType
from contextlib import contextmanager
from pathlib import Path
from .common_native_wrapper import load_module,bound,ast_hash
from . import private_response as prior
from . import link as bridge
CONTRACT={'schema':'literal-cardinality-C64-D4-M4-fixed100-current-role-v1','source':'make_single source0anchor plusprospective seeds1/2;train_epoch100,17full65536 minibatches/drop64940;270->16->16->1 counthead4625;nativeN64depth1;FP32 neural/FP64 complete-countDP','training':'literal native negative sampler and PermIterator;recordmask beforecoalesce;both fullquery laws/draws before completeTRAIN teacher;logsigmoid meanqueriesX4draws +lambda1 positive+negative fullpatternNLL/R; no cap or fallback','selection':'literal score_complete_valid60084positive/100000sharednegative;FirstStrictBest100orderedstates strict officialHits50;mean4rawlogits;no checkpoint0','serving':'D4mean4rawlogits;M4sameD4-selected checkpoint/slotuniforms/draw4/features/alpha1.05;independent actualmu Bernoulli;M4no fit/ownselection;one completeSELECTdiagnostic','randomness_adaptation':'paired seeds0/1/2; modelseed calls andfour counter seed tokens prospectively rebound; source0 law preserved; no HPO; same counterfamily perD4/M4bank',
 'runtime_compatibility':'Torch2.1 CountHead generator Xavier uses exact uniform(-sqrt6/(fan_in+fan_out),+bound) primitive;one ASTcall replacement;sourcebytes unchanged','adaptations':['current completeofficialVALID as SELECT/current raw-onlysealed evaluate;CALabsent/no SELECTreuse','fresh fixed-anchor nativeM1 seed0/ownI4 seed0+1000003body/originalGNNM seed0;controls independent selected','actual completeepoch work/loss trace wrapper around unchanged batch_forward;no change to gradients/draws/optimizer'],'historical_state_inherited':False}

def sources(torch,root,B,seed=0):
 if seed not in (0,1,2):raise ValueError('Declared threepaired C64seed roster required')
 values={};proof={}
 def xavier(value,generator):
  limit=math.sqrt(6./(value.shape[0]+value.shape[1]))
  with torch.no_grad():return value.uniform_(-limit,limit,generator=generator)
 for key in ('graph','counter','density','teacher','model','training','serving','diagnostic_recipe'):
  path=bound(root,B[key]);tree=ast.parse(path.read_text());changes={'seed_tokens':0,'native_seed_calls':0,'generator_Xavier_calls':0}
  for node in ast.walk(tree):
   if key=='counter' and isinstance(node,ast.Constant) and isinstance(node.value,str) and '|seed=0|' in node.value:
    node.value=node.value.replace('|seed=0|',f'|seed={seed}|');changes['seed_tokens']+=1
   if key=='model' and isinstance(node,ast.Call) and ast.unparse(node.func) in ('random.seed','np.random.seed','torch.manual_seed','torch.cuda.manual_seed_all'):
    if len(node.args)!=1 or not isinstance(node.args[0],ast.Constant) or node.args[0].value!=0:raise ValueError('Literal source initializer changed')
    node.args[0].value=seed;changes['native_seed_calls']+=1
   if key=='model' and isinstance(node,ast.Call) and ast.unparse(node)=='nn.init.xavier_uniform_(module.weight, generator=generator)':
    replacement=ast.parse('_generator_Xavier_primitive(module.weight,generator)').body[0].value;node.func,node.args,node.keywords=replacement.func,replacement.args,replacement.keywords;changes['generator_Xavier_calls']+=1
  if key=='counter' and changes['seed_tokens']!=4 or key=='model' and (changes['native_seed_calls']!=4 or changes['generator_Xavier_calls']!=1):raise ValueError('Exact source seed/API adaptation roster required')
  name='cardinality_'+key+'_'+B[key]['sha256'][:12]+'_seed'+str(seed);module=ModuleType(name);module.__file__=str(path);sys.modules[name]=module;module.__dict__['_generator_Xavier_primitive']=xavier
  with prior.aliases({'cardinality_'+k:v for k,v in values.items()}):exec(compile(ast.fix_missing_locations(tree),str(path)+'::declared_seed_API_bridge','exec'),module.__dict__)
  values[key]=module;proof[key]={'source_sha256':B[key]['sha256'],'bound_AST_sha256':ast_hash(tree),'changes':changes,'seed0_author_anchor':seed==0}
 native,utils=bridge.native_modules(root,B['native']);return values,native,utils,proof

class Adapter:
 def __init__(self,torch,numpy,model,optimizer,modules,utils,sampler,data,teacher,provenance):
  self.t,self.np,self.model,self.optimizer=torch,numpy,model,optimizer;self.modules,self.utils,self.sampler=modules,utils,sampler;self.data,self.teacher=data,teacher;self.device=data['x'].device;self.provenance=provenance;self.initialization={'source':'literalcoldC64_declared_seed','trained_donor':False};self._rng=bridge.rng_state(torch,numpy);self.selector=modules['serving'].FirstStrictBest();self._epoch=0;self.work=[];self.last_epoch_receipt=None
 @contextmanager
 def scope(self,training=False):
  outer=bridge.rng_state(self.t,self.np);bridge.restore_rng(self.t,self.np,self._rng)
  try:
   with prior.aliases({'cardinality_'+k:v for k,v in self.modules.items()}):yield
  finally:
   if training:self._rng=bridge.rng_state(self.t,self.np)
   bridge.restore_rng(self.t,self.np,outer)
 def train_step(self,epoch):
  if epoch!=self._epoch+1:raise ValueError('Literal completeorderedepoch required')
  m=self.modules['training'];original=m.batch_forward;losses=[]
  def traced(*args,**kwargs):
   value=original(*args,**kwargs);losses.append([float(x.detach()) for x in value[:3]]);return value
  with self.scope(True):
   m.batch_forward=traced
   try:work=m.train_epoch(self.model,self.optimizer,self.data,self.utils,self.sampler,self.teacher,epoch=epoch)
   finally:m.batch_forward=original
  if len(losses)!=17 or work['optimizer_steps']!=17:raise ValueError('Complete source17batch optimizerclock required')
  self._epoch=epoch;self.last_epoch_receipt=copy.deepcopy(work);self.work.append({'epoch':epoch,'stream':work['stream'],'wall_seconds':work['wall_seconds']})
  if hasattr(self,'receipt_output'):
   q=Path(self.receipt_output)/('epoch_'+str(epoch)+'.json');q.parent.mkdir(parents=True,exist_ok=True)
   with q.open('x') as f:json.dump(work,f,sort_keys=True,allow_nan=False)
  means=[sum(row[i] for row in losses)/17 for i in range(3)]
  return {'objective':means[2],'ce':means[0],'regularization':means[1],'operation_counts':{'optimizer_updates':17,'backward_calls':17,'encoder_calls':17,'positive_records':1114112,'negative_records':1114112,'dropped_positive_tail':64940,'source_D4_target_decodes':136,'source_complete_count_classes':sum(row['all_count_classes'] for row in work['batch_work']),'source_complete_residual_slots':sum(row['residual_slots'] for row in work['batch_work'])},'source_epoch_work':work}
 def metric(self,p,n):
  from ogb.linkproppred import Evaluator
  e=Evaluator.__new__(Evaluator);e.name='ogbl-collab';e.eval_metric='hits@50';e.K=50
  return self.modules['serving'].strict_official_hits50(e,p,n)
 def validate(self,epoch):
  with self.scope():p,n,work=self.modules['serving'].score_complete_valid(self.model,self.data,self.utils,route='D4')
  score=self.metric(p,n);ce=float(-self.t.nn.functional.logsigmoid(p).mean()-self.t.nn.functional.logsigmoid(-n).mean())
  return {'objective':ce,'ce':ce,'regularization':0.,'selection_metric':score,'hits@50':score,'source_complete_VALID_work':work}
 def observe_validation(self,epoch,value):
  better=self.selector.consider(epoch,value['selection_metric']);return {'save_selected':better,'stop_requested':False,'best_saved_metric':self.selector.best_value,'best_saved_step':self.selector.best_epoch,'full_budget':True,'states':self.selector.states}
 def predict_pairs(self,pairs,route='D4'):
  if route not in ('D4','M4'):raise ValueError('Literal declared servingroute required')
  if route=='M4' and not hasattr(self,'_M4_authority'):raise ValueError('Closed own D4-selected-bank authority required for M4')
  output=[];self.model.eval()
  with self.scope(),self.t.no_grad():
   graph=self.modules['graph'].Graph.from_pairs(self.data['pairs'],len(self.data['x']));h=self.model.encode(self.data['x'],graph)
   for start in range(0,len(pairs),131072):
    q=pairs[start:start+131072].to(self.device);z,detail,draw=self.model.query_forward(h,graph,q,self.modules['counter'].CompletionKeys('eval'),route=route);output.append(z.T.cpu())
  return self.t.cat(output,1)
 def complete_M4_SELECT(self,bank,digest):
  with self.scope():p,n,work=self.modules['serving'].score_complete_valid(self.model,self.data,self.utils,route='M4',selected_bank=bank,loaded_checkpoint_sha256=digest)
  return {'hits@50':self.metric(p,n),'source_work':work,'extra_fit_count':0,'selected_bank':bank.__dict__}
 def fixed_diagnostic_routes(self,bank,digest):
  if bank.checkpoint_sha256!=digest or not hasattr(self,'_M4_authority') or self._M4_authority!=bank:raise ValueError('Own closed selectedbank diagnostic authority required')
  payload=self.diagnostic;graph=self.modules['graph'].Graph(payload['nodes'],*[payload[k].to(self.device) for k in ('row','col','rowptr')]);queries=[q.to(self.device) for q in payload['queries']];result={};self.model.eval()
  with self.scope(),self.t.no_grad():
   h=self.model.encode(self.data['x'],graph)
   for route in ('D4','M4'):
    scores=[];work=[]
    for q in queries:
     logits,detail,draw=self.model.query_forward(h,graph,q,self.modules['counter'].CompletionKeys('eval'),route=route);scores.append(logits.T.cpu());work.append(draw['work'])
    result[route]={'logits':scores,'work':work}
  return result

 def capture_state(self):return {'provenance':copy.deepcopy(self.provenance),'fit_rng':copy.deepcopy(self._rng),'outer_rng':bridge.rng_state(self.t,self.np),'epoch':self._epoch,'selector':copy.deepcopy(self.selector.__dict__),'work':copy.deepcopy(self.work),'last_epoch_receipt':copy.deepcopy(self.last_epoch_receipt)}
 def restore_state(self,v):
  if v['provenance']!=self.provenance:raise ValueError('LiteralC64 currentrole coldancestry required')
  self._rng=copy.deepcopy(v['fit_rng']);self._epoch=v['epoch'];self.selector.__dict__.update(copy.deepcopy(v['selector']));self.work=copy.deepcopy(v['work']);self.last_epoch_receipt=copy.deepcopy(v['last_epoch_receipt']);bridge.restore_rng(self.t,self.np,v['outer_rng'])

def make_adapter(*,torch,project_root,study,run,device):
 import numpy
 from torch_geometric.utils import negative_sampling
 from structural_cardinality_semantics import validate_cardinality_run
 validate_cardinality_run(run);c=run['config'];B=c['cardinality_sources'];m,native,utils,digest=sources(torch,project_root,B,run['seed'])
 inp={k:torch.load(bound(project_root,row),map_location='cpu',weights_only=True) for k,row in c['link_inputs'].items()};g,tr,v=inp['graph'],inp['train'],inp['select']
 if study['dataset']!='ogbl-collab' or set(g)!={'x','raw_edge_index'} or set(tr)!={'positive'} or set(v)!={'positive','negative'}:raise ValueError('Complete currentrawNCNC rolecustody required')
 data={'x':g['x'].to(device),'raw_edge_index':g['raw_edge_index'].to(device),'pairs':tr['positive'].to(device),'valid_positive':v['positive'].to(device),'valid_negative':v['negative'].to(device)}
 data['digests']={'train_records':m['teacher'].tensor_sha(data['pairs'])}
 teacher=m['teacher'].TrainObservationTeacher.from_authenticated_train(data['pairs'],len(data['x']),expected_train_records_sha256=data['digests']['train_records'])
 diagnostic_root=Path(project_root)/c['source_diagnostic_output']/run['run_id'];payload_path=diagnostic_root/'TRAIN_DIAGNOSTIC.pt';seal_path=diagnostic_root/'SEAL.json'
 identity={'run_id':run['run_id'],'seed':run['seed'],'TRAIN':c['link_inputs']['train'],'GRAPH':c['link_inputs']['graph'],'cardinality_sources':B,'model_values_read':False}
 if diagnostic_root.exists():
  seal=json.loads(seal_path.read_text())
  if seal['identity']!=identity:raise ValueError('Retain conflicting pre-model diagnostic source custody')
  from .cardinality_endpoints import receipt
  if receipt(project_root,payload_path)!=seal['payload']:raise ValueError('Pre-model diagnostic bytes changed')
  diagnostic=torch.load(payload_path,map_location='cpu',weights_only=True)
 else:
  diagnostic_root.mkdir(parents=True)
  with prior.aliases({'cardinality_'+k:v for k,v in m.items()}):dg,dq,ds,dl,dr=m['diagnostic_recipe'].prepare_diagnostic_support(data,negative_sampling,teacher)
  diagnostic={'nodes':dg.nodes,'row':dg.row.cpu(),'col':dg.col.cpu(),'rowptr':dg.rowptr.cpu(),'queries':[q.cpu() for q in dq],'source_receipt':dr};torch.save(diagnostic,payload_path)
  from .cardinality_endpoints import receipt
  seal={'identity':identity,'payload':receipt(project_root,payload_path),'source_receipt':dr,'complete':True,'model_constructors':0,'optimizer_objects':0,'scores':0}
  with seal_path.open('x') as f:json.dump(seal,f,sort_keys=True,allow_nan=False)
 outer=bridge.rng_state(torch,numpy)
 try:model,opt=m['model'].make_single(native,device);rng=bridge.rng_state(torch,numpy)
 finally:bridge.restore_rng(torch,numpy,outer)
 a=Adapter(torch,numpy,model,opt,m,utils,negative_sampling,data,teacher,{'run_id':run['run_id'],'config':c,'contract':CONTRACT,'model_seed':run['seed'],'generator_API_AST_sha256':digest,'teacher':teacher.receipt()});a._rng=rng;a.diagnostic=diagnostic;a.receipt_output=Path(project_root)/c['source_epoch_receipt_output']/run['run_id'];return a
