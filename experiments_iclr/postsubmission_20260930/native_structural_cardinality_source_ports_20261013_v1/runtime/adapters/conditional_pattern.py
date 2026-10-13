"""Exact native PatternTwin + source J_K/J_K_sep/W_K complete TRAIN auxiliary."""
import copy
from contextlib import contextmanager
from pathlib import Path
from . import private_response as prior
from . import link as bridge
from .common_native_wrapper import load_module,bound
CONTRACT={'schema':'literal-native-conditional-pattern-JK-JKsep-WK-fixed100-v1','source':'PatternTwin make_pattern/v5 native pattern_forward;43790param;existing private clamped completion/count-free meanrawlogit serving unchanged','auxiliary':'v3 training_pattern_losses exactJ_K/J_K_sep/W_K;TrainPatternLabels complete_TRAIN_observation_membership;full raggedleft/right eta t=scale*(score-offset)+log(pt);denominator max(nL+nR,1);positive all-querymean+negative all-querymean coefficient1','training':'source native fullnegative draw/PermIterator17full65536/drop64940;maskTRAINrecords beforecoalesce;source mainBCE +oneauxiliary;oneAdam backward/map;native scorer checkpointautograd forauxiliary only','selection':'fixed100epochs complete officialSELECT60084positive/100000sharednegative strictHits50 everyepoch firstexacttie;source0anchor plusprospective paired1/2;noHPO/nohistoricalstate','bucket':'exact_cb_support_bucket_v2 is execution-equivalent sourcealias;no additionalscientific fits/independentobjective;inherited closed support arithmetic proofs retained','adaptations':['currentGRAPH/TRAIN/SELECT and raw-onlyopaqueevaluate/noCAL','fresh fixed-anchor nativeM1,I4,originalGNNM controls sharedwithC64 withinthisnewstudy;same recipe/role/selector/budget/seed','source2selectedcheckpoint/fullSELECT replay companions retained aspostselection readout tasks'],'historical_state_inherited':False}

def sources(root,B):
 m={}
 for name in ('graph_ops','prototype','pilot_common','pilot_data','pilot_model','design_spec','pattern_model','pattern_teacher','pilot_evaluate','conditional_loss'):
  with prior.aliases(m):m[name]=load_module(root,B[name])
 return m

def batch_step(torch,m,model,optimizer,data,teacher,negative,ids,arm):
 optimizer.zero_grad(set_to_none=True);graph=m['graph_ops'].Graph.mask_train_batch(data['pairs'],ids,len(data['x']));h=model.encoder(data['x'],graph);records=[]
 for query in (data['pairs'][ids],negative[ids]):
  logits,detail=model.decoder.pattern_forward(h,graph,query,auxiliary_grad=True);rows,labels=teacher.labels(query,detail['neighbors']);nl=len(detail['neighbors'].left[0]);labels=labels.to(detail['t'].dtype).detach();authority=m['conditional_loss'].TrainPatternLabels(labels,'complete_TRAIN_observation_membership')
  values=m['conditional_loss'].training_pattern_losses(detail['t'][:,:nl],detail['t'][:,nl:],detail['neighbors'].left[0],detail['neighbors'].right[0],authority,len(query));records.append({'query':query,'logits':logits,'detail':detail,'labels':labels,'values':values})
 main=-torch.nn.functional.logsigmoid(records[0]['logits']).mean()-torch.nn.functional.logsigmoid(-records[1]['logits']).mean();auxiliary=sum(row['values'][arm].mean() for row in records);total=main+auxiliary
 if not torch.isfinite(total):raise FloatingPointError('Nonfinite source native/conditional auxiliary')
 total.backward()
 for name,p in model.named_parameters():
  if not torch.isfinite(p).all() or p.grad is None and 'ptlin.' not in name or p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Incomplete/nonfinite source conditional nativegradient:'+name)
 optimizer.step()
 for p in model.parameters():
  if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite source conditional parameter')
 for state in optimizer.state.values():
  if any(torch.is_tensor(x) and not torch.isfinite(x).all() for x in state.values()):raise FloatingPointError('Nonfinite source conditional Adamstate')
 return {'main_loss':float(main.detach()),'selected_auxiliary_loss':float(auxiliary.detach()),'total_loss':float(total.detach()),'support_digest':[teacher.support_digest(row['query'],row['detail']['neighbors'],row['labels']) for row in records],'residual_slots':sum(row['detail']['t'].shape[1] for row in records)}

class Adapter:
 def __init__(self,torch,numpy,model,opt,m,utils,sampler,data,teacher,arm,provenance):
  self.t,self.np,self.model,self.optimizer=torch,numpy,model,opt;self.m,self.utils,self.sampler=m,utils,sampler;self.data,self.teacher,self.arm=data,teacher,arm;self.device=data['x'].device;self.provenance=provenance;self.initialization={'source':'literalcoldPatternTwin','trained_donor':False};self._rng=bridge.rng_state(torch,numpy);self._epoch=0;self._best=None;self._best_step=None;self.work=[]
 @contextmanager
 def scope(self,training=False):
  outer=bridge.rng_state(self.t,self.np);bridge.restore_rng(self.t,self.np,self._rng)
  try:
   with prior.aliases(self.m):yield
  finally:
   if training:self._rng=bridge.rng_state(self.t,self.np)
   bridge.restore_rng(self.t,self.np,outer)
 def train_step(self,epoch):
  if epoch!=self._epoch+1:raise ValueError('Exact source orderedepoch required')
  self.model.train();rows=[]
  with self.scope(True):
   self.t.cuda.synchronize(0);negative,iterator,stream=self.m['pilot_data'].epoch_stream(self.data,{'native_utils':self.utils},self.sampler)
   for ids in iterator:rows.append(batch_step(self.t,self.m,self.model,self.optimizer,self.data,self.teacher,negative,ids,self.arm))
   self.t.cuda.synchronize(0)
  if len(rows)!=17:raise ValueError('Complete source17minibatches required')
  self._epoch=epoch;work={'epoch':epoch,'stream':stream,'batches':rows,'full_batches':17,'optimizer_steps':17};self.work.append({'epoch':epoch,'stream':stream})
  if hasattr(self,'receipt_output'):
   q=Path(self.receipt_output)/('epoch_'+str(epoch)+'.json');q.parent.mkdir(parents=True,exist_ok=True)
   with q.open('x') as f:__import__('json').dump(work,f,sort_keys=True,allow_nan=False)
  return {'objective':sum(row['total_loss'] for row in rows)/17,'ce':sum(row['main_loss'] for row in rows)/17,'regularization':sum(row['selected_auxiliary_loss'] for row in rows)/17,'operation_counts':{'optimizer_updates':17,'backward_calls':17,'encoder_calls':17,'positive_records':1114112,'negative_records':1114112,'dropped_positive_tail':64940,'source_native_pattern_recursive_scorer_calls':272,'source_conditional_complete_residual_slots':sum(row['residual_slots'] for row in rows)},'source_epoch_work':work}
 def validate(self,epoch):
  from ogb.linkproppred import Evaluator
  with self.scope():p,n,work=self.m['pilot_evaluate'].score_valid(self.model,self.data,{'graph_ops':self.m['graph_ops'],'native_utils':self.utils},mode='private')
  e=Evaluator.__new__(Evaluator);e.name='ogbl-collab';e.eval_metric='hits@50';e.K=50;hits=self.m['pilot_evaluate'].hits50(e,p,n);ce=float(-self.t.nn.functional.logsigmoid(p).mean()-self.t.nn.functional.logsigmoid(-n).mean());return {'objective':ce,'ce':ce,'regularization':0.,'selection_metric':hits,'hits@50':hits,'source_complete_VALID_work':work}
 def observe_validation(self,epoch,v):
  better=self._best is None or v['selection_metric']>self._best
  if better:self._best,self._best_step=v['selection_metric'],epoch
  return {'save_selected':better,'stop_requested':False,'best_saved_metric':self._best,'best_saved_step':self._best_step,'full_budget':True}
 def predict_pairs(self,pairs,route='native_pattern'):
  if route!='native_pattern':raise ValueError('Native count-free privateclamp serving required')
  self.model.eval();scores=[]
  with self.scope(),self.t.no_grad():
   graph=self.m['graph_ops'].Graph.from_pairs(self.data['pairs'],len(self.data['x']));h=self.model.encoder(self.data['x'],graph)
   for start in range(0,len(pairs),131072):scores.append(self.model.decoder(h,graph,pairs[start:start+131072].to(self.device),'private').T.cpu())
  return self.t.cat(scores,1)
 def capture_state(self):return {'provenance':copy.deepcopy(self.provenance),'fit_rng':copy.deepcopy(self._rng),'outer_rng':bridge.rng_state(self.t,self.np),'epoch':self._epoch,'best':self._best,'best_step':self._best_step,'work':copy.deepcopy(self.work)}
 def restore_state(self,v):
  if v['provenance']!=self.provenance:raise ValueError('LiteralCB source/model/role ancestry required')
  self._rng=copy.deepcopy(v['fit_rng']);self._epoch=v['epoch'];self._best=v['best'];self._best_step=v['best_step'];self.work=copy.deepcopy(v['work']);bridge.restore_rng(self.t,self.np,v['outer_rng'])

def make_adapter(*,torch,project_root,study,run,device):
 import numpy
 from torch_geometric.utils import negative_sampling
 from structural_cardinality_semantics import validate_conditional_run
 validate_conditional_run(run);c=run['config'];m=sources(project_root,c['conditional_sources']);native,utils=bridge.native_modules(project_root,c['link_source']);outer=bridge.rng_state(torch,numpy)
 try:model,opt=m['pattern_model'].make_pattern({'prototype':m['prototype'],'design':m['design_spec']},run['seed'],device);rng=bridge.rng_state(torch,numpy)
 finally:bridge.restore_rng(torch,numpy,outer)
 inp={k:torch.load(bound(project_root,row),map_location='cpu',weights_only=True) for k,row in c['link_inputs'].items()};g,tr,v=inp['graph'],inp['train'],inp['select'];data={'x':g['x'].to(device),'raw_edge_index':g['raw_edge_index'].to(device),'pairs':tr['positive'].to(device),'valid_positive':v['positive'].to(device),'valid_negative':v['negative'].to(device)}
 if study['dataset']!='ogbl-collab' or data['pairs'].shape!=(1179052,2) or data['x'].shape!=(235868,128) or data['valid_positive'].shape!=(60084,2) or data['valid_negative'].shape!=(100000,2):raise ValueError('Complete native current CB rolecustody required')
 teacher=m['pattern_teacher'].ObservationTeacher.from_train(data['pairs'],len(data['x']));a=Adapter(torch,numpy,model,opt,m,utils,negative_sampling,data,teacher,c['conditional_arm'],{'run_id':run['run_id'],'config':c,'contract':CONTRACT,'teacher':teacher.receipt()});a._rng=rng;a.receipt_output=Path(project_root)/c['source_epoch_receipt_output']/run['run_id'];return a
