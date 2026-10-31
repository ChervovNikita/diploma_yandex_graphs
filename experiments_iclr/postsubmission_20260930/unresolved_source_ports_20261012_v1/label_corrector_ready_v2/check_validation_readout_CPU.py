"""New validation-engine interface gate; restore paid state if retained; never train."""
import copy,gc,json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;U=P.parent;R=U.parent
if len(sys.argv)==1:
 cases=[]
 for family in ('private_response','rank1','film','label_corrector'):
  prior=P/('VALIDATION_READOUT_'+family+'_ATTEMPT.json')
  if prior.is_file():
   old=json.loads(prior.read_text())
   if old['exit_code']==0:
    cases.append(json.loads(next(x for x in old['stdout'].splitlines() if x.startswith('{'))));continue
  v=subprocess.run([sys.executable,'-B',str(Path(__file__)),family],capture_output=True,text=True,timeout=180)
  (P/('VALIDATION_READOUT_'+family+'_ATTEMPT.json')).write_text(json.dumps({'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'new_optimizer_maps':0,'GO':False},indent=2)+'\n')
  if v.returncode:raise RuntimeError(v.stderr)
  cases.append(json.loads(next(x for x in v.stdout.splitlines() if x.startswith('{'))))
 report={'schema':'four-source-family-task-proper-validation-engine-readout-CPU-v1','status':'passed','cases':cases,'new_optimizer_maps':0,'training_repeats':0,'actual_input_arrays_opened':0,'CUDA_compute':0,'scientific_fits':0,'GO':False};(P/'VALIDATION_READOUT_ENGINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','families':4,'cases':sum(x['cases'] for x in cases),'new_optimizer_maps':0,'GO':False}));sys.exit()
family=sys.argv[1];packet=U/(family+'_ready_v2' if family in ('private_response','label_corrector') else family+'_ready_v1');sys.path.insert(0,str(packet/'runtime'))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.optim.Adam.step=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('New optimizer maps forbidden in readout gate'))
from engine import _metrics
results=[]
if family=='private_response':
 from adapters import private_response as port
 torch.manual_seed(323);ids=torch.arange(12);pairs=torch.stack((ids.repeat(2),torch.cat(((ids+1)%12,(ids+2)%12))),1);graph={'x':torch.randn(12,3703),'train_support_edge_index':torch.cat((pairs.t(),pairs.t().flip(0)),1)};train={'positive':pairs};neg=torch.stack((ids,(ids+6)%12),1);select={'positive':pairs[12:14],'negative':neg[:2,None,:].repeat(1,500,1)};B=json.loads((packet/'PRIVATE_RESPONSE_SOURCE_BINDINGS.json').read_text())
 for arm in ('shared_f4','capable_single','untied4','ordinary_native4','row0_single','native_M1','original_GNNM'):
  paid=U/'private_response_ready_v1/CPU_cases'/(arm+'_paid_episode.pt');assert paid.is_file();saved=torch.load(paid,map_location='cpu',weights_only=False);a=port.Adapter(torch,*port.construct(torch,R,B,arm,0,283136447778829174,'cpu'),graph,train,select,arm,0,0,{'synthetic':arm});a.model.load_state_dict(saved['model']);a.optimizer.load_state_dict(saved['optimizer']);a.restore_state(saved['adapter']);original=a.score;observed=[]
  def score(*args,**kwargs):
   z=original(*args,**kwargs);observed.append(z);return z
  a.score=score;v=a.validate(5);_metrics(torch,v);assert len(observed)==1;p,n=observed[0];expected=float(.5*(torch.nn.functional.softplus(-p.mean(1)).mean()+torch.nn.functional.softplus(n.mean(2)).mean()));assert v['objective']==v['ce']==expected and v['regularization']==0.;results.append({'arm':arm,'state':'retained passed paid synthetic episode','loss':'balanced mean-logit binary NLL','positive_denominator':len(p),'negative_denominator':n.shape[0]*n.shape[1],'engine_schema_and_actual_returned_logits_loss':True});del a,saved,p,n;gc.collect()
elif family in ('rank1','film'):
 if family=='rank1':from adapters import rank1_native as port
 else:from adapters import film_native as port
 torch.manual_seed(317 if family=='rank1' else 311);ids=torch.arange(11701);graph={'x':torch.randn(11701,300),'edge_index':torch.stack((ids.repeat(2),torch.cat((ids,(ids+1)%11701))))};train={'ids':ids[:580],'y':ids[:580]%10};select={'ids':ids[580:600],'y':ids[580:600]%10};B=json.loads((packet/('RANK1_SOURCE_BINDINGS.json' if family=='rank1' else 'FILM_SOURCE_BINDINGS.json')).read_text())
 for arm in (('point4','fixed_noise4','q4','q_M1') if family=='rank1' else ('shared4','original_GNNM')):
  a=port.attach(torch,*port.construct(torch,R,B,graph,arm,7301,0,'cpu'),graph,train,select,arm,7301,0,{'synthetic':arm} if family=='rank1' else {'fixture':arm});state='constructor-only synthetic state; prior in-memory trained checkpoint not retained'
  if family=='rank1':
   paid=packet/'CPU_cases'/(arm+'_paid_update.pt');assert paid.is_file();saved=torch.load(paid,map_location='cpu',weights_only=False);a.model.load_state_dict(saved['model']);a.optimizer.load_state_dict(saved['optimizer']);a.restore_state(saved['adapter']);state='retained passed paid synthetic update'
  observed=[]
  def read_output(module,args,output):
   z=output[0] if isinstance(output,tuple) else output;observed.append(z.detach() if z.ndim==3 else z.detach()[None])
  handle=a.model.register_forward_hook(read_output);v=a.validate(1);handle.remove();_metrics(torch,v);assert observed;z=torch.cat(observed)[:,select['ids']].double();lp=z.mean(0).log_softmax(-1) if a.fusion=='mean_logits' else torch.logsumexp(z.log_softmax(-1),0)-__import__('math').log(len(z));expected=float(-lp.gather(1,select['y'][:,None]).mean());assert v['ce']==expected and v['objective']==v['ce']+v['regularization'];results.append({'arm':arm,'state':state,'loss':'actual source pooled categorical SELECT NLL','SELECT_denominator':20,'routes':len(z),'engine_schema_and_actual_returned_logits_loss':True});del a,z;gc.collect()

else:
 from adapters import label_corrector as port
 B=json.loads((packet/'LABEL_CORRECTOR_SOURCE_BINDINGS.json').read_text());g=torch.Generator().manual_seed(83001);ids=torch.arange(600);graph={'x':torch.randn(600,300,generator=g),'edge_index':torch.stack([ids.repeat(3),torch.cat([(ids+1)%600,(ids+13)%600,ids])])};train={'ids':torch.arange(580),'y':torch.arange(580)%10};select={'ids':torch.arange(580,600),'y':torch.arange(20)%10}
 for arm in port.WORK:
  a=port.Adapter(torch,port.construct(torch,R,B,arm,6101,'cpu',graph,train),graph,train,select,arm,{'synthetic':True,'condition':arm,'actual_model_seed':6101});original=a.predict_rows;observed=[]
  def predict(*args,**kwargs):
   z=original(*args,**kwargs);observed.append(z);return z
  a.predict_rows=predict;v=a.validate(1);_metrics(torch,v);assert len(observed)==1;z=observed[0];lp=z.mean(0).log_softmax(-1) if arm=='original_GNNM' else torch.logsumexp(z.log_softmax(-1),0)-__import__('math').log(len(z));expected=float(-lp.gather(1,select['y'][:,None]).mean());assert v['objective']==v['ce']==expected and v['regularization']==0.;results.append({'arm':arm,'state':'constructor-only synthetic state; prior temporary trained checkpoint not retained','loss':'source pooled categorical SELECT NLL','SELECT_denominator':20,'routes':len(z),'engine_schema_and_actual_returned_logits_loss':True});del a,z;gc.collect()
print(json.dumps({'family':family,'status':'passed','cases':len(results),'records':results,'new_optimizer_maps':0,'training_repeats':0,'actual_input_arrays_opened':0,'CUDA_compute':0,'GO':False}))
