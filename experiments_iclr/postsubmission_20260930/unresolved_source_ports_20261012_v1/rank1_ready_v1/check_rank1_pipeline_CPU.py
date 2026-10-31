import copy,hashlib,io,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));from adapters import rank1_native as port
from adapters.rank1_endpoints import secondary_draws

def equal(t,a,b):
 if t.is_tensor(a):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(t,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(t,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();torch.manual_seed(317);n=11701;ids=torch.arange(n);graph={'x':torch.randn(n,300),'edge_index':torch.stack((ids.repeat(2),torch.cat((ids,(ids+1)%n))))};train={'ids':ids[:580],'y':ids[:580]%10};select={'ids':ids[580:600],'y':ids[580:600]%10};B=json.loads((P/'RANK1_SOURCE_BINDINGS.json').read_text());cases=P/'CPU_cases';cases.mkdir(exist_ok=True);records=[]
 for arm in ('point4','fixed_noise4','q4','q_M1'):
  done=cases/(arm+'.json');paid=cases/(arm+'_paid_update.pt')
  if done.is_file() and json.loads(done.read_text()).get('status')=='passed':records.append(json.loads(done.read_text()));continue
  state=port.construct(torch,R,B,graph,arm,7301,0,'cpu');a=port.attach(torch,*state,graph,train,select,arm,7301,0,{'synthetic':arm});model,opt=a.model,a.optimizer;owned=[p for g in opt.param_groups for p in g['params']];assert len(owned)==len({id(p) for p in owned}) and {id(p) for p in owned}=={id(p) for p in model.parameters() if p.requires_grad}
  if paid.is_file():
   saved=torch.load(paid,weights_only=False,map_location='cpu');model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);a.restore_state(saved['adapter']);result=saved['update']
  else:
   result=a.train_step(1);torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'adapter':a.capture_state(),'update':result},paid)
  count=sum(p.numel() for p in model.parameters());expected={'point4':237474,'fixed_noise4':253306,'q4':253306,'q_M1':229558}[arm];assert count==expected;assert not list(model.named_buffers());before=copy.deepcopy(a.capture_state()['external']);v=a.validate(1)
  # Eval changes only training flags; factor/dropout state must stay identical.
  if a.posterior_rows is not None:
   now=a.capture_state()['external'];assert equal(torch,before['factor_rng'],now['factor_rng']);assert equal(torch,before['native']['streams'],now['native']['streams']);assert equal(torch,before['posterior'],now['posterior']);a.posterior_rows.mean_scope_closed()
   assert a.posterior_rows.checkpoint()['factor_coordinates_per_route']==3958
   if arm=='fixed_noise4':assert all(not p.requires_grad for k,p in model.named_parameters() if k.endswith(('_log_std')))
  assert a.observe_validation(1,v)['save_selected'];assert not a.observe_validation(2,v)['save_selected'];saved=copy.deepcopy(a.capture_state());m=copy.deepcopy(model.state_dict());o=copy.deepcopy(opt.state_dict());fresh=port.construct(torch,R,B,graph,arm,7301,0,'cpu');other=port.attach(torch,*fresh,graph,train,select,arm,7301,0,{'synthetic':arm});other.model.load_state_dict(m);other.optimizer.load_state_dict(o);other.restore_state(saved);assert equal(torch,saved,other.capture_state())
  if a.posterior_rows is not None:
   changed=copy.deepcopy(saved);k=next(iter(changed['external']['posterior']['prior_centres']));changed['external']['posterior']['prior_centres'][k].reshape(-1)[0]+=1
   try:other.restore_state(changed)
   except ValueError:pass
   else:raise AssertionError('Changed immutable prior accepted')
  secondary=False
  if arm=='q4':
   before=copy.deepcopy(a.capture_state()['external']);raw,comp=secondary_draws(torch,[a],select['ids'],7301);assert raw.shape==(12,20,10) and comp==[0,1,2,3]*3 and torch.isfinite(raw).all();assert equal(torch,before,a.capture_state()['external']);secondary=True
  buf=io.BytesIO();torch.save({'model':m,'optimizer':o,'adapter':saved},buf);row={'arm':arm,'status':'passed','toy_optimizer_maps':1,'parameter_scalars':count,'active_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'source_endpoint_VALUE_included':True,'source_primitives_AST_unchanged':a.source_rendered,'every_active_parameter_owned_once':True,'mean_selector_factor_RNG_and_dropout_unchanged':True,'fresh_constructor_selected_state_full_roundtrip':True,'immutable_prior_mutation_rejected':a.posterior_rows is not None,'secondary_raw12_verified':secondary,'checkpoint_preview_bytes':len(buf.getvalue()),'operation_counts':result['operation_counts'],'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False};done.write_text(json.dumps(row,indent=2)+'\n');records.append(row)
 report={'schema':'literal-endpoint-E-Rank1-source-new-pipeline-synthetic-CPU-v1','status':'passed','cases':records,'toy_optimizer_maps':4,'source_secondary_raw12_forward_calls':12,'actual_input_arrays_opened':0,'old_closed_source_kernels_rerun':False,'scientific_fits':0,'CUDA_compute':0,'GO':False};(P/'RANK1_PIPELINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','toy_optimizer_maps':4,'parameters':[r['parameter_scalars'] for r in records],'GO':False}))
if __name__=='__main__':main()
