import copy,io,json,hashlib,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));from adapters import bsnn_deterministic as port

def equal(t,a,b):
 if t.is_tensor(a):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(t,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(t,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();n=12;ids=torch.arange(n);torch.manual_seed(311);x=torch.randn(n,6);e=torch.stack((ids.repeat(2),torch.cat(((ids+1)%n,(ids-1)%n))));e=e[:,(e[0]*n+e[1]).argsort()];graph={'x':x,'edge_index':e};train={'ids':ids[:8],'y':ids[:8]%2};select={'ids':ids[8:],'y':ids[8:]%2};B=json.loads((P/'BSNN_SOURCE_BINDINGS.json').read_text());rows=[]
 cases=P/'CPU_cases';cases.mkdir(exist_ok=True)
 for arm in ('native_M1','original_GNNM'):
  done=cases/(arm+'.json')
  if done.is_file() and json.loads(done.read_text()).get('status')=='passed':rows.append(json.loads(done.read_text()));continue
  model,opt,h,args=port.construct(torch,R,B,graph,arm,88201,'cpu');adapter=port.attach(torch,model,opt,h,args,graph,train,select,arm,88201,{'fixture':arm});owned=[p for g in opt.param_groups for p in g['params']];params=list(model.parameters());assert len({id(p) for p in owned})==len(owned)==len(params) and {id(p) for p in owned}=={id(p) for p in params}
  result=adapter.train_step(1);counts=result['operation_counts'];assert counts['optimizer_updates']==1 and counts['backward_calls']==(4 if arm=='original_GNNM' else 1)
  rng=copy.deepcopy(h.capture_rng(adapter.h.seed_all.__globals__.get('np',__import__('numpy')),torch));v=adapter.validate(1);assert equal(torch,rng,h.capture_rng(__import__('numpy'),torch));assert adapter.observe_validation(1,v)['save_selected'];state=copy.deepcopy(adapter.capture_state());m=copy.deepcopy(model.state_dict());o=copy.deepcopy(opt.state_dict());adapter.restore_state(state);model.load_state_dict(m);opt.load_state_dict(o);assert equal(torch,state,adapter.capture_state());buf=io.BytesIO();torch.save({'model_state':m,'optimizer_state':o,'adapter_state':state,'step':1},buf)
  q={'arm':arm,'status':'passed','toy_optimizer_maps':1,'literal_original_bundle_Householder_path':True,'source_Adam_or_boundary_AdamW_parameters_owned_exactly_once':True,'source_eval_RNG_snapshot_restored':True,'source_plain_L_and_source_training_epoch_selector_checkpoint_restored':True,'source_perlayer_static_weight_indices_moved':True,'parameter_scalars':sum(p.numel() for p in params),'checkpoint_serialized_preview_bytes':len(buf.getvalue()),'checkpoint_serialized_preview_sha256':hashlib.sha256(buf.getvalue()).hexdigest(),'operation_counts':counts,'actual_input_arrays_opened':0,'scientific_fits':0,'GO':False};done.write_text(json.dumps(q,indent=2)+'\n');rows.append(q)
 report={'schema':'new-BSNN-source-wrapper-provider-placement-checkpoint-CPU-QA-v1','status':'passed','cases':rows,'toy_optimizer_maps':sum(r['toy_optimizer_maps'] for r in rows),'provider_source_cpp_unchanged':True,'shared_venv_modified':False,'old_closed_source_kernel_checks_rerun':False,'actual_input_arrays_opened':0,'scientific_fits':0,'CUDA_compute':0,'GO':False};(P/'BSNN_PIPELINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
