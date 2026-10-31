import copy,hashlib,io,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'));from adapters import film_native as port

def equal(t,a,b):
 if t.is_tensor(a):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(t,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(t,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();torch.manual_seed(311);n=11701;ids=torch.arange(n);x=torch.randn(n,300);e=torch.stack((ids.repeat(2),torch.cat((ids,(ids+1)%n))));graph={'x':x,'edge_index':e};train={'ids':ids[:580],'y':ids[:580]%10};select={'ids':ids[580:600],'y':ids[580:600]%10};B=json.loads((P/'FILM_SOURCE_BINDINGS.json').read_text());cases=P/'CPU_cases';cases.mkdir(exist_ok=True);rows=[]
 for arm in ('shared4','original_GNNM'):
  done=cases/(arm+'.json')
  if done.is_file() and json.loads(done.read_text()).get('status')=='passed':rows.append(json.loads(done.read_text()));continue
  state=port.construct(torch,R,B,graph,arm,7301,0,'cpu');adapter=port.attach(torch,*state,graph,train,select,arm,7301,0,{'fixture':arm});model,opt=adapter.model,adapter.optimizer;owned=[p for g in opt.param_groups for p in g['params']];assert len(owned)==len({id(p) for p in owned}) and {id(p) for p in owned}=={id(p) for p in model.parameters()};result=adapter.train_step(1);assert result['operation_counts']['backward_calls']==(4 if arm=='original_GNNM' else 1);streams=copy.deepcopy(state[3]);v=adapter.validate(1);assert equal(torch,streams,state[3]);assert adapter.observe_validation(1,v)['save_selected'];assert not adapter.observe_validation(2,v)['save_selected'];assert not adapter.observe_validation(300,v)['stop_requested'];assert adapter.observe_validation(301,v)['stop_requested'];saved=copy.deepcopy(adapter.capture_state());m=copy.deepcopy(model.state_dict());o=copy.deepcopy(opt.state_dict());adapter.restore_state(saved);model.load_state_dict(m);opt.load_state_dict(o);assert equal(torch,saved,adapter.capture_state());buf=io.BytesIO();torch.save({'model':m,'optimizer':o,'adapter':saved},buf)
  row={'arm':arm,'status':'passed','toy_optimizer_maps':1,'full_synthetic_nodes':11701,'all_private_and_shared_parameters_owned_once':True,'eval_does_not_advance_dropout_streams':True,'strict_first_tie_and_stop301_after300_stale':True,'selected_model_optimizer_RNG_selector_factor_member_state_roundtrip':True,'actual_input_arrays_opened':0,'parameter_scalars':sum(p.numel() for p in model.parameters()),'checkpoint_preview_bytes':len(buf.getvalue()),'source_rendered':adapter.source_rendered,'operation_counts':result['operation_counts'],'scientific_fits':0,'GO':False}
  if arm=='shared4':assert row['parameter_scalars']==354146 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());row['all14_FactorLinear_sites_live']=True
  done.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 # Constructor-only exact genuine member0 identity and independent seed plans; no extra maps.
 states=[port.construct(torch,R,B,graph,'native_M1',7301,b,'cpu') for b in (0,0,1)];assert equal(torch,states[0][1].state_dict(),states[1][1].state_dict());assert equal(torch,states[0][3],states[1][3]);assert not equal(torch,states[0][1].state_dict(),states[2][1].state_dict());assert sum(p.numel() for p in states[0][1].parameters())==340928
 report={'schema':'literal-FiLM-source-port-fullshape-synthetic-CPU-v1','status':'passed','cases':rows,'toy_optimizer_maps':2,'constructor_only_M1_I4_body0_exact_seed_model_streams':True,'fresh_body1_different':True,'actual_input_arrays_opened':0,'old_closed_source_kernels_rerun':False,'scientific_fits':0,'CUDA_compute':0,'GO':False};(P/'FILM_PIPELINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','toy_optimizer_maps':2,'parameters':[r['parameter_scalars'] for r in rows],'GO':False}))
if __name__=='__main__':main()
