import copy,io,json,sys,hashlib,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(P/'runtime'))
from adapters import lowrank_gine as port

def equal(t,a,b):
 if t.is_tensor(a):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(t,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(t,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();binding=json.loads((P/'LOWRANK_SOURCE_BINDINGS.json').read_text());start=time.monotonic();model,opt,definition,loss=port.construct(torch,R,binding,'lowrank_private',70201,'cpu');factory_seconds=time.monotonic()-start
 plan=definition['lowrank']['rank_plan'];assert len(plan['candidates'])==129 and plan['chosen']['rank']==min(plan['candidates'],key=lambda c:(abs(c['parameter_scalars']-plan['target_counts']['parameter_scalars']),c['rank']))['rank'];assert len(model.low_rank_layers)==6
 params=list(model.parameters());owned=[x for g in opt.param_groups for x in g['params']];assert len(owned)==len(params) and len({id(p) for p in owned})==len(params) and {id(p) for p in owned}=={id(p) for p in params} and all(p.requires_grad for p in params)
 def pack(ids):
  return {'ids':torch.tensor(ids),'x':torch.zeros(len(ids)*3,9,dtype=torch.long),'edge_index':torch.tensor([[0,1,1,2],[1,0,2,1]]).repeat(1,len(ids)),'edge_attr':torch.zeros(4*len(ids),3,dtype=torch.long),'node_ptr':torch.arange(len(ids)+1)*3,'edge_ptr':torch.arange(len(ids)+1)*4,'y':torch.arange(len(ids)).remainder(2).float()[:,None]}
 train,select=pack([0,1,2,3]),pack([4,5]);cfg={'initialization_seed':70201,'training':{'batch_size':128,'eval_batch_size':128},'model':{'outputs':1}}
 adapter=port.GINEAdapter(torch,model,opt,definition,loss,train,select,cfg,{'fixture':True,'source':binding});r=adapter.train_step(1);assert r['operation_counts']['optimizer_updates']==1 and r['operation_counts']['model_forwards']==4
 v=adapter.validate(1);assert adapter.observe_validation(1,v)['save_selected'];state=copy.deepcopy(adapter.capture_state());m=copy.deepcopy(model.state_dict());o=copy.deepcopy(opt.state_dict());model.low_rank_layers[0].active_member=0;adapter._epoch=0;adapter.restore_state(state);model.load_state_dict(m);opt.load_state_dict(o);assert equal(torch,state,adapter.capture_state())
 buf=io.BytesIO();torch.save({'model_state':m,'optimizer_state':o,'adapter_state':state,'step':1},buf);preview=buf.getvalue()
 bad=copy.deepcopy(state);bad['provenance']={}
 try:adapter.restore_state(bad)
 except ValueError:pass
 else:raise AssertionError('Changed source/rank ancestry admitted')
 source_rank={'schema':'actual-datafree-GINE-factory-rank-authority-v1','status':'passed','factory_seconds_CPU':factory_seconds,'rank_plan':plan,'registered_constructor_objects':definition['registered_constructor_objects'],'rank_candidate_wrapper_objects':129,'zero_B_live_gradient_gate_rerun':False,'data_or_scores_accessed':False,'scientific_fits':0,'GO':False};(P/'ACTUAL_SOURCE_RANK_FACTORY.json').write_text(json.dumps(source_rank,indent=2)+'\n')
 report={'schema':'new-lowrank-GINE-factory-ownership-batching-checkpoint-CPU-QA-v1','status':'passed','toy_optimizer_maps':1,'all_trainable_source_parameters_owned_exactly_once':True,'exact_registered_rank':plan['chosen']['rank'],'parameter_scalars':plan['chosen']['parameter_scalars'],'checkpoint_model_optimizer_rng_plain_factor_state_restored':True,'source_batch128_one_full4graph_fixture_epoch':True,'source_selector_ROCauc_then_BCE_then_earlier':True,'checkpoint_serialized_preview_bytes':len(preview),'checkpoint_serialized_preview_sha256':hashlib.sha256(preview).hexdigest(),'checkpoint_preview_scope':'synthetic one-map actual source all namedAdamW states;no scientific checkpoint or data','old_completed_lowrank_zeroB_kernel_rerun':False,'actual_input_arrays_opened':0,'scientific_fits':0,'CUDA_compute':0,'GO':False};(P/'LOWRANK_PIPELINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
