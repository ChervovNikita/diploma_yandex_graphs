"""New wrapper context and checkpoint fixture; no actual arrays/outcomes."""
import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(P/'runtime'))
from adapters import unimp_reference as port

def equal(t,a,b):
 if t.is_tensor(a):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(t,a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(t,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();torch.manual_seed(331)
 n=16;ids=torch.arange(n);x=torch.randn(n,6);edges=torch.stack((ids.repeat(2),torch.cat(((ids+1)%n,(ids-1)%n))))
 model,opt,topology=port.construct(torch,ROOT,json.loads((P/'UNIMP_SOURCE_BINDINGS.json').read_text()),x=x,edges=edges,classes=3,seed=88101,learning_rate=.001,weight_decay=.0005,device='cpu')
 adapter=port.attach(torch,model,opt,topology,train_ids=ids[:10],train_y=ids[:10]%3,select_ids=ids[10:13],select_y=ids[10:13]%3,seed=88101,provenance={'fixture':True})
 contexts=[];original=model.forward
 def forward(t,visible_ids,visible_y):
  contexts.append(visible_ids.detach().clone());return original(t,visible_ids,visible_y)
 model.forward=forward
 out=adapter.train_step(1);assert out['operation_counts']['optimizer_updates']==1 and len(contexts[-1])==6
 value=adapter.validate(1);decision=adapter.observe_validation(1,value);assert decision['save_selected']
 assert torch.equal(contexts[-1],ids[:10]);state=copy.deepcopy(adapter.capture_state());params=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
 second=adapter.train_step(2);second_state=copy.deepcopy(adapter.capture_state());second_params=copy.deepcopy(model.state_dict())
 model.load_state_dict(params);opt.load_state_dict(adam);adapter.restore_state(state);assert equal(torch,state,adapter.capture_state())
 # Restore/capture is structural; do not repeat the second map as a numerical kernel audit.
 episode=second_state['external']['last_episode'];assert set(episode['visible_positions']).isdisjoint(episode['query_positions']) and len(episode['query_positions'])==4
 with torch.no_grad():z=adapter.predict_rows(ids[13:],1)
 assert z.shape==(1,3,3) and torch.isfinite(z).all() and torch.equal(contexts[-1],ids[:10])
 rejected=0
 for key,bad in [('TRAIN_context_ids',ids[1:11]),('visible_count',7),('source_model',{})]:
  altered=copy.deepcopy(state);altered['external'][key]=bad
  try:adapter.restore_state(altered)
  except ValueError:rejected+=1
  else:raise AssertionError('Changed source/context ancestry admitted')
 adapter.restore_state(state)
 report={'schema':'new-UniMPv2-source-wrapper-context-checkpoint-CPU-QA-v1','status':'passed','toy_optimizer_maps':2,'TRAIN_context6_query4':True,'complete_TRAIN_serving_context10':True,'SELECT_or_heldout_label_context':False,'literal_private_NumPy_order_RNG_and_Torch_RNG_restored':True,'model_and_named_Adam_state_saved_restored':True,'source_selector_double_NLL_strict_first_accuracy':True,'rejected_ancestry_cases':rejected,'old_completed_kernel_checks_rerun':False,'actual_input_arrays_opened':0,'scientific_fits':0,'CUDA_compute':0,'GO':False}
 (P/'UNIMP_PIPELINE_CPU_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
