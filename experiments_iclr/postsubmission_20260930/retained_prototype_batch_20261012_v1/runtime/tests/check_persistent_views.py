"""Tiny CPU new-binding/ownership/eight-pass/stage tests; no actual input."""
import copy,json,sys
from pathlib import Path
from unittest import mock
P=Path(__file__).resolve().parents[2];ROOT=P.parent;sys.path.insert(0,str(P/'runtime'))
from adapters import persistent_views as port
from adapters.common_native_wrapper import receipt

def equal_tree(torch,a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(torch,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(torch,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 bindings={k:receipt(ROOT,P/'sources'/v) for k,v in port.PATHS.items()};rows=[];steps=0
 cases=[(features,classes,arm) for features,classes in ((300,10),(745,8)) for arm in ('tied_persistent','untied_persistent','tied_shuffled','tied_random_null')]
 cases += [(300,10,'native_M1'),(745,8,'native_M1'),(300,10,'original_gnnm')]
 for features,classes,arm in cases:
  torch.manual_seed(7189);n=classes*2+4;x=torch.randn(n,features);src=torch.arange(n).repeat_interleave(3);dst=torch.stack((torch.arange(n),(torch.arange(n)+1)%n,(torch.arange(n)-1)%n),1).reshape(-1);edge=torch.stack((src,dst));graphs={'native':edge,'equal':edge[:,:-2],'different':edge[:,:-4],'null_equal':edge[:,:-6],'null_different':edge[:,:-8]};ids=torch.arange(classes*2);y=torch.arange(classes).repeat(2)
  rt,bank,schedule,driver,model,opt,details=port.construct(torch,ROOT,bindings,arm=arm,features=features,classes=classes,seed=88101,body=0,dropout=.3,learning_rate=.001,device='cpu',local=1,global_=2)
  adapter=port.attach(torch,rt,bank,schedule,driver,model,opt,details,x=x,graphs=graphs,train_ids=ids,train_y=y,select_ids=torch.arange(classes*2,n),select_y=torch.arange(4)%classes,arm=arm,seed=88101,provenance={'classes':classes})
  if arm=='untied_persistent':
   families=bank.families(model);assert len(families)==4
   assert all(not {p.untyped_storage().data_ptr() for p in families[a].parameters()}&{p.untyped_storage().data_ptr() for p in families[b].parameters()} for a in range(4) for b in range(a))
   for family in families[1:]:assert all(torch.equal(v,family.core.state_dict()[k]) for k,v in families[0].core.state_dict().items())
   assert not torch.equal(families[0].stem.R,families[1].stem.R)
  before=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(adapter.capture_state()));counter=[0];backward=torch.Tensor.backward
  def counted(value,*args,**kwargs):counter[0]+=1;return backward(value,*args,**kwargs)
  with mock.patch.object(torch.Tensor,'backward',counted):value=adapter.train_step(1)
  steps+=1;assert counter[0]==(2 if arm=='native_M1' else 8)
  after=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(adapter.capture_state()));model.load_state_dict(before[0]);opt.load_state_dict(before[1]);adapter.restore_state(before[2]);again=adapter.train_step(1);steps+=1
  assert value==again and equal_tree(torch,after[0],model.state_dict()) and equal_tree(torch,after[1],opt.state_dict()) and equal_tree(torch,after[2],adapter.capture_state())
  adapter.observe_validation(1,{'selection_metric':.5});adapter.train_step(2);steps+=1
  assert adapter.clock.actual_update==2 and (model._global if arm=='native_M1' else bank.stage(model))
  saved=copy.deepcopy(adapter.capture_state());adapter.clock.actual_update=-1
  if arm=='native_M1':model._global=False
  else:bank.set_stage(model,False)
  adapter.restore_state(saved);assert adapter.clock.actual_update==2 and (model._global if arm=='native_M1' else bank.stage(model))
  with torch.no_grad():
   z=adapter.logits_fn(model,False,2);expected=z.mean(0).softmax(-1) if arm=='original_gnnm' else z.softmax(-1).mean(0);assert torch.equal(adapter.probabilities(z),expected)
  rows.append({'arm':arm,'features':features,'classes':classes,'disjoint_untied_ownership_and_equal_slow_start':True,'exact_source_backwards':counter[0],'literal_Adam_RNG_replay':True,'selected_local_model_Adam_live_RNG_stage_handoff':True})
  print(json.dumps({'case':rows[-1],'completed_toy_steps':steps}),flush=True);del before,after,adapter,model,opt
 assert len(rows)==11 and steps==33 and not torch.cuda.is_initialized()
 result={'schema':'persistent-view-new-source-binding-CPU-QA-v1','status':'passed','cases':rows,'toy_optimizer_steps':steps,'prior_attempt_known_toy_steps':1,'all_attempt_toy_steps':steps+1,'benchmark_fits':0,'real_tensor_reads':0,'CUDA_compute':0,'existing_kernel_tests_repeated':False}
 (P/'PERSISTENT_VIEWS_CPU_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(rows),'toy_steps':steps,'scientific_fits':0}))

if __name__=='__main__':main()
