"""Tiny synthetic CPU checks of newly bound source mechanisms only."""
import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];ROOT=P.parent;sys.path.insert(0,str(P/'runtime'))
from adapters import polynormer_mechanisms as port
from adapters.common_native_wrapper import receipt

def equal_tree(torch,a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(torch,a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(torch,x,y) for x,y in zip(a,b))
 if hasattr(a,'shape'):return bool((a==b).all())
 return a==b

def main():
 import torch
 torch.set_num_threads(1)
 assert not torch.cuda.is_initialized()
 bindings={k:receipt(ROOT,P/'sources'/v) for k,v in port.PATHS.items()}
 prior=json.loads((P/'POLYNORMER_MECHANISMS_CPU_QA_TRANSPORT_attempt2.json').read_text())
 rows=[json.loads(line)['case'] for line in prior['stdout'].splitlines() if line.startswith('{')]
 assert len(rows)==2 and all(row['features']==300 and row['arm'] in ('alignment_only','residual_only') for row in rows)
 for row in rows:row['preserved_pass_receipt']='POLYNORMER_MECHANISMS_CPU_QA_TRANSPORT_attempt2.json'
 steps=0
 for features,classes in ((300,10),(745,8)):
  for arm in ('alignment_only','residual_only','independent_scorer_alphaF','independent_scorer_relationJ'):
   if features==300 and arm in ('alignment_only','residual_only'):continue
   torch.manual_seed(1234);n=classes*2+4;x=torch.randn(n,features);src=torch.arange(n).repeat_interleave(3);dst=torch.stack((torch.arange(n),(torch.arange(n)+1)%n,(torch.arange(n)-1)%n),1).reshape(-1);edges=torch.stack((src,dst));ids=torch.arange(classes*2);y=torch.arange(classes).repeat(2)
   session,details=port.construct(torch,ROOT,bindings,arm=arm,features=features,classes=classes,seed=88101,body=0,dropout=.5,learning_rate=.001,train_count=len(ids),device='cpu')
   adapter=port.attach(torch,session,details,x=x,edges=edges,train_ids=ids,train_y=y,select_ids=torch.arange(classes*2,n),select_y=torch.arange(4)%classes,arm=arm,provenance={'classes':classes},local_updates=1)
   assert len(session.optimizers)==1 and not session.optimizers[0].state
   assert all(torch.equal(m.r,torch.ones_like(m.r)) and torch.equal(m.s,torch.ones_like(m.s)) for m in session.model.modules() if isinstance(m,session.core['factors'].FactorLinear))
   if arm.startswith('independent_scorer_'):
    start=session.independent_native_start;assert start['default_CPU_CUDA_RNG_unchanged'] and start['before_original_Adam'] and len(start['rows'])==56
    assert len(session.relation_partition['relation'])==18 and len(session.relation_partition['internal_dense'])==56
    for _,p in session.relation_partition['relation']:
     if p.ndim==4:assert len({row.detach().numpy().tobytes() for row in p})==4
   before=(copy.deepcopy(session.model.state_dict()),copy.deepcopy(adapter.optimizer.state_dict()),copy.deepcopy(adapter.capture_state()))
   value=adapter.train_step(1);steps+=1;after=(copy.deepcopy(session.model.state_dict()),copy.deepcopy(adapter.optimizer.state_dict()),copy.deepcopy(adapter.capture_state()))
   session.model.load_state_dict(before[0]);adapter.optimizer.load_state_dict(before[1]);adapter.restore_state(before[2]);replay=adapter.train_step(1);steps+=1
   assert equal_tree(torch,value,replay) and equal_tree(torch,after[0],session.model.state_dict()) and equal_tree(torch,after[1],adapter.optimizer.state_dict()) and equal_tree(torch,after[2],adapter.capture_state())
   if arm.startswith('independent_scorer_'):
    assert value['operation_counts']['output_cotangent_collections']==2 and value['operation_counts']['member_reverse_collections']==16 and value['operation_counts']['optimizer_bank_updates']==1
    assert all(p.grad is None for name,p in session.model.named_parameters() if '.global_attn.' in name or '.pred_global.' in name)
   else:
    assert [session.unit_facade.alignment_source_calls,session.unit_facade.residual_source_calls]==([1,0] if arm=='alignment_only' else [0,1])
   adapter.observe_validation(1,{'selection_metric':.5})
   # Selected local snapshot restores exact model+Adam while source live RNG
   # remains at the end-local point, even if local winner was earlier.
   live=copy.deepcopy(session.streams);step_before=session.steps;adapter.train_step(2);steps+=1
   assert adapter.transition_completed and session.model.models[0].body._global and session.steps==step_before+1
   saved=copy.deepcopy(adapter.capture_state());session.model.set_global(False);session.steps=-1;adapter.restore_state(saved);assert session.model.models[0].body._global and session.steps==step_before+1
   rows.append({'arm':arm,'features':features,'classes':classes,'literal_source_update_and_Adam_RNG_replay':True,'source_local_selected_Adam_live_RNG_handoff':True,'exact_provider_permissions':True})
   print(json.dumps({'case':rows[-1],'completed_toy_steps':steps}),flush=True)
   del before,after,adapter,session
 assert len(rows)==8 and steps==18 and not torch.cuda.is_initialized()
 result={'schema':'retained-Polynormer-unit-private-credit-new-risk-CPU-QA-v1','status':'passed','cases':rows,'toy_optimizer_steps':steps,'prior_attempt_known_toy_steps':6,'all_attempt_toy_steps':steps+6,'benchmark_fits':0,'real_tensor_reads':0,'CUDA_compute':0,'existing_kernel_tests_repeated':False}
 (P/'POLYNORMER_MECHANISMS_CPU_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(rows),'toy_steps':steps,'scientific_fits':0}))

if __name__=='__main__':main()
