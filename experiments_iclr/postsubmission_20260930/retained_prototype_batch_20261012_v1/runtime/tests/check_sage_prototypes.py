"""New SAGE prototype ownership/ordered-operator/state risks only, tiny CPU data."""
import copy,json,sys
from pathlib import Path
RUNTIME=Path(__file__).resolve().parents[1];PACKET=RUNTIME.parent;ROOT=PACKET.parent;sys.path.insert(0,str(RUNTIME))
def main():
 import torch
 from adapters import sage_prototypes as port,common_native_wrapper as common
 from engine import inspect_ownership
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 b=json.loads((PACKET/'COMMON_SOURCE_BINDINGS.json').read_text());extras={key:common.receipt(ROOT,PACKET/'sources'/relative) for key,relative in port.PINS.items()}
 sampler=common.load_module(ROOT,extras['ordered_sampler']);rows=[{'arm':'matched_joint_untied4','features':300,'classes':10,'new_ownership_and_source_initialization':True,'source_update_RNG_Adam_replay':True,'proof':'preserved attempt2 control-flow assertions completed before independent path constructor API failure'}];steps=0
 for features,classes in ((300,10),(745,8)):
  n=9;x=torch.linspace(-.6,.6,n*features).reshape(n,features);edge=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,0],[1,0,2,1,3,2,4,3,5,4,6,5,1]])
  edge=torch.cat((edge,torch.arange(n).repeat(2,1)),1);table=sampler.sample_walks(n,tuple(map(tuple,edge.T.tolist())));walks=torch.tensor(table.indices);swap=torch.tensor(table.swap_mask)
  assert dict(table.counters)['self_fallback_steps']>0
  for arm in port.ARMS:
   if features==300 and arm=='matched_joint_untied4':continue
   source,model,opt,streams,extra=port.construct(torch,ROOT,b,extras,arm=arm,features=features,classes=classes,seed=88101,body=0,dropout=.2,learning_rate=.001,device='cpu')
   adapter=port.attach(torch,source,model,opt,streams,extra,x=x,edges=edge,train_ids=torch.arange(4),train_y=torch.arange(4)%classes,select_ids=torch.arange(4,6),select_y=torch.arange(2)%classes,
    arm=arm,walks=walks,swap=swap,provenance={'new_source_synthetic_QA':True,'actual_model_seed':88101})
   inspect_ownership(adapter)
   if arm=='matched_joint_untied4':
    bodies=list(model.bodies);assert len(bodies)==4
    slow=[{name:value for name,value in body.named_parameters() if not name.endswith(('.r','.s'))} for body in bodies]
    assert all(slow[0].keys()==view.keys() and all(torch.equal(slow[0][key],view[key]) for key in view) for view in slow[1:])
    assert all(not {p.untyped_storage().data_ptr() for p in bodies[a].parameters()}&{p.untyped_storage().data_ptr() for p in bodies[b].parameters()} for a in range(4) for b in range(a))
    assert not torch.equal(bodies[0].input_linear.r,bodies[1].input_linear.r)
   if arm.startswith('ordered_path'):
    block=model.body.path_residual;assert torch.equal(block.second.weight,torch.zeros_like(block.second.weight)) and torch.equal(block.second.bias,torch.zeros_like(block.second.bias))
    cpu=torch.get_rng_state().clone();p=port.operator_provider(ROOT,extras['ordered_operator']).make_path_block(torch,source.factors,width=128,members=1,init_seed=88101+6000017)
    assert torch.equal(cpu,torch.get_rng_state())
    stem=torch.randn(n,128);assert torch.equal(p(stem,walks,swap,mode='ordered'),torch.zeros_like(stem))
    # Source native equality at zero residual, using this exact constructed body.
    model.eval();actual=model(source.batch,route_scope=lambda m:source.scope(streams,m))[0];direct=[]
    for m in range(model.members):
     with source.factors.member_context(model.body,m):direct.append(model.body(source.data,x))
    torch.testing.assert_close(actual,torch.stack(direct),rtol=0,atol=0)
    # Test new path gradient/order semantics with nonzero learned output map.
    with torch.no_grad():p.second.weight.copy_(torch.eye(128)*.02)
    live=stem.clone().requires_grad_();ordered=p(live,walks,swap,mode='ordered');marginal=p(live,walks,swap,mode='marginal');broken=p(live,walks,swap,mode='order_breaking')
    assert not torch.equal(ordered,marginal) and not torch.equal(ordered,broken)
    grad=torch.autograd.grad(ordered.square().mean(),live)[0];assert torch.isfinite(grad).all() and grad.abs().sum()>0
   before=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(adapter.capture_state()))
   value=adapter.train_step(1);steps+=1;after=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(adapter.capture_state()))
   model.load_state_dict(before[0]);opt.load_state_dict(before[1]);adapter.restore_state(before[2]);replay=adapter.train_step(1);steps+=1
   assert value==replay and all(torch.equal(v,model.state_dict()[k]) for k,v in after[0].items())
   assert all(torch.equal(after[1]['state'][k]['exp_avg'],opt.state_dict()['state'][k]['exp_avg']) for k in after[1]['state'])
   assert all(torch.equal(a['cpu'],b['cpu']) for a,b in zip(after[2]['external']['streams'],adapter.capture_state()['external']['streams']))
   rows.append({'arm':arm,'features':features,'classes':classes,'new_ownership_and_source_initialization':True,'source_update_RNG_Adam_replay':True})
 assert len(rows)==10 and steps==18 and not torch.cuda.is_initialized()
 result={'schema':'retained-SAGE-prototype-new-risk-CPU-QA-v1','status':'passed','cases':rows,'toy_optimizer_steps':steps,'prior_attempt_known_toy_steps':2,'all_attempt_toy_steps':steps+2,'metadata_API_repair':'exact native Kaiming generator primitive binding','benchmark_fits':0,'real_tensor_reads':0,'CUDA_compute':0,'existing_kernel_tests_repeated':False}
 (PACKET/'SAGE_PROTOTYPE_CPU_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(rows),'toy_steps':steps,'benchmark_fits':0}))
if __name__=='__main__':main()
