"""New prospective namespace/head/trial custody fixture, not closed Q21 audit."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(P))
import curvature_lifecycle as port

def equal(torch,a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(torch,a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(torch,x,y) for x,y in zip(a,b))
 return a==b

def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 bindings=json.loads((P/'SOURCE_BINDINGS.json').read_text());point={'learning_rate':.001,'dropout':.7};sources=port.modules(ROOT,bindings,point);api=sources['native_api'];adapter=sources['integration'];selector=sources['selector'];boundary=sources['backbone_boundary_adapter']
 constants=selector.FrozenConstants(**json.loads((P/'PROSPECTIVE_CONSTANTS_UNADMITTED.json').read_text())['values']);constants.validate()
 with port.aliases(sources):native=api.TeacherFamily(api.specification('polynormer_r','single_author',0,88101));opt=api.optimizer_for(native)
 assert native.specification['native']['weight_decay']==5e-5 and native.specification['native']['heads']==8 and native.specification['native']['hidden_per_head']==64
 n=12;torch.manual_seed(187);x=torch.randn(n,745);ids=torch.arange(n);edge=torch.stack((ids.repeat(2),torch.cat((ids,(ids+1)%n))));graph=SimpleNamespace(teacher_input=x,teacher_edge_index=edge,teacher_backbone='polynormer_r',classes=8);train=SimpleNamespace(nodes=torch.arange(10),labels=torch.arange(10)%8)
 adapter.train_update(native,opt,graph,train);native.set_global_stage(True);adapter.train_update(native,opt,graph,train);steps=2
 snapshot=adapter.named_optimizer_snapshot(native,opt);k1=adapter.clone_boundary(native,boundary,1);k4=adapter.clone_boundary(native,boundary,4);transported,transport=adapter.transport_optimizer(native,snapshot,k4);frozen=adapter.named_optimizer_snapshot(k4,transported)
 assert transport['native_steps_preserved'] and transport['all_native_inactive_state_retained'] and transport['private_bias_eps_scale']==.25 and transport['private_bias_coupled_decay_scale']==.25
 k1.eval();theta0,logits,binding=selector.bind_head_only(k1,'polynormer_r',(x,edge));theta=theta0.clone().requires_grad_();value=logits(theta);gradient=torch.autograd.grad(value.square().sum(),theta)[0];assert gradient.shape==(512,) and torch.isfinite(gradient).all() and gradient.abs().sum()>0
 before=(copy.deepcopy(k4.state_dict()),copy.deepcopy(frozen));rng=adapter.rng_snapshot();trial=selector.one_adam_trial(k4,frozen,theta0.expand(4,-1).clone(),'polynormer_r',(x,edge),train.nodes,train.labels,rng,adapter);steps+=trial['updates']
 assert trial['status']=='eligible' and trial['updates']==1 and trial['trial_state_discarded'] and equal(torch,before[0],k4.state_dict()) and equal(torch,before[1],adapter.named_optimizer_snapshot(k4,transported)) and equal(torch,rng,adapter.rng_snapshot())
 assert steps==3 and not torch.cuda.is_initialized()
 result={'schema':'new-curvature-typed-namespace-head-trial-custody-CPU-fixture-v1','status':'passed','scope':'new seed/HPO/continuation namespace and source named-Adam/head binding; one tiny discarded trial; no closed Q12/Q21 numerical audit rerun','binding_receipt':sources['binding_receipt'],'source_new_seed':88101,'source_Photo_heads8_H64_recipe_verified':True,'source_global_head_width512_functional_gradient':True,'source_named_Adam_B_epsilon_decay_quarter_moment_step_transport':True,'source_discarded_trial_model_optimizer_and_RNG_unchanged':True,'toy_optimizer_steps':steps,'scientific_fits':0,'actual_input_arrays_opened':0,'CUDA_compute':0,'constants_are_admitted':False,'typed_master_integration_admitted':False,'GO':False}
 (P/'CPU_SOURCE_CUSTODY_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','toy_steps':steps,'scientific_fits':0,'typed_disabled':True}))

if __name__=='__main__':main()
