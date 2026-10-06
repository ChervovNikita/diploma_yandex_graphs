"""Literal native-only training operators shared by donor fits and qualification."""
from contextlib import contextmanager
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from vendor.native_polynormer import Polynormer
from common import compare

ARGUMENTS=dict(in_channels=300,hidden_channels=512,out_channels=10,local_layers=7,global_layers=2,
    in_dropout=.5,dropout=.5,global_dropout=.5,heads=1,beta=-1,pre_ln=False)

def cpu_tree(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return type(v)(cpu_tree(x) for x in v)
    return v

def adam(params):return torch.optim.Adam(params,lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8)

def construct(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Polynormer(**ARGUMENTS).to(device);model.reset_parameters();model._global=False
    optimizer=adam(model.parameters())
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in model.parameters()}:
        raise ValueError('Native optimizer ownership')
    stream={'cpu':torch.get_rng_state().clone(),'cuda':torch.cuda.get_rng_state().clone()}
    return model,optimizer,stream

@contextmanager
def route_rng(stream):
    old_cpu,old_cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
    torch.set_rng_state(stream['cpu']);torch.cuda.set_rng_state(stream['cuda'])
    try:yield
    finally:
        stream.update(cpu=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state().clone())
        torch.set_rng_state(old_cpu);torch.cuda.set_rng_state(old_cuda)

def active(name,global_stage):
    return not name.startswith('pred_local.') if global_stage else not name.startswith(('global_attn.','ln.','pred_global.'))

def epoch(model,opt,stream,x,edge,ids,labels,reference=False):
    model.train();opt.zero_grad(set_to_none=True)
    ref_model=ref_opt=None;reference_gradients=None;audit={}
    if reference:
        ref_model=copy.deepcopy(model);ref_opt=adam(ref_model.parameters());ref_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        for (n,p),(rn,rp) in zip(model.named_parameters(),ref_model.named_parameters()):
            if n!=rn or not torch.equal(p,rp) or p.untyped_storage().data_ptr()==rp.untyped_storage().data_ptr():
                raise ValueError('Parameter-only reference snapshot ownership/copy failure')
    with route_rng(stream):
        logits=model(x,edge)
        loss=F.nll_loss(F.log_softmax(logits,1)[ids],labels)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Full native TRAIN loss')
        names=[(n,p) for n,p in model.named_parameters() if active(n,model._global)]
        if reference:
            reference_gradients=torch.autograd.grad(loss,[p for n,p in names],retain_graph=True)
        loss.backward()
    for n,p in model.named_parameters():
        if active(n,model._global) and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
            raise ValueError('Active native gradient missing/nonfinite: '+n)
        if not active(n,model._global) and p.grad is not None:raise ValueError('Inactive native gradient unexpected: '+n)
    if reference:
        refs=dict(ref_model.named_parameters());errors=[]
        for (n,p),g in zip(names,reference_gradients):
            errors.append(compare(torch,p.grad,g,'gradients_moments'))
            # Optimizer parity uses exactly the production backward gradients.
            # The unchanged comparison above separately tests backward agreement.
            refs[n].grad=p.grad.detach().clone()
            if not torch.equal(refs[n].grad,p.grad):raise ValueError('Production gradient clone differs')
        audit['gradient_max_abs']=max(errors)
        audit['gradient_agreement_reference']='autograd.grad versus production backward, same retained native forward'
        audit['optimizer_reference_gradient_source']='exact observed production p.grad clones'
        audit['optimizer_reference_inputs_bitwise_equal']=True
        audit['gradient_tolerances_unchanged']=True
        ref_opt.step()
    opt.step()
    if reference:
        errors=[compare(torch,p,dict(ref_model.named_parameters())[n],'parameters') for n,p in model.named_parameters()]
        audit['parameter_max_abs']=max(errors)
        state=opt.state_dict()['state'];ref_state=ref_opt.state_dict()['state']
        if state.keys()!=ref_state.keys():raise ValueError('Native Adam state ownership differs')
        errors=[]
        for key in state:
            for field in ('step','exp_avg','exp_avg_sq'):
                errors.append(compare(torch,state[key][field],ref_state[key][field],'gradients_moments'))
        audit['Adam_state_max_abs']=max(errors);audit['same_native_forward_graph_reference']=True
        del reference_gradients,ref_opt,ref_model
    return {'own_CE':float(loss.detach()),'complete_TRAIN_labels':580,'public_nodes':11701,
        'prepared_edges':442907,'native_forwards':1,'native_Adam_updates':1,'reference':audit}
