"""Raw-target SSL and frozen-backbone classification, imported after custody."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

BASE=Path(__file__).resolve().parents[1]/'wikics_staged_private_graph_residual_method_preparation_20261007_v4/method.py'
BASE_SHA='d5874791fa6b6a323f1a672553ce22297df2fb45d083afc50dc7b16947bf8a1c'
def base_module():
    if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Pinned graph-path source changed')
    name='ssl_original_graph_method_v4'
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=BASE.resolve():raise ValueError('Graph method shadowed')
        return module
    spec=importlib.util.spec_from_file_location(name,BASE);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

BASE_METHOD=base_module()
ARMS=('B0','BR','BC','S0','SC')
SSL_ARMS=('BR','BC','SC')

def finite(parameters,gradients=False):
    norm=0.
    for p in parameters:
        if not bool(torch.isfinite(p).all()):raise FloatingPointError('Nonfinite private parameters')
        if gradients:
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Disconnected/nonfinite active gradient')
            norm+=float(p.grad.detach().double().square().sum())
    return norm**.5

class Predictor(nn.Module):
    def __init__(self,donor,seed,arm,device):
        super().__init__();self.donor=donor;self.arm=arm;self.single=arm.startswith('S')
        self.donor.eval()
        for p in self.donor.parameters():p.requires_grad_(False)
        graphs=[];heads=[];self.streams=[]
        for m in range(4):
            path,stream=BASE_METHOD.isolated_path(seed+1009*m,device)
            graphs.append(path.graph);heads.append(path.output);self.streams.append(stream)
        self.paths=nn.ModuleList(graphs)
        if self.single:
            saved=BASE_METHOD.rng_snapshot()
            try:
                torch.manual_seed(seed+790001);torch.cuda.manual_seed_all(seed+790001)
                self.classifier=nn.Linear(2048,10).to(device)
                nn.init.zeros_(self.classifier.weight);nn.init.zeros_(self.classifier.bias)
            finally:BASE_METHOD.rng_restore(saved)
        else:self.classifiers=nn.ModuleList(heads)
        self.clean_cache=None;self.decoders=None
        ptrs=[p.untyped_storage().data_ptr() for p in self.parameters()]
        if len(ptrs)!=len(set(ptrs)):raise ValueError('Private parameter storage aliases')
        self.assert_zero_classes()
    def class_heads(self):return [self.classifier] if self.single else list(self.classifiers)
    def assert_zero_classes(self):
        if any(bool((p!=0).any()) for head in self.class_heads() for p in head.parameters()):raise ValueError('Class outputs must remain zero before class fitting')
    def train(self,mode=True):
        super().train(mode);self.donor.eval();return self
    def encode(self,x,edge):
        if any(p.requires_grad for p in self.donor.parameters()):raise ValueError('Backbone must be frozen')
        self.donor.eval()
        with torch.no_grad():h,z=BASE_METHOD.native_hidden_logits(self.donor,x,edge)
        if not bool(torch.isfinite(h).all() and torch.isfinite(z).all()):raise FloatingPointError('Nonfinite frozen encoding')
        return h.detach(),z.detach()
    def make_clean_cache(self,x,edge):self.clean_cache=self.encode(x,edge)
    def ssl_representation(self,m,xmask,hmask,edge):
        if self.clean_cache is not None:raise ValueError('Clean H0/z0 forbidden during SSL')
        return self.paths[m](xmask,hmask,edge)
    def logits(self,m,x,edge,training=False):
        if self.clean_cache is None:raise ValueError('Clean class cache absent')
        h,z=self.clean_cache
        self.paths[m].train(training)
        if training:
            with BASE_METHOD.route_rng(self.streams[m]):r=self.paths[m](x,h,edge)
        else:r=self.paths[m](x,h,edge)
        return z+self.classifiers[m](r)
    def single_logits(self,x,edge,training=False):
        if self.clean_cache is None:raise ValueError('Clean class cache absent')
        h,z=self.clean_cache;representations=[]
        for m in range(4):
            self.paths[m].train(training)
            if training:
                with BASE_METHOD.route_rng(self.streams[m]):r=self.paths[m](x,h,edge)
            else:r=self.paths[m](x,h,edge)
            representations.append(r)
        return z+self.classifier(torch.cat(representations,1))
    def serving(self,x,edge):
        self.eval()
        with torch.no_grad():
            if self.single:return F.softmax(self.single_logits(x,edge),1).unsqueeze(0)
            return torch.stack([F.softmax(self.logits(m,x,edge),1) for m in range(4)])

class Decoder(nn.Module):
    def __init__(self):super().__init__();self.affine=nn.Linear(512,300)
    def forward(self,representation,masked,adjacency):
        # Only corrupted graph representations enter this forward.
        remasked=representation.clone();remasked[masked]=0
        return self.affine(torch.sparse.mm(adjacency,remasked))

def adjacency(edge,nodes):
    source,target=edge
    degree=torch.bincount(target,minlength=nodes).to(torch.float32)
    if bool((degree==0).any()):raise ValueError('Prepared selfloop graph must cover every node')
    scale=degree.rsqrt();values=scale[target]*scale[source]
    return torch.sparse_coo_tensor(torch.stack((target,source)),values,(nodes,nodes),device=edge.device).coalesce()

def mask_block(x,train_ids,generator):
    allowed=torch.ones(len(x),dtype=torch.bool,device=x.device);allowed[train_ids]=False
    allowed&=x.detach().square().sum(1)>0
    eligible=allowed.nonzero().flatten().cpu();size=4*((len(eligible)//2)//4)
    if size==0:raise ValueError('No nonzero unlabeled raw feature targets')
    sampled=eligible[torch.randperm(len(eligible),generator=generator)[:size]].to(x.device)
    quarters=sampled.reshape(4,-1)
    xmask=x.detach().clone();xmask[sampled]=0
    return xmask,sampled,quarters,len(eligible)

def start_ssl(model,seed,device):
    saved=BASE_METHOD.rng_snapshot()
    try:
        torch.manual_seed(seed+870001);torch.cuda.manual_seed_all(seed+870001)
        model.decoders=nn.ModuleList([Decoder().to(device) for _ in range(4)])
    finally:BASE_METHOD.rng_restore(saved)
    if sum(p.numel() for p in model.decoders.parameters())!=615600:raise ValueError('Four actual reconstruction heads required')
    for head in model.class_heads():
        for p in head.parameters():p.requires_grad_(False)
    parameters=list(model.paths.parameters())+list(model.decoders.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8)
    if len({id(p) for p in parameters})!=len(parameters):raise ValueError('SSL optimizer ownership overlap')
    return optimizer,parameters

def ssl_update(model,opt,parameters,xmask,hmask,edge,masked,quarters,t,raw_targets,adj):
    model.eval();model.assert_zero_classes();opt.zero_grad(set_to_none=True);losses=[]
    counts=[]
    for m in range(4):
        target=quarters[t if model.arm=='BR' else (m+t)%4]
        representation=model.ssl_representation(m,xmask,hmask,edge)
        reconstructed=model.decoders[m](representation,masked,adj)
        # Original raw x enters only here as a loss target, never a forward input.
        cosine=F.cosine_similarity(raw_targets[target],reconstructed[target],dim=1,eps=1e-8)
        losses.append((1-cosine).square().mean());counts.append(len(target))
    loss=torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite raw scaled-cosine SSL loss')
    loss.backward();grad_norm=finite(parameters,True);opt.step();finite(parameters)
    model.assert_zero_classes()
    if any(p.grad is not None for head in model.class_heads() for p in head.parameters()):raise ValueError('SSL class heads received gradients')
    return {'scaled_cosine_squared':float(loss.detach()),'branch_losses':[float(v.detach()) for v in losses],
        'branch_target_counts':counts,'active_private_gradient_norm':grad_norm,'private_forward_backward':4,
        'decoder_forward_backward':4,'SSL_AdamW_updates':1,'clean_H0_z0_received':False}

def end_ssl(model):
    model.assert_zero_classes();model.decoders=None
    for head in model.class_heads():
        for p in head.parameters():p.requires_grad_(True)
    # Classification receives a fresh source Adam, with no SSL moments.
    for p in model.paths.parameters():p.grad=None

def bank_class_step(model,m,opt,x,edge,ids,labels,peers):
    opt.zero_grad(set_to_none=True);logits=model.logits(m,x,edge,True)
    loss,own,brier=BASE_METHOD.correction_loss(logits,peers,ids,labels,True)
    loss.backward();parameters=list(model.paths[m].parameters())+list(model.classifiers[m].parameters())
    norm=finite(parameters,True);opt.step();finite(parameters)
    return {'own_CE':float(own.detach()),'pool_Brier':float(brier.detach()),'J':float(loss.detach()),'active_gradient_norm':norm}

def single_class_step(model,opt,x,edge,ids,labels):
    opt.zero_grad(set_to_none=True);logits=model.single_logits(x,edge,True)
    p=F.softmax(logits,1)[ids];target=F.one_hot(labels,10).to(p.dtype)
    own=F.nll_loss(F.log_softmax(logits,1)[ids],labels);brier=(p-target).square().sum(1).mean();loss=own+2*brier
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite capable single class objective')
    loss.backward();parameters=list(model.paths.parameters())+list(model.classifier.parameters())
    norm=finite(parameters,True);opt.step();finite(parameters)
    return {'own_CE':float(own.detach()),'served_Brier':float(brier.detach()),'J':float(loss.detach()),'active_gradient_norm':norm}
