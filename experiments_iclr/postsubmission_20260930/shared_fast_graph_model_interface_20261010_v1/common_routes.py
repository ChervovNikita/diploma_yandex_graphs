"""Small graph BatchEnsemble/route-exchange prototype; stdlib-only import.

Callers supply their existing torch, PyG nn and FactorLinear module. This file
owns no data reader, trainer, selector, optimizer, launcher or job admission.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math


@dataclass
class NodeState:
    node_ids: object
    hidden: object
    continuation: object = None


def node_ids(torch, batch):
    ids = batch.get('node_ids', torch.arange(len(batch['x']), device=batch['x'].device))
    if ids.dtype != torch.long or ids.shape != (len(batch['x']),) or len(torch.unique(ids)) != len(ids):
        raise ValueError('Unique label-free identity for every feature row is required')
    return ids


@contextmanager
def existing_member_rng_scope(torch, streams, member, cuda_index=None):
    """Resume/update the caller's existing Session streams; no new RNG owner."""
    stream = streams[member]
    with torch.random.fork_rng(devices=[] if cuda_index is None else [cuda_index]):
        torch.set_rng_state(stream['cpu'])
        if cuda_index is not None:
            torch.cuda.set_rng_state(stream['cuda'], cuda_index)
        try:
            yield
        finally:
            stream['cpu'] = torch.get_rng_state()
            if cuda_index is not None:
                stream['cuda'] = torch.cuda.get_rng_state(cuda_index)


class NativeModelAdapter:
    """Original models.Model GCN/GAT/SAGE residual stack, without rewriting convs."""
    def __init__(self,torch,body):
        expected={'GCN':('GCNModule','GCNConv',('lin',)),
                  'GAT':('GATModule','GATConv',('lin',)),
                  'SAGE':('SAGEModule','SAGEConv',('lin_l','lin_r'))}
        if getattr(body,'model_name',None) not in expected:
            raise ValueError('Explicit original Model GCN/GAT/SAGE only; not TABMModel or GAT-sep')
        required=('input_linear','dropout','act','residual_modules','output_normalization','output_linear')
        if any(not hasattr(body,k) for k in required) or len(body.residual_modules)<1:
            raise ValueError('Original layout with at least one complete residual module')
        module_name,conv_name,maps=expected[body.model_name]
        for residual in body.residual_modules:
            module=residual.module
            if module.__class__.__name__!=module_name or module.conv.__class__.__name__!=conv_name:
                raise ValueError('Native residual/convolution layout differs')
            if any(getattr(module.conv,k,None) is None or not hasattr(getattr(module.conv,k),'weight') for k in maps):
                raise ValueError('Concrete native PyG dense projection is missing')
        norms=[r.normalization for r in body.residual_modules]+[body.output_normalization]
        if any(n.__class__.__name__ not in ('LayerNorm','Identity','GraphNorm') for n in norms):
            raise ValueError('Running-stat or custom normalization needs a separate adapter')
        self.torch,self.body=torch,body;self.width=body.input_linear.weight.shape[0]
        self.name=body.model_name;self.site='after_complete_residual_module0'
        self.projection_maps=maps;self.continuing_blocks=len(body.residual_modules)-1
    def check(self,batch):
        if 'graph' not in batch or batch['graph'].edge_index is not batch['edge_index']:
            raise ValueError('Supply the same factual graph edge_index in graph and batch')
        if 'edge_attr' in batch or 'edge_weight' in batch:
            raise ValueError('Original Model does not consume these edge fields')
    def native(self,batch):
        captured=[];head=self.body.output_linear
        hook=head.register_forward_pre_hook(lambda _module,args:captured.append(args[0]))
        try:logits=self.body(batch['graph'],batch['x'])
        finally:hook.remove()
        if len(captured)!=1:raise ValueError('Unique original output head required')
        return logits.unsqueeze(-1) if logits.ndim==1 else logits,captured[0]
    def prefix(self,batch):
        b=self.body;h=b.act(b.dropout(b.input_linear(batch['x'])))
        h=b.residual_modules[0](batch['graph'],h)
        return NodeState(node_ids(self.torch,batch),h)
    def tail(self,state,hidden,batch):
        b=self.body;h=hidden
        for i in range(1,len(b.residual_modules)):
            h=b.residual_modules[i](batch['graph'],h)
        representation=b.output_normalization(h)
        return b.output_linear(representation),representation


class PolynormerAdapter:
    """Saved native local/global equation mapping; no WikiCS source mutation.

    Supply a fresh unmodified native Polynormer body, not the running V3 bank.
    Native local attention vectors remain shared slow parameters in this port.
    """
    def __init__(self,torch,body):
        required=('lin_in','local_convs','h_lins','lins','lns','betas','global_attn','pred_local','pred_global','ln')
        if any(not hasattr(body,k) for k in required) or len(body.local_convs)<2:
            raise ValueError('Explicit saved native Polynormer layout required')
        self.torch,self.body=torch,body;self.width=body.lin_in.weight.shape[0]
        self.name='polynormer';self.site='after_local_block0_before_block1'
    def check(self,batch):
        if 'edge_attr' in batch or 'edge_weight' in batch:
            raise ValueError('Saved native Polynormer consumes unweighted edge_index only')
    def local(self,i,x,batch):
        b=self.body;F=self.torch.nn.functional
        if b.pre_ln:x=b.pre_lns[i](x)
        h=F.relu(b.h_lins[i](x));x=F.relu(b.local_convs[i](x,batch['edge_index'])+b.lins[i](x))
        x=F.dropout(x,p=b.dropout,training=b.training)
        beta=F.sigmoid(b.betas[i]).unsqueeze(0) if b.beta<0 else b.betas[i].unsqueeze(0)
        return (1-beta)*b.lns[i](h*x)+beta*x
    def native(self,batch):
        head=self.body.pred_global if self.body._global else self.body.pred_local;captured=[]
        hook=head.register_forward_pre_hook(lambda _module,args:captured.append(args[0]))
        try:logits=self.body(batch['x'],batch['edge_index'])
        finally:hook.remove()
        if len(captured)!=1:raise ValueError('Unique native head representation required')
        return logits,captured[0]
    def prefix(self,batch):
        b=self.body;F=self.torch.nn.functional
        x=F.dropout(batch['x'],p=b.in_drop,training=b.training)
        x=F.dropout(b.lin_in(x),p=b.dropout,training=b.training)
        h=self.local(0,x,batch)
        return NodeState(node_ids(self.torch,batch),h,h)
    def tail(self,state,hidden,batch):
        b=self.body;x=hidden;x_local=state.continuation
        for i in range(1,len(b.local_convs)):
            x=self.local(i,x,batch);x_local=x_local+x
        representation=b.global_attn(b.ln(x_local)) if b._global else x_local
        head=b.pred_global if b._global else b.pred_local
        return head(representation),representation


def make_route_block(torch,width,members,kind,rank,seed):
    """Known attention/separable residuals with identical4*width*rank weights."""
    if kind not in ('exchange','separable') or min(width,rank)<=0 or members<2:
        raise ValueError('Positive widths and at least two routes')
    nn=torch.nn;generator=torch.Generator(device='cpu').manual_seed(seed)
    def parameter(rows,columns,zero=False):
        value=torch.empty(rows,columns,dtype=torch.float32,device='cpu')
        if zero:value.zero_()
        else:value.uniform_(-math.sqrt(6/(rows+columns)),math.sqrt(6/(rows+columns)),generator=generator)
        return nn.Parameter(value)
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            if kind=='exchange':
                self.Q=parameter(width,rank);self.K=parameter(width,rank)
                self.V=parameter(width,rank);self.U=parameter(rank,width,True)
            else:self.B=parameter(width,2*rank);self.C=parameter(2*rank,width,True)
        def forward(self,raw):
            z=nn.functional.layer_norm(raw,(width,),eps=1e-5)
            if kind=='separable':return raw+nn.functional.relu(z@self.B)@self.C
            scores=((z@self.Q)@(z@self.K).transpose(-2,-1))/math.sqrt(rank)
            scores=scores.masked_fill(torch.eye(members,dtype=torch.bool,device=raw.device)[None],float('-inf'))
            return raw+(scores.softmax(-1)@(z@self.V))@self.U
    return Block()


def wrap_shared(torch,factors,body,adapter,*,members=4,kind='baseline',rank=16,block_seed=1):
    """Register one native body and private fast banks; install before optimizer.

    `factors` is the existing core/factors.py module. Training callers provide
    a persistent per-member RNG context factory, reusable from their Session.
    """
    if adapter.body is not body or kind not in ('baseline','separable','exchange') or members not in (1,4):
        raise ValueError('Explicit matching adapter; M1 or M4; fixed three kinds')
    if members==1 and kind!='baseline':raise ValueError('M1 reference has no peer exchange')
    if kind!='baseline' and getattr(adapter,'continuing_blocks',len(getattr(body,'local_convs',[]))-1)<1:
        raise ValueError('Exchange site must have a continuing native graph block')
    if torch.get_default_dtype()!=torch.float32:raise ValueError('Use the existing float32 provider convention')
    if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in body.parameters()):
        raise ValueError('Wrap fresh materialized float32 CPU body, then move the whole wrapper before optimizer')
    if any(isinstance(m,factors.FactorLinear) for m in body.modules()):raise ValueError('Already factorized body; fresh native Model required')
    if list(body.named_buffers()):raise ValueError('Running/statistical or static buffers need a separately reviewed adapter')
    modules=list(body.named_modules(remove_duplicate=False))
    if len({id(m) for _,m in modules})!=len(modules):raise ValueError('Tied module aliases need explicit factor placement')
    native=list(body.named_parameters(remove_duplicate=False));native_ids={id(p) for _,p in native}
    if len(native_ids)!=len(native):raise ValueError('Tied parameter aliases need an explicit ownership contract')
    for _,module in modules:
        if getattr(module,'cached',False):raise ValueError('Mutable adjacency caching is unsupported')
        if hasattr(module,'has_uninitialized_params') and module.has_uninitialized_params():raise ValueError('Materialize lazy maps before wrapping')
    maps=factors.install_factors(body,members)
    if not native_ids.issubset({id(p) for p in body.parameters()}):raise ValueError('Native parameter identity changed')
    if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in body.parameters()):
        raise ValueError('Default allocation must keep the new factor banks on float32 CPU')
    native_names={'body.'+name for name,p in body.named_parameters() if id(p) in native_ids}
    nn=torch.nn
    class SharedRoutes(nn.Module):
        def __init__(self):
            super().__init__();self.body=body;self.members=members;self.kind=kind
            self.block=make_route_block(torch,adapter.width,members,kind,rank,block_seed) if kind!='baseline' else None
            self.factorized_maps=maps
        def parameter_roles(self):
            private={id(p) for module in self.body.modules() if isinstance(module,factors.FactorLinear) for p in (module.r,module.s)}
            return {role:[name for name,p in self.named_parameters() if (id(p) in private if role=='private_fast' else name in native_names if role=='native_slow' else id(p) not in private and name not in native_names)]
                for role in ('native_slow','private_fast','shared_block')}
        def check_optimizer_ownership(self,optimizer):
            expected={id(p) for p in self.parameters() if p.requires_grad}
            actual=[id(p) for group in optimizer.param_groups for p in group['params']]
            if len(actual)!=len(set(actual)) or set(actual)!=expected:
                raise ValueError('Create optimizer after wrapping: all registered trainable parameters exactly once')
        def forward(self,batch,*,route_scope=None):
            if any(k in batch for k in ('y','labels','train_mask','valid_mask','test_mask')):raise ValueError('Model interface is label-free')
            if self.training and route_scope is None:raise ValueError('Training needs the caller persistent member RNG scope')
            if batch['x'].ndim!=2 or batch['edge_index'].ndim!=2 or batch['edge_index'].shape[0]!=2:
                raise ValueError('One homogeneous factual feature matrix and edge_index')
            adapter.check(batch);versions=(batch['x']._version,batch['edge_index']._version)
            def call(member,operation):
                with (route_scope(member) if route_scope else nullcontext()),factors.member_context(self.body,member):
                    return operation()
            if self.kind=='baseline':rows=[call(m,lambda:adapter.native(batch)) for m in range(members)]
            else:
                states=[call(m,lambda:adapter.prefix(batch)) for m in range(members)]
                if any(not torch.equal(s.node_ids,states[0].node_ids) or s.hidden.shape!=(len(batch['x']),adapter.width) for s in states):
                    raise ValueError('Every live prefix must align the same label-free nodes and width')
                mixed=self.block(torch.stack([s.hidden for s in states],dim=1))
                rows=[call(m,lambda m=m:adapter.tail(states[m],mixed[:,m,:],batch)) for m in range(members)]
            if versions!=(batch['x']._version,batch['edge_index']._version):raise ValueError('A route mutated factual graph/input')
            selected=batch.get('ids',torch.arange(len(batch['x']),device=batch['x'].device))
            return torch.stack([r[0][selected] for r in rows]),torch.stack([r[1][selected] for r in rows])
        def serve(self,logits,mode='logit_mean'):
            if mode=='logit_mean':return logits.mean(0)
            if mode=='probability_mean':return (logits.sigmoid() if logits.shape[-1]==1 else logits.softmax(-1)).mean(0)
            raise ValueError('Freeze raw-logit or native probability mean before training/selection')
    return SharedRoutes()
