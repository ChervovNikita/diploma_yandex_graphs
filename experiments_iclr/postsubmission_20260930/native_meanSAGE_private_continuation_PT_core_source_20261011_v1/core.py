"""Disabled source-only continuing P/T core; Torch/F and bound inputs injected."""
import copy
import math
from contextlib import nullcontext

ARMS=('head_only','mixed_single','private4','middle_state_mean','I4_mixed')
ROLES=('PT','TP','TT1','TT2')
OPERATIONS=('PP','PT','TP','TT')

def parameter_law(features,width,classes,arm):
    if arm not in ARMS or any(type(v) is not int or v<=0 for v in (features,width,classes)):raise ValueError('Positive source dimensions and defined architecture interface')
    source=2*features*width+width+2*width*classes+classes
    head=2*width*classes+classes
    shared=0 if arm=='head_only' else 15*width**2+18*width+8
    private=32 if arm in ('private4','middle_state_mean') else 0
    bodies=4 if arm=='I4_mixed' else 1
    return dict(source_per_body=source,live_head_per_body=head,shared_continuation_per_body=shared,private_e_per_body=private,resident_parameters=bodies*(source+shared+private),trainable_parameters=bodies*(head+shared+private),bodies=bodies)

def graph_check(torch,adjacency,degrees,nodes):
    if adjacency.layout!=torch.sparse_coo or not adjacency.is_coalesced() or tuple(adjacency.shape)!=(nodes,nodes):raise ValueError('Inherited complete coalesced COO A[dst,src] required')
    if adjacency.requires_grad or degrees.requires_grad or adjacency.dtype!=torch.float32 or degrees.dtype!=torch.float32 or tuple(degrees.shape)!=(nodes,1) or adjacency.device!=degrees.device:raise ValueError('Fixed native FP32 graph and Nx1 degree on one device')
    if not torch.equal(adjacency.values(),torch.ones_like(adjacency.values())):raise ValueError('Inherited binary support; no weights or support changes')
    counts=torch.bincount(adjacency.indices()[0],minlength=nodes).to(torch.float32).unsqueeze(1)
    if not torch.equal(degrees,counts) or not bool((degrees>0).all().item()):raise ValueError('Exact positive inherited row degree; no clamp')
    if int((adjacency.indices()[0]==adjacency.indices()[1]).sum().item())!=nodes:raise ValueError('Inherited one self entry per node; add no new loops')

def native_layers(ancestor):
    if getattr(ancestor,'members',None)!=1 or len(ancestor.bodies)!=1 or len(ancestor.bodies[0].layers)!=2 or getattr(ancestor,'feedback',None)!='none':raise ValueError('Selected ordinary native-core P1 ancestor required; no wrapper approximation')
    first,last=ancestor.bodies[0].layers
    if any(layer.factorized for layer in (first,last)):raise ValueError('Ordinary unfactorized selected source only')
    features,width=first.aggregate_weight.shape;classes=last.aggregate_weight.shape[1]
    if first.skip_weight.shape!=(features,width) or first.bias.shape!=(1,width) or last.aggregate_weight.shape!=(width,classes) or last.skip_weight.shape!=(width,classes) or last.bias.shape!=(1,classes):raise ValueError('Exact native U0/V0/b0 and U1/V1/b1 shapes')
    expected={id(p) for layer in (first,last) for p in (layer.aggregate_weight,layer.skip_weight,layer.bias)}
    if {id(p) for p in ancestor.parameters()}!=expected or len(expected)!=6:raise ValueError('Complete six native ordinary parameter objects, no auxiliary maps')
    return first,last,(features,width,classes)

def cache_stem(torch,F,ancestor,x,adjacency,degrees):
    """Deferred source-only helper: exact undropped native clip-then-ReLU H0.

    Caller binds source/checkpoint/input/cache custody before any admitted use.
    No data load, source acquisition, dropout draw or optimizer is performed.
    """
    first,last,(features,width,classes)=native_layers(ancestor);nodes=x.shape[0];graph_check(torch,adjacency,degrees,nodes)
    if tuple(x.shape)!=(nodes,features) or x.requires_grad or x.dtype!=torch.float32 or x.device!=adjacency.device or any(p.dtype!=torch.float32 or p.device!=x.device for p in ancestor.parameters()):raise ValueError('Unchanged complete native X and selected FP32 source on graph device')
    if x.is_sparse and not x.is_coalesced():raise ValueError('Inherited raw sparse X must remain coalesced')
    with torch.no_grad():
        aggregate=first._transform(x,first.aggregate_weight,None,None)
        z=torch.sparse.mm(adjacency,aggregate)/degrees+first._transform(x,first.skip_weight,None,None)+first.bias[0]
        clipped=z/torch.linalg.vector_norm(z,ord=2,dim=1,keepdim=True).clamp_min(1.)
        hidden=F.relu(clipped).detach()
    if tuple(hidden.shape)!=(nodes,width) or not torch.isfinite(hidden).all().item():raise ValueError('Complete finite undropped native H0')
    return hidden

def build_arm(torch,F,arm,ancestors,cached_H0,adjacency,degrees,*,initialization_seeds):
    """Fresh continuation/head copies, never a fit or source acquisition.

    Seeds are REQUIRED caller bindings: one isolated CPU generator per actual
    body. Draw order is block1 PT/TP/TT1/TT2/G1 then block2 same. No offsets or
    global constructor draws. Each own TRAIN model needs context(view) streams.
    Independent bodies remain separate own-loss/selector/optimizer interfaces.
    """
    if arm not in ARMS:raise ValueError('Defined continuing P/T interface required')
    bodies=4 if arm=='I4_mixed' else 1
    if len(ancestors)!=bodies or len(cached_H0)!=bodies or len(initialization_seeds)!=bodies or any(type(s) is not int for s in initialization_seeds):raise ValueError('Exactly one selected source/cache/caller seed per actual body')
    if torch.get_default_dtype()!=torch.float32 or torch.empty(0).device.type!='cpu':raise ValueError('FP32 CPU initialization required')
    shapes=[native_layers(source)[2] for source in ancestors]
    if len(set(shapes))!=1:raise ValueError('Matched complete native source dimensions')
    features,width,classes=shapes[0];nodes=cached_H0[0].shape[0];graph_check(torch,adjacency,degrees,nodes)
    for source,H0 in zip(ancestors,cached_H0):
        if tuple(H0.shape)!=(nodes,width) or H0.requires_grad or H0.dtype!=torch.float32 or H0.device!=adjacency.device or any(p.dtype!=torch.float32 or p.device!=H0.device for p in source.parameters()) or not torch.isfinite(H0).all().item():raise ValueError('Audited fixed finite H0 and ordinary FP32 source on inherited graph device')
    if bodies==4:
        supplied=[{(str(p.device),p.untyped_storage().data_ptr()) for p in a.parameters()} for a in ancestors]
        if len({id(a) for a in ancestors})!=4 or any(supplied[i]&supplied[j] for i in range(4) for j in range(i)) or len({H.untyped_storage().data_ptr() for H in cached_H0})!=4:raise ValueError('Four distinct supplied independent source bodies/caches; no source aliases')
    private=arm in ('private4','middle_state_mean')
    def clip(z):return z/torch.linalg.vector_norm(z,ord=2,dim=1,keepdim=True).clamp_min(1.)

    class Block(torch.nn.Module):
        def __init__(self,generator):
            super().__init__();self.W=torch.nn.ParameterDict();self.b=torch.nn.ParameterDict()
            for role in ROLES:
                weight=torch.empty(width,width,device='cpu');bound=math.sqrt(6/(2*width))
                with torch.no_grad():weight.uniform_(-bound,bound,generator=generator)
                self.W[role]=torch.nn.Parameter(weight);self.b[role]=torch.nn.Parameter(torch.zeros(width,device='cpu'))
            first=torch.empty(3*width,width,device='cpu');bound=math.sqrt(6/(4*width))
            with torch.no_grad():first.uniform_(-bound,bound,generator=generator)
            self.G1=torch.nn.Parameter(first);self.beta1=torch.nn.Parameter(torch.zeros(width,device='cpu'))
            self.G2=torch.nn.Parameter(torch.zeros(width,4,device='cpu'));self.beta2=torch.nn.Parameter(torch.zeros(4,device='cpu'))
        def T(self,role,z):
            relu=F.relu(z@self.W[role]+self.b[role]);return relu,clip(relu)
        def bank(self,H,P):
            PH=P(H);P2H=P(PH);retained={}
            retained['PT']=self.T('PT',PH);retained['TP']=self.T('TP',H)
            retained['TT1']=self.T('TT1',H);retained['TT2']=self.T('TT2',retained['TT1'][1])
            operations=torch.stack((clip(P2H),retained['PT'][1],clip(P(retained['TP'][1])),retained['TT2'][1]))
            context=torch.cat((H,(PH-H).abs(),(P2H-H).abs()),dim=1)
            gate=F.relu(context@self.G1+self.beta1)@self.G2+self.beta2
            return dict(H=H,PH=PH,P2H=P2H,T_relu_and_clipped=retained,operations=operations,context=context,gate_logits=gate)

    class Continuation(torch.nn.Module):
        def __init__(self,seed):
            super().__init__();generator=torch.Generator(device='cpu').manual_seed(seed)
            self.blocks=torch.nn.ModuleList((Block(generator),Block(generator)))
            self.Wo=torch.nn.Parameter(torch.zeros(width,width,device='cpu'))
            self.e=torch.nn.Parameter((torch.eye(4,device='cpu')-.25).unsqueeze(0).repeat(2,1,1)) if private else None
        def mix(self,bank,block,route):
            gate=bank['gate_logits'] if self.e is None else bank['gate_logits']+self.e[block,route]
            weights=gate.softmax(dim=-1);return (weights.T.unsqueeze(-1)*bank['operations']).sum(0),weights
        def forward(self,H0,P):
            first=self.blocks[0].bank(H0,P);routes=4 if private else 1
            first_pairs=[self.mix(first,0,m) for m in range(routes)];first_states=[pair[0] for pair in first_pairs]
            if arm=='middle_state_mean':
                incoming=torch.stack(first_states).mean(0);second=self.blocks[1].bank(incoming,P)
                banks=[second]*4
            else:banks=[self.blocks[1].bank(H,P) for H in first_states]
            second_pairs=[self.mix(bank,1,m) for m,bank in enumerate(banks)]
            final=[pair[0] for pair in second_pairs]
            hidden=[clip(H0+.25*(H@self.Wo)) for H in final]
            return hidden,dict(first_bank=first,first_states=first_states,first_weights=[p[1] for p in first_pairs],second_banks=banks,second_states=final,second_weights=[p[1] for p in second_pairs])

    class Model(torch.nn.Module):
        def __init__(self,body):
            super().__init__();self.arm='mixed_single' if arm=='I4_mixed' else arm;self.body_index=body;self.initialization_seed=initialization_seeds[body]
            self.native=copy.deepcopy(ancestors[body]);first,last,_=native_layers(self.native)
            for p in first.parameters():p.requires_grad_(False)
            for p in last.parameters():p.requires_grad_(True)
            self.register_buffer('fixed_H0',cached_H0[body],persistent=False);self.register_buffer('fixed_A',adjacency,persistent=False);self.register_buffer('fixed_degree',degrees,persistent=False)
            self.memory=None if arm=='head_only' else Continuation(initialization_seeds[body]).to(cached_H0[body].device)
            self.TRAIN_views=4;self.eval_views=1 if self.arm=='mixed_single' else 4
            self._fixed_versions=tuple(x._version for x in (self.fixed_H0,self.fixed_A,self.fixed_degree));self.train(True)
        def train(self,mode=True):
            super().train(mode);self.native.bodies[0].layers[0].eval();return self
        def P(self,z):return torch.sparse.mm(self.fixed_A,z)/self.fixed_degree
        def fixed_guard(self):
            if self._fixed_versions!=tuple(x._version for x in (self.fixed_H0,self.fixed_A,self.fixed_degree)):raise ValueError('Inherited H0/A/degree mutated')
        def regularizer(self):
            first=self.native.bodies[0].layers[0]
            return .05*(first.aggregate_weight.square().sum()+first.skip_weight.square().sum())
        def continued_hidden(self):
            self.fixed_guard()
            return ([self.fixed_H0]*4,dict(head_only=True)) if self.memory is None else self.memory(self.fixed_H0,self.P)
        def decode(self,hidden,view,route_rng_context):
            if self.training and route_rng_context is None:raise ValueError('Caller-owned persistent private tail context(view) required')
            with route_rng_context(view) if self.training else nullcontext():
                D=F.dropout(hidden,p=.4,training=self.training);last=self.native.bodies[0].layers[1]
                # The SAME dropped D feeds the native aggregate and skip maps.
                aggregate=last._transform(D,last.aggregate_weight,None,None)
                return clip(self.P(aggregate)+last._transform(D,last.skip_weight,None,None)+last.bias[0])
        def forward(self,route_rng_context=None,*,return_details=False):
            hidden,details=self.continued_hidden();views=self.TRAIN_views if self.training else self.eval_views
            logits=torch.stack([self.decode(hidden[0] if self.arm=='mixed_single' else hidden[v],v,route_rng_context) for v in range(views)])
            self.fixed_guard();return (logits,details) if return_details else logits
        def objective(self,logits,train_ids,train_y):
            if not self.training or logits.shape!=(4,nodes,classes):raise ValueError('Complete own four TRAIN logits and caller-bound role only')
            return torch.stack([F.cross_entropy(z[train_ids],train_y) for z in logits]).mean()+self.regularizer()

    models=tuple(Model(body) for body in range(bodies));law=parameter_law(features,width,classes,arm)
    result=dict(arm=arm,models=models,genuine_independent=bodies==4,optimizer_trajectories=bodies,TRAIN_tail_draws=4*bodies,eval_predictions=4 if arm!='mixed_single' else 1,parameter_law=law,initialization_seeds=tuple(initialization_seeds),dimensions=dict(nodes=nodes,features=features,width=width,classes=classes),no_optimizer_or_fit_policy=True)
    inspect_parameter_ownership(result);return result

def inspect_parameter_ownership(core):
    objects=set();storage=set();resident=live=private=0
    for model in core['models']:
        for parameter in model.parameters():
            key=(str(parameter.device),parameter.untyped_storage().data_ptr())
            if id(parameter) in objects or key in storage:raise ValueError('Learners share parameter objects/storage')
            objects.add(id(parameter));storage.add(key);resident+=parameter.numel();live+=parameter.numel() if parameter.requires_grad else 0
        if model.memory is not None and model.memory.e is not None:private+=model.memory.e.numel()
    law=core['parameter_law']
    if resident!=law['resident_parameters'] or live!=law['trainable_parameters'] or private!=law['private_e_per_body']*law['bodies']:raise ValueError('Exact native/common/private parameter law violated')
    return dict(resident_parameters=resident,trainable_parameters=live,private_e_parameters=private,genuine_independent=core['genuine_independent'],all_registered_parameter_objects_and_storage_disjoint=True)

def inspect_optimizer_ownership(core,optimizers):
    """Inspect only caller-created optimizers; choose no optimizer law here."""
    if len(optimizers)!=len(core['models']):raise ValueError('One separately owned optimizer per actual body')
    seen=set()
    for model,optimizer in zip(core['models'],optimizers):
        owned=[p for group in optimizer.param_groups for p in group['params']];ids={id(p) for p in owned}
        if len(ids)!=len(owned) or ids!={id(p) for p in model.parameters() if p.requires_grad} or seen&ids:raise ValueError('Exact disjoint live-only own optimizer ownership')
        seen.update(ids)
    return dict(own_optimizers=len(optimizers),live_only=True,disjoint=True,optimizer_law_supplied_by_caller=True)

def assemble_eval(torch,core):
    """Dropout-off original probability mean; never an independent fit API."""
    if any(model.training for model in core['models']):raise ValueError('Individually selected/restored eval models only')
    with torch.no_grad():logits=torch.cat([model() for model in core['models']],dim=0)
    if logits.shape[0]!=core['eval_predictions']:raise ValueError('Complete actual eval interface')
    return logits,logits.softmax(-1).mean(0)
