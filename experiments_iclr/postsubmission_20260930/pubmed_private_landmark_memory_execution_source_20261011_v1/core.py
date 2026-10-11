"""Fixed PubMed landmark memory; Torch/F/exact FactorLinear injected by owner."""
import copy
import math
from contextlib import nullcontext

SEEDS=(9101,9203,9307)
ARMS=('B1','B4','T16','S64','FS64','LM4','I4[S64]','I4[FS64]')
COUNTS={'B1':(2083818,1030),'B4':(2083818,1030),'T16':(2127338,44550),'S64':(2128874,46086),'FS64':(2130442,47654),'LM4':(2128874,46086),'I4[S64]':(8515496,184344),'I4[FS64]':(8521768,190616)}
DIMENSIONS={'Aq':(256,32),'Ak':(256,32),'Av':(256,32),'Bq':(256,32),'Bk':(32,32),'Bv':(32,32),'Wo':(32,256)}

def build_arm(torch,F,arm,seed,ancestors,caches,geometry,*,FactorLinear):
    """New heads/memory from exact selected sources, no checkpoint/data access."""
    if arm not in ARMS or seed not in SEEDS:raise ValueError('Fixed LM4 arm/seed')
    if torch.__version__.split('+')[0]!='2.1.2' or torch.get_default_dtype()!=torch.float32 or torch.empty(0).device.type!='cpu':raise ValueError('Bound Torch2.1.2 FP32 CPU construction')
    independent=arm.startswith('I4[');needed=4 if independent else 1
    if len(ancestors)<needed or len(caches)<needed:raise ValueError('Declared individually selected source ancestors')
    if geometry['anchors'].shape!=(64,) or geometry['bias'].shape!=(19717,64):raise ValueError('Qualified complete same64 landmark geometry')

    class EmptyLinear(torch.nn.Linear):
        # No discarded constructor draws, including zero Wo. The six actual
        # initializers consume ONLY the isolated generator in fixed map order.
        def reset_parameters(self):pass

    class Memory(torch.nn.Module):
        def __init__(self,body):
            super().__init__();self.maps=torch.nn.ModuleDict()
            generator=torch.Generator(device='cpu').manual_seed(seed+491003041+1000003*body)
            factorized=arm in ('FS64','I4[FS64]')
            for name,(input_dim,output_dim) in DIMENSIONS.items():
                layer=EmptyLinear(input_dim,output_dim,bias=False)
                with torch.no_grad():
                    if name=='Wo':layer.weight.zero_()
                    else:torch.nn.init.xavier_uniform_(layer.weight,generator=generator)
                self.maps[name]=FactorLinear(layer,1) if factorized else layer
            shape=(4,16,32) if arm=='LM4' else ((16,32) if arm=='T16' else (64,32))
            self.E=torch.nn.Parameter(torch.zeros(shape))
        def constants(self,U):
            # Fresh differentiable products ONCE per bank and incoming state.
            # Caller retains only this shared tape across that bank's backwards.
            return dict(K=self.maps['Ak'](U),V=self.maps['Av'](U),Q=self.maps['Bq'](U))
        def forward(self,H,U,anchors,bias,view,constants):
            if arm in ('LM4','T16'):
                start=16*view;ids=anchors[start:start+16];distance_bias=bias[:,start:start+16]
                offset=self.E[view] if arm=='LM4' else self.E
            else:ids=anchors;distance_bias=bias;offset=self.E
            query=self.maps['Aq'](U.index_select(0,ids))+offset
            write=(query@constants['K'].T/math.sqrt(32)+distance_bias.T).softmax(-1)@constants['V']
            memory_key=self.maps['Bk'](write);memory_value=self.maps['Bv'](write)
            read=(constants['Q']@memory_key.T/math.sqrt(32)+distance_bias).softmax(-1)@memory_value
            return H+.25*self.maps['Wo'](read)

    class Model(torch.nn.Module):
        def __init__(self,body):
            super().__init__();self.arm=arm;self.body_index=body;self.seed=seed
            self.views=self.members=4 if arm in ('B4','T16','LM4') else 1
            self.native=copy.deepcopy(ancestors[body]);head=self.native.lin3
            # The exact source factory loads its factor module separately per
            # source session, so equivalent bound classes have distinct Python
            # identities. Match the pinned implementation and parameter law.
            if type(head).__name__!='FactorLinear' or type(head).forward.__code__.co_code!=FactorLinear.forward.__code__.co_code or set(dict(head.named_parameters()))!={'weight','bias','r','s'} or head.weight.shape!=(3,256) or head.bias.shape!=(3,) or head.r.shape!=(1,256) or head.s.shape!=(1,3) or head.member!=0:raise ValueError('Exact selected native factorized lin3/member0')
            for p in self.native.parameters():p.requires_grad_(False)
            for p in head.parameters():p.requires_grad_(True)
            H,U=caches[body]['H'],caches[body]['U']
            if H.shape!=U.shape or H.shape!=(19717,256) or H.dtype!=torch.float32 or U.dtype!=torch.float32 or H.requires_grad or U.requires_grad:raise ValueError('Exact immutable raw H and memory-only U')
            # External immutable payloads are bound by checkpoint metadata and
            # guard versions/fingerprints; restoration never copies into them.
            self.register_buffer('fixed_H',H,persistent=False);self.register_buffer('fixed_U',U,persistent=False)
            self.register_buffer('anchors',geometry['anchors'],persistent=False);self.register_buffer('distance_bias',geometry['bias'],persistent=False)
            self.memory=Memory(body) if arm not in ('B1','B4') else None
            self.train(True)
        def train(self,mode=True):
            super().train(mode);self.native.eval();self.native.lin3.train(mode);return self
        def bank_constants(self):return {} if self.memory is None else self.memory.constants(self.fixed_U)
        def view_forward(self,view,constants,route_rng_context=None):
            if view not in range(self.views):raise ValueError('Actual own view only')
            if self.training and route_rng_context is None:raise ValueError('Persistent independently owned tail stream')
            context=route_rng_context(view) if self.training else nullcontext()
            with context:
                hidden=self.fixed_H if self.memory is None else self.memory(self.fixed_H,self.fixed_U,self.anchors,self.distance_bias,view,constants)
                if self.native.lin3.member!=0:raise ValueError('Source head always factual factor member0')
                return self.native.lin3(F.dropout(hidden,p=.5,training=self.training))
        def forward(self,route_rng_context=None):
            constants=self.bank_constants()
            return torch.stack([self.view_forward(v,constants,route_rng_context) for v in range(self.views)])

    models=tuple(Model(b) for b in (range(4) if independent else (0,)))
    return dict(arm=arm,seed=seed,models=models,genuine_independent=independent,optimizer_trajectories=needed,members=4 if independent else models[0].views,expected_parameters=COUNTS[arm][0],expected_trainable_parameters=COUNTS[arm][1])

def inspect_parameter_ownership(core):
    seen=set();storage=set();resident=live=0
    for model in core['models']:
        for p in model.parameters():
            key=(str(p.device),p.untyped_storage().data_ptr())
            if id(p) in seen or key in storage:raise ValueError('Independent learners share parameter objects/storage')
            seen.add(id(p));storage.add(key);resident+=p.numel();live+=p.numel() if p.requires_grad else 0
    if resident!=core['expected_parameters'] or live!=core['expected_trainable_parameters']:raise ValueError('Frozen LM4 resident/live counts')
    return dict(resident_parameters=resident,trainable_parameters=live,optimizer_trajectories=core['optimizer_trajectories'],actual_served_views=core['members'],genuine_independent=core['genuine_independent'],common_projections_reused_once_per_bank=True,H_unclipped=True)

def make_optimizers(torch,core):
    result=[]
    for model in core['models']:
        head=list(model.native.lin3.parameters());memory=[] if model.memory is None else list(model.memory.parameters())
        groups=[dict(params=head,lr=.005,weight_decay=.001)]
        if memory:groups.append(dict(params=memory,lr=.0005,weight_decay=1e-8))
        optimizer=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8)
        actual=[p for g in optimizer.param_groups for p in g['params']]
        if len({id(p) for p in actual})!=len(actual) or {id(p) for p in actual}!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('Exact explicit head/memory optimizer ownership')
        result.append(optimizer)
    return tuple(result)
