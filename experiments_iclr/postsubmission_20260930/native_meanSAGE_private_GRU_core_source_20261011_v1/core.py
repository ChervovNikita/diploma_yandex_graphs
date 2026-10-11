"""Frozen staged common-native/private-GRU allocation; numerical modules injected."""
import copy
import math
from contextlib import nullcontext

BLOCKS=(10103,10427,10949)
ARMS=('B4','G1','GF1','GM4','GN4','GI16','GFI16')
COUNTS={'B4':(538917,325),'G1':(546277,7685),'GF1':(546565,7973),'GM4':(548005,9413),'GN4':(614297,75705),'GI16':(2185108,30740),'GFI16':(2186260,31892)}

def build_arm(torch,F,arm,block,ancestors,cached_H,*,gru_backend):
    """Four selected ordinary native P1 ancestors; never construct/fetch here.

    Each returned model owns its live parameters; fixed H may be readonly
    shared. GI16/GFI16 return four genuine own learners, each four views.
    """
    if arm not in ARMS or block not in BLOCKS:raise ValueError('Frozen arm/block')
    needed=4 if arm in ('GI16','GFI16') else 1
    if len(ancestors)<needed or len(cached_H)<needed:
        raise ValueError('One candidate ancestor, or four audited independent reference ancestors/caches')
    if torch.get_default_dtype()!=torch.float32 or torch.empty(0).device.type!='cpu':raise ValueError('Native FP32 CPU construction')
    gru_backend.require_runtime(torch);independent=arm in ('GI16','GFI16')
    def clip(z):return z/torch.linalg.vector_norm(z,ord=2,dim=1,keepdim=True).clamp_min(1.)

    class Memory(torch.nn.Module):
        def __init__(self,body):
            super().__init__();self.factorized=arm in ('GF1','GM4','GFI16');self.rows=4 if arm=='GM4' else 1
            self.W_ih=torch.nn.Parameter(torch.empty(96,32));self.W_hh=torch.nn.Parameter(torch.empty(96,32))
            generator=torch.Generator(device='cpu').manual_seed(block+459000131+1000003*body)
            bound=1/math.sqrt(32)
            with torch.no_grad():self.W_ih.uniform_(-bound,bound,generator=generator);self.W_hh.uniform_(-bound,bound,generator=generator)
            input_bias=torch.zeros(self.rows,3,32);input_bias[:,1,:]=math.log(99)
            self.b_i=torch.nn.Parameter(input_bias);self.b_h=torch.nn.Parameter(torch.zeros(self.rows,3,32))
            if self.factorized:
                self.r_x=torch.nn.Parameter(torch.ones(self.rows,3,32));self.r_h=torch.nn.Parameter(torch.ones(self.rows,3,32));self.s=torch.nn.Parameter(torch.ones(self.rows,3,32))
            self.W_o=torch.nn.Parameter(torch.zeros(32,32))
        def effective_weights(self,view):
            row=view if self.rows==4 else 0
            if self.factorized:
                # Fresh differentiable products at the current optimizer state.
                a=(self.s[row,:, :,None]*self.W_ih.reshape(3,32,32)*self.r_x[row,:,None,:]).reshape(96,32)
                b=(self.s[row,:, :,None]*self.W_hh.reshape(3,32,32)*self.r_h[row,:,None,:]).reshape(96,32)
            else:a,b=self.W_ih,self.W_hh
            return [a,b,self.b_i[row].reshape(96),self.b_h[row].reshape(96)]
        def forward(self,hidden,packed,view):
            weights=self.effective_weights(view)
            node=gru_backend.node_memory(torch,hidden,weights,packed,self.training)
            return clip(hidden+.25*(node@self.W_o))

    class Model(torch.nn.Module):
        def __init__(self,body):
            super().__init__();self.arm=arm;self.body_index=body;self.members=4;self.views=4
            self.native=copy.deepcopy(ancestors[body]);self.memory=Memory(body) if arm!='B4' else None
            if self.native.members!=1 or len(self.native.bodies)!=1 or any(layer.factorized for layer in self.native.bodies[0].layers):raise ValueError('Selected ordinary M1 P1 ancestor only')
            first,last=self.native.bodies[0].layers
            for p in first.parameters():p.requires_grad_(False)
            for p in last.parameters():p.requires_grad_(True)
            if arm=='GN4':
                for layer,ni,no in ((first,8415,32),(last,32,5)):
                    layer.factorized=True
                    for name,dimension in (('aggregate_r',ni),('aggregate_s',no),('skip_r',ni),('skip_s',no)):
                        setattr(layer,name,torch.nn.Parameter(torch.ones(4,dimension)))
                    layer.bias_offset=torch.nn.Parameter(torch.zeros(4,no))
            else:
                if tuple(cached_H[body].shape)!=(34493,32) or cached_H[body].requires_grad:raise ValueError('Audited fixed eval common H cache only')
                self.register_buffer('fixed_H',cached_H[body])
            self.train(True)
        @property
        def bodies(self):return self.native.bodies
        def train(self,mode=True):
            super().train(mode);self.native.bodies[0].layers[0].eval();return self
        def native_layer(self,layer,x,adjacency,degrees,view,dropout):
            if dropout:x=F.dropout(x,p=.4,training=self.training)
            if arm=='GN4':
                r,s=layer.aggregate_r[view],layer.aggregate_s[view];rr,ss=layer.skip_r[view],layer.skip_s[view];offset=layer.bias_offset[view]
            else:r=s=rr=ss=None;offset=0
            agg=layer._transform(x,layer.aggregate_weight,r,s)
            return clip(torch.sparse.mm(adjacency,agg)/degrees+layer._transform(x,layer.skip_weight,rr,ss)+layer.bias[0]+offset)
        def regularizer(self):
            if arm=='GN4':return .05*torch.stack([self.native.bodies[0].layers[0].effective_square(m) for m in range(4)]).mean()
            return self.native.regularizer()
        def view_forward(self,x,adjacency,degrees,tours,view,route_rng_context=None):
            if view not in range(4) or tuple(x.shape)!=(34493,8415) or len(tours)!=4:raise ValueError('Full qualified graph/four tours/view')
            if self.training and route_rng_context is None:raise ValueError('Persistent private tail stream required')
            context=route_rng_context(view) if self.training else nullcontext()
            with context:
                first,last=self.native.bodies[0].layers
                hidden=F.relu(self.native_layer(first,x,adjacency,degrees,view,False)) if arm=='GN4' else self.fixed_H
                if self.memory is not None:hidden=self.memory(hidden,tours[view],view)
                return self.native_layer(last,hidden,adjacency,degrees,view,True)
        def forward(self,x,adjacency,degrees,tours,route_rng_context=None):
            # Evaluation only in controller; TRAIN streams each live view backward.
            return torch.stack([self.view_forward(x,adjacency,degrees,tours,v,route_rng_context) for v in range(4)])

    models=tuple(Model(body) for body in (range(4) if independent else (0,)))
    return dict(arm=arm,block=block,models=models,genuine_independent=independent,optimizer_trajectories=4 if independent else 1,members=16 if independent else 4,expected_parameters=COUNTS[arm][0],expected_trainable_parameters=COUNTS[arm][1],recurrent_contexts=0 if arm=='B4' else (16 if independent else 4))

def inspect_parameter_ownership(core):
    seen=set();stores=set();resident=trainable=0
    for model in core['models']:
        for p in model.parameters():
            key=(str(p.device),p.untyped_storage().data_ptr())
            if id(p) in seen or key in stores:raise ValueError('Distinct learners share registered parameter objects/storage')
            seen.add(id(p));stores.add(key);resident+=p.numel();trainable+=p.numel() if p.requires_grad else 0
    if resident!=core['expected_parameters'] or trainable!=core['expected_trainable_parameters']:raise ValueError('Frozen staged resident/trainable count differs')
    return dict(resident_parameters=resident,trainable_parameters=trainable,optimizer_trajectories=core['optimizer_trajectories'],members=core['members'],genuine_independent=core['genuine_independent'],native_factors_live=core['arm']=='GN4',shared_readout_unfactorized=True)

def make_optimizers(core,optimizer_class):
    result=tuple(optimizer_class(p for p in m.parameters() if p.requires_grad) for m in core['models'])
    for model,opt in zip(core['models'],result):
        owned=[p for g in opt.param_groups for p in g['params']]
        if {id(p) for p in owned}!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('Exact live-only native source optimizer ownership')
    return result
