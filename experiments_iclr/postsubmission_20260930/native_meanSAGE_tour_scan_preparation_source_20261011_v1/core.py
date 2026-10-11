"""Frozen ten-arm tour scanner core; injected torch/F/native core/scan backend.

No numeric imports, constructors, data, optimizer, selector or execution at
module import. Existing competent native meanSAGE source bytes are reused.
"""
from contextlib import nullcontext
import math

BLOCKS=(10103,10427,10949)
NODES,FEATURES,HIDDEN,CLASSES=34493,8415,32,5
SPEC={
 'P1':dict(native='P1',views=1,scan=False,tied=False,independent=False,parameters=538917),
 'F1':dict(native='F1',views=1,scan=False,tied=False,independent=False,parameters=555885),
 'I4':dict(native='I4',views=1,scan=False,tied=False,independent=True,parameters=2223540),
 'S4':dict(native='S4',views=4,scan=False,tied=False,independent=False,parameters=606900),
 'PV1':dict(native='P1',views=4,scan=True,tied=True,independent=False,parameters=542053),
 'WI4':dict(native='I4',views=1,scan=True,tied=False,independent=True,parameters=2236084),
 'V1':dict(native='F1',views=4,scan=True,tied=True,independent=False,parameters=559021),
 'SW4':dict(native='S4',views=4,scan=True,tied=False,independent=False,parameters=610036),
 'C4':dict(native='S4',views=4,scan=True,tied=False,independent=False,parameters=610036),
 'R4':dict(native='S4',views=4,scan=True,tied=False,independent=False,parameters=610036),
}

def build_arm(torch,F,arm,block,*,native_core,scan_backend):
    """Return explicit joint or four-own learner objects; no optimizer creation.

    PV1/V1 use native parameter rows1 with logical views4. WI4 uses four native
    M1 objects/scanner objects; body m always scans tour m, local RNG index0.
    Shared scanner maps have no per-view factor/bias dimension.
    """
    if arm not in SPEC or block not in BLOCKS or type(block) is not int:
        raise ValueError('Exact frozen arm/block required')
    if torch.get_default_dtype()!=torch.float32 or torch.empty(0).device.type!='cpu':
        raise ValueError('Native float32 CPU construction before device transfer')
    spec=SPEC[arm];ancestor=native_core.build_arm(torch,F,spec['native'],block)
    scan=scan_backend.make_segmented_scan(torch)

    def clip_rows(z):
        return z/torch.linalg.vector_norm(z,ord=2,dim=1,keepdim=True).clamp_min(1.)

    class Scanner(torch.nn.Module):
        def __init__(self,body_index):
            super().__init__()
            self.W_u=torch.nn.Parameter(torch.empty(HIDDEN,HIDDEN,device='cpu'))
            self.W_a=torch.nn.Parameter(torch.empty(HIDDEN,HIDDEN,device='cpu'))
            self.W_o=torch.nn.Parameter(torch.zeros(HIDDEN,HIDDEN,device='cpu'))
            self.b_u=torch.nn.Parameter(torch.zeros(HIDDEN,device='cpu'))
            self.b_a=torch.nn.Parameter(torch.full((HIDDEN,),math.log(99),device='cpu'))
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(block+337000043+1000003*body_index)
                torch.nn.init.xavier_uniform_(self.W_u);torch.nn.init.xavier_uniform_(self.W_a)
        def forward(self,hidden,tour,reset_every_token=False):
            # q uses the ORIGINAL learned gate; scan masks carry only.
            u=torch.tanh(hidden@self.W_u+self.b_u)
            a=torch.sigmoid(hidden@self.W_a+self.b_a)
            q=(1-a)*u
            z=scan_backend.bidirectional_context(torch,scan,a,q,tour,reset_every_token)
            return clip_rows(hidden+.25*(z@self.W_o))

    class Model(torch.nn.Module):
        def __init__(self,native,body_index):
            super().__init__();self.native=native;self.body_index=body_index
            self.members=spec['views'];self.views=spec['views'];self.arm=arm
            self.genuine_independent=spec['independent'];self.complete_parameter_tying=spec['tied']
            self.native_parameter_routes=native.members
            self.parameter_route_ids=(0,)*self.views if self.native_parameter_routes==1 else tuple(range(self.views))
            self.tour_ids=tuple((body_index if spec['independent'] else (0 if arm=='C4' else j)) for j in range(self.views))
            self.scanner=Scanner(body_index) if spec['scan'] else None
        @property
        def bodies(self):return self.native.bodies
        def regularizer(self):
            # Native rows1: one FULL ordinary/effective penalty, even views4.
            # Native rows4: mean of four own effective penalties, exactly once.
            return self.native.regularizer()
        def forward(self,x,adjacency,degrees,tours=None,route_rng_context=None):
            if tuple(x.shape)!=(NODES,FEATURES) or tuple(adjacency.shape)!=(NODES,NODES) or tuple(degrees.shape)!=(NODES,1):
                raise ValueError('Bound full native graph required')
            if self.training and route_rng_context is None:
                raise ValueError('Persistent own view/body TRAIN RNG context required')
            if self.scanner is not None and (tours is None or len(tours)!=4):
                raise ValueError('Exactly four fixed admitted graph tours required')
            context=route_rng_context if self.training else lambda j:nullcontext()
            logits=[]
            for view,parameter_route in enumerate(self.parameter_route_ids):
                body,local=self.native._body(parameter_route)
                with context(view):
                    hidden=body.first(x,adjacency,degrees,local)
                    if self.scanner is not None:
                        hidden=self.scanner(hidden,tours[self.tour_ids[view]],arm=='R4')
                    logits.append(body.last(hidden,adjacency,degrees,local))
            # No hidden averaging. Each view gets its own complete final logits.
            return torch.stack(logits)

    models=tuple(Model(model,m if spec['independent'] else 0) for m,model in enumerate(ancestor['models']))
    return dict(arm=arm,block=block,models=models,genuine_independent=spec['independent'],optimizer_trajectories=4 if spec['independent'] else 1,native_bodies=4 if spec['independent'] else 1,members=4 if spec['independent'] else spec['views'],scan_contexts=4 if spec['scan'] else 0,expected_parameters=spec['parameters'],higher_order_scan_supported=False,serving='mean_class_probabilities',hidden_averaging=False)


def inspect_parameter_ownership(core):
    """Deferred exact count, disjoint genuine learners and complete view tying."""
    models=core['models'];params=[p for m in models for p in m.parameters()]
    count=sum(p.numel() for p in params)
    if count!=core['expected_parameters']:raise ValueError('Frozen parameter count differs')
    objects,storage=set(),set()
    for model in models:
        group=list(model.parameters());ids={id(p) for p in group};stores={(str(p.device),p.untyped_storage().data_ptr()) for p in group}
        if ids & objects or stores & storage:raise ValueError('Own learners share parameters/storage')
        objects.update(ids);storage.update(stores)
        if model.complete_parameter_tying:
            if len(model.bodies)!=1 or model.native_parameter_routes!=1 or model.parameter_route_ids!=(0,0,0,0):
                raise ValueError('PV1/V1 must tie ALL native/factor/bias rows')
            for layer in model.bodies[0].layers:
                if layer.bias.shape[0]!=1:raise ValueError('No private per-view bias rows')
                if layer.factorized and any(getattr(layer,name).shape[0]!=1 for name in ('aggregate_r','aggregate_s','skip_r','skip_s')):
                    raise ValueError('No private per-view fast factors')
        if model.scanner is not None:
            if {n:tuple(p.shape) for n,p in model.scanner.named_parameters()}!={'W_u':(32,32),'W_a':(32,32),'W_o':(32,32),'b_u':(32,),'b_a':(32,)}:
                raise ValueError('One scalar-free map/bias set per actual learner')
    return dict(parameters=count,optimizer_trajectories=core['optimizer_trajectories'],native_bodies=core['native_bodies'],logical_member_or_view_outputs=core['members'],genuine_independent=core['genuine_independent'],complete_tying=core['arm'] in ('PV1','V1'),hidden_averaging=False)


def own_objective(torch,F,model,logits,train_ids,targets):
    """One model's mean own CE plus ONE native/effective penalty, never pool CE."""
    return torch.stack([F.cross_entropy(value[train_ids],targets) for value in logits]).mean()+model.regularizer()


def make_optimizers(core,source_optimizer_class):
    """One exact source optimizer per returned own learner; I4/WI4 have four."""
    optimizers=tuple(source_optimizer_class(model.parameters()) for model in core['models'])
    owned=set()
    for model,opt in zip(core['models'],optimizers):
        params=[p for group in opt.param_groups for p in group['params']];ids={id(p) for p in params}
        if len(ids)!=len(params) or ids!={id(p) for p in model.parameters() if p.requires_grad} or ids & owned:
            raise ValueError('Exact complete disjoint own optimizer parameters')
        owned.update(ids)
    if len(optimizers)!=core['optimizer_trajectories']:raise ValueError('Own optimizer trajectory count differs')
    return optimizers
