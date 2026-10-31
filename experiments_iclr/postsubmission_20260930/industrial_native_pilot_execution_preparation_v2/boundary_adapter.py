"""UNEXECUTED source-only adapter. Runtime/model/label use requires admission.

Preserves GraphPFN paper GATModule's exact [N,d_head,n_heads] channel packing.
The optional Torch COO backend is an algebraic compatibility candidate; native
DGL forward/backward and finite-difference qualification must pass first.
"""
from __future__ import annotations
import copy


def build_boundary_family(donor, members: int):
    import torch
    from torch import nn
    from torch.nn import functional as F
    if members not in (1, 4):
        raise ValueError('Only admitted K1/K4 route families')
    if donor.num_embedding is not None or donor.pearl is not None:
        raise ValueError('Native Tolokers config has no embeddings or PEARL')
    if len(donor.input) != 4 or len(donor.output) != 4:
        raise ValueError('Need exact native MLP input/output topology')
    if not isinstance(donor.input[0], nn.Linear) or not isinstance(donor.output[3], nn.Linear):
        raise ValueError('Boundary maps differ from pinned native model')
    if donor.input[0].out_features != 512 or donor.output[3].in_features != 512 or donor.output[3].out_features != 2:
        raise ValueError('Fixed width512/two logits required')

    class PrivateBoundary(nn.Module):
        def __init__(self, native):
            super().__init__()
            if native.bias is None:
                raise ValueError('Native boundary bias must be present')
            self.weight = nn.Parameter(native.weight.detach().clone())
            self.R = nn.Parameter(torch.ones((members, native.in_features), device=native.weight.device, dtype=native.weight.dtype))
            self.S = nn.Parameter(torch.ones((members, native.out_features), device=native.weight.device, dtype=native.weight.dtype))
            self.B = nn.Parameter(native.bias.detach().clone().expand(members, -1).clone())
        def forward(self, x, member):
            return F.linear(x*self.R[member], self.weight, None)*self.S[member] + self.B[member]

    class BoundaryFamily(nn.Module):
        def __init__(self):
            super().__init__()
            self.members = members
            self.core = copy.deepcopy(donor)
            self.stem = PrivateBoundary(self.core.input[0])
            self.head = PrivateBoundary(self.core.output[3])
            self.core.input[0] = nn.Identity()
            self.core.output[3] = nn.Identity()
        def forward(self, graph, x_num=None, x_other=None):
            values = [x for x in (x_num, x_other) if x is not None]
            if not values:
                raise ValueError('Missing admitted features')
            x = torch.cat(values, dim=1)
            logits = []
            for member in range(self.members):
                z = self.stem(x, member)
                z = self.core.input(z)
                z = self.core.backbone(graph, z)
                z = self.core.output(z)
                logits.append(self.head(z, member))
            return torch.stack(logits)
    return BoundaryFamily()


def replace_graph_primitives_for_ad(family, source_ids, destination_ids, num_nodes: int):
    """Replace only native GATModule's DGL primitives; parameter names stay intact.

    COO ids must be captured from the admitted native graph in its eid order.
    No graph modification, reordering, attention approximation or hidden pooling.
    Qualification against the original DGL family is mandatory before admission.
    """
    import torch
    from torch import nn
    if source_ids.dtype != torch.int64 or destination_ids.dtype != torch.int64:
        raise ValueError('Int64 node ids required for index_select/index_add')
    if source_ids.ndim != 1 or destination_ids.shape != source_ids.shape:
        raise ValueError('COO endpoint shape mismatch')
    if source_ids.device != destination_ids.device:
        raise ValueError('COO devices differ')

    class TorchGAT(nn.Module):
        def __init__(self, native):
            super().__init__()
            self.d, self.n_heads, self.d_head = native.d, native.n_heads, native.d_head
            if (self.d, self.n_heads, self.d_head) != (512, 4, 128):
                raise ValueError('Native GAT dimensions changed')
            for name in ('input_linear','attn_linear_u','attn_linear_v','attn_act','ffn'):
                setattr(self, name, getattr(native, name))
            self.register_buffer('_src', source_ids.detach().clone())
            self.register_buffer('_dst', destination_ids.detach().clone())
            self.num_nodes = num_nodes
        def forward(self, graph, x, edge_weights=None):
            del graph
            if edge_weights is not None:
                raise ValueError('Native GAT rejects supplied edge weights')
            if x.shape != (self.num_nodes, self.d):
                raise ValueError('Full admitted graph shape required')
            x = self.input_linear(x)
            scores = self.attn_act(self.attn_linear_u(x).index_select(0,self._src)
                                  + self.attn_linear_v(x).index_select(0,self._dst))
            # A detached per-destination shift leaves softmax and its derivative
            # unchanged algebraically. Explicit scatter avoids legacy DGL Function.
            dst_heads = self._dst[:,None].expand(-1,self.n_heads)
            maxima = torch.full((self.num_nodes,self.n_heads), -torch.inf,
                                 device=x.device,dtype=x.dtype)
            maxima = maxima.scatter_reduce(0,dst_heads,scores,reduce='amax',include_self=True).detach()
            numerator = (scores-maxima.index_select(0,self._dst)).exp()
            denom = torch.zeros_like(maxima).scatter_add(0,dst_heads,numerator)
            probs = numerator/denom.index_select(0,self._dst)
            values = x.reshape(self.num_nodes,self.d_head,self.n_heads)
            messages = values.index_select(0,self._src)*probs[:,None,:]
            aggregate = torch.zeros_like(values).index_add(0,self._dst,messages)
            return self.ffn(None, aggregate.reshape(self.num_nodes,self.d))
    layers = family.core.backbone.layers
    if len(layers) != 3:
        raise ValueError('Three native pre-LN residual blocks required')
    for layer in layers:
        native = layer.base
        if native.__class__.__name__ != 'GATModule':
            raise ValueError('Unexpected graph operator')
        layer.base = TorchGAT(native)
    return family


def map_native_parameter_name(name: str):
    if name == 'input.0.weight': return 'stem.weight', False
    if name == 'input.0.bias': return 'stem.B', True
    if name == 'output.3.weight': return 'head.weight', False
    if name == 'output.3.bias': return 'head.B', True
    return 'core.'+name, False


def transport_adamw(donor_model, donor_optimizer, family):
    """New named AdamW transport; never uses R17 coupled-Adam decay scaling.

    At identical K routes with mean member CE: private B gradients/moments scale
    1/K and1/K², eps scales1/K, and decoupled weight decay remains unchanged.
    New R/S have empty state. Dropout-off populated-history one-step parity is
    a required gate with all new R/S frozen. No equivalence is asserted after
    private routes diverge or new factors receive optimizer updates.
    """
    import torch
    if type(donor_optimizer) is not torch.optim.AdamW:
        raise ValueError('Exact ordinary AdamW required')
    k=family.members
    aliases={id(p):[] for p in donor_model.parameters()}
    for name,p in donor_model.named_parameters(remove_duplicate=False): aliases[id(p)].append(name)
    target=dict(family.named_parameters())
    mapped,replicated={},set()
    for pid,names in aliases.items():
        translations=[map_native_parameter_name(n) for n in names]
        if len({n for n,_ in translations}) != 1:
            raise ValueError('Native parameter aliases need explicit retained mapping')
        mapped[pid]=translations[0][0]
        if translations[0][1]:replicated.add(pid)
    groups=[];seen=set();target_seen=set();group_by_native={}
    for index,group in enumerate(donor_optimizer.param_groups):
        options=copy.deepcopy({n:v for n,v in group.items() if n not in ('params','param_names')})
        if options.get('capturable') or options.get('fused') or options.get('differentiable'):
            raise ValueError('Only ordinary uncapturable/unfused AdamW admitted')
        shared,bias=[],[]
        for p in group['params']:
            if id(p) in seen or id(p) not in mapped:raise ValueError('Optimizer coverage/alias mismatch')
            seen.add(id(p));group_by_native[id(p)]=options
            name=mapped[id(p)]
            if name not in target or name in target_seen:raise ValueError('Mapping missing or merged native params')
            target_seen.add(name);q=target[name]
            expected=p.detach().expand(k,-1) if id(p) in replicated else p.detach()
            if not torch.equal(q.detach(),expected):raise ValueError('Warm parameter copy changed')
            (bias if id(p) in replicated else shared).append(q)
        if shared:groups.append(dict(params=shared,**options))
        if bias:
            opts=copy.deepcopy(options);opts['eps']=options['eps']/k
            # AdamW's shrinkage is parameter-based, independent of gradient scale.
            # Preserve weight_decay, including native zero-decay bias group.
            groups.append(dict(params=bias,**opts))
    if seen != {id(p) for p in donor_model.parameters()}:raise ValueError('Native optimizer omits a parameter')
    native=dict(donor_model.named_parameters())
    for boundary,native_weight in [('stem','input.0.weight'),('head','output.3.weight')]:
        opts=copy.deepcopy(group_by_native[id(native[native_weight])])
        for suffix in ('R','S'):
            name=boundary+'.'+suffix
            if name in target_seen:raise ValueError('New factor overlaps native mapping')
            target_seen.add(name);groups.append(dict(params=[target[name]],**copy.deepcopy(opts)))
    if target_seen!=set(target):raise ValueError('Incomplete family optimizer coverage')
    optimizer=torch.optim.AdamW(groups)
    for group in donor_optimizer.param_groups:
        for p in group['params']:
            q=target[mapped[id(p)]];old=donor_optimizer.state.get(p,{})
            if not set(old)<= {'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}:raise ValueError('Unknown AdamW state')
            state={}
            for key,value in old.items():
                if torch.is_tensor(value):
                    if key=='step':state[key]=value.detach().clone().cpu()
                    else:
                        if value.shape!=p.shape:raise ValueError('Moment shape mismatch')
                        v=value.detach().to(device=q.device,dtype=q.dtype)
                        if id(p) in replicated:
                            scale=k if key=='exp_avg' else k*k
                            v=v.expand_as(q).clone()/scale
                        else:v=v.clone()
                        state[key]=v
                else:state[key]=copy.deepcopy(value)
            optimizer.state[q]=state
    return optimizer
