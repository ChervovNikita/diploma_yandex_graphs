"""Independent Torch expression of pinned HGB HGT and attributed BE/CP factors.

Reference: THUDM/HGB ca6fd5bb, NC/benchmark/methods/HGT/model.py.
No DGL, graph acquisition, label loading, optimizer or training launch.
"""
from dataclasses import dataclass
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Relation:
    raw_id: int
    source: str
    target: str
    src: torch.Tensor
    dst: torch.Tensor

    @property
    def name(self):
        return f"raw{self.raw_id}__{self.source}__{self.target}"

    @property
    def canonical(self):
        return (self.source, self.name, self.target)


class HeteroGraph:
    """Shared topology only. Every forward allocates its own hidden/scratch frames.

    Node/canonical order is lexical. Supply actual DGL etype order when importing
    native parameter rows; otherwise canonical order is an explicit new schema.
    Raw IDs are retained, including distinct relations with identical endpoints.
    Caller owns immutable, local, int64 edge tensors; no self/reverse edges added.
    """
    def __init__(self, node_counts, relations, relation_names=None):
        self.ntypes = tuple(sorted(node_counts))
        self.node_counts = dict(node_counts)
        self.node_dict = {t:i for i,t in enumerate(self.ntypes)}
        self.relations = tuple(sorted(relations, key=lambda r:r.canonical))
        names = [r.name for r in self.relations]
        if len(names)!=len(set(names)) or len({r.raw_id for r in self.relations})!=len(names):
            raise ValueError('Each released raw relation ID must be distinct')
        if relation_names is not None:
            if len(relation_names)!=len(names) or set(relation_names)!=set(names):
                raise ValueError('Exact canonical relation-name row order required')
            names = list(relation_names)
        self.edge_dict = {name:i for i,name in enumerate(names)}
        self.raw_relation_to_row = {r.raw_id:self.edge_dict[r.name] for r in self.relations}
        for r in self.relations:
            if r.source not in node_counts or r.target not in node_counts:
                raise ValueError('Unknown endpoint type')
            if r.src.dtype!=torch.long or r.dst.dtype!=torch.long or r.src.ndim!=1 or r.src.shape!=r.dst.shape:
                raise ValueError('Local int64 edge vectors required')
            if r.src.device!=r.dst.device or r.src.numel()==0:
                raise ValueError('Nonempty relation required by actual author edge id[0]')
            if int(r.src.min())<0 or int(r.src.max())>=node_counts[r.source] or int(r.dst.min())<0 or int(r.dst.max())>=node_counts[r.target]:
                raise ValueError('Edge ID outside local type range')
        if {r.target for r in self.relations}!=set(self.ntypes):
            raise ValueError('Actual author model needs an incoming relation for every type')

    def to(self, device):
        return HeteroGraph(self.node_counts, [Relation(r.raw_id,r.source,r.target,
                          r.src.to(device),r.dst.to(device)) for r in self.relations],tuple(self.edge_dict))


def relation_sum(scores, values, destination, count):
    """Softmax over incoming edges within one canonical relation, per head.

    Zero-in-degree rows remain zero and participate in cross-relation mean.
    Requires Torch>=2.1 scatter_reduce; no dense node x node attention matrix.
    """
    index = destination[:,None].expand_as(scores)
    maximum = scores.new_full((count,scores.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0,index,scores,reduce='amax',include_self=True)
    numerator = (scores-maximum[destination]).exp()
    denominator = scores.new_zeros(maximum.shape).scatter_add(0,index,numerator)
    attention = numerator/denominator[destination]
    summed = values.new_zeros((count,)+values.shape[1:])
    return summed.index_add(0,destination,attention[...,None]*values)


class HGTLayer(nn.Module):
    """Native biased typed maps, relation maps, sigmoid skip, norm then dropout."""
    def __init__(self, width, num_types, num_relations, heads, use_norm=True, dropout=0.2):
        super().__init__()
        if width%heads:
            raise ValueError('Hidden width must be divisible by heads')
        self.out_dim,self.n_heads,self.d_k = width,heads,width//heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.k_linears,self.q_linears,self.v_linears,self.a_linears,self.norms = (nn.ModuleList() for _ in range(5))
        for _ in range(num_types):
            self.k_linears.append(nn.Linear(width,width))
            self.q_linears.append(nn.Linear(width,width))
            self.v_linears.append(nn.Linear(width,width))
            self.a_linears.append(nn.Linear(width,width))
            if use_norm:
                self.norms.append(nn.LayerNorm(width))
        self.relation_pri = nn.Parameter(torch.ones(num_relations,heads))
        self.relation_att = nn.Parameter(torch.empty(num_relations,heads,self.d_k,self.d_k))
        self.relation_msg = nn.Parameter(torch.empty_like(self.relation_att))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, graph, hidden, factors=None, member=None):
        keys,queries,values = {},{},{}
        for t in graph.ntypes:
            i = graph.node_dict[t]
            keys[t] = self.k_linears[i](hidden[t]).view(-1,self.n_heads,self.d_k)
            queries[t] = self.q_linears[i](hidden[t]).view(-1,self.n_heads,self.d_k)
            # Input scale precedes biased V; shared V bias is not input-scaled.
            value_input = hidden[t] if factors is None else hidden[t]*factors.a[member]
            values[t] = self.v_linears[i](value_input).view(-1,self.n_heads,self.d_k)
        incoming = {t:[] for t in graph.ntypes}
        for relation in graph.relations:
            r = graph.edge_dict[relation.name]
            key = torch.bmm(keys[relation.source][relation.src].transpose(1,0),
                            self.relation_att[r]).transpose(1,0)
            scores = (queries[relation.target][relation.dst]*key).sum(-1)*self.relation_pri[r]/self.sqrt_dk
            message = torch.bmm(values[relation.source][relation.src].transpose(1,0),
                                self.relation_msg[r]).transpose(1,0)
            # Output scale follows relation transform, including transformed bias.
            if factors is not None:
                message = message*factors.output(member,r).view(1,self.n_heads,self.d_k)
            reduced = relation_sum(scores,message,relation.dst,graph.node_counts[relation.target])
            incoming[relation.target].append(reduced.reshape(-1,self.out_dim))
        result = {}
        for t in graph.ntypes:
            i = graph.node_dict[t]
            # Mean over all incoming relation types, including zero-degree rows.
            aggregate = torch.stack(incoming[t]).mean(0)
            alpha = self.skip[i].sigmoid()
            mixed = self.a_linears[i](aggregate)*alpha+hidden[t]*(1-alpha)
            result[t] = self.drop(self.norms[i](mixed) if self.use_norm else mixed)
        return result


class NativeHGT(nn.Module):
    """Factor-off HGB HGT core. State keys match the preserved actual author model."""
    def __init__(self, graph, input_dims, width, classes, layers, heads=8, use_norm=True):
        super().__init__()
        self.gcs = nn.ModuleList()
        self.adapt_ws = nn.ModuleList(nn.Linear(input_dims[t],width) for t in graph.ntypes)
        self.gcs.extend(HGTLayer(width,len(graph.ntypes),len(graph.edge_dict),heads,use_norm)
                        for _ in range(layers))
        self.out = nn.Linear(width,classes)

    def forward(self, graph, features, target, factors=None, member=None, return_states=False):
        # Re-enter biased adapters from original immutable features for each route.
        hidden = {t:self.adapt_ws[graph.node_dict[t]](features[t]).tanh() for t in graph.ntypes}
        states = [hidden] if return_states else None
        for index,layer in enumerate(self.gcs):
            hidden = layer(graph,hidden,None if factors is None else factors[index],member)
            if return_states:
                states.append(hidden)
        logits = self.out(hidden[target])
        return (logits,states) if return_states else logits


class PrivateFactors(nn.Module):
    """Same-site global BE, rank-one CP residual, or unrestricted output table."""
    def __init__(self, a, b, c, q, u, mode):
        super().__init__()
        if mode not in ('be','cp','unrestricted'):
            raise ValueError('Factor mode must be be, cp or unrestricted')
        self.mode = mode
        self.a = nn.Parameter(a.clone())
        if mode=='unrestricted':
            self.s = nn.Parameter((b[:,None,:]+c[:,None,None]*q[None,:,None]*u[None,None,:]).clone())
        else:
            self.b = nn.Parameter(b.clone())
            if mode=='cp':
                self.c,self.q,self.u = (nn.Parameter(value.clone()) for value in (c,q,u))

    def output(self, member, relation):
        if self.mode=='unrestricted':
            return self.s[member,relation]
        if self.mode=='cp':
            return self.b[member]+self.c[member]*self.q[relation]*self.u
        return self.b[member]


def initialized_factors(width, relations, layers, generic_seed, dtype=torch.float32):
    """Generic supplied seed; no study seed or label/cotangent initialization.

    Independent CPU generator per layer: seed+1009*l. Base BE a,b are iid signs;
    q uses balanced signs shuffled once, u iid signs, c exactly saved delta*gamma.
    Returns all matching factor arms without perturbing the global core RNG.
    """
    if relations<2:
        raise ValueError('CP candidate requires at least two canonical relations')
    result = {mode:nn.ModuleList() for mode in ('be','cp','unrestricted')}
    for layer in range(layers):
        generator = torch.Generator(device='cpu').manual_seed(generic_seed+1009*layer)
        signs = lambda shape:torch.randint(0,2,shape,generator=generator).to(dtype)*2-1
        a,b = signs((4,width)),signs((4,width))
        c = torch.tensor([-3.,-1.,1.,3.],dtype=dtype)*(0.01/math.sqrt(5))
        q = torch.cat((-torch.ones(relations//2,dtype=dtype),torch.ones(relations-relations//2,dtype=dtype)))
        q = q[torch.randperm(relations,generator=generator)]
        u = signs((width,))
        for mode in result:
            result[mode].append(PrivateFactors(a,b,c,q,u,mode))
    return result


class MemberStreams:
    """Persistent independent dropout states; save alongside model/optimizer.

    Global caller RNG is restored. A stream advances across all layers and calls.
    The root must prospectively bind supplied stream seeds and generic factor seed.
    """
    def __init__(self, seeds, device='cpu'):
        if len(seeds)!=4 or len(set(seeds))!=4:
            raise ValueError('Four distinct supplied member seeds required')
        self.device = torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:
            self.device = torch.device('cuda',torch.cuda.current_device())
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('Only pinned CPU/CUDA dropout RNG streams supported')
        self.states = [torch.Generator(device=self.device).manual_seed(seed).get_state() for seed in seeds]

    def state_dict(self):
        return dict(device=str(self.device),states=[state.clone() for state in self.states])

    def load_state_dict(self, value):
        if value['device']!=str(self.device) or len(value['states'])!=4:
            raise ValueError('Member RNG device/count changed')
        self.states = [state.clone() for state in value['states']]

    def run(self, member, callback):
        devices = [] if self.device.type=='cpu' else [self.device.index]
        with torch.random.fork_rng(devices=devices):
            if self.device.type=='cpu':
                torch.set_rng_state(self.states[member])
            else:
                torch.cuda.set_rng_state(self.states[member],self.device)
            value = callback()
            self.states[member] = (torch.get_rng_state() if self.device.type=='cpu'
                                   else torch.cuda.get_rng_state(self.device))
        return value


class PrivateHGT(nn.Module):
    """Four complete shared-core trajectories; member state is never pooled early."""
    def __init__(self, core, factors=None):
        super().__init__()
        self.core,self.factors = core,factors

    def forward(self, graph, features, target, streams=None, return_states=False):
        device = next(self.parameters()).device
        if self.training and (streams is None or streams.device!=device):
            raise ValueError('Training requires persistent member RNG streams on model device')
        members = []
        for member in range(4):
            callback = lambda member=member:self.core(graph,features,target,self.factors,member,return_states)
            members.append(callback() if streams is None else streams.run(member,callback))
        if return_states:
            return torch.stack([value[0] for value in members]),[value[1] for value in members]
        return torch.stack(members)


def matched_arms(core, factor_arms):
    """Copies preserve shared-core initialization; free table starts exactly at CP."""
    return {mode:PrivateHGT(copy.deepcopy(core),copy.deepcopy(factors)) for mode,factors in factor_arms.items()}


def mean_member_ce(member_logits, train_ids, train_labels):
    """Labels are compact TRAIN labels aligned to train_ids, not full-node labels."""
    return torch.stack([F.cross_entropy(logits[train_ids],train_labels) for logits in member_logits]).mean()


def mean_logits(member_logits):
    return member_logits.mean(0)


def parameter_count(module):
    return sum(value.numel() for value in module.parameters() if value.requires_grad)


def untied_core_hook(core):
    """Same native constructor copied four times; comparison recipe remains root work."""
    return nn.ModuleList(copy.deepcopy(core) for _ in range(4))


def wider_be_hook(graph, input_dims, classes, layers, base_width, generic_seed, heads=8, use_norm=True):
    """Smallest head-divisible wider BE meeting actual CP parameter count.

    Constructor/count hook only, not an independently qualified stronger baseline.
    Separate RNG scope avoids altering the caller's study initialization stream.
    """
    with torch.random.fork_rng(devices=[]):
        cp = PrivateHGT(NativeHGT(graph,input_dims,base_width,classes,layers,heads,use_norm),
                        initialized_factors(base_width,len(graph.edge_dict),layers,generic_seed)['cp'])
        budget = parameter_count(cp)
        width = base_width+heads
        while True:
            wider = PrivateHGT(NativeHGT(graph,input_dims,width,classes,layers,heads,use_norm),
                               initialized_factors(width,len(graph.edge_dict),layers,generic_seed)['be'])
            count = parameter_count(wider)
            if count>=budget:
                return wider,dict(width=width,cp_parameters=budget,wider_parameters=count,gap=count-budget)
            width += heads
