"""Runnable HGT families reusing the exact separately pinned core, not new backbones."""
import copy
import torch
from torch import nn


class SharedRelationFactors(nn.Module):
    """Capacity control: common relation residual; no member CP coordinate.

    Shared rho=0.01 matches CP's RMS c at initialization, not its initial pooled
    function. q/u/base signs match CP. Different coordinates/decay are disclosed.
    """
    def __init__(self, cp):
        super().__init__()
        self.a,self.b,self.q,self.u = (nn.Parameter(value.detach().clone()) for value in (cp.a,cp.b,cp.q,cp.u))
        self.rho = nn.Parameter(cp.c.new_tensor(0.01))

    def output(self, member, relation):
        return self.b[member]+self.rho*self.q[relation]*self.u


class UntiedMembers(nn.Module):
    def __init__(self, cores):
        super().__init__(); self.cores = nn.ModuleList(cores)

    def forward(self, graph, features, target, streams=None):
        if self.training and streams is None:
            raise ValueError('Untied training requires persistent member streams')
        rows = []
        for member,core in enumerate(self.cores):
            callback = lambda core=core:core(graph,features,target)
            rows.append(callback() if streams is None else streams.run(member,callback))
        return torch.stack(rows)


class NativeMember(nn.Module):
    def __init__(self, core):
        super().__init__(); self.core = core

    def forward(self, graph, features, target, streams=None):
        return self.core(graph,features,target).unsqueeze(0)


def build(implementation, graph, input_dims, classes, seed, factor_seed, device, requested):
    """One paired core/base-factor draw; untied seeds follow explicit saved policy."""
    torch.manual_seed(seed)
    core = implementation.NativeHGT(graph,input_dims,64,classes,3,8,True)
    factors = implementation.initialized_factors(64,len(graph.edge_dict),3,factor_seed)
    models = {}
    for arm in requested:
        if arm=='native_HGT':
            model = NativeMember(copy.deepcopy(core))
        elif arm in ('global_BE','CP','unrestricted'):
            mode = {'global_BE':'be','CP':'cp','unrestricted':'unrestricted'}[arm]
            model = implementation.PrivateHGT(copy.deepcopy(core),copy.deepcopy(factors[mode]))
        elif arm=='shared_relation':
            model = implementation.PrivateHGT(copy.deepcopy(core),nn.ModuleList(SharedRelationFactors(cp) for cp in factors['cp']))
        elif arm=='wider_BE':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model,_ = implementation.wider_be_hook(graph,input_dims,classes,3,64,factor_seed,8,True)
        elif arm=='untied_HGT':
            cores = [copy.deepcopy(core)]
            for member in range(1,4):
                torch.manual_seed(seed+200003*member)
                cores.append(implementation.NativeHGT(graph,input_dims,64,classes,3,8,True))
            model = UntiedMembers(cores)
        else:
            raise ValueError('Unsupported HGT arm: '+arm)
        models[arm] = model.to(device)
    return models
