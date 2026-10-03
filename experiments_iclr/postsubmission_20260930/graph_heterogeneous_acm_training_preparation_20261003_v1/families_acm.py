"""ACM constructor adaptation; immutable core, wrappers, controls and draw order."""
import copy
import torch
from torch import nn


def build(implementation, existing, graph, input_dims, classes, seed, factor_seed, device, requested):
    """DBLP v2 family build order with only ACM layers2/use_normFalse substituted."""
    torch.manual_seed(seed)
    core = implementation.NativeHGT(graph, input_dims, 64, classes, 2, 8, False)
    factors = implementation.initialized_factors(64, len(graph.edge_dict), 2, factor_seed)
    models = {}
    for arm in requested:
        if arm == 'native_HGT':
            model = existing.NativeMember(copy.deepcopy(core))
        elif arm in ('global_BE', 'CP', 'unrestricted'):
            mode = {'global_BE': 'be', 'CP': 'cp', 'unrestricted': 'unrestricted'}[arm]
            model = implementation.PrivateHGT(copy.deepcopy(core), copy.deepcopy(factors[mode]))
        elif arm == 'shared_relation':
            model = implementation.PrivateHGT(copy.deepcopy(core),
                nn.ModuleList(existing.SharedRelationFactors(cp) for cp in factors['cp']))
        elif arm == 'wider_BE':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model, _ = implementation.wider_be_hook(graph, input_dims, classes, 2, 64, factor_seed, 8, False)
        elif arm == 'untied_HGT':
            cores = [copy.deepcopy(core)]
            for member in range(1, 4):
                torch.manual_seed(seed + 200003 * member)
                cores.append(implementation.NativeHGT(graph, input_dims, 64, classes, 2, 8, False))
            model = existing.UntiedMembers(cores)
        else:
            raise ValueError('Unsupported HGT arm: ' + arm)
        models[arm] = model.to(device)
    return models
