"""Native TRAIN-negative/outer-order draw with independent prospective data RNG.

Delayed numerical imports only. Restores model/global random state. This source
does not load data or draw queries at import, and creates no model or fit.
"""
import random


def native_cycle(train, nodes, seed, cycle=0):
    if type(seed) is not int or type(cycle) is not int or cycle < 0:
        raise ValueError("Require fixed integer seed and nonnegative cycle")
    import numpy as np
    import torch
    from torch_geometric.utils import negative_sampling, to_undirected
    data_seed = (seed+cycle) % (2**32)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(data_seed); np.random.seed(data_seed)
        with torch.random.fork_rng(devices=[]):
            # Seed only CPU: torch.manual_seed would also alter CUDA generators.
            torch.random.default_generator.manual_seed(data_seed)
            edge = to_undirected(torch.tensor(train, dtype=torch.long).t())
            bank = negative_sampling(edge, nodes)
            order = torch.randperm(len(train)).tolist()
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
    return bank.t().tolist(), order
