"""Established rank-one BatchEnsemble maps around shared dense matrices.

No teacher acquisition. One native body is evaluated once per member because
factors change inputs to nonlinear attention/message stages, not just outputs.
"""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.dense.linear import Linear as PyGLinear


class FactorLinear(nn.Module):
    def __init__(self, source, members):
        super().__init__()
        if source.weight.ndim != 2:
            raise ValueError("Only materialized dense maps")
        self.weight = source.weight
        self.bias = source.bias
        self.r = nn.Parameter(torch.ones(members, source.weight.shape[1]))
        self.s = nn.Parameter(torch.ones(members, source.weight.shape[0]))
        self.member = 0

    def forward(self, x):
        y = F.linear(x * self.r[self.member], self.weight) * self.s[self.member]
        return y if self.bias is None else y + self.bias


def install_factors(module, members):
    """Replace each dense map, including PyG GAT/GCN projections, once.

    Run AFTER native construction/reset; source parameter objects are retained.
    Shared normalizations, attention vectors and all other parameters stay live.
    """
    seen = {}
    def visit(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.Linear, PyGLinear)):
                key = id(child)
                if key not in seen:
                    seen[key] = FactorLinear(child, members)
                setattr(parent, name, seen[key])
            else:
                visit(child)
    visit(module)
    if not seen:
        raise ValueError("No internal dense map was factorized")
    return len(seen)


def initialize_first_factor(module, seed):
    """TabM-inspired first-adapter only Rademacher start; later factors = 1.

    First is identified by architecture construction order, not alphabetic
    parameter names. Caller supplies the explicit stem module.
    """
    if not isinstance(module, FactorLinear):
        raise TypeError("Exact factorized stem required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        signs = torch.randint(0, 2, module.r.shape, generator=generator) * 2 - 1
        module.r.copy_(signs.to(module.r))


@contextmanager
def member_context(module, member):
    maps = [x for x in module.modules() if isinstance(x, FactorLinear)]
    old = [x.member for x in maps]
    for x in maps:
        if member < 0 or member >= x.r.shape[0]:
            raise IndexError(member)
        x.member = member
    try:
        yield
    finally:
        for x, value in zip(maps, old):
            x.member = value


def factor_counts(module):
    private = {id(p) for x in module.modules() if isinstance(x, FactorLinear)
               for p in (x.r, x.s)}
    private |= {id(p) for name,p in module.named_parameters() if name.endswith('message_factors')}
    return {"total": sum(p.numel() for p in module.parameters()),
            "private_factors": sum(p.numel() for p in module.parameters() if id(p) in private),
            "non_factor_parameters": sum(p.numel() for p in module.parameters() if id(p) not in private)}
