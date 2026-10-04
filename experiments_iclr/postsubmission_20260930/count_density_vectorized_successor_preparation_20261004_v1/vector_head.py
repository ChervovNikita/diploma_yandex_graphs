"""Batched hypothetical count evaluation using the unchanged C_mu head.

Pass the existing CountPotential.count module; no added parameters. Only actual
feasible pairs enter neural evaluation. head_state_chunk changes workspace.
"""
import torch
from torch import nn
from torch.nn import functional as F
from vector_density import require


def potential(head, context, swapped, nl, nr, *, head_state_chunk=4096, padded_shape=None):
    require(context.shape == swapped.shape == (len(nl), 520) and context.dtype == swapped.dtype == torch.float32,
            "Unchanged FP32 full context required")
    require(nl.dtype == nr.dtype == torch.long and nl.shape == nr.shape
            and len(nl) > 0 and head_state_chunk > 0, "Positive batch and state chunk required")
    require(len(head) == 3 and isinstance(head[0], nn.Linear) and isinstance(head[1], nn.ReLU)
            and isinstance(head[2], nn.Linear) and head[0].weight.shape == (64,530)
            and head[2].weight.shape == (1,64), "Frozen count MLP structure required")
    # Exact affine decomposition, same parameter views/gradient sharing:
    # W[C,f]+b = W_context C+b + W_count f. Context's expensive520->64
    # projection is paid once per query, instead of once per count pair.
    original_context = F.linear(context, head[0].weight[:, :520], head[0].bias)
    exchanged_context = F.linear(swapped, head[0].weight[:, :520], head[0].bias)
    a, b = (int(nl.max()), int(nr.max())) if padded_shape is None else padded_shape
    k, l = torch.meshgrid(torch.arange(a + 1, device=context.device), torch.arange(b + 1, device=context.device), indexing="ij")
    valid = (k[None] <= nl[:, None, None]) & (l[None] <= nr[:, None, None])
    row, k, l = torch.nonzero(valid, as_tuple=True)
    values = []
    def features(x, y, nx, ny):
        x, y, nx, ny = x.to(torch.float32), y.to(torch.float32), nx.to(torch.float32), ny.to(torch.float32)
        return torch.stack((x / nx.clamp_min(1), y / ny.clamp_min(1), torch.log1p(x), torch.log1p(y),
                            torch.log1p(nx - x), torch.log1p(ny - y), (x == 0).to(x.dtype), (y == 0).to(x.dtype),
                            (x == nx).to(x.dtype), (y == ny).to(x.dtype)), 1)
    for start in range(0, len(row), head_state_chunk):
        r, x, y = row[start:start+head_state_chunk], k[start:start+head_state_chunk], l[start:start+head_state_chunk]
        original_hidden = F.relu(original_context[r] + F.linear(features(x, y, nl[r], nr[r]), head[0].weight[:,520:]))
        exchanged_hidden = F.relu(exchanged_context[r] + F.linear(features(y, x, nr[r], nl[r]), head[0].weight[:,520:]))
        original = F.linear(original_hidden, head[2].weight, head[2].bias).flatten()
        exchanged = F.linear(exchanged_hidden, head[2].weight, head[2].bias).flatten()
        values.append((.5 * (original + exchanged)).to(torch.float64))
    flat_index = (row * (a + 1) + k) * (b + 1) + l
    # Invalid padded cells are finite dummy zeros; coefficients exclude them.
    result = context.new_zeros(len(nl) * (a + 1) * (b + 1), dtype=torch.float64)
    return result.index_copy(0, flat_index, torch.cat(values)).reshape(len(nl), a + 1, b + 1)
