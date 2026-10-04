"""HLGNN affine members on a shared fixed graph-power computation.

Source only: this module has not been imported or numerically checked here.
M=1 follows the native KI HLGNN computation and parameter layout exactly.
M>1 is an explicit shared-embedding/shared-dropout adaptation.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul


class SharedPowerHLGNN(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=512, K=15,
                 dropout=0.3, alpha=0.5, members=1, private_alpha=False):
        super().__init__()
        if members < 1 or K < 0:
            raise ValueError("members must be positive and K nonnegative")
        self.K, self.dropout, self.alpha, self.init = K, dropout, alpha, "KI"
        self.members = members
        self.private_alpha = bool(private_alpha and members > 1)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        initial = torch.tensor([alpha ** k for k in range(K + 1)], dtype=torch.float64)
        if self.private_alpha:
            initial = initial.repeat(members, 1)
        self.temp = nn.Parameter(initial)
        if members > 1:
            self.r = nn.Parameter(torch.ones(members, in_channels))
            self.s = nn.Parameter(torch.ones(members, hidden_channels))
            self.bias_delta = nn.Parameter(torch.zeros(members, hidden_channels))

    def reset_parameters(self):
        # Native KI reset preserves lin1; fresh fits must create fresh objects.
        with torch.no_grad():
            for k in range(self.K + 1):
                self.temp[..., k].fill_(self.alpha ** k)
            if self.members > 1:
                self.r.fill_(1)
                self.s.fill_(1)
                self.bias_delta.zero_()

    def coefficients(self, member):
        return self.temp[member] if self.private_alpha else self.temp

    def context(self, x, adj_t, edge_weight=None, dropped_x=None):
        z = F.dropout(x, p=self.dropout, training=self.training) if dropped_x is None else dropped_x
        if z.shape != x.shape or z.dtype != x.dtype or z.device != x.device:
            raise ValueError("dropped_x must preserve the input shape, dtype and device")
        P = gcn_norm(adj_t, edge_weight, adj_t.size(0), dtype=torch.float)
        return z, P

    def affine_after_filter(self, aggregate, member):
        features, constant = aggregate[:, :-1], aggregate[:, -1:]
        if self.members == 1:
            return F.linear(features, self.lin1.weight, None) + constant * self.lin1.bias
        projected = F.linear(features * self.r[member], self.lin1.weight, None) * self.s[member]
        effective_bias = self.lin1.bias * self.s[member] + self.bias_delta[member]
        return projected + constant * effective_bias

    def factored(self, z, P, storage="aggregates"):
        # The last channel is P^k 1, required for exact affine bias propagation.
        current = torch.cat((z, z.new_ones((z.shape[0], 1))), dim=-1)
        routes = self.members if self.private_alpha else 1
        if storage == "aggregates":
            aggregates = [current * self.coefficients(m)[0] for m in range(routes)]
            for k in range(1, self.K + 1):
                current = matmul(P, current, reduce="add")
                aggregates = [a + current * self.coefficients(m)[k]
                              for m, a in enumerate(aggregates)]
            outputs = [self.affine_after_filter(aggregates[m if self.private_alpha else 0], m)
                       for m in range(self.members)]
        elif storage == "powers":
            powers = [current]
            for _ in range(self.K):
                current = matmul(P, current, reduce="add")
                powers.append(current)
            if not self.private_alpha:
                aggregate = sum(h * a for h, a in zip(powers, self.temp))
                outputs = [self.affine_after_filter(aggregate, m) for m in range(self.members)]
            else:
                outputs = [self.affine_after_filter(
                    sum(h * a for h, a in zip(powers, self.temp[m])), m)
                    for m in range(self.members)]
        else:
            raise ValueError("storage must be aggregates or powers")
        return outputs[0] if self.members == 1 else torch.stack(outputs, dim=0)

    def forward(self, x, adj_t, edge_weight=None, *, dropped_x=None, storage="aggregates"):
        z, P = self.context(x, adj_t, edge_weight, dropped_x)
        if self.members == 1:
            # Preserve native operation order as well as the mathematical reduction.
            h = self.lin1(z)
            hidden = h * self.temp[0]
            for k in range(self.K):
                h = matmul(P, h, reduce="add")
                hidden = hidden + self.temp[k + 1] * h
            return hidden
        return self.factored(z, P, storage)


def direct_reference(model, x, adj_t, edge_weight=None, *, dropped_x=None):
    """Deliberately propagate every member after its affine map.

    Uses identical parameters, fixed P and dropped input. It performs M*K
    sparse applications in H channels, unlike the factored D+1-channel route.
    """
    z, P = model.context(x, adj_t, edge_weight, dropped_x)
    outputs = []
    for member in range(model.members):
        if model.members == 1:
            h = model.lin1(z)
        else:
            h = F.linear(z * model.r[member], model.lin1.weight, model.lin1.bias)
            h = h * model.s[member] + model.bias_delta[member]
        coefficient = model.coefficients(member)
        hidden = h * coefficient[0]
        for k in range(model.K):
            h = matmul(P, h, reduce="add")
            hidden = hidden + coefficient[k + 1] * h
        outputs.append(hidden)
    return outputs[0] if model.members == 1 else torch.stack(outputs, dim=0)
