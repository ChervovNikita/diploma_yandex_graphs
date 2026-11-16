"""OGB Collab GCN/GraphSAGE with a link predictor, and GNNM with four routes.

GNNM follows the node models: route factors sit at the entry and at the exit of
the network, and everything between them is shared. The entry is the first
message-passing layer: its input is multiplied by the route's R (random signs)
and its output by S, then B is added. The exit is the predictor's final linear
layer, replaced by the original ``BEBlock`` with the native weight and bias
copied in. With R = S = 1 and B = 0 every route equals the native model.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

from models import BEBlock
from .protocol import RECIPE, ROUTES


def make_conv(backbone, inputs, outputs):
    if backbone == 'gcn':
        return GCNConv(inputs, outputs, normalize=False, cached=False)
    if backbone == 'sage':
        return SAGEConv(inputs, outputs)
    raise ValueError(f'Unknown backbone {backbone}')


class LinkModel(nn.Module):
    def __init__(self, backbone, num_features, routes=1):
        super().__init__()
        hidden, layers = RECIPE['hidden_channels'], RECIPE['num_layers']
        self.backbone, self.routes = backbone, routes
        sizes = [num_features] + [hidden] * layers
        self.convs = nn.ModuleList(make_conv(backbone, a, b) for a, b in zip(sizes, sizes[1:]))
        self.lins = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(RECIPE['predictor_layers'] - 1))
        final = nn.Linear(hidden, 1)
        if routes == 1:
            self.final = final
        else:
            self.R = nn.Parameter(torch.empty(routes, num_features).bernoulli_(0.5).mul_(2).sub_(1))
            self.S = nn.Parameter(torch.ones(routes, hidden))
            self.B = nn.Parameter(torch.zeros(routes, hidden))
            self.final = BEBlock(hidden, 1, False, 'cpu', routes)
            with torch.no_grad():
                self.final.W.weight.copy_(final.weight)
                self.final.B.copy_(final.bias.expand(routes, -1))

    def private_parameters(self):
        if self.routes == 1:
            return ()
        return (self.R, self.S, self.B, self.final.R, self.final.S, self.final.B)

    def conv(self, layer, x, graph):
        if self.backbone == 'gcn':
            return layer(x, graph['gcn_index'], graph['gcn_weight'])
        return layer(x, graph['edge_index'])

    def embed(self, x, graph, route=0):
        if self.routes > 1:
            x = x * self.R[route]
        for index, layer in enumerate(self.convs):
            x = self.conv(layer, x, graph)
            if index == 0 and self.routes > 1:
                x = x * self.S[route] + self.B[route]
            if index < len(self.convs) - 1:
                x = F.dropout(F.relu(x), p=RECIPE['dropout'], training=self.training)
        return x

    def score(self, h, links, route=0, return_prehead=False):
        x = h[links[:, 0]] * h[links[:, 1]]
        for lin in self.lins:
            x = F.dropout(F.relu(lin(x)), p=RECIPE['dropout'], training=self.training)
        logits = self.final(x) if self.routes == 1 else self.final(x, route)
        logits = logits.reshape(-1)
        return (logits, x) if return_prehead else logits


def build_model(backbone, variant, num_features):
    if variant == 'native':
        return LinkModel(backbone, num_features, routes=1)
    if variant in ('gnnm', 'gnnm_contrast'):
        return LinkModel(backbone, num_features, routes=ROUTES)
    raise ValueError(f'Unknown Collab variant {variant}')


def link_loss(positive_logits, negative_logits):
    """The OGB example's loss, -log(sigmoid) terms with its 1e-15 guard."""
    positive = -torch.log(torch.sigmoid(positive_logits) + 1e-15).mean()
    negative = -torch.log(1 - torch.sigmoid(negative_logits) + 1e-15).mean()
    return positive + negative
