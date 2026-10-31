"""Disposable Torch COO candidate; baseline fits always use native DGL."""
from __future__ import annotations
import torch
from torch import nn


class TorchGAT(nn.Module):
    def __init__(self, native, graph):
        super().__init__()
        self.d, self.n_heads, self.d_head = native.d, native.n_heads, native.d_head
        if (self.d, self.n_heads, self.d_head) != (512, 4, 128):
            raise ValueError('Native GAT dimensions differ')
        for name in ('input_linear', 'attn_linear_u', 'attn_linear_v', 'attn_act', 'ffn'):
            setattr(self, name, getattr(native, name))
        src, dst = graph.edges(order='eid')
        self.register_buffer('_src', src.to(torch.int64).clone())
        self.register_buffer('_dst', dst.to(torch.int64).clone())
        self.num_nodes = graph.num_nodes()

    def forward(self, graph, x, edge_weights=None):
        if edge_weights is not None or x.shape != (self.num_nodes, self.d):
            raise ValueError('Exact full-graph native GAT contract required')
        if graph.num_nodes() != self.num_nodes or graph.num_edges() != self._src.numel():
            raise ValueError('Disposable graph identity changed')
        x = self.input_linear(x)
        scores = self.attn_act(self.attn_linear_u(x).index_select(0, self._src)
                              + self.attn_linear_v(x).index_select(0, self._dst))
        heads = self._dst[:, None].expand(-1, self.n_heads)
        maximum = torch.full((self.num_nodes, self.n_heads), -torch.inf, device=x.device, dtype=x.dtype)
        maximum = maximum.scatter_reduce(0, heads, scores, reduce='amax', include_self=True).detach()
        numerator = (scores - maximum.index_select(0, self._dst)).exp()
        denominator = torch.zeros_like(maximum).scatter_add(0, heads, numerator)
        attention = numerator / denominator.index_select(0, self._dst)
        values = x.reshape(self.num_nodes, self.d_head, self.n_heads)
        messages = values.index_select(0, self._src) * attention[:, None, :]
        summed = torch.zeros_like(values).index_add(0, self._dst, messages)
        return self.ffn(None, summed.reshape(self.num_nodes, self.d))


def replace_native_primitives(model, graph):
    if len(model.backbone.layers) != 3:
        raise ValueError('Native three-block contract differs')
    for layer in model.backbone.layers:
        if layer.base.__class__.__name__ != 'GATModule':
            raise ValueError('Native GAT operator differs')
        layer.base = TorchGAT(layer.base, graph)
    return model
