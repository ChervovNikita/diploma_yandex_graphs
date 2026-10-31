"""Source-copied pinned native GAT closure; UNEXECUTED preparation source."""
from __future__ import annotations
import dgl
import torch
import torch.nn as nn
from dgl import ops
from torch import Tensor

def get_activation_module(activation: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }[activation]()


class ResidualModule(nn.Module):
    def __init__(
        self,
        base_class: type[nn.Module],
        norm_class: type[nn.Module],
        d_hidden: int,
        *,
        residual: bool = True,
        **module_kwargs,
    ):
        super().__init__()
        self.norm = norm_class(d_hidden)
        self.base = base_class(d=d_hidden, **module_kwargs)
        self.residual = residual

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        x_res = self.norm(x)
        x_res = self.base(graph, x_res, edge_weights)
        x = x + x_res if self.residual else x_res
        return x


class FFNModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_inputs: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d * n_inputs, d),
            nn.Dropout(dropout),
            get_activation_module(activation),
            nn.Linear(d, d),
            nn.Dropout(dropout),
        )

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        x = self.layers(x)
        return x


def _check_d_and_n_heads_consistency(d: int, n_heads: int) -> None:
    if d % n_heads != 0:
        raise ValueError("Dimension mismatch: d should be a multiple of n_head.")


class GATModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        _check_d_and_n_heads_consistency(d, n_heads)
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.input_linear = nn.Linear(d, d)
        self.attn_linear_u = nn.Linear(d, n_heads)
        self.attn_linear_v = nn.Linear(d, n_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.ffn = FFNModule(d, dropout=dropout, activation=activation)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        assert edge_weights is None, "So far, we can not apply edge weights."

        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.d_head, self.n_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.d)
        x = self.ffn(graph, x)
        return x


CONVOLUTIONAL_MODULES = {"gat": [GATModule]}
NORM_MODULES = {"layer": nn.LayerNorm}

class BaseGraphBackbone(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_hidden: int,
        conv_name: str,
        norm_name: str,
        activation: str,
        *,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        norm_class = NORM_MODULES[norm_name]

        for _ in range(n_layers):
            for base_class in CONVOLUTIONAL_MODULES[conv_name]:
                self.layers.append(
                    ResidualModule(
                        base_class,
                        norm_class,
                        d_hidden=d_hidden,
                        residual=residual,
                        dropout=dropout,
                        activation=activation,
                    )
                )

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(graph, x, edge_weights)
        return x


GRAPH_MODULES = {"BaseGraphBackbone": BaseGraphBackbone}

def make_module(type: str, *args, **kwargs):
    graph_module = GRAPH_MODULES[type]
    return graph_module(*args, **kwargs)

