"""Exact pinned Model class via narrow namespace shim; no unused package imports."""
from __future__ import annotations
from types import SimpleNamespace
import dgl
import torch
import torch.nn as nn
from torch import Tensor
import native_layers

lib = SimpleNamespace(graph=SimpleNamespace(deep=native_layers))

def PEARL(*args, **kwargs):
    raise ValueError("PEARL is outside the frozen native configuration")

class Model(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        n_num_features: int,
        n_classes: None | int,
        backbone: dict,
        bin_edges: list[Tensor] | None = None,
        num_features: None | dict = None,
        inp_out_type: str = "mlp",
        pearl: dict | None = None,
    ):
        super().__init__()

        # >>> Continuous (numerical) features
        if num_features is None:
            assert bin_edges is None
            self.num_embedding = None
            d_flat = n_features

        else:
            if num_features["name"] == "piecewise":
                assert bin_edges is not None

            self.num_embedding = lib.graph.deep.make_module(
                **num_features,
                n_features=n_num_features,
                bin_edges=bin_edges,
            )
            d_embed = num_features["d_embed"]
            d_flat = n_features + n_num_features * (d_embed - 1)

        # >>> PEARL
        if pearl is not None:
            self.pearl = PEARL(**pearl)
            d_flat += pearl["d_output"]
        else:
            self.pearl = None

        # >>> Input
        d_hidden = backbone["d_hidden"]
        dropout = backbone["dropout"]

        self.input = {
            "mlp": (
                nn.Sequential(
                    nn.Linear(d_flat, d_hidden),
                    nn.Dropout(dropout),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_hidden),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_flat, d_hidden),
                )
            ),
        }[inp_out_type]

        # >>> Backbone
        self.backbone = lib.graph.deep.make_module(
            **backbone,
        )

        # >>> Output
        d_out = 1 if n_classes is None else n_classes
        self.output = {
            "mlp": (
                nn.Sequential(
                    lib.graph.deep.NORM_MODULES[backbone["norm_name"]](d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_out),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_hidden, d_out),
                )
            ),
        }[inp_out_type]

    def forward(
        self,
        graph: dgl.DGLGraph,
        x_num: None | Tensor = None,
        x_other: None | Tensor = None,
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_embedding is None else self.num_embedding(x_num))

        if x_other is not None:
            x.append(x_other)

        if self.pearl is not None:
            x.append(self.pearl(graph))

        x = torch.cat(x, dim=1)
        x = self.input(x)
        x = self.backbone(graph, x)
        x = self.output(x)
        return x

