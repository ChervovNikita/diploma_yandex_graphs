"""Source-only SAGE adaptation of all-layer BatchEnsemble.

Reuses the existing hidden-factor port and preserved phase model/graph readout.
This is a prospectively declarable comparator, not a Kim (2023) reproduction.
No dataset access, runner, optimizer, or accelerator selection is included.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These are preserved copies, so importing this prototype does not depend on
# live runner edits. The existing adapter loads its adjacent models.py copy.
BASELINE = _load("closest_control_frozen_baseline", HERE / "sources" / "phase" /
                 "protocols" / "baseline_adapters.py")
MNIST = _load("closest_control_frozen_mnist", HERE / "sources" / "phase" /
             "new_tasks" / "mnist_adapters.py")


class MemberFactorLinear(nn.Module):
    """Member-specific rank-one factors around an existing shared affine map.

    The constructor and forward are copied from the existing Roman SAGE port.
    R indexes input features; S and B index output features.
    """

    def __init__(self, shared: nn.Module, members: int):
        super().__init__()
        weight = getattr(shared, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError("Expected a linear map with a two-dimensional weight")
        outputs, inputs = weight.shape
        self.shared = shared
        self.R = nn.Parameter(weight.new_ones((members, inputs)))
        self.S = nn.Parameter(weight.new_ones((members, outputs)))
        self.B = nn.Parameter(weight.new_zeros((members, outputs)))
        self.members = members
        self.active_member: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        member = self.active_member
        if member is None or not 0 <= member < self.members:
            raise RuntimeError("Select a BatchEnsemble member before forwarding")
        return self.shared(x * self.R[member]) * self.S[member] + self.B[member]

    @property
    def added_parameters(self) -> int:
        return self.R.numel() + self.S.numel() + self.B.numel()


class AllLayerSAGEAdapter(BASELINE.ProjectorEnsemble):
    """[M,N,C] logits with four factorized hidden maps in every SAGE block.

    Constructor adaptation of validation_tuning_prepared/AllLayerFactorModel;
    member stacking follows the preserved ProjectorEnsemble ABI. Member calls
    are sequential and stateful; concurrent/reentrant forwarding is unsupported.
    """

    def __init__(self, options, members=4):
        if options["model_name"] != "SAGE":
            raise ValueError("This declared adaptation supports SAGE only")
        if options["normalization"] != "LayerNorm":
            raise ValueError("The declared adaptation uses shared LayerNorm")
        if options["hidden_dim_multiplier"] != 1:
            raise ValueError("The declared adaptation uses SAGE MLP multiplier one")
        if members < 2 or options["num_layers"] < 1:
            raise ValueError("Require at least two members and one residual layer")
        super().__init__(options, members=members)
        factors = []
        if len(self.model.residual_modules) != options["num_layers"]:
            raise RuntimeError("Unexpected SAGE residual-block count")
        for residual in self.model.residual_modules:
            sage = residual.module
            if not isinstance(sage, BASELINE._MODELS.SAGEModule):
                raise RuntimeError("Unexpected residual module in SAGE model")
            for owner, name in (
                (sage.conv, "lin_l"), (sage.conv, "lin_r"),
                (sage.feed_forward_module, "linear_1"),
                (sage.feed_forward_module, "linear_2"),
            ):
                wrapped = MemberFactorLinear(getattr(owner, name), members)
                setattr(owner, name, wrapped)
                factors.append(wrapped)
        # These modules are registered inside self.model only.
        self.factor_layers = tuple(factors)
        if len(factors) != 4 * options["num_layers"]:
            raise RuntimeError("A hidden linear map was not factorized")

    @property
    def added_parameters(self):
        return sum(layer.added_parameters for layer in self.factor_layers)

    def forward_member(self, graph, features, member):
        if not 0 <= member < self.members:
            raise ValueError(member)
        for layer in self.factor_layers:
            layer.active_member = member
        return BASELINE._matrix(
            self.model(graph, features, tabm_seed=member), self.output_dim)

    def forward(self, graph, features):
        return torch.stack([self.forward_member(graph, features, member)
                            for member in range(self.members)])


def pooled_probabilities(member_logits, task):
    """Probability pooling is explicit and occurs after the per-member readout."""
    if task == "ppi":
        return member_logits.sigmoid().mean(0)
    if task == "mnist":
        return member_logits.softmax(-1).mean(0)
    raise ValueError("Task must be ppi or mnist")


class AllLayerMNISTGraphAdapter(MNIST.GraphClassifier):
    """Reuse the exact existing mean-node-logit graph readout."""

    def pooled_forward(self, graph, features=None):
        return pooled_probabilities(self(graph, features), "mnist")


def make_mnist_model(hidden_dim=128, layers=3, members=4, dropout=0.2):
    options = BASELINE.model_options(3, 10, hidden_dim, layers, dropout)
    node_model = AllLayerSAGEAdapter(options, members)
    definition = {
        "arm": "all_layer_be_sage_adaptation_prob",
        "prior_method": "all-layer BatchEnsemble; SAGE adaptation",
        "thesis_reproduction": False,
        "hidden_dim": hidden_dim, "layers": layers, "members": members,
        "dropout": dropout, "normalization": "shared LayerNorm per member",
        "readout": "mean_node_logits_after_node_LayerNorm",
        "pooling": "mean_member_softmax_probabilities_after_graph_readout",
        "execution": "source_sequential_members",
        "parameters": BASELINE.parameter_count(node_model),
        "added_hidden_factor_parameters": node_model.added_parameters,
    }
    return AllLayerMNISTGraphAdapter(node_model, "source", members, definition), definition


def make_ppi_model(hidden_dim=128, layers=3, members=4, dropout=0.2):
    options = BASELINE.model_options(50, 121, hidden_dim, layers, dropout)
    return AllLayerSAGEAdapter(options, members)
