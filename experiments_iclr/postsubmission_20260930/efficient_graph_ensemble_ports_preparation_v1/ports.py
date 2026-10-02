"""Prospective ports; never imported or executed during preparation.

Consume the existing prepared public graph (teacher_backbone, teacher_input,
teacher_edge_index) and compact role packs (nodes, labels). No data loading,
preprocessing, fitting loop, checkpoint selection or experiment launch.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, stack_module_state, vmap

from native_polyformer_outer import PolyFormer
from native_polynormer import Polynormer


MEMBERS = 4
# Complete-source counts, not the interior-only alpha^2/M approximation.
FIXED_WIDTH = {"polyformer_mono": 116, "polynormer_r": 248}
MIMO_WIDTH = 208
FEATURES = {"polyformer_mono": 2089, "polynormer_r": 745}
CLASSES = {"polyformer_mono": 5, "polynormer_r": 8}
BOUNDARY_CAP = {"polyformer_mono": 4_430_644, "polynormer_r": 7_773_732}


def native_parameter_count(backbone, width):
    """Exact for these pinned configurations, including heads/norms/biases."""
    if backbone == "polyformer_mono":
        if width <= 0 or width % 4:
            raise ValueError("PolyFormer width must be a positive multiple of four")
        return 59 * width**2 + 2159 * width + 109
    if backbone == "polynormer_r":
        if width <= 0 or width % 8:
            raise ValueError("Polynormer total width must be a positive multiple of eight")
        return 28 * width**2 + 826 * width + 16
    raise ValueError("Unknown retained backbone")


def mimo_parameter_count(width):
    native_parameter_count("polyformer_mono", width)  # divisibility
    return 59 * width**2 + 8441 * width + 124


def new_native(backbone, width, seed):
    """CPU construction with isolated Torch RNG; caller moves the result."""
    native_parameter_count(backbone, width)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)  # CPU only; no CUDA RNG mutation
        if backbone == "polyformer_mono":
            args = SimpleNamespace(hidden=width, d_ffn=width // 2, K=12,
                nlayer=2, n_head=4, q=1.4, multi=1.0, dropout=0.3,
                dprate=0.8, base="mono", dataset="squirrel_filtered",
                num_features=2089, num_classes=5)
            return PolyFormer(None, args)
        model = Polynormer(745, width // 8, 8, local_layers=7,
            global_layers=2, in_dropout=0.2, dropout=0.7,
            global_dropout=0.7, heads=8, beta=-1, pre_ln=False)
        model.reset_parameters()  # retained native main.py construction contract
        return model


class NativeInput(nn.Module):
    def __init__(self, backbone, core):
        super().__init__()
        self.backbone, self.core = backbone, core

    def forward(self, inputs, edges):
        if self.backbone == "polyformer_mono":
            return self.core(SimpleNamespace(list_mat=inputs.unbind(1)))
        return self.core(inputs, edges)


def graph_inputs(graph, backbone):
    if graph.teacher_backbone != backbone:
        raise ValueError("Prepared public graph uses a different backbone")
    inputs = graph.teacher_input
    if inputs.shape[-1] != FEATURES[backbone]:
        raise ValueError("Wrong retained input feature count")
    if backbone == "polyformer_mono":
        if inputs.ndim != 3 or inputs.shape[1] != 13:
            raise ValueError("Expected complete [node,13,feature] token rows")
        return inputs, None
    if inputs.ndim != 2:
        raise ValueError("Expected full Polynormer node feature matrix")
    return inputs, graph.teacher_edge_index


class PackedBackendUnavailable(RuntimeError):
    """vmap is unsupported or failed; no hidden sequential retry is performed."""


class IndependentMembers(nn.Module):
    """M4 gamma1 untied graph port; explicit sequential or prospective vmap.

    The stacked parameters and Adam slots retain a leading member axis. Each
    member's complete attention/norm/beta/bias computation remains separate.
    Synchronized updates/stages are required for stacked Adam step equivalence.
    """
    def __init__(self, backbone, models, backend="vmap"):
        super().__init__()
        if backend not in ("vmap", "sequential") or len(models) != MEMBERS:
            raise ValueError("Explicit backend and exactly four native members required")
        expected = native_parameter_count(backbone, FIXED_WIDTH[backbone])
        if any(sum(p.numel() for p in model.parameters()) != expected for model in models):
            raise ValueError("Members do not match the fixed complete-source parameter contract")
        self.backbone, self.backend = backbone, backend
        self.members, self.global_stage = MEMBERS, False
        wrapped = [NativeInput(backbone, model) for model in models]
        if backend == "sequential":
            self.models = nn.ModuleList(wrapped)
        else:
            parameters, buffers = stack_module_state(wrapped)
            self._parameter_keys = {name: f"p{i}" for i, name in enumerate(parameters)}
            self.stacked = nn.ParameterDict({self._parameter_keys[name]: value
                for name, value in parameters.items()})
            self._buffer_keys = {name: f"b{i}" for i, name in enumerate(buffers)}
            for name, value in buffers.items():
                self.register_buffer(self._buffer_keys[name], value)
            # A stateless meta template is deliberately NOT a registered owner.
            object.__setattr__(self, "template", copy.deepcopy(wrapped[0]).to("meta"))
        self.specification = dict(backbone=backbone, members=MEMBERS,
            family="packed_native_small_graph_port", backend=backend,
            width=FIXED_WIDTH[backbone], gamma=1)

    @classmethod
    def build(cls, backbone, seed, backend="vmap"):
        width = FIXED_WIDTH[backbone]
        return cls(backbone, [new_native(backbone, width, seed + 100003 * m)
            for m in range(MEMBERS)], backend)

    def train(self, mode=True):
        super().train(mode)
        if self.backend == "vmap":
            self.template.train(mode)
        return self

    def set_global_stage(self, enabled):
        if self.backbone != "polynormer_r" and enabled:
            raise ValueError("PolyFormer has no global stage")
        self.global_stage = bool(enabled)
        if self.backbone == "polynormer_r":
            if self.backend == "sequential":
                for member in self.models:
                    member.core._global = self.global_stage
            else:
                self.template.core._global = self.global_stage

    def native_parameters(self):
        """Semantic names preserve the retained attention-specific Adam groups."""
        if self.backend == "sequential":
            yield from self.named_parameters()
        else:
            for name, key in self._parameter_keys.items():
                yield name, self.stacked[key]

    def forward(self, graph, x=None):
        if x is not None:
            raise ValueError("Only the caller's existing prepared public graph is consumed")
        inputs, edges = graph_inputs(graph, self.backbone)
        if self.backend == "sequential":
            return torch.stack([member(inputs, edges) for member in self.models])
        parameters = {name: self.stacked[key] for name, key in self._parameter_keys.items()}
        buffers = {name: getattr(self, key) for name, key in self._buffer_keys.items()}

        def one_member(params, bufs):
            return functional_call(self.template, (params, bufs),
                (inputs, edges), strict=True)

        try:
            # Different dropout streams, never member-shared randomness.
            return vmap(one_member, randomness="different")(parameters, buffers)
        except (RuntimeError, NotImplementedError) as error:
            raise PackedBackendUnavailable(
                f"{self.backbone} vmap is unqualified/unsupported: {error}. "
                "Select the explicit sequential reference separately; it is not a measured packed backend."
            ) from error


def check_compact_role(role, node_count):
    if role.nodes.dtype != torch.int64 or role.labels.dtype != torch.int64:
        raise ValueError("Compact nodes and corresponding labels must be int64")
    if role.nodes.ndim != 1 or role.labels.shape != role.nodes.shape:
        raise ValueError("Expected parallel compact node/label vectors")
    if len(role.nodes) == 0 or not bool(((role.nodes >= 0) & (role.nodes < node_count)).all()):
        raise ValueError("Empty or out-of-range compact role")


def member_loss(logits, role):
    """Retained mean member CE; logits [member,node,class]."""
    check_compact_role(role, logits.shape[1])
    return F.cross_entropy(logits[:, role.nodes].flatten(0, 1),
        role.labels.repeat(logits.shape[0]))


def pooled_log_probabilities(logits, reducer):
    if reducer == "native_probability":
        return torch.logsumexp(logits.log_softmax(-1), 0) - math.log(logits.shape[0])
    if reducer == "common_logit":
        return logits.mean(0).log_softmax(-1)
    raise ValueError("Declare native_probability or common_logit explicitly")


def validation_nll(logits, role, *, reducer):
    check_compact_role(role, logits.shape[1])
    return F.nll_loss(pooled_log_probabilities(logits, reducer)[role.nodes], role.labels)


def optimizer_for(model):
    """Native Adam settings, including all private biases/norms/betas."""
    if model.backbone == "polyformer_mono":
        items = model.native_parameters() if hasattr(model, "native_parameters") else model.named_parameters()
        groups = [dict(params=[p], lr=0.001 if "attnmodule" in name else 0.0001,
            weight_decay=1e-7 if "attnmodule" in name else 0.0) for name, p in items]
        return torch.optim.Adam(groups)
    return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)


@dataclass(frozen=True)
class MIMOTuples:
    tokens: torch.Tensor  # [tuple,13,4F], complete cached rows concatenated by slot
    labels: torch.Tensor  # [tuple,4], from the corresponding compact TRAIN positions
    nodes: torch.Tensor   # [tuple,4], public node identities for qualification
    positions: torch.Tensor  # [tuple,4], compact TRAIN positions, not global labels


def independent_tuple_positions(train_size, *, generator, device="cpu"):
    """rho=0, one full TRAIN exposure per slot; no replacement or repetitions."""
    if train_size <= 0:
        raise ValueError("Positive compact TRAIN size required")
    return torch.stack([torch.randperm(train_size, generator=generator, device=device)
        for _ in range(MEMBERS)], dim=1)


def make_mimo_tuples(graph, train, positions):
    inputs, _ = graph_inputs(graph, "polyformer_mono")
    check_compact_role(train, inputs.shape[0])
    if positions.dtype != torch.int64 or positions.ndim != 2 or positions.shape[1] != MEMBERS:
        raise ValueError("Tuple positions must be [tuple,4] int64")
    if len(positions) == 0 or not bool(((positions >= 0) & (positions < len(train.nodes))).all()):
        raise ValueError("Tuple positions must index the compact TRAIN pack")
    nodes, labels = train.nodes[positions], train.labels[positions]
    tokens = inputs[nodes].permute(0, 2, 1, 3).flatten(2, 3)
    return MIMOTuples(tokens, labels, nodes, positions)


class PolyFormerMIMO(nn.Module):
    """M4 on cached token examples; body width208 fixed before validation."""
    backbone, members = "polyformer_mono", MEMBERS

    def __init__(self, seed):
        super().__init__()
        self.core = new_native(self.backbone, MIMO_WIDTH, seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 50000)
            self.core.lin1 = nn.Linear(MEMBERS * 2089, MIMO_WIDTH)
            self.core.lin3 = nn.Linear(MIMO_WIDTH, MEMBERS * 5)
        self.specification = dict(backbone=self.backbone, members=MEMBERS,
            family="cached_token_mimo_graph_port", width=MIMO_WIDTH,
            input_repetition=0, batch_repetitions=1,
            train_reduction="mean tuples, sum matching heads")

    def forward_tuples(self, tuples):
        if tuples.tokens.ndim != 3 or tuples.tokens.shape[1:] != (13, MEMBERS * 2089):
            raise ValueError("Expected complete concatenated MIMO token examples")
        outputs = self.core(SimpleNamespace(list_mat=tuples.tokens.unbind(1)))
        return outputs.reshape(len(outputs), MEMBERS, 5)

    def forward(self, graph, x=None):
        if x is not None:
            raise ValueError("Consume the existing complete cached token rows")
        inputs, _ = graph_inputs(graph, self.backbone)
        repeated = inputs.unsqueeze(2).expand(-1, -1, MEMBERS, -1).flatten(2, 3)
        outputs = self.core(SimpleNamespace(list_mat=repeated.unbind(1)))
        return outputs.reshape(len(outputs), MEMBERS, 5).permute(1, 0, 2)


def mimo_loss(logits, tuples):
    if logits.ndim != 3 or logits.shape[:2] != tuples.labels.shape:
        raise ValueError("MIMO heads must correspond to the tuple-label slots")
    return F.cross_entropy(logits.flatten(0, 1), tuples.labels.flatten()) * MEMBERS
