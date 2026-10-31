# SPDX-License-Identifier: Apache-2.0
"""Inactive adapter for the exact licensed NSD source vendored in this packet.

Propagation, reverse-edge pairing, normalized block degrees, SVD, augmentation,
training jitter, clamping and residual updates are delegated to the author code.
There is no dataset loader or training entry point in this module.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

NSD_COMMIT = "11e21b561d884713ab1a18a521a7dc2fb26b9361"
PACKET = Path(__file__).resolve().parent
NATIVE_ROOT = PACKET / "source_custody" / "nsd"


def load_native_general_class():
    """Load only the pinned author class; fail on hashes or namespace collision."""
    pins = json.loads((PACKET / "PINNED_CORE_SHA256.json").read_text())
    for relative, expected in pins.items():
        actual = hashlib.sha256((NATIVE_ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Native source hash mismatch: {relative}")
    for name, loaded in list(sys.modules.items()):
        if name in ("models", "lib") or name.startswith(("models.", "lib.")):
            origin = getattr(loaded, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(NATIVE_ROOT):
                raise RuntimeError(f"Cannot safely import native namespace {name!r}")
    native_path = str(NATIVE_ROOT)
    if native_path not in sys.path:
        sys.path.insert(0, native_path)
    module = importlib.import_module("models.disc_models")
    if Path(module.__file__).resolve() != NATIVE_ROOT / "models" / "disc_models.py":
        raise RuntimeError("Unexpected native model module")
    return module.DiscreteGeneralSheafDiffusion


def make_native_factory(edge_index, native_args: dict) -> Callable[[], nn.Module]:
    """Return a fresh original constructor; graph/split acquisition is external.

    This packet supports nonlinear, full-map, nonsparse learners, with both native
    left/right feature maps enabled. The source forward indexes those lists before
    its helper checks the flags, so unsupported flag combinations are rejected.
    """
    args = dict(native_args)
    if args["linear"] or args["sparse_learner"]:
        raise ValueError("This adapter requires nonlinear LocalConcatSheafLearner")
    if not args["left_weights"] or not args["right_weights"]:
        raise ValueError("Both native left/right map lists must be populated")
    if args["sheaf_act"] != "tanh" or args["d"] <= 1:
        raise ValueError("This packet is scoped to full tanh incidence maps, d>1")
    if not args["normalised"] or args["deg_normalised"]:
        raise ValueError("Keep the original augmented block-degree normalization")
    if args["add_lp"] or args["add_hp"]:
        raise ValueError("The fixed packet/control budgets exclude LP/HP dimensions")
    if str(edge_index.device) != str(args["device"]):
        raise ValueError("Construct native graph indices on the final device first")
    validate_native_graph(edge_index, args["graph_size"])
    native_class = load_native_general_class()
    graph_version = edge_index._version

    def factory():
        if edge_index._version != graph_version:
            raise RuntimeError("The validated static graph was modified in place")
        return native_class(edge_index, dict(args)).to(args["device"])

    return factory


def validate_native_graph(edge_index, graph_size: int):
    """Validate support without coalescing, sorting, or replacing caller edges.

    The exact author pairing uses a directed-edge dictionary and therefore cannot
    represent duplicates/self-loops. Temporary sorted keys are checks only.
    This function is an unexecuted future qualification gate in this packet.
    """
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Native edge_index must be int64 [2, E_directed]")
    if not 0 < graph_size <= 3037000499:
        raise ValueError("Invalid node count or int64 pair-key overflow risk")
    if edge_index.shape[1] == 0 or edge_index.shape[1] % 2:
        raise ValueError("Native support needs nonempty paired directed incidences")
    row, col = edge_index
    if row.min().item() < 0 or col.min().item() < 0:
        raise ValueError("Negative node index")
    if row.max().item() >= graph_size or col.max().item() >= graph_size:
        raise ValueError("Node index outside graph_size")
    if torch.any(row == col).item():
        raise ValueError("Native paired support must exclude self-loops")
    ordered = torch.sort(row * graph_size + col).values
    if torch.any(ordered[1:] == ordered[:-1]).item():
        raise ValueError("Duplicate directed incidence")
    reverse = torch.sort(col * graph_size + row).values
    if not torch.equal(ordered, reverse):
        raise ValueError("Every directed incidence needs exactly one reverse")


def _shared_copy_memo(prototype):
    """Share learned base parameters and immutable topology, not member modules."""
    memo = {id(parameter): parameter for parameter in prototype.parameters()}
    for value in (prototype.edge_index, prototype.time_range):
        memo[id(value)] = value
    builder = prototype.laplacian_builder
    for name in ("edge_index", "full_left_right_idx", "left_right_idx",
                 "vertex_tril_idx", "diag_indices", "tril_indices", "deg",
                 "fixed_diag_indices", "fixed_tril_indices"):
        value = getattr(builder, name, None)
        if isinstance(value, torch.Tensor):
            memo[id(value)] = value
    return memo


class SharedBELinear(nn.Module):
    """Original affine map with shared W/b and private endpoint/feature factors."""
    def __init__(self, native: nn.Linear):
        super().__init__()
        self.in_features = native.in_features
        self.out_features = native.out_features
        self.weight = native.weight
        self.bias = native.bias
        self.r = nn.Parameter(native.weight.new_ones(native.in_features))
        self.s = nn.Parameter(native.weight.new_ones(native.out_features))

    def forward(self, x):
        y = F.linear(x * self.r, self.weight, self.bias)
        # Algebraically dense*s+b, while unit factors use the original biased
        # affine kernel rather than replacing it by an unfused bias addition.
        if self.bias is None:
            return y * self.s
        return y * self.s + self.bias * (1 - self.s)

    def reset_factors(self):
        """Factor reset only; native weight reset requires a new bank/factory."""
        with torch.no_grad():
            self.r.fill_(1)
            self.s.fill_(1)


def _module_at(root: nn.Module, path: str):
    module = root
    for component in path.split("."):
        module = module._modules[component]
    return module


def _replace(root: nn.Module, path: str, value: nn.Module):
    parts = path.split(".")
    parent = root if len(parts) == 1 else _module_at(root, ".".join(parts[:-1]))
    parent._modules[parts[-1]] = value


class AssignedIncidences(nn.Module):
    """Reassign paired reverse-edge incidences before the original builder.

    fixed: one persisted, label-blind permutation; resampled: one fresh global-RNG
    permutation per forward, including evaluation. No multi-draw serving shortcut.
    """
    def __init__(self, original, left, right, edge_index, mode: str, seed: int):
        super().__init__()
        if mode not in ("fixed", "resampled"):
            raise ValueError(mode)
        edge_count = edge_index.shape[1]
        if left.numel() * 2 != edge_count:
            raise ValueError("Native pair index does not cover exactly two incidences")
        both = torch.cat((left, right))
        if torch.unique(both).numel() != edge_count:
            raise ValueError("Repeated/missing incidence indices")
        self.original = original
        self.mode = mode
        self.bound_edge_index = edge_index
        self.bound_graph_version = edge_index._version
        self.register_buffer("left", left.clone())
        self.register_buffer("right", right.clone())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(left.numel(), generator=generator).to(left.device)
        self.register_buffer("fixed_permutation", permutation)

    @property
    def L(self):
        return self.original.L

    def set_L(self, value):
        self.original.set_L(value)

    def forward(self, x, edge_index):
        if edge_index is not self.bound_edge_index or edge_index._version != self.bound_graph_version:
            raise ValueError("Assignment control is bound to the exact static graph/order")
        maps = self.original(x, edge_index)
        if maps.shape[0] != self.left.numel() * 2:
            raise ValueError("Graph/order changed after native pairing was bound")
        permutation = self.fixed_permutation
        if self.mode == "resampled":
            permutation = torch.randperm(self.left.numel(), device=maps.device)
        reassigned = maps.clone()
        reassigned[self.left] = maps[self.left[permutation]]
        reassigned[self.right] = maps[self.right[permutation]]
        return reassigned


class LayerSharedIncidence(nn.Module):
    """One full incidence matrix per member/layer, with native normalization."""
    def __init__(self, d: int, reference: torch.Tensor):
        super().__init__()
        self.raw = nn.Parameter(torch.eye(d, device=reference.device, dtype=reference.dtype))
        self.L = None

    def set_L(self, value):
        self.L = value.clone().detach()

    def forward(self, x, edge_index):
        return self.raw.tanh().unsqueeze(0).expand(edge_index.shape[1], -1, -1)


class NodeCapacityRight(nn.Module):
    """Node-only residual capacity before the unchanged native right map."""
    def __init__(self, native_right: nn.Module, graph_size: int, final_d: int,
                 channels: int, width: int, reference: torch.Tensor):
        super().__init__()
        self.native_right = native_right
        self.graph_size, self.final_d, self.channels = graph_size, final_d, channels
        c = final_d * channels
        self.down = nn.Linear(c, width, bias=False).to(reference)
        self.up = nn.Linear(width, c, bias=False).to(reference)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        node = x.reshape(self.graph_size, self.final_d * self.channels)
        node = node + self.up(F.silu(self.down(node)))
        return self.native_right(node.reshape(self.graph_size * self.final_d, self.channels))


class SharedNativeBank(nn.Module):
    """M complete native forwards; large affine weights shared by object identity.

    Each member has its own source graph builder, map diagnostics, private states,
    factor parameters, and native dropout/normalization random draws. No averaging
    of states/maps is inserted. Immutable graph indices and native epsilons are
    shared. States are recomputed privately on each forward, not cached across it.
    """
    def __init__(self, factory: Callable[[], nn.Module], members: int = 4,
                 assignment: Optional[str] = None, assignment_seed: int = 71237,
                 layer_shared: bool = False, capacity_control: bool = False):
        super().__init__()
        if members < 1:
            raise ValueError("members must be positive")
        prototype = factory()  # native initialization exactly once for common W
        if type(prototype) is not load_native_general_class():
            raise TypeError("Only the pinned original general-map class is supported")
        clones = [prototype] + [copy.deepcopy(prototype, _shared_copy_memo(prototype))
                                for _ in range(members - 1)]
        self.members = nn.ModuleList(clones)
        paths = [(name, module) for name, module in prototype.named_modules()
                 if isinstance(module, nn.Linear)]
        self.shared_linear_paths = tuple(name for name, _ in paths)
        for name, native in paths:
            for member in clones:
                _replace(member, name, SharedBELinear(native))
        for member in clones[1:]:
            for layer in range(prototype.layers):
                member.epsilons[layer] = prototype.epsilons[layer]
        self.members_count = members
        self.native_commit = NSD_COMMIT
        self.bound_edge_index = prototype.edge_index
        self.bound_graph_version = prototype.edge_index._version
        self.capacity_width = None
        if assignment and (layer_shared or capacity_control):
            raise ValueError("Choose a single mechanism control")
        if assignment:
            for m, member in enumerate(clones):
                left, right = member.laplacian_builder.left_right_idx
                for layer, learner in enumerate(list(member.sheaf_learners)):
                    member.sheaf_learners[layer] = AssignedIncidences(
                        learner, left, right, member.edge_index, assignment,
                        assignment_seed + 1009 * m + 65537 * layer)
        if layer_shared or capacity_control:
            # Original nonsparse general builder has no bias. Across the bank,
            # B + fast r/s cost 2c*d² + M(2c+d²). Constant maps cost M*d²;
            # bias-free c→width→c node adapters cost M*2c*width.
            if capacity_control and prototype.d * prototype.d % members:
                raise ValueError("Exact control budget requires d² divisible by M")
            width = 1 + prototype.d * prototype.d // members
            self.capacity_width = width if capacity_control else None
            for member in clones:
                for layer in range(member.layers):
                    reference = member.lin_right_weights[layer].weight
                    member.sheaf_learners[layer] = LayerSharedIncidence(member.d, reference)
                    if capacity_control:
                        member.lin_right_weights[layer] = NodeCapacityRight(
                            member.lin_right_weights[layer], member.graph_size,
                            member.final_d, member.hidden_channels, width, reference)

    def forward(self, x):
        self.assert_static_graph()
        return torch.stack([member(x) for member in self.members], dim=0)

    def assert_static_graph(self):
        if self.bound_edge_index._version != self.bound_graph_version:
            raise RuntimeError("The bound graph/order was modified in place")
        for member in self.members:
            if (member.edge_index is not self.bound_edge_index or
                    member.laplacian_builder.edge_index is not self.bound_edge_index):
                raise RuntimeError("Native update_edge_index is outside this packet's scope")

    @staticmethod
    def own_F(member_log_probabilities, train_index, train_labels):
        """F is mean original NLL; labels supplied only for TRAIN rows."""
        return torch.stack([F.nll_loss(p[train_index], train_labels)
                            for p in member_log_probabilities]).mean()

    @staticmethod
    def probability_pool(member_log_probabilities):
        return member_log_probabilities.exp().mean(dim=0)

    def parameter_groups(self, sheaf_decay: float, weight_decay: float):
        """Deduplicated shared parameters; original map/other decay separation."""
        sheaf, other = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (sheaf if "sheaf_learners" in name else other).append(parameter)
        return [{"params": sheaf, "weight_decay": sheaf_decay},
                {"params": other, "weight_decay": weight_decay}]


class IndependentNativeBank(nn.Module):
    """Ordinary complete native models with separate weights and optimizers.

    Train each member with its own unscaled original NLL and native parameter
    groups. Do not use shared-bank mean F with one common optimizer as a shortcut.
    """
    def __init__(self, factory: Callable[[], nn.Module], members: int = 4):
        super().__init__()
        if members < 1:
            raise ValueError("members must be positive")
        self.members = nn.ModuleList([factory() for _ in range(members)])
        if any(type(member) is not load_native_general_class() for member in self.members):
            raise TypeError("Only complete pinned native general-map members are supported")
        seen = set()
        for member in self.members:
            current = {id(p) for p in member.parameters()}
            if seen.intersection(current):
                raise ValueError("Independent factory reused learned parameters")
            seen.update(current)

    def forward(self, x):
        return torch.stack([member(x) for member in self.members], dim=0)

    def member_parameter_groups(self, index: int, sheaf_decay: float, weight_decay: float):
        """Use these groups with a separate optimizer for this complete member."""
        sheaf, other = self.members[index].grouped_parameters()
        return [{"params": sheaf, "weight_decay": sheaf_decay},
                {"params": other, "weight_decay": weight_decay}]


class JointMultiTransportSingle(nn.Module):
    """Same native transport states, with a capable joint nonlinear classifier.

    A terminal pre-hook reads raw pre-classifier states. It does not replace the
    native propagation. Unused native classifiers still compute and their cost
    must be charged; they are frozen/excluded from training. This is a joint
    readout control, not an author-native single-model performance claim.
    """
    def __init__(self, shared_bank: SharedNativeBank, classes: int, width: int = 64):
        super().__init__()
        self.bank = shared_bank
        c = shared_bank.members[0].hidden_dim
        for member in shared_bank.members:
            for parameter in member.lin2.parameters():
                parameter.requires_grad_(False)
        reference = shared_bank.members[0].lin1.weight
        self.readout = nn.Sequential(nn.Linear(shared_bank.members_count * c, width),
                                     nn.ELU(), nn.Linear(width, classes)).to(reference)

    def forward(self, x):
        self.bank.assert_static_graph()
        states = []
        for member in self.bank.members:
            captured = []

            def tap(_module, inputs):
                captured.append(inputs[0])  # retain the live graph; no detach

            handle = member.lin2.register_forward_pre_hook(tap)
            try:
                unused_native_logp = member(x)
                del unused_native_logp
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("Unexpected native terminal classifier path")
            states.append(captured[0])
        return F.log_softmax(self.readout(torch.cat(states, dim=-1)), dim=-1)

    @staticmethod
    def own_F(log_probabilities, train_index, train_labels):
        return F.nll_loss(log_probabilities[train_index], train_labels)

    def parameter_groups(self, sheaf_decay: float, weight_decay: float):
        groups = self.bank.parameter_groups(sheaf_decay, weight_decay)
        groups[1]["params"].extend(parameter for parameter in self.readout.parameters()
                                    if parameter.requires_grad)
        return groups
