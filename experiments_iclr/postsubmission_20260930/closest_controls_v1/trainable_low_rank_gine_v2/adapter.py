"""Declared TabLoRA-inspired hidden-map GINE control; not author reproduction."""
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
FIXED_OPTIONS = {"output_dim": 1, "hidden_dim": 128, "num_layers": 3,
                 "dropout": 0.1, "hidden_dim_multiplier": 2}
ARM = "tab_lora_inspired_hidden_gine_nearest_private_block"


def verify_source_bindings():
    registry = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for name, digest in registry["phase_files_sha256"].items():
        path = (PHASE / name).resolve()
        if not path.is_relative_to(PHASE) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Retained source changed: " + name)
    return registry


verify_source_bindings()
_SPEC = importlib.util.spec_from_file_location(
    "trainable_low_rank_preserved_molecular_gine", PHASE / "new_tasks/gine_adapters.py")
GINE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GINE)


class MemberLowRankLinear(nn.Module):
    """Shared trainable W/bias plus separately trainable B_m A_m per member.

    alpha=rank gives unit correction scaling. Random A / zero B preserves
    the base function while dL/dB is live on step one; dL/dA is initially zero.
    Initialization uses a private CPU generator and consumes no dropout RNG.
    """
    def __init__(self, shared, members, rank, init_seed):
        super().__init__()
        if not isinstance(shared, nn.Linear) or shared.weight.device.type != "cpu":
            raise TypeError("Wrap the existing CPU dense affine before moving to device")
        if type(rank) is not int or not 0 <= rank <= min(shared.in_features, shared.out_features):
            raise ValueError("Rank must be an integer within the dense-map dimensions")
        if type(members) is not int or members < 1:
            raise ValueError("Positive integer member count required")
        self.shared = shared
        self.members, self.rank = members, rank
        self.active_member = 0
        self.alpha = rank
        self.scaling = 1.0 if rank else 0.0
        if rank:
            generator = torch.Generator(device="cpu").manual_seed(init_seed)
            self.A = nn.Parameter(shared.weight.new_empty(members, rank, shared.in_features))
            self.B = nn.Parameter(shared.weight.new_zeros(members, shared.out_features, rank))
            # Torch2.1's nn.init.kaiming_uniform_ lacks a generator argument.
            # Reproduce its exact leaky-ReLU a=sqrt5 gain/std/bound formula,
            # then use the supported Tensor.uniform_ private-generator API.
            std = nn.init.calculate_gain("leaky_relu", math.sqrt(5)) / math.sqrt(shared.in_features)
            bound = math.sqrt(3.0) * std  # algebraically1/sqrt(fan_in)
            with torch.no_grad():
                for member in range(members):
                    self.A[member].uniform_(-bound, bound, generator=generator)
        else:
            # No empty trainable parameters: rank zero has exactly source storage.
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    @property
    def added_parameters(self):
        return 0 if self.rank == 0 else self.A.numel() + self.B.numel()

    def forward(self, x):
        member = self.active_member
        if not 0 <= member < self.members:
            raise IndexError("Member index out of range")
        shared = self.shared(x)
        if self.rank == 0:
            return shared
        return shared + self.scaling * F.linear(F.linear(x, self.A[member]), self.B[member])


class TrainableLowRankGINE(nn.Module):
    """Same full member graph paths and boundary projectors; six private maps.

    Bias, LayerNorm, atom/bond encoders and GINE epsilon remain shared and
    trainable. No member activation is pooled until the final raw-logit mean.
    Mutable member context requires sequential execution, as in the source.
    """
    def __init__(self, original, rank, init_seed=29001):
        super().__init__()
        if not isinstance(original, GINE.MolecularProjectorEnsemble):
            # A module imported under another name has a distinct Python class;
            # accept its exact preserved class/source structure, not arbitrary ABI.
            cls = type(original)
            if cls.__name__ != "MolecularProjectorEnsemble" or Path(
                    inspect.getfile(cls)).resolve() != PHASE / "new_tasks/gine_adapters.py":
                raise TypeError("Use the exact preserved MolecularProjectorEnsemble")
        if original.options != FIXED_OPTIONS or original.members != 4 or original.private_blocks:
            raise ValueError("Require fully tied width128/L3/M4 molecular source")
        self.base = copy.deepcopy(original)
        self.options = dict(original.options)
        self.members, self.output_dim, self.rank = original.members, original.output_dim, rank
        self.init_seed = init_seed
        layers = []
        for block_index, block in enumerate(self.base.blocks):
            mlp = block.conv.nn
            if len(mlp) != 5 or not isinstance(mlp[0], nn.Linear) or not isinstance(mlp[3], nn.Linear):
                raise ValueError("Expected original two-affine GINE MLP")
            for map_index in (0, 3):
                wrapped = MemberLowRankLinear(mlp[map_index], self.members, rank,
                                              init_seed + 101 * block_index + map_index)
                mlp[map_index] = wrapped
                layers.append(wrapped)
        # Already registered inside base; avoid duplicate state/storage registration.
        self.low_rank_layers = tuple(layers)

    @property
    def added_parameters(self):
        return sum(layer.added_parameters for layer in self.low_rank_layers)

    def forward_member(self, graph, features=None, member=0):
        if not 0 <= member < self.members:
            raise IndexError("Member index out of range")
        for layer in self.low_rank_layers:
            layer.active_member = member
        return self.base.forward_member(graph, features, member)

    def forward(self, graph, features=None):
        return torch.stack([self.forward_member(graph, features, member)
                            for member in range(self.members)])

    forward_members = forward

    def pooled_forward(self, graph, features=None):
        return self.forward(graph, features).mean(0)

    def probe_blocks(self):
        return self.base.probe_blocks()


def counts(model):
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {"parameter_scalars": sum(p.numel() for p in parameters),
            "trainable_parameter_scalars": sum(p.numel() for p in parameters if p.requires_grad),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
            "buffer_scalars": sum(b.numel() for b in buffers),
            "buffer_bytes": sum(b.numel() * b.element_size() for b in buffers)}


def rank_plan(original, init_seed=29001):
    """Nearest ACTUAL registered count; no data, forward, optimizer or scores.

    The target is exactly one copied-private middle GINE block (index1).
    Enumerate legal ranks0..128 by constructed-module counts; ties lower rank.
    """
    if original.options != FIXED_OPTIONS or original.members != 4 or original.private_blocks:
        raise ValueError("Rank rule requires fixed fully tied production architecture")
    source = counts(original)
    target_model = copy.deepcopy(original)
    target_model.privatize([1])
    target = counts(target_model)
    del target_model
    candidates = []
    for rank in range(129):
        candidate = TrainableLowRankGINE(original, rank, init_seed)
        candidates.append({"rank": rank, **counts(candidate),
                           "added_low_rank_parameters": candidate.added_parameters})
        del candidate
    chosen = min(candidates, key=lambda c: (abs(c["parameter_scalars"] - target["parameter_scalars"]), c["rank"]))
    mismatch = (chosen["parameter_scalars"] - target["parameter_scalars"]) / target["parameter_scalars"]
    return {"rule": "nearest actual registered parameter count to one fully copied-private middle GINE block; ranks0..128; ties lower rank",
            "target_private_block_indices": [1], "source_counts": source, "target_counts": target,
            "chosen": chosen, "signed_parameter_mismatch": mismatch,
            "absolute_parameter_mismatch": abs(mismatch), "within_one_percent": abs(mismatch) <= 0.01,
            "exact_parameter_match": chosen["parameter_scalars"] == target["parameter_scalars"],
            "candidates": candidates, "data_or_scores_accessed": False}


def make_model(original=None, init_seed=29001):
    if original is None:
        original = GINE.MolecularProjectorEnsemble(FIXED_OPTIONS, members=4)
    plan = rank_plan(original, init_seed)
    model = TrainableLowRankGINE(original, plan["chosen"]["rank"], init_seed)
    if counts(model) != {k: plan["chosen"][k] for k in counts(model)}:
        raise ValueError("Chosen actual constructed counts differ")
    definition = {"arm": ARM, "options": FIXED_OPTIONS, "members": 4,
                  "rank_plan": plan, "counts": counts(model),
                  "private_maps": [f"blocks.{i}.conv.nn.{j}" for i in range(3) for j in (0, 3)],
                  "every_private_factor": {name: {"shape": list(p.shape), "scalars": p.numel(),
                      "bytes": p.numel() * p.element_size(), "trainable": p.requires_grad}
                      for name, p in model.named_parameters()
                      if name.startswith("base.blocks.") and name.endswith((".A", ".B"))},
                  "all_buffers": {name: {"shape": list(b.shape), "scalars": b.numel(),
                      "bytes": b.numel() * b.element_size()} for name, b in model.named_buffers()},
                  "additional_frozen_basis_buffer_scalars": 0,
                  "initialization": "copied common source; per-member random Kaiming A, zero B; private local CPU generator; alpha=rank => scaling1",
                  "init_seed": init_seed, "joint_trainable": "shared original W/bias/encoders/norm/epsilon/boundaries plus every private A/B",
                  "pooling": "mean raw logits only after full member graph readout",
                  "loss": "exact inherited mean-member equal-graph finite-target BCE",
                  "all_member_activations_separate": True, "graph_propagations_per_forward": 12,
                  "propagation_or_activation_savings_claimed": False,
                  "prior_method": "TabLoRA-inspired trainable low-rank ensemble; declared GINE adaptation with existing BE boundaries retained",
                  "exact_TabLoRA_author_reproduction": False, "novelty_claim": False, "predictive_evidence": False}
    return model, definition
