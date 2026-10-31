"""Source-only coordinate ensemble preparation; runtime qualification is pending.

No models, graphs, datasets, or fits are constructed at module import. The frozen
dependency is loaded only by build_model(). Permutations are checkpoint buffers.
"""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


FROZEN_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "propagation_cost_impl_v1" / "_pinned" / "frozen_models.py"
)
FROZEN_SHA256 = "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"
ARMS = (
    "original", "factor", "coordinate", "permutation", "bias_only",
    "single", "untied", "heads", "gt_sep_single",
)


@lru_cache(maxsize=1)
def _load_frozen():
    if hashlib.sha256(FROZEN_SOURCE.read_bytes()).hexdigest() != FROZEN_SHA256:
        raise RuntimeError("The immutable frozen_models.py dependency changed")
    spec = importlib.util.spec_from_file_location("coordinate_frozen_models", FROZEN_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen model dependency: {FROZEN_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContextLinear(nn.Module):
    """One shared hidden matrix with explicit member contexts.

    Column-vector map: D_s P_out W P_in D_r h + b_m.
    P[i, index[i]]=1. In row-batch code, scale the original input coordinates,
    gather input_index, apply W, gather output_index, scale output coordinates,
    then add b_m. Input and output permutations have independent buffers.

    Identity permutations and constant identity factors are omitted safely:
    there is no identity gather, tensor buffer, or free-seed storage claim.
    Bias presence follows the original map, including bias-free SAGE lin_r.
    """

    def __init__(self, source: nn.Module, members: int, mode: str,
                 permutation_generator: torch.Generator):
        super().__init__()
        if mode not in ("factor", "coordinate", "permutation", "bias_only"):
            raise ValueError(f"Unsupported hidden-map mode: {mode}")
        self.weight = source.weight
        self.out_features, self.in_features = self.weight.shape
        self.members = members
        self.mode = mode
        source_bias = getattr(source, "bias", None)
        if source_bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(source_bias.detach().clone().repeat(members, 1))

        if mode in ("factor", "coordinate"):
            self.R = nn.Parameter(self.weight.new_ones(members, self.in_features))
            self.S = nn.Parameter(self.weight.new_ones(members, self.out_features))
        else:
            self.register_parameter("R", None)
            self.register_parameter("S", None)

        if mode in ("coordinate", "permutation"):
            # The local CPU generator never consumes the model/dropout RNG.
            self.register_buffer("input_index", torch.stack([
                torch.randperm(self.in_features, generator=permutation_generator)
                for _ in range(members)
            ]))
            self.register_buffer("output_index", torch.stack([
                torch.randperm(self.out_features, generator=permutation_generator)
                for _ in range(members)
            ]))
        else:
            self.register_buffer("input_index", None)
            self.register_buffer("output_index", None)

    def forward(self, x: torch.Tensor, member: int) -> torch.Tensor:
        if not isinstance(member, int) or not 0 <= member < self.members:
            raise IndexError("member must be one fixed integer in [0, members)")
        if self.R is not None:
            x = x * self.R[member]
        if self.input_index is not None:
            x = x.index_select(-1, self.input_index[member])
        x = F.linear(x, self.weight)
        if self.output_index is not None:
            x = x.index_select(-1, self.output_index[member])
        if self.S is not None:
            x = x * self.S[member]
        if self.bias is not None:
            x = x + self.bias[member]
        return x

    def private_parameters(self):
        return tuple(p for p in (self.R, self.S, self.bias) if p is not None)


def _member_check(member: int, members: int) -> None:
    if not isinstance(member, int) or not 0 <= member < members:
        raise IndexError("member must be one fixed integer in [0, members)")


def _common_trunk(base, graph, x: torch.Tensor) -> torch.Tensor:
    x = base.act(base.dropout(base.input_linear(x)))
    for residual in base.residual_modules:
        x = residual(graph, x)
    return base.output_normalization(x)


def _copy_shared_tabm_to_single(tabm, ordinary) -> None:
    """Match common maps/norms; an ordinary stem cannot match private R."""
    ordinary.residual_modules.load_state_dict(tabm.residual_modules.state_dict())
    ordinary.output_normalization.load_state_dict(tabm.output_normalization.state_dict())
    with torch.no_grad():
        ordinary.input_linear.weight.copy_(tabm.input_be_block.W.weight)
        ordinary.input_linear.bias.copy_(tabm.input_be_block.B[0])
        ordinary.output_linear.weight.copy_(tabm.output_be_block.W.weight)
        ordinary.output_linear.bias.copy_(tabm.output_be_block.B[0])


class TABMEnsemble(nn.Module):
    """Frozen TABM boundaries; optional explicit hidden contexts.

    LayerNorm affine parameters and identity residual skips remain common and
    unpermuted. This is a deliberate non-gauge contract. The selected map output
    is gathered/scaled and privately biased before subsequent SAGE additions,
    concatenation, GELU, dropout, or the residual addition.
    """

    def __init__(self, base, arm: str, members: int, permutation_seed: int):
        super().__init__()
        self.base = base
        self.arm = arm
        self.members = members
        self.permutation_seed = permutation_seed
        self.hidden_context_paths: list[str] = []
        self._private_parameter_ids = {
            id(p) for block in (base.input_be_block, base.output_be_block)
            for p in (block.R, block.S, block.B)
        }
        if arm == "original":
            return
        generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
        for layer, residual in enumerate(base.residual_modules):
            module = residual.module
            conv = module.conv
            if (conv.aggr != "mean" or conv.project or conv.normalize
                    or not conv.root_weight):
                raise ValueError("Only the original default homogeneous SAGE contract is supported")
            sites = (
                (conv, "lin_l", f"residual_modules.{layer}.module.conv.lin_l"),
                (conv, "lin_r", f"residual_modules.{layer}.module.conv.lin_r"),
                (module.feed_forward_module, "linear_1",
                 f"residual_modules.{layer}.module.feed_forward_module.linear_1"),
                (module.feed_forward_module, "linear_2",
                 f"residual_modules.{layer}.module.feed_forward_module.linear_2"),
            )
            for owner, name, path in sites:
                original_map = getattr(owner, name)
                if isinstance(original_map, ContextLinear):
                    raise RuntimeError("Attempted to wrap a hidden map twice")
                context = ContextLinear(original_map, members, arm, generator)
                setattr(owner, name, context)
                self.hidden_context_paths.append("base." + path)
                self._private_parameter_ids.update(id(p) for p in context.private_parameters())
        # BEBlock.W is deliberately untouched: no input/class-axis permutation.

    def forward_member(self, graph, x: torch.Tensor, member: int) -> torch.Tensor:
        _member_check(member, self.members)
        x = self.base.input_be_block(x, tabm_seed=member)
        x = self.base.act(self.base.dropout(x))
        for residual in self.base.residual_modules:
            if self.arm == "original":
                x = residual(graph, x)
                continue
            normalized = residual.normalization(x)
            module = residual.module
            conv = module.conv
            # Exactly the default SAGEConv order: mean neighbors, lin_l,
            # lin_r(root), addition. ContextLinear needs an explicit member.
            neighbors = conv.propagate(
                graph.edge_index, x=(normalized, normalized), size=None,
            )
            message = conv.lin_l(neighbors, member) + conv.lin_r(normalized, member)
            ff = module.feed_forward_module
            branch = ff.linear_1(torch.cat([normalized, message], dim=1), member)
            branch = ff.act(ff.dropout_1(branch))
            branch = ff.dropout_2(ff.linear_2(branch, member))
            x = x + branch
        x = self.base.output_normalization(x)
        # Unlike the original class, retain the C axis even when C=1.
        return self.base.output_be_block(x, tabm_seed=member)

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward_member(graph, x, m) for m in range(self.members)])

    def storage_report(self) -> dict[str, Any]:
        return storage_report(self)


class OrdinaryEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module], arm: str):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.arm = arm
        self.members = len(models)
        self._private_parameter_ids = (
            {id(p) for p in self.parameters()} if arm == "untied" else set()
        )

    def forward_member(self, graph, x: torch.Tensor, member: int) -> torch.Tensor:
        _member_check(member, self.members)
        base = self.models[member]
        return base.output_linear(_common_trunk(base, graph, x))

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward_member(graph, x, m) for m in range(self.members)])

    def storage_report(self) -> dict[str, Any]:
        return storage_report(self)


class SharedTrunkHeads(nn.Module):
    """One complete shared SAGE trunk with M capable h->h->C GELU heads."""

    def __init__(self, base, hidden_dim: int, output_dim: int, members: int, dropout: float):
        super().__init__()
        self.base = base
        # Remove the unused original readout and its registered parameters.
        self.base.output_linear = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
            for _ in range(members)
        ])
        self.arm = "heads"
        self.members = members
        self._private_parameter_ids = {id(p) for p in self.heads.parameters()}

    def forward_member(self, graph, x: torch.Tensor, member: int) -> torch.Tensor:
        _member_check(member, self.members)
        return self.heads[member](_common_trunk(self.base, graph, x))

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        # Credit this arm's real reuse: one trunk, followed by M heads.
        h = _common_trunk(self.base, graph, x)
        return torch.stack([head(h) for head in self.heads])

    def storage_report(self) -> dict[str, Any]:
        return storage_report(self)


def build_model(arm: str, input_dim: int, hidden_dim: int, output_dim: int,
                seed: int, permutation_seed: int, members: int = 4,
                num_layers: int = 2, dropout: float = 0.5, *,
                hidden_dim_multiplier: float = 1.0,
                normalization: str = "layer", num_heads: int = 4) -> nn.Module:
    """Construct CPU source architecture after explicit runtime admission.

    A caller may move the returned model to its admitted device. Root's JSON
    protocol supplies dimensions/recipe; this module has no dataset names.
    All TABM arms preserve live boundary BE factors. `permutation` therefore
    means boundary_BE_plus_hidden_permutations_biases: only interior factors
    are identities. `untied`/`heads` widths are selected analytically by caller.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if normalization not in ("layer", "LayerNorm"):
        raise ValueError("This preparation fixes the original affine LayerNorm contract")
    if min(input_dim, hidden_dim, output_dim, members) < 1 or num_layers < 0:
        raise ValueError("Dimensions/members must be positive and layers nonnegative")
    if not 0 <= dropout < 1 or int(hidden_dim * hidden_dim_multiplier) < 1:
        raise ValueError("Invalid dropout or hidden_dim_multiplier")
    if arm == "gt_sep_single" and (num_heads < 1 or hidden_dim % num_heads):
        raise ValueError("GT-sep hidden_dim must be divisible by num_heads")
    kwargs = dict(
        model_name="GT-sep" if arm == "gt_sep_single" else "SAGE",
        num_layers=num_layers, input_dim=input_dim, hidden_dim=hidden_dim,
        output_dim=output_dim, hidden_dim_multiplier=hidden_dim_multiplier,
        num_heads=num_heads, normalization="LayerNorm", dropout=dropout,
    )
    # Preserve caller RNG. Every TABM-derived arm starts from identical matrix,
    # boundary-factor, bias, and norm initialization for a common seed.
    with torch.random.fork_rng(devices=[]):
        frozen = _load_frozen()
        # Only seed CPU construction; do not mutate caller CUDA RNG states.
        torch.default_generator.manual_seed(seed)
        if arm in ("original", "factor", "coordinate", "permutation", "bias_only"):
            base = frozen.TABMModel(**kwargs, tabm_inits=members, device="cpu")
            model = TABMEnsemble(base, arm, members, permutation_seed)
        elif arm == "single":
            reference = frozen.TABMModel(**kwargs, tabm_inits=members, device="cpu")
            ordinary = frozen.Model(**kwargs)
            _copy_shared_tabm_to_single(reference, ordinary)
            model = OrdinaryEnsemble([ordinary], arm)
        elif arm == "gt_sep_single":
            model = OrdinaryEnsemble([frozen.Model(**kwargs)], arm)
        elif arm == "untied":
            model = OrdinaryEnsemble([frozen.Model(**kwargs) for _ in range(members)], arm)
        else:
            model = SharedTrunkHeads(frozen.Model(**kwargs), hidden_dim, output_dim, members, dropout)
    model.specification = dict(
        arm=arm, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
        seed=seed, permutation_seed=permutation_seed, requested_members=members,
        actual_members=model.members, num_layers=num_layers, dropout=dropout,
        hidden_dim_multiplier=hidden_dim_multiplier, normalization="layer", num_heads=num_heads,
        boundary_be_live=arm in ("original", "factor", "coordinate", "permutation", "bias_only"),
        permutation_scope="hidden maps only" if arm in ("coordinate", "permutation") else "none",
        initialization_contract=("shared TABM maps/norms copied; ordinary boundaries omit private R/S"
                                 if arm == "single" else "matched TABM base seed" if arm in
                                 ("original", "factor", "coordinate", "permutation", "bias_only")
                                 else "native paired construction seed; shapes/functions may differ"),
        frozen_source=str(FROZEN_SOURCE), frozen_sha256=FROZEN_SHA256,
    )
    return model


def parameter_count_formula(arm: str, input_dim: int, hidden_dim: int, output_dim: int,
                            members: int = 4, num_layers: int = 2,
                            hidden_dim_multiplier: float = 1.0) -> int:
    """Pure arithmetic for prospective SAGE width matching; no construction.

    Counts include affine LayerNorm, bias flags, private biases, boundaries and
    capable heads. This count does not include integer permutation storage.
    GT-sep counts require a separately admitted construction/storage report.
    """
    h, t, d, c, m, l = (hidden_dim, int(hidden_dim * hidden_dim_multiplier),
                         input_dim, output_dim, members, num_layers)
    ordinary_block = 2 * h * h + 3 * h * t + 4 * h + t
    single = d * h + h + l * ordinary_block + 2 * h + h * c + c
    if arm == "single":
        return single
    if arm == "untied":
        return m * single
    if arm == "heads":
        return single - (h * c + c) + m * (h * h + h + h * c + c)
    shared_boundaries = d * h + h * c
    private_boundaries = m * (d + 3 * h + 2 * c)
    if arm == "original":
        return shared_boundaries + l * ordinary_block + 2 * h + private_boundaries
    if arm in ("factor", "coordinate", "permutation", "bias_only"):
        shared_blocks = l * (2 * h * h + 3 * h * t + 2 * h)
        private_biases = m * l * (2 * h + t)
        factors = m * l * (7 * h + 2 * t) if arm in ("factor", "coordinate") else 0
        return shared_boundaries + shared_blocks + 2 * h + private_boundaries + private_biases + factors
    raise ValueError("No qualified analytic count for this arm")


def storage_report(model: nn.Module) -> dict[str, Any]:
    """JSON-serializable actual tensor counts; never infers storage from seeds.

    Logical bytes count unique registered tensor objects. unique_storage_bytes
    additionally deduplicates aliases/views across parameter and buffer storage.
    Checkpoints include all context indices. Graph/activation/optimizer memory is
    outside this model-only report and must be measured/charged by the runner.
    """
    private_ids = getattr(model, "_private_parameter_ids", set())
    parameters, buffers = [], []
    allocations: dict[tuple[str, int], int] = {}
    for name, parameter in model.named_parameters():
        classification = "private" if id(parameter) in private_ids else "shared"
        row = dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                   numel=parameter.numel(), bytes=parameter.numel() * parameter.element_size(),
                   trainable=parameter.requires_grad, classification=classification)
        parameters.append(row)
        storage = parameter.untyped_storage()
        allocations[(str(parameter.device), storage.data_ptr())] = storage.nbytes()
    for name, buffer in model.named_buffers():
        kind = "index" if name.endswith(("input_index", "output_index")) else "other"
        buffers.append(dict(name=name, shape=list(buffer.shape), dtype=str(buffer.dtype),
                            numel=buffer.numel(), bytes=buffer.numel() * buffer.element_size(), kind=kind))
        storage = buffer.untyped_storage()
        allocations[(str(buffer.device), storage.data_ptr())] = storage.nbytes()
    trainable = [row for row in parameters if row["trainable"]]
    parameter_bytes = sum(row["bytes"] for row in parameters)
    buffer_bytes = sum(row["bytes"] for row in buffers)
    return dict(
        trainable_parameters=sum(row["numel"] for row in trainable),
        shared_trainable_parameters=sum(row["numel"] for row in trainable if row["classification"] == "shared"),
        private_trainable_parameters=sum(row["numel"] for row in trainable if row["classification"] == "private"),
        parameter_bytes=parameter_bytes,
        shared_parameter_bytes=sum(row["bytes"] for row in parameters if row["classification"] == "shared"),
        private_parameter_bytes=sum(row["bytes"] for row in parameters if row["classification"] == "private"),
        buffer_bytes=buffer_bytes,
        index_buffer_bytes=sum(row["bytes"] for row in buffers if row["kind"] == "index"),
        index_buffer_values=sum(row["numel"] for row in buffers if row["kind"] == "index"),
        total_model_tensor_bytes=parameter_bytes + buffer_bytes,
        unique_storage_bytes=sum(allocations.values()), parameters=parameters, buffers=buffers,
        specification=getattr(model, "specification", {}),
    )
