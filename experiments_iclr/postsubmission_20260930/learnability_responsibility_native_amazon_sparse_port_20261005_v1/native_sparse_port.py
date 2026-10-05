"""Disabled Amazon one-member callback/sparse affinity port; source only.

No loader, native import, model construction/reset, trainer, RNG call or CLI.
Supply the pinned native family and public context through a separately gated
caller. Underscored helpers are for separately authorized engineering checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from torch import Tensor

PORT_RELEASED = False
OPERATOR_SHA256 = "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"
PRIVATE_NAMES = tuple(f"{owner}.{factor}" for owner in
                      ("stem", "local_head", "global_head")
                      for factor in ("R", "S", "B"))
PRIVATE_SHAPES = {
    "stem.R": (300,), "stem.S": (512,), "stem.B": (512,),
    "local_head.R": (512,), "local_head.S": (5,), "local_head.B": (5,),
    "global_head.R": (512,), "global_head.S": (5,), "global_head.B": (5,),
}


def _backend():
    import torch
    return torch


def _check_operator(operator, source_sha256):
    # Caller must independently verify exact loaded source bytes before binding.
    if source_sha256 != OPERATOR_SHA256 or operator.SOURCE_RELEASED is not False:
        raise ValueError("Require the exact disabled repaired G0V2 operator")


def _native_callback_and_state(family, features, edge_index, *,
                               expected_nodes=24492, global_stage=True):
    """Extract complete state and call ONLY one native forward_member route."""
    torch = _backend()
    if type(family).__name__ != "PolynormerBoundaryFamily" or family.members != 4:
        raise ValueError("Only the pinned native Amazon shared4 family is bound")
    if any(module.training for module in family.modules()):
        raise ValueError("Caller must set the complete family to eval beforehand")
    if tuple(family.named_buffers()):
        raise ValueError("Registered buffers are not admitted by this pure callback")
    core = family.core
    if (bool(core._global) != global_stage or len(core.local_convs) != 10
            or core.global_attn.num_layers != 1 or core.global_attn.hidden_channels != 256
            or core.global_attn.heads != 2 or core.pre_ln
            or not core.global_attn.qk_shared or core.beta != -1
            or core.global_attn.beta != -1 or core.in_drop != 0.2
            or core.dropout != 0.3 or core.global_attn.dropout != 0.3):
        raise ValueError("Native architecture/stage differs")
    if (features.shape != (expected_nodes, 300) or features.requires_grad
            or edge_index.ndim != 2 or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long or edge_index.requires_grad):
        raise ValueError("Fixed complete public context required")
    parameters = dict(family.named_parameters())
    if not set(PRIVATE_NAMES).issubset(parameters):
        raise ValueError("Missing private boundary bank")
    for name in PRIVATE_NAMES:
        if parameters[name].shape != (4, *PRIVATE_SHAPES[name]):
            raise ValueError("Wrong private bank shape: " + name)
    if any(p.dtype != features.dtype or p.device != features.device
           for p in parameters.values()) or edge_index.device != features.device:
        raise ValueError("All parameters/public context must share dtype/device")
    shared_names = tuple(name for name in parameters if name not in PRIVATE_NAMES)
    theta = {name: parameters[name].detach().clone() for name in shared_names}
    phis = tuple({name: parameters[name][m].detach().clone() for name in PRIVATE_NAMES}
                 for m in range(4))

    class OneRoute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.family = family
            self.training = False

        def forward(self, x, edges):
            return self.family.forward_member(x, edges, 0)

    route = OneRoute()

    def forward(shared, private):
        if set(shared) != set(shared_names) or set(private) != set(PRIVATE_NAMES):
            raise ValueError("Complete theta and private row dictionaries required")
        if bool(core._global) != global_stage or any(m.training for m in family.modules()):
            raise ValueError("Stage/eval mode changed after callback binding")
        state = {"family." + name: value for name, value in shared.items()}
        # Row0 now IS the supplied member, not the original bank's member0.
        state.update({"family." + name: value.unsqueeze(0)
                      for name, value in private.items()})
        return torch.func.functional_call(route, (state, {}),
                                          (features, edge_index),
                                          tie_weights=True, strict=True)

    return forward, theta, phis, shared_names


@dataclass(frozen=True)
class SparseLaplacian:
    row: Tensor
    column: Tensor
    weight: Tensor
    degree: Tensor

    def __matmul__(self, q):
        """Exactly diag(W1)Q-WQ; fixed edges, differentiable dense columns."""
        torch = _backend()
        messages = self.weight[:, None] * q[self.column]
        adjacency_q = torch.zeros_like(q).index_add(0, self.row, messages)
        return self.degree[:, None] * q - adjacency_q


def _sparse_pairs(operator, inner_indices, inner_labels, edge_index, *,
                  node_count, dtype, affinity_permutation=None):
    """Same K restriction/norm as dense G0V2; no same-label edge filter.

    Native model edges already are unique reciprocal undirected edges plus loops.
    Remove loops ONLY for K. If supplied, permutation p means K'[i,j]=K[p[i],p[j]].
    p must be the frozen within-class control from the separate protocol.
    """
    torch = _backend()
    device = edge_index.device
    if (inner_indices.dtype != torch.long or inner_labels.dtype != torch.long
            or inner_indices.ndim != 1 or inner_indices.shape != inner_labels.shape
            or not inner_indices.numel()
            or inner_indices.unique().numel() != inner_indices.numel()
            or inner_indices.device != device or inner_labels.device != device
            or bool(((inner_indices < 0) | (inner_indices >= node_count)).any())
            or bool(((inner_labels < 0) | (inner_labels >= 5)).any())
            or not bool((torch.bincount(inner_labels, minlength=5) > 0).all())):
        raise ValueError("Unique permitted innerS with all five classes required")
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Native public edges must be [2,E] long")
    if bool(((edge_index < 0) | (edge_index >= node_count)).any()):
        raise ValueError("Invalid public graph node identity")
    encoded = edge_index[0] * node_count + edge_index[1]
    reverse = edge_index[1] * node_count + edge_index[0]
    if (encoded.unique().numel() != encoded.numel()
            or not torch.equal(encoded.sort().values, reverse.sort().values)):
        raise ValueError("Require exact unique reciprocal native edge support")
    size = inner_indices.shape[0]
    to_inner = torch.full((node_count,), -1, dtype=torch.long, device=device)
    to_inner[inner_indices] = torch.arange(size, device=device)
    source, target = to_inner[edge_index[0]], to_inner[edge_index[1]]
    keep = (source >= 0) & (target >= 0) & (source != target)
    source, target = source[keep], target[keep]
    if affinity_permutation is not None:
        p = affinity_permutation
        if (p.shape != inner_indices.shape or p.dtype != torch.long or p.device != device
                or p.unique().numel() != size or bool(((p < 0) | (p >= size)).any())
                or not torch.equal(inner_labels[p], inner_labels)):
            raise ValueError("Require a fixed within-innerS-class permutation")
        inverse = torch.empty_like(p)
        inverse[p] = torch.arange(size, device=device)
        source, target = inverse[source], inverse[target]
    pairs = []
    for left in range(5):
        for right in range(left + 1, 5):
            nodes = torch.nonzero((inner_labels == left) | (inner_labels == right)).flatten()
            labels = inner_labels[nodes]
            competitors = torch.where(labels == left, right, left)
            to_pair = torch.full((size,), -1, dtype=torch.long, device=device)
            to_pair[nodes] = torch.arange(nodes.shape[0], device=device)
            row, column = to_pair[source], to_pair[target]
            keep = (row >= 0) & (column >= 0)
            row, column = row[keep], column[keep]
            raw_weight = torch.ones(row.shape, dtype=dtype, device=device)
            raw_degree = torch.zeros(nodes.shape[0], dtype=dtype, device=device).index_add(
                0, row, raw_weight)
            weight = raw_weight / (1.0 + raw_degree.max())
            degree = torch.zeros_like(raw_degree).index_add(0, row, weight)
            laplacian = SparseLaplacian(row, column, weight, degree)
            pairs.append(operator.Pair((left, right), nodes, labels, competitors, laplacian))
    return tuple(pairs)


def bind_native_callback(family, features, edge_index, **kwargs):
    if not PORT_RELEASED:
        raise RuntimeError("Disabled source-only native port")
    return _native_callback_and_state(family, features, edge_index, **kwargs)


def prepare_sparse_pairs(operator, source_sha256, *args, **kwargs):
    if not PORT_RELEASED:
        raise RuntimeError("Disabled source-only native port")
    _check_operator(operator, source_sha256)
    return _sparse_pairs(operator, *args, **kwargs)


def native_sparse_episode(operator, source_sha256, theta, phis, forward, pairs,
                          inner_indices, inner_labels, query_indices, query_labels,
                          *, control="live"):
    """Same G0V2 state map; only the prepared L object is sparse.

    Caller validates/binds fixed S/R/A roles and all graph/task provenance before
    entry. Pair objects must be from the matching prepare_sparse_pairs call.
    No acquisition, scoring, selection, buffer/parameter commit or I/O occurs.
    """
    if not PORT_RELEASED:
        raise RuntimeError("Disabled source-only native port")
    _check_operator(operator, source_sha256)
    torch = _backend()
    cfg = operator.Config()  # Exact fixed G0, no coefficient/iteration override.
    if (len(phis) != 4 or len(pairs) != 10 or control not in operator.CONTROLS
            or query_indices.dtype != torch.long or query_labels.dtype != torch.long
            or query_indices.shape != query_labels.shape or query_indices.ndim != 1
            or not query_indices.numel()
            or query_indices.unique().numel() != query_indices.numel()
            or bool((query_indices < 0).any())
            or bool(torch.isin(inner_indices, query_indices).any())
            or bool(((query_labels < 0) | (query_labels >= 5)).any())):
        raise ValueError("Bound shared4 S/R episode required")
    theta_start = operator._detach(theta)
    private_start = tuple(operator._detach(phi) for phi in phis)

    def outer(core):
        adapted, _ = operator._private_response(
            core, private_start, forward, inner_indices, inner_labels, pairs, 5, cfg, control)
        return operator._query_objective(
            core, adapted, forward, query_indices, query_labels, cfg)

    gradient, virtual_loss = torch.func.grad_and_value(outer)(theta_start)
    theta_next = operator._detach(operator._sgd(theta_start, gradient, cfg.eta_core))
    recomputed, diagnostics = operator._private_response(
        theta_next, private_start, forward, inner_indices, inner_labels, pairs,
        5, cfg, control, collect_diagnostics=True)
    phis_next = tuple(operator._detach(phi) for phi in recomputed)
    diagnostics["virtual_query_loss"] = virtual_loss.detach()
    return theta_next, phis_next, diagnostics
