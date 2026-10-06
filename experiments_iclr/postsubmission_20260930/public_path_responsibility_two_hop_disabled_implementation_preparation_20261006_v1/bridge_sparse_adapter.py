"""DISABLED pair/Laplacian adapter for the pinned G0/native sparse port.

No trainer, supervision rule, callback, loader, model, RNG, CLI or source import.
The caller independently authenticates loaded source and public/S-role custody.
Only supplied copied S labels enter fixed pair membership/permutation checks.
No implementation execution or native/backend/resource qualification is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

ADAPTER_RELEASED = False
OPERATOR_SHA256 = "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"
NATIVE_PORT_SHA256 = "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"
PROTOCOL_SHA256 = "27f8a8d85aec4e81114ed9ecd8554fcfaa6dd4ccf1eaf7678cc0f5eeeb477d99"
CENSUS_SUMMARY_SHA256 = "015fb3ee020f1f57f04aa050ed74a577a39eb150a1dc2979fbc363fa790f4d88"
EXPECTED_CLASS_COUNTS = (631, 906, 578, 231, 103)
# (terminal rows, direct undirected edges, isolates, direct components).
EXPECTED_DIRECT_CENSUS = {
    (0, 1): (1537, 436, 904, 1140), (0, 2): (1209, 234, 834, 992),
    (0, 3): (862, 127, 645, 740), (0, 4): (734, 108, 557, 632),
    (1, 2): (1484, 385, 895, 1120), (1, 3): (1137, 224, 790, 927),
    (1, 4): (1009, 187, 712, 832), (2, 3): (809, 140, 589, 679),
    (2, 4): (681, 82, 532, 599), (3, 4): (334, 35, 275, 300),
}


def _backend():
    import torch
    return torch


def _check_sources(operator, native_port, operator_sha256, native_port_sha256,
                   protocol_sha256, census_summary_sha256):
    """Exact caller attestations; not a replacement for upstream custody."""
    if (operator_sha256 != OPERATOR_SHA256
            or native_port_sha256 != NATIVE_PORT_SHA256
            or protocol_sha256 != PROTOCOL_SHA256
            or census_summary_sha256 != CENSUS_SUMMARY_SHA256
            or native_port.OPERATOR_SHA256 != OPERATOR_SHA256):
        raise ValueError("Require the exact pinned G0/port/protocol/census")
    native_port._check_operator(operator, operator_sha256)


def _guard_fixed_permutation(inner_indices, inner_labels, permutation):
    """Verify the single literal seed17 protocol rule; never choose a recipe.

    As in the existing port, p means K_control[i,j] = K[p[i],p[j]].
    Sorted public S IDs and copied S labels are inspected only at static setup.
    """
    torch = _backend()
    size = inner_indices.numel()
    if (permutation.dtype != torch.long or permutation.device != inner_indices.device
            or permutation.shape != inner_indices.shape or permutation.requires_grad
            or not torch.equal(permutation.sort().values,
                               torch.arange(size, device=permutation.device))
            or not torch.equal(inner_labels[permutation], inner_labels)):
        raise ValueError("Require a complete class-preserving S permutation")
    ids = inner_indices.detach().cpu().tolist()
    labels = inner_labels.detach().cpu().tolist()
    if ids != sorted(ids):
        raise ValueError("Protocol S rows must be increasing public node IDs")
    expected = list(range(size))
    for label in range(5):
        ordered = [i for i, value in enumerate(labels) if value == label]
        hashed = sorted(ordered, key=lambda i: (
            sha256(("amazon-response-G0|split=0|seed=17|perm|"
                    + str(label) + "|" + str(ids[i])).encode("utf-8")).digest(), ids[i]))
        for position, replacement in zip(ordered, hashed):
            expected[position] = replacement
    required = torch.tensor(expected, dtype=torch.long, device=permutation.device)
    if not torch.equal(permutation, required):
        raise ValueError("Permutation differs from the one frozen protocol rule")


def _component_summary(size, direct_rows, direct_columns, *,
                       star_rows=(), star_columns=(), inverse=None):
    """Integer DSU: direct edges plus each interior's positive bridge clique.

    No clique edge matrix is materialized. Singleton stars do not add edges.
    inverse maps an original bridge row to its permuted-control row.
    """
    parent = list(range(size))
    mass = [1] * size

    def find(value):
        while value != parent[value]:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if mass[left] < mass[right]:
            left, right = right, left
        parent[right] = left
        mass[left] += mass[right]

    for left, right in zip(direct_rows, direct_columns):
        union(left, right)
    first = {}
    for row, interior in zip(star_rows, star_columns):
        row = row if inverse is None else inverse[row]
        if interior in first:
            union(first[interior], row)
        else:
            first[interior] = row
    sizes = [mass[i] for i in range(size) if find(i) == i]
    return {"components": len(sizes), "largest_component": max(sizes),
            "connected_support": len(sizes) == 1,
            "singleton_components": sum(value == 1 for value in sizes)}


@dataclass(frozen=True)
class StarBridgeLaplacian:
    """C=A_TU D_U^-1 A_UT; apply the Laplacian of offdiag(C).

    row/column encode unique unit terminal-interior incidences, with compact U.
    full-degree inverses and integer terminal-neighbour counts are fixed.
    degree_i = sum_{u adjacent i} (k_u-1)/d_u, accumulated nonnegatively.
    No C diagonal subtraction is used to construct degree or normalization.
    """

    row: Tensor
    column: Tensor
    inverse_full_degree: Tensor
    terminal_neighbours: Tensor
    degree: Tensor

    def __matmul__(self, q):
        torch = _backend()
        if (q.ndim != 2 or q.shape[0] != self.degree.numel()
                or q.dtype != self.degree.dtype or q.device != self.row.device):
            raise ValueError("Bridge action requires matching dense terminal rows")
        selected = q[self.row]
        interior_sum = q.new_zeros((self.terminal_neighbours.numel(), q.shape[1]))
        interior_sum = interior_sum.index_add(0, self.column, selected)
        # k_u*q_i - sum_j(q_j) equals (k_u-1)*q_i - sum_{j != i}(q_j).
        # Thus self affinity cancels algebraically; Q is never detached/repaired.
        messages = (self.terminal_neighbours[self.column, None].to(q.dtype) * selected
                    - interior_sum[self.column])
        messages = self.inverse_full_degree[self.column, None] * messages
        return torch.zeros_like(q).index_add(0, self.row, messages)


@dataclass(frozen=True)
class DirectAndBridgeLaplacian:
    direct: Any                  # Existing native_port.SparseLaplacian, RAW units.
    bridge: StarBridgeLaplacian
    permutation: Tensor
    inverse_permutation: Tensor
    normalization: Tensor        # Same scalar object in live and control.

    def __matmul__(self, q):
        # Let Pq=q[inverse_permutation]. Then P^T L_B P acts as below,
        # with control affinity W_B[p[i],p[j]], matching the native convention.
        # Identity p in live keeps exactly the same gather/aggregation work.
        bridged = self.bridge @ q[self.inverse_permutation]
        return (self.direct @ q + bridged[self.permutation]) / self.normalization


@dataclass(frozen=True)
class PairBanks:
    live: tuple[Any, ...]
    bridge_permuted: tuple[Any, ...]
    static_coverage: tuple[dict, ...]
    full_public_summary: dict
    minimum_restoration_gate: bool


def _coverage(classes, direct, factor, partner_count, permutation, inverse):
    """Exact integer support counts; weighted values are not used as cutoffs."""
    rows, columns = direct.row.cpu().tolist(), direct.column.cpu().tolist()
    star_rows, star_columns = factor.row.cpu().tolist(), factor.column.cpu().tolist()
    n = factor.degree.numel()
    direct_degree = [0] * n
    for row in rows:
        direct_degree[row] += 1
    partners = partner_count.cpu().tolist()
    p, inv = permutation.cpu().tolist(), inverse.cpu().tolist()
    direct_summary = _component_summary(n, rows, columns)
    observed = (n, len(rows) // 2, sum(value == 0 for value in direct_degree),
                direct_summary["components"])
    if observed != EXPECTED_DIRECT_CENSUS[classes]:
        raise ValueError("Direct support differs from the pinned actual census")
    result = {"classes": classes, "pair_nodes": n, "direct_edges": len(rows) // 2,
              "direct_isolates": observed[2], "direct_connectivity": direct_summary,
              "terminal_interior_incidences": len(star_rows),
              "interior_nodes_used": factor.terminal_neighbours.numel(),
              "bridge_two_hop_path_instances": sum(
                  k * (k - 1) // 2 for k in factor.terminal_neighbours.cpu().tolist()),
              "bridge_unique_edge_count": None}
    for name, values, mapping in (("live", partners, None),
                                  ("bridge_permuted", [partners[i] for i in p], inv)):
        result[name] = {
            "bridge_supported_rows": sum(value > 0 for value in values),
            "newly_coupled_direct_isolates": sum(
                d == 0 and b > 0 for d, b in zip(direct_degree, values)),
            "combined_isolates": sum(
                d == 0 and b == 0 for d, b in zip(direct_degree, values)),
            "combined_connectivity": _component_summary(
                n, rows, columns, star_rows=star_rows, star_columns=star_columns,
                inverse=mapping)}
    return result


def _prepare_bridge_pair_banks(operator, native_port, operator_sha256,
                               native_port_sha256, inner_indices, inner_labels,
                               edge_index, *, node_count, dtype,
                               affinity_permutation, protocol_sha256,
                               census_summary_sha256):
    """Static preparation only; no cost, model, callback or episode is evaluated.

    Underscored entry is solely for a later separately authorized engineering
    caller. It does not authenticate physical dataset/role artifacts itself.
    All tensors/context must first pass the existing custodian/admission gates.
    """
    _check_sources(operator, native_port, operator_sha256, native_port_sha256,
                   protocol_sha256, census_summary_sha256)
    torch = _backend()
    if (node_count != 24492 or inner_indices.numel() != 2449
            or dtype not in (torch.float32, torch.float64)
            or edge_index.requires_grad):
        raise ValueError("Require the pinned fixed Amazon context and floating dtype")
    # Reuse its exact pair membership/edge validation. Do NOT permute direct edges.
    original = native_port._sparse_pairs(
        operator, inner_indices, inner_labels, edge_index,
        node_count=node_count, dtype=dtype, affinity_permutation=None)
    if (len(original) != 10 or not torch.equal(
            torch.bincount(inner_labels, minlength=5),
            torch.tensor(EXPECTED_CLASS_COUNTS, dtype=torch.long,
                         device=inner_labels.device))):
        raise ValueError("S support differs from the fixed census")
    _guard_fixed_permutation(inner_indices, inner_labels, affinity_permutation)
    nonloop = edge_index[0] != edge_index[1]
    public_row, public_column = edge_index[0, nonloop], edge_index[1, nonloop]
    full_degree = torch.bincount(public_row, minlength=node_count)
    full_summary = _component_summary(node_count, public_row.cpu().tolist(),
                                     public_column.cpu().tolist())
    if (public_row.numel() != 186100 or bool((full_degree == 0).any())
            or not full_summary["connected_support"]):
        raise ValueError("Full public graph differs from the actual connected census")
    full_summary.update({"nodes": node_count, "undirected_nonloop_edges": 93050})
    live, controls, coverage = [], [], []
    for pair in original:
        n = pair.nodes.numel()
        terminal = inner_indices[pair.nodes]
        to_terminal = torch.full((node_count,), -1, dtype=torch.long,
                                 device=edge_index.device)
        to_terminal[terminal] = torch.arange(n, device=edge_index.device)
        row, destination = to_terminal[public_row], to_terminal[public_column]
        keep = (row >= 0) & (destination < 0)
        row, interior = row[keep], public_column[keep]
        unique_interior, column = torch.unique(interior, sorted=True, return_inverse=True)
        counts = torch.bincount(column, minlength=unique_interior.numel())
        if counts.numel() and bool((counts < 1).any()):
            raise ValueError("Invalid compact interior incidence")
        inverse_degree = full_degree[unique_interior].to(dtype).reciprocal()
        partners = counts[column] - 1  # Integer nonnegative; never a float repair.
        partner_count = torch.zeros(n, dtype=torch.long, device=edge_index.device)
        partner_count = partner_count.index_add(0, row, partners)
        degree = torch.zeros(n, dtype=dtype, device=edge_index.device)
        degree = degree.index_add(0, row, partners.to(dtype) * inverse_degree[column])
        factor = StarBridgeLaplacian(row.clone(), column.clone(), inverse_degree.clone(),
                                    counts.clone(), degree.clone())
        raw_degree = torch.zeros(n, dtype=torch.long, device=edge_index.device)
        raw_degree = raw_degree.index_add(0, pair.laplacian.row,
                                         torch.ones_like(pair.laplacian.row))
        direct = native_port.SparseLaplacian(
            pair.laplacian.row.clone(), pair.laplacian.column.clone(),
            torch.ones_like(pair.laplacian.weight), raw_degree.to(dtype))
        normalization = 1.0 + direct.degree.max() + factor.degree.max()
        if (not bool(torch.isfinite(factor.degree).all())
                or bool((factor.degree < 0).any())
                or not bool(torch.isfinite(normalization))):
            raise ValueError("Invalid fixed bridge degree/normalization; no repair allowed")
        to_pair = torch.full(inner_indices.shape, -1, dtype=torch.long,
                             device=edge_index.device)
        to_pair[pair.nodes] = torch.arange(n, device=edge_index.device)
        p = to_pair[affinity_permutation[pair.nodes]].clone()
        if (bool((p < 0).any()) or not torch.equal(p.sort().values,
                                                   torch.arange(n, device=p.device))
                or not torch.equal(pair.targets[p], pair.targets)):
            raise ValueError("Fixed permutation does not restrict within this pair")
        inverse = torch.empty_like(p)
        inverse[p] = torch.arange(n, device=p.device)
        identity = torch.arange(n, device=p.device)
        for bank, perm, inv in ((live, identity, identity), (controls, p, inverse)):
            laplacian = DirectAndBridgeLaplacian(direct, factor, perm, inv, normalization)
            bank.append(operator.Pair(pair.classes, pair.nodes.clone(), pair.targets.clone(),
                                      pair.competitors.clone(), laplacian))
        coverage.append(_coverage(pair.classes, direct, factor, partner_count, p, inverse))
    return PairBanks(tuple(live), tuple(controls), tuple(coverage), full_summary,
                     all(c["live"]["newly_coupled_direct_isolates"] > 0 for c in coverage))


def prepare_bridge_pair_banks(*args, **kwargs):
    """Release-gated preparation; no callable trainer or episode is added."""
    if not ADAPTER_RELEASED:
        raise RuntimeError("Disabled source-only bridge adapter; not qualified/released")
    return _prepare_bridge_pair_banks(*args, **kwargs)
