"""Prepared exact undirected mask semantics; no data read or sampler run here.

Simple reciprocal unweighted native edge records are required. Unsupported
weighted/duplicate graphs stop; no silent coalescing, directed removal or repair.
Native normalization is rebuilt after paired-record removal. All self-loops stay.
"""
from dataclasses import dataclass
from collections import Counter, defaultdict
from hashlib import sha256
from typing import Mapping
from core.groups import LabelRoles

VIEWS = ("same_removal", "other_removal")


class MaskInfeasible(ValueError):
    pass


def _priority(seed: str, purpose: str, unit):
    return sha256(f"{seed}|{purpose}|{unit[0]}|{unit[1]}".encode()).hexdigest(), unit


@dataclass(frozen=True)
class EdgeUniverse:
    node_count: int
    records: tuple[tuple[int, int], ...]
    source_normalization_receipt: str
    edge_attributes_are_none: bool

    def __post_init__(self):
        if self.node_count <= 0 or not self.source_normalization_receipt or not self.edge_attributes_are_none:
            raise ValueError("Pinned simple unweighted native graph required; weighted semantics unqualified")
        directed = Counter()
        for u, v in self.records:
            if type(u) is not int or type(v) is not int or not (0 <= u < self.node_count and 0 <= v < self.node_count):
                raise ValueError("Native edge endpoint invalid")
            if u != v:
                directed[(u, v)] += 1
        if any(count != 1 or directed[(v, u)] != 1 for (u, v), count in directed.items()):
            raise ValueError("Reciprocal simple records required; duplicate/asymmetric case must be explicitly source-qualified")

    @property
    def units(self):
        return tuple(sorted({(min(u, v), max(u, v)) for u, v in self.records if u != v}))

    @property
    def native_degrees_with_one_loop(self):
        # Pinned gcn_norm(add_self_loops=True, edge_weight=None) retains/replaces
        # self-loop records with exactly one unit loop per node before degree.
        degree = [1] * self.node_count
        for u, v in self.units:
            degree[u] += 1; degree[v] += 1
        return tuple(degree)

    @property
    def nonloop_neighbor_map(self):
        neighbors = {n: [] for n in range(self.node_count)}
        for u, v in self.units:
            neighbors[u].append(v); neighbors[v].append(u)
        return {n: tuple(sorted(values)) for n, values in neighbors.items()}

    def retained_records(self, removed_units):
        removed = set(removed_units)
        if len(removed) != len(removed_units) or not removed <= set(self.units):
            raise ValueError("Removed units must be unique native nonloop pairs")
        return tuple((u, v) for u, v in self.records if u == v or (min(u, v), max(u, v)) not in removed)


@dataclass(frozen=True)
class MaskPair:
    same_removal: tuple[tuple[int, int], ...]
    other_removal: tuple[tuple[int, int], ...]
    rule_receipt: str
    seed_receipt: str

    def __post_init__(self):
        if not self.rule_receipt or not self.seed_receipt:
            raise ValueError("Explicit mask rule and fixed seed receipts required")
        if set(self.same_removal) & set(self.other_removal):
            raise ValueError("Same/different removals must be disjoint")
        if any(len(set(getattr(self, v))) != len(getattr(self, v)) for v in VIEWS):
            raise ValueError("Repeated removed unit")


def _eligible(universe: EdgeUniverse, fit_labels: Mapping[int, int]):
    if not fit_labels or any(type(n) is not int or not 0 <= n < universe.node_count or
                            type(y) is not int or y < 0 for n, y in fit_labels.items()):
        raise ValueError("Exactly declared fit-role TRAIN labels required")
    same, other = [], []
    for u, v in universe.units:
        if u in fit_labels and v in fit_labels:
            (same if fit_labels[u] == fit_labels[v] else other).append((u, v))
    return {"same_removal": tuple(same), "other_removal": tuple(other)}


def actual_fit_labels(roles: LabelRoles):
    labels = dict(roles.train_labels)
    return {n: labels[n] for n in roles.fit}


def _permutation_boundary(roles: LabelRoles, labels: Mapping[int, int]):
    original = actual_fit_labels(roles)
    if set(labels) != set(original) or Counter(labels.values()) != Counter(original.values()):
        raise ValueError("Auxiliary permutation must cover exactly fit TRAIN labels and preserve class counts")


def sample_fixed_ten_percent(universe: EdgeUniverse, roles: LabelRoles, *,
                            seed: str, resolved_mask_rule_receipt: str):
    """Proposed successor rule: floor(E_eligible/10) distinct unordered units.

    One fixed seed per block; no per-class balancing, retries or minimum-one
    repair. Remove both original reciprocal records. Empty probes remain empty
    and may fail the response active-coverage screen. Adoption is still pending.
    """
    if not seed or not resolved_mask_rule_receipt:
        raise ValueError("Seed and explicit prospective mask adoption required")
    eligible = _eligible(universe, actual_fit_labels(roles))
    selected = {view: tuple(sorted(sorted(edges, key=lambda e: _priority(seed, view, e))[:len(edges) // 10]))
                for view, edges in eligible.items()}
    return MaskPair(**selected, rule_receipt=resolved_mask_rule_receipt, seed_receipt=seed)


def permute_fit_labels_once(roles: LabelRoles, *, seed: str):
    """One fixed pseudorandom permutation of fit label slots, no seed search.

    These auxiliary labels are used only for permuted-mask edge categories.
    Actual TRAIN target/group/CE labels must remain unchanged. Class counts are
    preserved. Identity/noninformative draws are reported, never retried silently.
    """
    if not seed:
        raise ValueError("Fixed permutation seed required")
    fit_labels = actual_fit_labels(roles)
    nodes = sorted(fit_labels)
    destination = sorted(nodes, key=lambda n: _priority(seed, "fit_label_permutation", (n, n)))
    result = {n: fit_labels[source] for n, source in zip(destination, nodes)}
    if Counter(result.values()) != Counter(fit_labels.values()):
        raise RuntimeError("Permutation changed fit class counts")
    return result


def removed_degree_vector(universe: EdgeUniverse, units):
    degree = [0] * universe.node_count
    for u, v in units:
        degree[u] += 1; degree[v] += 1
    return tuple(degree)


def _original_boundary(universe: EdgeUniverse, original: MaskPair, roles: LabelRoles):
    eligible = _eligible(universe, actual_fit_labels(roles))
    for view in VIEWS:
        if not set(getattr(original, view)) <= set(eligible[view]) or len(getattr(original, view)) != len(eligible[view]) // 10:
            raise ValueError("Original mask violates fit-only category/native-unit/floor-ten-percent rules")


@dataclass(frozen=True)
class StrictDegreeInstance:
    eligible: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    required_counts: tuple[tuple[str, int], ...]
    required_per_node_removed_degrees: tuple[tuple[str, tuple[int, ...]], ...]
    priority_order: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    instance_receipt: str


def strict_per_node_instance(universe: EdgeUniverse, original: MaskPair,
                             roles: LabelRoles, permuted_fit_labels: Mapping[int, int], *, seed: str,
                             instance_receipt: str):
    """Alternative A: exact per-node/per-view degree and original total counts.

    Binary variables x_e^view are limited to the once-permuted category. Require
    sum x=count_original and sum_{e incident v}x=degree_original(v), every node.
    Fixed hash priority defines lexicographic selected-set tie breaking; a pinned
    exact b-matching/MILP feasibility oracle is still required, not implemented.
    Solve one instance only. Infeasible/timeout/unqualified oracle stops the arm;
    no alternative permutation, masks, relaxed degrees or coarse-bin fallback.
    """
    _permutation_boundary(roles, permuted_fit_labels)
    _original_boundary(universe, original, roles)
    if not seed or not instance_receipt:
        raise ValueError("Fixed oracle priority seed and source-qualified instance receipt required")
    eligible = _eligible(universe, permuted_fit_labels)
    required_degrees = {v: removed_degree_vector(universe, getattr(original, v)) for v in VIEWS}
    required_counts = {v: len(getattr(original, v)) for v in VIEWS}
    for view in VIEWS:
        capacity = removed_degree_vector(universe, eligible[view])
        if len(eligible[view]) < required_counts[view] or any(need > cap for need, cap in zip(required_degrees[view], capacity)):
            raise MaskInfeasible("Necessary exact per-node category capacity already fails; no retry")
    return StrictDegreeInstance(tuple(eligible.items()), tuple(required_counts.items()),
                                tuple(required_degrees.items()),
                                tuple((v, tuple(sorted(es, key=lambda e: _priority(seed, v, e)))) for v, es in eligible.items()),
                                instance_receipt)


def verify_strict_degree_certificate(universe: EdgeUniverse, instance: StrictDegreeInstance,
                                     selected: MaskPair):
    allowed, counts, degrees = dict(instance.eligible), dict(instance.required_counts), dict(instance.required_per_node_removed_degrees)
    for view in VIEWS:
        units = getattr(selected, view)
        if not set(units) <= set(allowed[view]) or len(units) != counts[view] or removed_degree_vector(universe, units) != degrees[view]:
            raise MaskInfeasible("Exact degree/count/once-permuted category certificate fails")
    # This checks feasibility only; oracle optimal/lex tie-breaking and genuine
    # infeasibility certificates remain separately source-qualified requirements.
    return True


def match_exact_native_degree_pairs(universe: EdgeUniverse, original: MaskPair,
                                    roles: LabelRoles, permuted_fit_labels: Mapping[int, int], *, seed: str,
                                    resolved_alternative_b_receipt: str):
    """Alternative B: exact native-degree-pair edge distribution/count matching.

    Distinct scientific null: counts per (view,min native degree,max native degree)
    exactly match original masks. Not per-node removed-degree matching. One fixed
    permutation and hash-priority subset per stratum; insufficient capacity stops.
    Requires explicit prospective adoption and relabeled falsifier scope, not a
    weak stand-in silently substituted for A. No coarse bins or matching tolerance.
    """
    if not resolved_alternative_b_receipt:
        raise ValueError("Explicit distinct degree-distribution null adoption required")
    _permutation_boundary(roles, permuted_fit_labels)
    _original_boundary(universe, original, roles)
    if not seed:
        raise ValueError("Fixed once-permuted mask priority seed required")
    degree = universe.native_degrees_with_one_loop
    key = lambda e: tuple(sorted((degree[e[0]], degree[e[1]])))
    eligible, selected = _eligible(universe, permuted_fit_labels), {}
    for view in VIEWS:
        needed = Counter(key(e) for e in getattr(original, view))
        pools = defaultdict(list)
        for edge in eligible[view]:
            pools[key(edge)].append(edge)
        result = []
        for stratum, count in sorted(needed.items()):
            if len(pools[stratum]) < count:
                raise MaskInfeasible("Exact native-degree-pair stratum capacity fails; no retry")
            result.extend(sorted(pools[stratum], key=lambda e: _priority(seed, view, e))[:count])
        selected[view] = tuple(sorted(result))
    return MaskPair(**selected, rule_receipt=resolved_alternative_b_receipt, seed_receipt=seed)
