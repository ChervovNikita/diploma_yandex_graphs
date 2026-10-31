"""Source-only complete graph, fixed order and fixed mixture contracts.

No graph/label/RNG generation, model import or numerical execution occurred.
"""
from dataclasses import dataclass, field
from math import isfinite, log
from typing import Sequence


@dataclass(frozen=True)
class ScheduleDomain:
    """Entire simple undirected conflict graph and theta-independent node order."""

    n_nodes: int
    edges: tuple[tuple[int, int], ...]
    order: tuple[int, ...]
    graph_id: str
    order_id: str
    neighbors: tuple[tuple[int, ...], ...] = field(init=False)

    def __post_init__(self):
        if type(self.n_nodes) is not int or self.n_nodes < 0:
            raise ValueError("Explicit nonnegative full-instance node count required")
        order = tuple(self.order)
        if any(type(node) is not int for node in order) or sorted(order) != list(range(self.n_nodes)):
            raise ValueError("Caller must supply a complete node permutation")
        if not self.graph_id or not self.order_id:
            raise ValueError("Recorded graph and order identities required")
        edges = []
        neighbors = [set() for _ in range(self.n_nodes)]
        for u, v in self.edges:
            if type(u) is not int or type(v) is not int or not (0 <= u < self.n_nodes and 0 <= v < self.n_nodes) or u == v:
                raise ValueError("Simple integer-indexed conflict edges required")
            edges.append((min(u, v), max(u, v)))
            neighbors[u].add(v)
            neighbors[v].add(u)
        if len(set(edges)) != len(edges):
            raise ValueError("Duplicate undirected edges must be resolved upstream")
        object.__setattr__(self, "edges", tuple(sorted(edges)))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "neighbors", tuple(tuple(sorted(row)) for row in neighbors))

    def assignment(self, supplied: Sequence[int]) -> tuple[int, ...]:
        result = tuple(supplied)
        if len(result) != self.n_nodes or any(type(label) is not int or label not in (0, 1) for label in result):
            raise ValueError("Complete binary assignment in original node IDs required")
        return result

    def feasible(self, supplied: Sequence[int]) -> bool:
        assignment = self.assignment(supplied)
        return all(not (assignment[u] and assignment[v]) for u, v in self.edges)

    def blocked(self, node: int, prefix: Sequence[int | None]) -> bool:
        """Active emitted neighbors force zero; future nodes are still None."""
        return any(prefix[neighbor] == 1 for neighbor in self.neighbors[node])


@dataclass(frozen=True)
class FixedMixture:
    """Positive externally supplied masses, normalized into fixed pi.

    These are plain immutable numbers, never trainable tensors. No uniform
    weights or branch count are chosen by this source packet.
    """

    masses: tuple[float, ...]

    def __post_init__(self):
        masses = tuple(float(value) for value in self.masses)
        total = sum(masses)
        if not masses or any(not isfinite(value) or value <= 0 for value in masses) or not isfinite(total):
            raise ValueError("Finite positive fixed mixture masses required")
        if any(value / total == 0 for value in masses):
            raise ArithmeticError("Positive mixture weight lost floating-point representability")
        object.__setattr__(self, "masses", masses)

    @property
    def probabilities(self) -> tuple[float, ...]:
        total = sum(self.masses)
        return tuple(value / total for value in self.masses)

    @property
    def log_probabilities(self) -> tuple[float, ...]:
        log_total = log(sum(self.masses))
        return tuple(log(value) - log_total for value in self.masses)


@dataclass(frozen=True)
class GeneratedSchedule:
    """Future generated record; no pathwise derivative through discrete labels."""

    assignment: tuple[int, ...]
    generating_branch: int
    graph_id: str
    order_id: str


@dataclass(frozen=True)
class IndependentBaseline:
    """External certification; source cannot prove statistical independence.

    One graph/branch-specific value per branch, fixed before this sample batch.
    Current-batch means (including the scored sample) are forbidden. The root
    must audit the provenance before admitting an estimator call.
    """

    values_by_branch: tuple[float, ...]
    basis: str
    provenance_id: str
    certified_excludes_current_batch: bool

    def __post_init__(self):
        values = tuple(float(value) for value in self.values_by_branch)
        if not values or any(not isfinite(value) for value in values):
            raise ValueError("Finite externally supplied baseline values required")
        if self.basis not in ("predeclared_constant", "previous_independent_batch"):
            raise ValueError("Only a valid current-sample-independent baseline is allowed")
        if self.certified_excludes_current_batch is not True or not self.provenance_id:
            raise ValueError("Baseline independence/provenance certification required")
        object.__setattr__(self, "values_by_branch", values)
