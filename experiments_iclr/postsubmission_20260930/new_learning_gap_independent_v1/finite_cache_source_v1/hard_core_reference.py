"""Source-only exact reference for one complete supplied conflict graph.

No graph generator, RNG, executable example, file I/O or default resource limit.
Nothing in this module has been numerically executed as part of this proposal.
Rational fugacities make enumeration/conditioning exact before log conversion.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, log
from typing import Iterable, Sequence


Locks = tuple[tuple[int, int], ...]


def canonical_locks(n_nodes: int, locks: Iterable[tuple[int, int]]) -> Locks:
    result: dict[int, int] = {}
    for node, label in locks:
        if type(node) is not int or not 0 <= node < n_nodes:
            raise ValueError("Lock node is outside the complete graph")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Binary task label required")
        if node in result and result[node] != label:
            raise ValueError("Contradictory locks")
        result[node] = label
    return tuple(sorted(result.items()))


def merge_locks(n_nodes: int, left: Locks, right: Locks) -> Locks:
    return canonical_locks(n_nodes, (*left, *right))


@dataclass(frozen=True)
class EnumerationBudget:
    """Required future limits inherited from the admitted resource plan."""

    max_states: int
    max_rational_bits: int

    def __post_init__(self) -> None:
        if type(self.max_states) is not int or self.max_states <= 0:
            raise ValueError("A positive explicitly admitted state cap is required")
        if type(self.max_rational_bits) is not int or self.max_rational_bits <= 0:
            raise ValueError("An explicitly admitted rational-size cap is required")

    def check_fraction(self, value: Fraction) -> None:
        size = max(value.numerator.bit_length(), value.denominator.bit_length())
        if size > self.max_rational_bits:
            raise ValueError("Exact arithmetic exceeds the declared rational-size cap")


@dataclass(frozen=True)
class CompleteConflictGraph:
    """The entire instance, including isolated nodes; no clique assumption."""

    n_nodes: int
    edges: tuple[tuple[int, int], ...]
    fugacities: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if type(self.n_nodes) is not int or self.n_nodes < 0:
            raise ValueError("Explicit nonnegative node count required")
        if len(self.fugacities) != self.n_nodes:
            raise ValueError("One supplied fugacity per node required")
        if any(not isinstance(x, Fraction) or x <= 0 for x in self.fugacities):
            raise ValueError("Positive rational fugacities required for exact reference")
        object.__setattr__(self, "fugacities", tuple(self.fugacities))
        normalized = []
        for u, v in self.edges:
            if type(u) is not int or type(v) is not int:
                raise ValueError("Integer endpoint indices required")
            if not (0 <= u < self.n_nodes and 0 <= v < self.n_nodes) or u == v:
                raise ValueError("Simple undirected conflict edges required")
            normalized.append((min(u, v), max(u, v)))
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate undirected edges must be removed upstream")
        object.__setattr__(self, "edges", tuple(sorted(normalized)))

    def feasible(self, assignment: Sequence[int]) -> bool:
        if len(assignment) != self.n_nodes:
            raise ValueError("A complete graph assignment is required")
        if any(type(x) is not int or x not in (0, 1) for x in assignment):
            raise ValueError("Binary complete assignment required")
        return all(not (assignment[u] and assignment[v]) for u, v in self.edges)


def _matches(mask: int, locks: Locks) -> bool:
    return all(((mask >> node) & 1) == label for node, label in locks)


def _assignment(mask: int, n_nodes: int) -> tuple[int, ...]:
    return tuple((mask >> node) & 1 for node in range(n_nodes))


@dataclass(frozen=True)
class ExactHardCoreLaw:
    graph: CompleteConflictGraph
    budget: EnumerationBudget
    weighted_states: tuple[tuple[int, Fraction], ...]
    partition: Fraction
    context: Locks
    states_considered: int

    @classmethod
    def enumerate(cls, graph: CompleteConflictGraph, budget: EnumerationBudget):
        states_considered = 1 << graph.n_nodes
        if states_considered > budget.max_states:
            raise ValueError("Complete-graph enumeration exceeds the existing state cap")
        for value in graph.fugacities:
            budget.check_fraction(value)
        weighted = []
        partition = Fraction(0)
        for mask in range(states_considered):
            if any((mask & (1 << u)) and (mask & (1 << v)) for u, v in graph.edges):
                continue
            weight = Fraction(1)
            for node, fugacity in enumerate(graph.fugacities):
                if mask & (1 << node):
                    weight *= fugacity
                    budget.check_fraction(weight)
            partition += weight
            budget.check_fraction(partition)
            weighted.append((mask, weight))
        return cls(graph, budget, tuple(weighted), partition, (), states_considered)

    def condition(self, locks: Iterable[tuple[int, int]]):
        context = merge_locks(
            self.graph.n_nodes, self.context, canonical_locks(self.graph.n_nodes, locks)
        )
        weighted = tuple((mask, weight) for mask, weight in self.weighted_states
                         if _matches(mask, context))
        partition = sum((weight for _, weight in weighted), Fraction(0))
        self.budget.check_fraction(partition)
        if partition == 0:
            raise ValueError("The supplied context is incompatible with the hard-core law")
        return ExactHardCoreLaw(
            self.graph, self.budget, weighted, partition, context, self.states_considered
        )

    def event_probability(self, locks: Iterable[tuple[int, int]]) -> Fraction:
        event = canonical_locks(self.graph.n_nodes, locks)
        context = dict(self.context)
        if any(node in context and context[node] != label for node, label in event):
            return Fraction(0)
        weight = sum((weight for mask, weight in self.weighted_states
                      if _matches(mask, event)), Fraction(0))
        self.budget.check_fraction(weight)
        probability = weight / self.partition
        self.budget.check_fraction(probability)
        return probability

    def event_log_probability(self, locks: Iterable[tuple[int, int]]) -> float:
        probability = self.event_probability(locks)
        if probability == 0:
            return float("-inf")
        return log(probability.numerator) - log(probability.denominator)

    def complete_probability(self, assignment: Sequence[int]) -> Fraction:
        if len(assignment) != self.graph.n_nodes:
            raise ValueError("All nodes of the instance must be supplied")
        return self.event_probability(enumerate(assignment))

    def marginals(self) -> tuple[Fraction, ...]:
        return tuple(self.event_probability(((node, 1),))
                     for node in range(self.graph.n_nodes))

    def integer_ticket_intervals(self):
        """Future exact sampler table; no randomness is produced here.

        A caller could provide one unbiased integer in [0,total). Integer size,
        table construction, RNG generation and reuse all need resource accounting.
        """
        probabilities = tuple((mask, weight / self.partition)
                              for mask, weight in self.weighted_states)
        for _, probability in probabilities:
            self.budget.check_fraction(probability)
        total = 1
        for _, probability in probabilities:
            total = total // gcd(total, probability.denominator) * probability.denominator
            if total.bit_length() > self.budget.max_rational_bits:
                raise ValueError("Exact ticket denominator exceeds the declared cap")
        cursor = 0
        intervals = []
        for mask, probability in probabilities:
            width = probability.numerator * (total // probability.denominator)
            intervals.append((mask, cursor, cursor + width))
            cursor += width
        if cursor != total:
            raise ArithmeticError("Exact ticket table failed normalization")
        return total, tuple(intervals)

    def assignment_for_ticket(self, ticket: int) -> tuple[int, ...]:
        """Deterministic future mapping from a caller-supplied integer ticket."""
        total, intervals = self.integer_ticket_intervals()
        if type(ticket) is not int or not 0 <= ticket < total:
            raise ValueError("Ticket outside the exact reference distribution")
        for mask, lower, upper in intervals:
            if lower <= ticket < upper:
                return _assignment(mask, self.graph.n_nodes)
        raise ArithmeticError("Ticket table failed to cover the ticket")


def conditioned_residual_nodes(graph: CompleteConflictGraph, locks: Locks):
    """Standard hard-core residual graph rule; preserves original node IDs.

    Active locks force neighbors inactive. Returns remaining free IDs and forced
    inactive IDs; it does not generate a graph instance or a schedule.
    """
    context = dict(canonical_locks(graph.n_nodes, locks))
    forced_inactive = set()
    for u, v in graph.edges:
        if context.get(u) == 1:
            forced_inactive.add(v)
        if context.get(v) == 1:
            forced_inactive.add(u)
    if any(context.get(node) == 1 for node in forced_inactive):
        raise ValueError("Active locks conflict")
    free = tuple(node for node in range(graph.n_nodes)
                 if node not in context and node not in forced_inactive)
    return free, tuple(sorted(forced_inactive))
