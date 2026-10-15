"""Exact fixed TRAIN quartile responsibilities; no tensor/model imports."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import random

MEMBERS = 4
PERMUTATION_SEED_OFFSET = 7030031


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ResponsibilityTable:
    ids: tuple
    labels: tuple
    metrics: tuple
    q: tuple
    weights: tuple
    permutation: tuple
    control: bool
    permutation_seed: int | None

    def record(self):
        encode = lambda rows: [[[value.numerator, value.denominator] for value in row] for row in rows]
        return {"ids": list(self.ids), "labels": list(self.labels), "metrics": list(self.metrics),
            "q_fraction_pairs": encode(self.q), "weight_fraction_pairs": encode(self.weights),
            "destination_to_source_position": list(self.permutation), "whole_vector_permutation": self.control,
            "permutation_seed": self.permutation_seed, "role": "TRAIN", "members": MEMBERS}

    def digest(self):
        return canonical_hash(self.record())

    def summary(self):
        classes = []
        for label in sorted(set(self.labels)):
            positions = [index for index, value in enumerate(self.labels) if value == label]
            classes.append({"label": label, "records": len(positions),
                "q_mass_per_member": [[sum((self.q[m][i] for i in positions), Fraction()).numerator,
                                        sum((self.q[m][i] for i in positions), Fraction()).denominator] for m in range(MEMBERS)],
                "weight_mass_per_member": [str(sum((self.weights[m][i] for i in positions), Fraction())) for m in range(MEMBERS)],
                "metric_tie_groups": len(set(self.metrics[i] for i in positions)),
                "unique_weight_vectors": len({tuple(self.weights[m][i] for m in range(MEMBERS)) for i in positions})})
        return {"table_sha256": self.digest(), "records": len(self.ids), "classes": classes,
            "whole_vector_permutation": self.control, "permutation_seed": self.permutation_seed,
            "moved_positions": sum(i != old for i, old in enumerate(self.permutation)),
            "weight_range": [.5, 2.5], "exact_record_mean_weight": 1,
            "exact_member_class_weight_mass": "class record count", "role": "TRAIN"}

    def tensor_weights(self, torch, *, device="cpu"):
        return torch.tensor([[float(value) for value in row] for row in self.weights], dtype=torch.float64, device=device)


def make_table(ids, labels, metrics, *, role, permutation_seed=None):
    """Fractional mass for tied rank intervals; source object order is retained.

    A tied group occupying ranks[a,a+t) receives the same q for every member of
    that group. q_m is its interval overlap with quartile[m*N/4,(m+1)*N/4),
    divided by t. No object-ID tie breaking, target losses or validation data.
    """
    ids, labels, metrics = tuple(ids), tuple(labels), tuple(metrics)
    count = len(ids)
    if role != "TRAIN" or not count or len(labels) != count or len(metrics) != count or len(set(ids)) != count:
        raise ValueError("Unique complete observed TRAIN IDs/classes/metrics in source order required")
    if any(type(value) is not int or value < 0 for value in ids+labels+metrics):
        raise ValueError("Observed class IDs, object IDs and structural counts must be nonnegative integers")
    q = [[Fraction() for _ in ids] for _ in range(MEMBERS)]
    for label in sorted(set(labels)):
        positions = sorted((i for i, value in enumerate(labels) if value == label), key=lambda i: metrics[i])
        n, begin = len(positions), 0
        while begin < n:
            end = begin+1
            while end < n and metrics[positions[end]] == metrics[positions[begin]]:
                end += 1
            for member in range(MEMBERS):
                overlap4 = max(0, min(4*end, (member+1)*n)-max(4*begin, member*n))
                share = Fraction(overlap4, 4*(end-begin))
                for position in positions[begin:end]:
                    q[member][position] = share
            begin = end
    weights = [[Fraction(1, 2)+2*value for value in row] for row in q]
    permutation = list(range(count))
    if permutation_seed is not None:
        if type(permutation_seed) is not int or permutation_seed < 0:
            raise ValueError("An explicit nonnegative isolated control seed is required")
        generator = random.Random(permutation_seed+PERMUTATION_SEED_OFFSET)
        for label in sorted(set(labels)):
            destinations = [i for i, value in enumerate(labels) if value == label]
            sources = list(destinations); generator.shuffle(sources)
            for destination, source in zip(destinations, sources):
                permutation[destination] = source
        q = [[row[source] for source in permutation] for row in q]
        weights = [[row[source] for source in permutation] for row in weights]
    result = ResponsibilityTable(ids, labels, metrics, tuple(map(tuple, q)), tuple(map(tuple, weights)),
                                 tuple(permutation), permutation_seed is not None, permutation_seed)
    validate_table(result)
    return result


def validate_table(table):
    n = len(table.ids)
    for i in range(n):
        if sum(table.q[m][i] for m in range(MEMBERS)) != 1 \
                or sum(table.weights[m][i] for m in range(MEMBERS)) != MEMBERS:
            raise ValueError("Every TRAIN object requires exact q mass1 and member mean weight1")
    for label in set(table.labels):
        positions = [i for i, value in enumerate(table.labels) if value == label]
        for member in range(MEMBERS):
            if sum(table.q[member][i] for i in positions) != Fraction(len(positions), MEMBERS) \
                    or sum(table.weights[member][i] for i in positions) != len(positions):
                raise ValueError("Exact member/class masses violated")
    if any(not Fraction(1, 2) <= value <= Fraction(5, 2) for row in table.weights for value in row):
        raise ValueError("All native supervision must retain positive declared weight")


def native_incoming_counts(num_nodes, edge_sources, edge_targets, train_ids):
    """Count native graph edge records at target IDs; no topology rewrite.

    Pass the actual post-source-preprocessing edge_index rows. Duplicate edge
    records retain multiplicity; present self loops count once. No additional
    self loop, coalescing, symmetrization or propagation-normalization edge is
    inserted here. Public full-graph topology may contribute to TRAIN degree;
    only TRAIN labels/assignments belong to the table.
    """
    if type(num_nodes) is not int or num_nodes < 1 or len(edge_sources) != len(edge_targets):
        raise ValueError("Declared native graph and aligned directed edge records required")
    degree = [0]*num_nodes
    for source, target in zip(edge_sources, edge_targets):
        if type(source) is not int or type(target) is not int or not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError("Native edge record out of range")
        degree[target] += 1
    if len(set(train_ids)) != len(train_ids) or any(type(i) is not int or not 0 <= i < num_nodes for i in train_ids):
        raise ValueError("Unique authentic TRAIN IDs required")
    return tuple(degree[i] for i in train_ids)
