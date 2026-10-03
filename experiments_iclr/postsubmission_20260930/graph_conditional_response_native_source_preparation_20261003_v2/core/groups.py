"""Label boundaries and explicitly resolved fixed group/cell membership.

No fallback membership is inferred: the sealed prose admits incompatible choices.
Use an externally reviewed resolution manifest and explicit membership lists.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence


class ResolutionRequired(ValueError):
    pass


@dataclass(frozen=True)
class LabelRoles:
    fit: tuple[int, ...]
    control: tuple[int, ...]
    train_labels: tuple[tuple[int, int], ...]
    class_count: int
    forbidden_nodes: tuple[int, ...] = ()

    def __post_init__(self):
        fit, control, forbidden = set(self.fit), set(self.control), set(self.forbidden_nodes)
        labels = dict(self.train_labels)
        if type(self.class_count) is not int or self.class_count < 2 or len(fit) != len(self.fit) or len(control) != len(self.control):
            raise ValueError("Nonempty distinct class labels and unique role targets required")
        if not fit or not control or fit & control or (fit | control) & forbidden:
            raise ValueError("Fit/control must be nonempty disjoint TRAIN roles")
        if len(labels) != len(self.train_labels) or set(labels) != fit | control:
            raise ValueError("Labels must cover exactly fit/control TRAIN targets")
        if any(type(n) is not int for n, _ in self.train_labels):
            raise ValueError("TRAIN label keys must be integer node ids")
        if any(type(n) is not int or n < 0 for n in fit | control | forbidden):
            raise ValueError("Node ids must be nonnegative integers")
        if any(type(y) is not int or not 0 <= y < self.class_count for y in labels.values()):
            raise ValueError("Invalid TRAIN class id")

    def targets(self, role: str) -> tuple[int, ...]:
        if role == "fit":
            return self.fit
        if role == "control":
            return self.control
        raise ValueError("Only fit/control supervision is admissible here")

    def labels_for(self, role: str) -> tuple[int, ...]:
        labels = dict(self.train_labels)
        return tuple(labels[n] for n in self.targets(role))


def raw_group_keys(roles: LabelRoles, role: str,
                   qualified_neighbors: Mapping[int, Sequence[int]], *,
                   adjacency_semantics_receipt: str) -> dict[int, tuple[int, str]]:
    """Class and >=1/2 same-class fit-neighbor fraction on fixed native adjacency.

    The caller must qualify direction/duplicates/self-loops before providing this
    adjacency. Only fit labels are consulted for neighbors. Control target labels
    are TRAIN supervision. No validation/test label is accepted by this API.
    """
    if not adjacency_semantics_receipt:
        raise ResolutionRequired("Native adjacency semantics need source qualification")
    labels = dict(roles.train_labels)
    fit_labels = {n: labels[n] for n in roles.fit}
    out = {}
    for n in roles.targets(role):
        if n not in qualified_neighbors:
            raise ValueError("Adjacency must explicitly include isolated targets")
        eligible = [v for v in qualified_neighbors[n] if v in fit_labels]
        if not eligible:
            key = "no_fit_labeled_neighbors"
        else:
            same = sum(fit_labels[v] == labels[n] for v in eligible)
            key = "same_fraction_ge_half" if 2 * same >= len(eligible) else "same_fraction_lt_half"
        out[n] = (labels[n], key)
    return out


@dataclass(frozen=True)
class FixedPlan:
    role: str
    target_ids: tuple[int, ...]
    cells: tuple[tuple[str, tuple[int, ...]], ...]
    kinds: tuple[tuple[str, str], ...]
    resolution: tuple[tuple[str, str], ...]
    source_receipt: str

    def __post_init__(self):
        if self.role not in ("fit", "control") or not self.source_receipt:
            raise ResolutionRequired("Fixed TRAIN role and explicit source receipt required")
        decisions = dict(self.resolution)
        required = {"fallback_membership", "no_neighbor_fallback", "undersized_global",
                    "overlap_weighting", "control_redundant_cells"}
        if not required <= decisions.keys() or any(not decisions[k] or
                decisions[k].upper() in ("UNRESOLVED", "TBD") for k in required):
            raise ResolutionRequired("All ambiguous grouping rules need explicit decisions")
        target_set, kinds = set(self.target_ids), dict(self.kinds)
        ids = [name for name, _ in self.cells]
        if not target_set or len(target_set) != len(self.target_ids):
            raise ValueError("Unique complete role targets required")
        if not ids or len(set(ids)) != len(ids) or set(ids) != set(kinds):
            raise ValueError("Unique cells and one declared kind per cell required")
        covered = set()
        minimum = 32 if self.role == "fit" else 16
        for name, members in self.cells:
            if not members or len(set(members)) != len(members) or not set(members) <= target_set:
                raise ValueError("Cell memberships must be nonempty unique role targets")
            if kinds[name] not in ("fine", "class", "global", "no_neighbors"):
                raise ValueError("Unknown cell kind")
            if kinds[name] != "global" and len(members) < minimum:
                raise ValueError("Unsupported cell: explicit class/global fallback required")
            if kinds[name] == "global" and len(members) < minimum and decisions["undersized_global"] != "allow_terminal_global":
                raise ResolutionRequired("Undersized global support is unresolved")
            covered.update(members)
        if covered != target_set:
            raise ValueError("Every target needs membership; native supervision never drops targets")
        if decisions["overlap_weighting"] == "disjoint_partition":
            total = sum(len(members) for _, members in self.cells)
            if total != len(target_set):
                raise ValueError("Disjoint resolution conflicts with overlapping membership")
        elif decisions["overlap_weighting"] != "equal_cells_with_declared_overlap":
            raise ResolutionRequired("Declare partition or equal-cell overlap weighting")
        if self.role == "control" and not any(kinds[n] == "global" and
                set(members) == target_set for n, members in self.cells):
            raise ValueError("Competence guards require an explicit global control cell")
        if decisions["control_redundant_cells"] == "deduplicate_memberships":
            memberships = [frozenset(m) for _, m in self.cells]
            if len(set(memberships)) != len(memberships):
                raise ValueError("Duplicate memberships conflict with deduplication resolution")
        elif decisions["control_redundant_cells"] != "retain_declared_cells":
            raise ResolutionRequired("Declare how redundant control guards are treated")

    @property
    def fingerprint(self) -> str:
        payload = {"role": self.role, "targets": self.target_ids, "cells": self.cells,
                   "kinds": self.kinds, "resolution": self.resolution,
                   "source_receipt": self.source_receipt}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def validate_labels(self, roles: LabelRoles,
                        raw_keys: Mapping[int, tuple[int, str]] | None = None):
        """Check memberships against labels; no fallback or label policy is selected."""
        if self.target_ids != roles.targets(self.role):
            raise ValueError("Plan target order differs from fixed role order")
        labels, kinds = dict(roles.train_labels), dict(self.kinds)
        for name, members in self.cells:
            if kinds[name] in ("class", "fine", "no_neighbors"):
                if len({labels[n] for n in members}) != 1:
                    raise ValueError("Non-global cell mixes target classes")
            if kinds[name] in ("fine", "no_neighbors"):
                if raw_keys is None or len({raw_keys[n] for n in members}) != 1:
                    raise ValueError("Fine cell requires one declared raw group")
                raw_key = raw_keys[members[0]][1]
                if (kinds[name] == "no_neighbors") != (raw_key == "no_fit_labeled_neighbors"):
                    raise ValueError("No-neighbor cell kind mismatch")

    def indices(self, name: str) -> tuple[int, ...]:
        return dict(self.cells)[name]
