"""Prospective explicit successor rules from parent; never read outcomes/data.

Fit membership is disjoint. Terminal fallback support is not silently relaxed.
Control CE guards overlap by design; identical memberships are deduplicated.
"""
from dataclasses import dataclass
from collections import defaultdict
from typing import Mapping
from core.groups import FixedPlan, LabelRoles

RULE_RECEIPT = "parent_explicit_successor_rules_20261003"
ALLOWED_KEYS = {"same_fraction_ge_half", "same_fraction_lt_half", "no_fit_labeled_neighbors"}


class UnsupportedRole(ValueError):
    pass


@dataclass(frozen=True)
class GroupBuild:
    plan: FixedPlan
    raw_memberships: tuple[tuple[tuple[int, str], tuple[int, ...]], ...]
    membership_aliases: tuple[tuple[str, str], ...]
    coverage: tuple[tuple[str, float], ...]
    fallback_lineage: tuple[tuple[str, tuple[str, ...]], ...]


def _raw(roles: LabelRoles, role: str, keys: Mapping[int, tuple[int, str]]):
    if set(keys) != set(roles.targets(role)):
        raise ValueError("Raw keys must cover exactly the declared TRAIN role")
    labels, groups = dict(roles.train_labels), defaultdict(list)
    for n in roles.targets(role):
        y, kind = keys[n]
        if y != labels[n] or kind not in ALLOWED_KEYS:
            raise ValueError("Raw key violates target label or neighborhood semantics")
        groups[(y, kind)].append(n)
    return tuple((key, tuple(ids)) for key, ids in sorted(groups.items()))


def _name(key):
    return f"class{key[0]}:{key[1]}"


def build_fit_plan(roles: LabelRoles, keys: Mapping[int, tuple[int, str]]) -> GroupBuild:
    if len(roles.fit) < 32:
        raise UnsupportedRole("Complete fit role is below 32; stop")
    raw = _raw(roles, "fit", keys)
    cells, kinds, lineage = [], [], []
    rare = defaultdict(list)
    for key, ids in raw:
        if len(ids) >= 32:
            name = _name(key)
            cells.append((name, ids))
            kinds.append((name, "no_neighbors" if key[1] == "no_fit_labeled_neighbors" else "fine"))
        else:
            rare[key[0]].append((key, ids))
    remaining, remaining_lineage = [], []
    role_order = {n: i for i, n in enumerate(roles.fit)}
    for y, records in sorted(rare.items()):
        union = tuple(sorted((n for _, ids in records for n in ids), key=role_order.__getitem__))
        source_names = tuple(_name(key) for key, _ in records)
        if len(union) >= 32:
            name = f"class{y}:rare_only_fallback"
            cells.append((name, union)); kinds.append((name, "class")); lineage.append((name, source_names))
        else:
            remaining.extend(union); remaining_lineage.extend(source_names)
    if remaining:
        if len(remaining) < 32:
            raise UnsupportedRole("Terminal rare-only global fallback is below 32; stop without dropping targets")
        name = "global:remaining_rare_only"
        cells.append((name, tuple(sorted(remaining, key=role_order.__getitem__))))
        kinds.append((name, "global")); lineage.append((name, tuple(remaining_lineage)))
    resolution = (("fallback_membership", "merge_only_rare_raw_groups_class_then_global"),
                  ("no_neighbor_fallback", "retain_supported_else_merge_with_rare_only"),
                  ("undersized_global", "stop_if_global_unsupported"),
                  ("overlap_weighting", "disjoint_partition"),
                  ("control_redundant_cells", "deduplicate_memberships"))
    plan = FixedPlan("fit", roles.fit, tuple(cells), tuple(kinds), resolution, RULE_RECEIPT)
    plan.validate_labels(roles, keys)
    fallback_ids = {n for name, ids in cells for n in ids if name.endswith("fallback") or name.startswith("global:")}
    return GroupBuild(plan, raw, (), (("all_fit_targets", 1.0),
                                     ("fallback_targets", len(fallback_ids) / len(roles.fit))), tuple(lineage))


def build_control_plan(roles: LabelRoles, keys: Mapping[int, tuple[int, str]]) -> GroupBuild:
    if len(roles.control) < 16:
        raise UnsupportedRole("Complete global control role is below 16; stop")
    raw = _raw(roles, "control", keys)
    labels, by_class = dict(roles.train_labels), defaultdict(list)
    for n in roles.control:
        by_class[labels[n]].append(n)
    proposals = [("global:complete_control", roles.control, "global")]
    class_covered, fine_covered = set(), set()
    for y, ids in sorted(by_class.items()):
        if len(ids) >= 16:
            proposals.append((f"class{y}:complete_control", tuple(ids), "class"))
            class_covered.update(ids)
    for key, ids in raw:
        if len(ids) >= 16:
            proposals.append((_name(key), ids, "no_neighbors" if key[1] == "no_fit_labeled_neighbors" else "fine"))
            fine_covered.update(ids)
    seen, cells, kinds, aliases = {}, [], [], []
    for name, ids, kind in proposals:
        identity = frozenset(ids)
        if identity in seen:
            aliases.append((name, seen[identity]))
        else:
            seen[identity] = name; cells.append((name, ids)); kinds.append((name, kind))
    resolution = (("fallback_membership", "global_plus_all_supported_complete_class_and_raw_cells"),
                  ("no_neighbor_fallback", "supported_separate_cell_else_supported_class_or_global"),
                  ("undersized_global", "stop_if_global_unsupported"),
                  ("overlap_weighting", "equal_cells_with_declared_overlap"),
                  ("control_redundant_cells", "deduplicate_memberships"))
    plan = FixedPlan("control", roles.control, tuple(cells), tuple(kinds), resolution, RULE_RECEIPT)
    plan.validate_labels(roles, keys)
    total = len(roles.control)
    coverage = (("global_targets", 1.0), ("supported_class_targets", len(class_covered) / total),
                ("supported_raw_targets", len(fine_covered) / total),
                ("rare_raw_with_class_fallback", len(class_covered - fine_covered) / total),
                ("global_only_targets", len(set(roles.control) - class_covered) / total))
    return GroupBuild(plan, raw, tuple(aliases), coverage, ())
