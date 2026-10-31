"""Prospective exact-molecule groups, balanced by observed GRAPH count.

No labels, SMILES files, metrics, torch or dataset provider are accessed.
The caller supplies a frozen canonical-molecule digest for each train ID.
"""
import random


GROUP_RULE = "union of same canonical isomeric SMILES and same stereo-free Bemis-Murcko scaffold; empty scaffolds stay together; no similarity threshold"


def balance(groups, counts, seed):
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    sides, totals = [[], []], [0, 0]
    for group in order:
        side = 0 if totals[0] <= totals[1] else 1
        sides[side].append(group)
        totals[side] += counts[group]
    if not all(sides):
        raise ValueError("Need at least two distinct observed molecule groups")
    return [sorted(side) for side in sides], totals


def grouped_folds(unit_to_group):
    if not unit_to_group or any(not str(k) or not str(v) for k, v in unit_to_group.items()):
        raise ValueError("Need nonempty observed training-unit/group identities")
    members = {}
    for unit, group in sorted(unit_to_group.items()):
        members.setdefault(group, []).append(unit)
    counts = {group: len(units) for group, units in members.items()}
    (p, q), pq_counts = balance(members, counts, 700)
    if len(q) < 2:
        raise ValueError("Q has too few molecule groups; cannot make A/B folds")
    expand = lambda groups: sorted(unit for group in groups for unit in members[group])
    partitions = []
    for seed in (701, 702, 703):
        (a, b), sizes = balance(q, counts, seed)
        partitions.append({"partition_id": str(seed), "A": expand(a), "B": expand(b),
                           "A_groups": a, "B_groups": b, "observed_graph_counts": sizes})
    result = {"probe_fit_unit_ids": expand(p), "score_unit_ids": expand(q),
              "P_groups": p, "Q_groups": q, "partitions": partitions,
              "observed_graph_counts_P_Q": pq_counts,
              "unit_to_group": dict(sorted(unit_to_group.items())), "grouping": GROUP_RULE,
              "balance_rule": "shuffle sorted groups with seed; append to lower-count side; ties P/A",
              "seed_P_Q": 700, "seeds_A_B": [701, 702, 703],
              "fold_objective": "equal observed-graph means, regardless of group/node/label counts",
              "not_independent_confirmation": True}
    validate_grouped(result)
    return result


def validate_grouped(folds):
    mapping = folds["unit_to_group"]
    p, q = set(folds["probe_fit_unit_ids"]), set(folds["score_unit_ids"])
    if p & q or p | q != set(mapping) or not p or not q:
        raise ValueError("Incomplete/disjoint P/Q graph allocation")
    group_set = lambda ids: {mapping[unit] for unit in ids}
    if group_set(p) & group_set(q):
        raise ValueError("Exact molecule repeats cross P/Q")
    if len(folds["partitions"]) != 3:
        raise ValueError("Require the unchanged three scoring partitions")
    for part in folds["partitions"]:
        a, b = set(part["A"]), set(part["B"])
        if a & b or a | b != q or not a or not b or group_set(a) & group_set(b):
            raise ValueError("Invalid molecule-disjoint A/B graph allocation")
    return True
