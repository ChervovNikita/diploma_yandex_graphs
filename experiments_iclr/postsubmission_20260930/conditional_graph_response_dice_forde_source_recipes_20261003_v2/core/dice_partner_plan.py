"""Prepared, unexecuted same-class partner planner; standard library only.

Requires already authorized TRAIN labels from caller. No dataset loading.
Plans are made at base state before predictor updates and sealed as ephemeral
inputs. Partner IDs are reused across replicas, while feature noise is not.
"""
from collections import defaultdict
from hashlib import sha256
import random


def _keyed_seed(seed, cycle, auxiliary_step):
    material = f"dice_partners_v2:{seed}:{cycle}:{auxiliary_step}".encode()
    return int.from_bytes(sha256(material).digest(), "big")


def draw_partner_plans(train_labels_by_id, target_ids, *, seed, cycle, members=4,
                       auxiliary_steps=4, negatives=2):
    """Return immutable tuples (aux,i,j,anchor,negative_index,partner_id)."""
    if (type(seed) is not int or type(cycle) is not int or cycle < 0
            or type(members) is not int or members < 2
            or auxiliary_steps != 4 or negatives != 2):
        raise ValueError("Explicit cycle/seed and fixed source-style settings required")
    targets = tuple(target_ids)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Distinct target IDs required")
    classes = defaultdict(list)
    for node, label in sorted(train_labels_by_id.items()):
        classes[label].append(node)
    if any(len(ids) < 2 for ids in classes.values()):
        raise ValueError("Every represented TRAIN class requires two distinct IDs")
    if any(node not in train_labels_by_id for node in targets):
        raise ValueError("Target outside authorized TRAIN population")
    allowed = {node: tuple(other for other in classes[train_labels_by_id[node]] if other != node)
               for node in targets}
    plan = []
    for auxiliary in range(auxiliary_steps):
        rng = random.Random(_keyed_seed(seed, cycle, auxiliary))
        for first in range(members):
            for second in range(first + 1, members):
                for node in targets:
                    for negative in range(negatives):
                        partner = allowed[node][rng.randrange(len(allowed[node]))]
                        plan.append((auxiliary, first, second, node, negative, partner))
    return tuple(plan)

