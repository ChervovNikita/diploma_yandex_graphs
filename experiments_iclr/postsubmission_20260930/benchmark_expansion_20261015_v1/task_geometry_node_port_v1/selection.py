"""Fixed bounded TRAIN auxiliary cohort; no model or numerical imports."""
import hashlib
import json

BUDGET = 512


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select(ids, labels, *, seed, role):
    ids, labels = tuple(ids), tuple(labels)
    if role != "TRAIN" or not ids or len(ids) != len(labels) or len(set(ids)) != len(ids) \
            or any(type(v) is not int or v < 0 for v in ids+labels) or type(seed) is not int or seed < 0:
        raise ValueError("Unique authentic observed categorical TRAIN IDs/classes and fit seed required")
    classes = sorted(set(labels))
    queues = {c: sorted([i for i,v in enumerate(labels) if v == c],
        key=lambda i: (hashlib.sha256(f"c1-node-class-geometry-v1|{seed}|{c}|{ids[i]}".encode()).digest(), ids[i])) for c in classes}
    if sum(min(2, len(queues[c])) for c in classes) > BUDGET:
        raise ValueError("Fixed512 budget cannot retain all classes and each available distinct-node variance pair")
    chosen, level = set(), 0
    while len(chosen) < min(BUDGET, len(ids)):
        for c in classes:
            if level < len(queues[c]) and len(chosen) < min(BUDGET, len(ids)):
                chosen.add(queues[c][level])
        level += 1
    positions = [i for i in range(len(ids)) if i in chosen]
    record = {"role": "TRAIN", "fit_seed": seed, "fixed_budget": BUDGET,
        "full_TRAIN_nodes": len(ids), "selected_nodes": len(positions),
        "positions": positions, "selected_ids": [ids[i] for i in positions],
        "selected_classes": [labels[i] for i in positions],
        "full_ids_classes_sha256": digest({"ids": ids, "labels": labels}),
        "class_counts": [{"label": c, "full": len(queues[c]),
            "selected": sum(labels[i] == c for i in positions), "full_singleton": len(queues[c]) == 1} for c in classes],
        "missing_labels": 0, "one_class": len(classes) == 1,
        "selection": "seed/class/ID SHA256 order within each class, round-robin across sorted classes, retain native source order",
        "epoch_change": False, "loss_or_VALID_dependent": False,
        "native_CE_population_changed": False, "unbiased_full_auxiliary_estimator_claim": False}
    return positions, record, digest(record)
