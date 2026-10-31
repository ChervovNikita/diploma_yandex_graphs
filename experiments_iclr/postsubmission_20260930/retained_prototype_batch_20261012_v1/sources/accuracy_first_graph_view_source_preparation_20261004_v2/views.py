"""Deterministic, stdlib-only views. Inputs contain TRAIN labels only."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

DEGREE_UPPER_BOUNDS = (0, 1, 3, 7, 15, 31, 63, 127, 255)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def integer(value):
    if type(value) is not int:
        raise ValueError("Primitive integer required")
    return value


@dataclass(frozen=True)
class TrainRole:
    nodes: int
    split: int
    train_ids: tuple
    train_labels: tuple
    val_ids: tuple
    test_ids: tuple

    def __post_init__(self):
        integer(self.nodes)
        integer(self.split)
        domains = [tuple(integer(i) for i in ids) for ids in
                   (self.train_ids, self.val_ids, self.test_ids)]
        if self.nodes <= 0 or any(len(set(ids)) != len(ids) for ids in domains):
            raise ValueError("Invalid nodes or duplicate role IDs")
        if any(tuple(sorted(ids)) != ids for ids in domains):
            raise ValueError("Preserve ascending official mask row order")
        flat = sum(domains, ())
        if len(flat) != self.nodes or set(flat) != set(range(self.nodes)):
            raise ValueError("Official masks must partition all nodes")
        if len(self.train_labels) != len(self.train_ids) or not self.train_ids:
            raise ValueError("Exactly every TRAIN label required")
        if any(integer(y) not in range(5) for y in self.train_labels):
            raise ValueError("Only five-class TRAIN labels admitted")

    @classmethod
    def from_label_map(cls, nodes, split, train_ids, label_map, val_ids, test_ids):
        if set(label_map) != set(train_ids):
            raise ValueError("Label domain must equal TRAIN: VALIDATION/TEST labels forbidden")
        return cls(nodes, split, tuple(train_ids), tuple(label_map[i] for i in train_ids),
                   tuple(val_ids), tuple(test_ids))

    def identity(self):
        return {"nodes": self.nodes, "split": self.split,
                "TRAIN_ids_sha256": digest(self.train_ids),
                "TRAIN_labels_sha256": digest(self.train_labels),
                "TRAIN_count": len(self.train_ids),
                "VAL_ids_sha256": digest(self.val_ids), "VAL_count": len(self.val_ids),
                "TEST_mask_ids_sha256": digest(self.test_ids), "TEST_count": len(self.test_ids)}


def native_edges(nodes, raw_edges):
    """Stdlib equivalent edge identities/order for native undirect/remove/add loops.

    Actual execution still verifies against the admitted native preprocessing.
    """
    pairs = set()
    for a, b in raw_edges:
        integer(a)
        integer(b)
        if a not in range(nodes) or b not in range(nodes):
            raise ValueError("Endpoint outside graph")
        if a != b:
            pairs.add((a, b))
            pairs.add((b, a))
    return tuple(sorted(pairs)) + tuple((i, i) for i in range(nodes))


def validate_native(nodes, edges):
    edges = tuple(tuple(e) for e in edges)
    if len(set(edges)) != len(edges):
        raise ValueError("Duplicate native edge")
    for edge in edges:
        if len(edge) != 2 or any(integer(i) not in range(nodes) for i in edge):
            raise ValueError("Invalid native edge")
    edge_set = set(edges)
    if any((b, a) not in edge_set for a, b in edges):
        raise ValueError("Native graph must be bidirected")
    if {a for a, b in edges if a == b} != set(range(nodes)):
        raise ValueError("Exactly one native self-loop per node required")
    return edges


def degree_bin(degree):
    for index, upper in enumerate(DEGREE_UPPER_BOUNDS):
        if degree <= upper:
            return index
    return len(DEGREE_UPPER_BOUNDS)


def rank_pairs(pairs, seed, salt):
    return sorted(pairs, key=lambda p: (digest([seed, salt, p]), p))


def matched_null(pool, requested, strata, seed, salt):
    """No replacement, borrowing, reseeding or retry on undersupply.

    Keep all available pairs in an undersupplied stratum and mark the result
    ineligible for scientific freezing. The normal TRAIN-pair pool includes
    semantic deletions, so supplying each individual target is guaranteed.
    """
    buckets = {}
    for pair in pool:
        buckets.setdefault(strata[pair], []).append(pair)
    selected, records = [], []
    for key, count in sorted(requested.items()):
        candidates = rank_pairs(buckets.get(key, ()), seed, [salt, key])
        take = min(count, len(candidates))
        selected.extend(candidates[:take])
        records.append({"stratum": list(key), "requested": count,
                        "available": len(candidates), "selected": take,
                        "shortfall": count - take})
    return tuple(sorted(selected)), records


def construct_views(role, edges, seed):
    """Delete floor(category_count/10) unordered TRAIN-TRAIN pairs per category."""
    integer(seed)
    edges = validate_native(role.nodes, edges)
    pairs = tuple(sorted((a, b) for a, b in edges if a < b))
    degree = Counter(i for pair in pairs for i in pair)
    strata = {p: tuple(sorted((degree_bin(degree[p[0]]), degree_bin(degree[p[1]]))))
              for p in pairs}
    labels = dict(zip(role.train_ids, role.train_labels))
    eligible = tuple(p for p in pairs if p[0] in labels and p[1] in labels)
    categories = {"equal": tuple(p for p in eligible if labels[p[0]] == labels[p[1]]),
                  "different": tuple(p for p in eligible if labels[p[0]] != labels[p[1]])}
    deleted, counts = {}, {}
    null_records = {}
    for name, candidates in categories.items():
        chosen = tuple(sorted(rank_pairs(candidates, seed, [role.split, name])[:len(candidates) // 10]))
        deleted[name] = chosen
        requested = Counter(strata[p] for p in chosen)
        null, records = matched_null(eligible, requested, strata, seed, [role.split, "null", name])
        deleted["null_" + name] = null
        null_records[name] = records
        counts[name] = {"eligible_pairs": len(candidates), "requested_10_percent_floor": len(candidates) // 10,
                        "deleted_pairs": len(chosen), "directed_nonloop_edges_deleted": 2 * len(chosen),
                        "distinct_TRAIN_endpoints": len({i for p in chosen for i in p}),
                        "null_deleted_pairs": len(null),
                        "null_semantic_overlap_pairs": len(set(chosen).intersection(null))}
    graph_views = {}
    for name, deletions in deleted.items():
        removed = set(deletions)
        graph_views[name] = tuple((a, b) for a, b in edges
                                  if a == b or (min(a, b), max(a, b)) not in removed)
    ready = all(c["deleted_pairs"] > 0 for c in counts.values()) and not any(
        row["shortfall"] for rows in null_records.values() for row in rows)
    coverage = {"schema": "accuracy-first-TRAIN-view-coverage-v1", "role": role.identity(),
                "view_seed": seed, "native_edges_sha256": digest(edges),
                "native_directed_edge_count": len(edges), "native_nonloop_pairs": len(pairs),
                "self_loops_retained": role.nodes, "TRAIN_TRAIN_pairs": len(eligible),
                "non_TRAIN_pair_exclusion_count": len(pairs) - len(eligible),
                "deletion_fraction": {"numerator": 1, "denominator": 10, "rounding": "floor per category"},
                "degree_definition": "native undirected non-loop pair incidence; all nodes/topology; no labels",
                "degree_strata_upper_bounds": list(DEGREE_UPPER_BOUNDS) + ["overflow"],
                "pair_stratum": "sorted endpoint degree-bin indices",
                "null_pool": "all TRAIN-TRAIN non-loop pairs, ignores classes; overlap with semantic mask allowed",
                "undersupply": "take available, record shortfall, disallow scientific freeze; no stratum borrowing",
                "categories": counts, "null_strata": null_records,
                "views": {name: {"edges_sha256": digest(value), "edge_count": len(value),
                                  "deleted_pairs_sha256": digest(deleted[name])}
                          for name, value in graph_views.items()},
                "coverage_eligible": ready}
    return {"role": role, "native": edges, "views": graph_views,
            "deleted_pairs": deleted, "coverage": coverage}


def save_coverage(bundle, path, *, origin, execute=False):
    if origin not in ("synthetic_fixture", "official_TRAIN"):
        raise ValueError("Declare coverage origin")
    if origin == "official_TRAIN" and not execute:
        raise RuntimeError("Real coverage requires deliberate root-admitted data access")
    record = dict(bundle["coverage"], coverage_origin=origin,
                  scientific_freeze_eligible=origin == "official_TRAIN" and bundle["coverage"]["coverage_eligible"])
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return record


def verify_bundle(bundle):
    """Check live graph/mask custody against saved coverage, without reading labels elsewhere."""
    coverage = bundle["coverage"]
    if bundle["role"].identity() != coverage["role"] or \
            digest(bundle["native"]) != coverage["native_edges_sha256"] or \
            len(bundle["native"]) != coverage["native_directed_edge_count"] or \
            set(bundle["views"]) != {"equal", "different", "null_equal", "null_different"}:
        raise ValueError("Role/native graph/view inventory changed after coverage")
    for name, edges in bundle["views"].items():
        row = coverage["views"][name]
        if digest(edges) != row["edges_sha256"] or len(edges) != row["edge_count"] or \
                digest(bundle["deleted_pairs"][name]) != row["deleted_pairs_sha256"]:
            raise ValueError("Fixed view/mask changed after coverage")
