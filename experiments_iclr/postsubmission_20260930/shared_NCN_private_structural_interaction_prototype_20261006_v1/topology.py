"""Exact CPU enumeration on the CURRENT target-masked support; no numerical imports.

No restored targets, edge cap, sampling, skip, or topology cache. This prototype
trades CPU work/transfer for transparency; full costs must be measured.
"""
from itertools import combinations
from math import log1p, log2
from time import monotonic


def enumerate_queries(neighbors, queries, kind):
    """Return unordered witnesses, symmetric gate features, and count features.

The blind branch accesses neighbor sets to form C, then enumerates ALL C pairs;
it never accesses an internal adjacency or induced degree. Each support is
symmetric, unweighted, and loopless; queried edges must be absent.
"""
    if kind not in ("structural", "blind", "count"):
        raise ValueError("Explicit witness kind required")
    qids, left, right, gates, stats = [], [], [], [], []
    work = {"topology_queries": len(queries), "CN_node_instances": 0,
            "CN_intersection_probe_upper_bound": 0,
            "internal_neighbor_entries_inspected": 0,
            "internal_edge_instances_enumerated": 0,
            "blind_pair_instances_enumerated": 0,
            "count_query_rows": 0}
    n = len(neighbors)
    for q, (u, v) in enumerate(queries):
        if not 0 <= u < n or not 0 <= v < n:
            raise ValueError("Query outside current support")
        if v in neighbors[u]:
            raise ValueError("Queried target edge is present in current support")
        common = sorted(neighbors[u] & neighbors[v])
        common_set = set(common)
        cn = len(common)
        work["CN_node_instances"] += cn
        work["CN_intersection_probe_upper_bound"] += min(len(neighbors[u]), len(neighbors[v]))
        if kind == "blind":
            for w, z in combinations(common, 2):
                qids.append(q); left.append(w); right.append(z)
                gates.append((1., log1p(cn), 0., 0.))
                work["blind_pair_instances_enumerated"] += 1
            continue
        edges, degrees = [], {w: 0 for w in common}
        for w in common:
            work["internal_neighbor_entries_inspected"] += len(neighbors[w])
            for z in sorted(neighbors[w]):
                if z > w and z in common_set:
                    edges.append((w, z)); degrees[w] += 1; degrees[z] += 1
        lcl = len(edges)
        work["internal_edge_instances_enumerated"] += lcl
        if kind == "count":
            # deg_A is from CURRENT support. Each common node has deg_A>=2.
            cra = sum(degrees[w]/len(neighbors[w]) for w in common)
            caa = sum(degrees[w]/log2(len(neighbors[w])) for w in common)
            stats.append((float(cn), float(lcl), float(cn*lcl), cra, caa))
            qids.append(q); gates.append((1., log1p(cn), log1p(lcl), 0.))
            work["count_query_rows"] += 1
        else:
            for w, z in edges:
                qids.append(q); left.append(w); right.append(z)
                gates.append((1., log1p(cn), log1p(degrees[w])+log1p(degrees[z]),
                              float(abs(degrees[w]-degrees[z]))))
    return {"query_ids": qids, "left": left, "right": right, "gates": gates,
            "statistics": stats, "queries": queries, "work": work, "kind": kind}


def from_support(support, edges, nodes):
    """CPU transfer and exact support validation are paid on every forward."""
    started = monotonic()
    if tuple(support.sparse_sizes()) != (nodes, nodes):
        raise ValueError("Current support geometry differs")
    row, col, values = support.coo()
    if values is not None and not bool((values == 1).all()):
        raise ValueError("Structural prototype requires unweighted support")
    pairs = list(zip(row.detach().cpu().tolist(), col.detach().cpu().tolist()))
    neighbors = [set() for _ in range(nodes)]
    for u, v in pairs:
        if u == v or v in neighbors[u]:
            raise ValueError("Support must be coalesced and loopless")
        neighbors[u].add(v)
    if any(u not in neighbors[v] for u, v in pairs):
        raise ValueError("Support must be symmetric")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("Query tensor must have shape 2 by Q")
    queries = list(zip(*edges.detach().cpu().tolist()))
    return neighbors, queries, {"support_CPU_copies": 1, "support_directed_entries_copied": len(pairs),
                                 "query_rows_copied": len(queries),
                                 "support_transfer_validation_seconds": monotonic()-started}
