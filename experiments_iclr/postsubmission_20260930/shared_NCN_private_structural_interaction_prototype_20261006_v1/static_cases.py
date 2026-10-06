"""Small stdlib-only topology/algebra certificates. No dataset/model execution."""
from math import isclose
import topology


def graph(matching):
    neighbors = [set() for _ in range(6)]
    for u, v in [(a, b) for a in (0, 1) for b in range(2, 6)] + matching:
        neighbors[u].add(v); neighbors[v].add(u)
    return neighbors


def run_cases():
    a, b = graph([(2, 3), (4, 5)]), graph([(2, 4), (3, 5)])
    qa = topology.enumerate_queries(a, [(0, 1), (1, 0)], "structural")
    qb = topology.enumerate_queries(b, [(0, 1), (1, 0)], "structural")
    assert list(zip(qa["left"][:2], qa["right"][:2])) == [(2, 3), (4, 5)]
    assert list(zip(qb["left"][:2], qb["right"][:2])) == [(2, 4), (3, 5)]
    assert qa["gates"][:2] == qa["gates"][2:] == qb["gates"][:2]
    ca = topology.enumerate_queries(a, [(0, 1)], "count")["statistics"][0]
    cb = topology.enumerate_queries(b, [(0, 1)], "count")["statistics"][0]
    assert all(isclose(x, y) for x, y in zip(ca, cb)) and ca[:3] == (4., 2., 8.)
    ba = topology.enumerate_queries(a, [(0, 1)], "blind")
    bb = topology.enumerate_queries(b, [(0, 1)], "blind")
    assert [(ba[key], bb[key]) for key in ("left", "right", "gates")] == [(ba[key], ba[key]) for key in ("left", "right", "gates")]
    assert len(ba["query_ids"]) == 6 and ba["work"]["internal_neighbor_entries_inspected"] == 0
    h = {2: 1, 3: 1, 4: -1, 5: -1}
    message_a = sum(h[w]*h[z] for w, z in zip(qa["left"][:2], qa["right"][:2]))
    message_b = sum(h[w]*h[z] for w, z in zip(qb["left"][:2], qb["right"][:2]))
    assert (message_a, message_b) == (2, -2)
    # Product has a nonzero mixed difference; it is not f(h_w)+g(h_z).
    assert 1*1+(-1)*(-1)-1*(-1)-(-1)*1 == 4
    empty = topology.enumerate_queries([set(), set()], [(0, 1)], "structural")
    assert empty["query_ids"] == []
    bad = graph([(2, 3)]); bad[0].add(1); bad[1].add(0)
    try: topology.enumerate_queries(bad, [(0, 1)], "structural")
    except ValueError: pass
    else: raise AssertionError("Present target was not rejected")
    return {"passed": True, "cases": ["exact_internal_edges", "endpoint_swap_symmetry", "count_collision",
        "blind_adjacency_invariance", "fixed_H_nonseparable_product_distinction", "empty_CN", "present_target_rejection"],
        "count_statistics": ca, "edge_product_sums": [message_a, message_b],
        "scope": "conditional_fixed_H_interface_certificate; not full_GNN_expressiveness_or_novelty"}


if __name__ == "__main__":
    import json
    print(json.dumps(run_cases(), indent=2))
