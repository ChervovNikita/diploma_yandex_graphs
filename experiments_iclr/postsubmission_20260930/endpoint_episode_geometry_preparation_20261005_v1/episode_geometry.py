"""Pure-stdlib proposed TRAIN episode geometry, shared by diagnostics/runners.

No model, numerical library, data load, scheduling loop or fit runs at import.
Outer/inner sizes and paired-support policy are candidates, not adopted recipes.
"""
from collections import Counter
import random


def canonical(pair):
    return tuple(sorted(pair))


def _ids(ids, population, role, unique=False):
    if any(type(i) is not int or not 0 <= i < population for i in ids):
        raise ValueError("Invalid " + role + " index")
    if unique and len(set(ids)) != len(ids):
        raise ValueError("Repeated " + role + " index")


def _streams(streams):
    if not streams or any(set(stream) != {"positive", "negative"} or
                          any(not isinstance(stream[role], random.Random) for role in stream)
                          for stream in streams):
        raise ValueError("Require nonempty member/class Python Random streams")


def neighbors(train, nodes):
    if type(nodes) is not int or nodes < 1:
        raise ValueError("Require full positive integer node population")
    graph = [set() for _ in range(nodes)]
    seen = set()
    for u, v in train:
        if not 0 <= u < nodes or not 0 <= v < nodes or u == v or canonical((u, v)) in seen:
            raise ValueError("Require native unique nonself TRAIN pairs")
        seen.add(canonical((u, v)))
        graph[u].add(v); graph[v].add(u)
    return graph


def validate_negative_bank(train, negative, nodes):
    facts = {canonical(pair) for pair in train}
    for u, v in negative:
        if not 0 <= u < nodes or not 0 <= v < nodes or u == v or canonical((u, v)) in facts:
            raise ValueError("Native negative bank differs from full-TRAIN visibility contract")
    if len(negative) < len(train):
        raise ValueError("Native sampler supplied too few index-paired negatives")


def route_streams(seed, members, control=False):
    if type(seed) is not int or type(members) is not int or members < 1 or type(control) is not bool:
        raise ValueError("Require fixed integer seed, route count and boolean control flag")
    offset = 1000003 if control else 0
    return [{role: random.Random(seed+offset+1009*member+salt)
             for role, salt in (("positive", 17), ("negative", 31))} for member in range(members)]


def outer_batches(order, edges, outer_size):
    if type(outer_size) is not int or outer_size < 1 or type(edges) is not int or edges < 1:
        raise ValueError("Invalid outer size or TRAIN population")
    _ids(order, edges, "outer order", unique=True)
    if len(order) != edges or set(order) != set(range(edges)):
        raise ValueError("Require one complete prospective outer permutation")
    return [order[start:start+outer_size] for start in range(0, edges, outer_size)]


def endpoint_episode(train, negative, outer_ids, streams, inner_size):
    _ids(outer_ids, len(train), "outer", unique=True)
    _ids(outer_ids, len(negative), "outer negative", unique=True)
    _streams(streams)
    if type(inner_size) is not int or inner_size < 1 or not outer_ids:
        raise ValueError("Invalid candidate query sizes/outer IDs")
    outer_negative_ids = list(outer_ids)  # native index-paired convention
    endpoints = {node for i in outer_ids for node in train[i]}
    endpoints.update(node for i in outer_negative_ids for node in negative[i])
    eligible_pos = [i for i, pair in enumerate(train) if endpoints.isdisjoint(pair)]
    eligible_neg = [i for i, pair in enumerate(negative) if endpoints.isdisjoint(pair)]
    output = {"outer_pos_ids": list(outer_ids), "outer_neg_ids": outer_negative_ids,
              "outer_endpoints": sorted(endpoints), "eligible_positive": len(eligible_pos),
              "eligible_negative": len(eligible_neg), "inner": [],
              "feasible": len(eligible_pos) >= inner_size and len(eligible_neg) >= inner_size}
    if not output["feasible"]:
        output["reason"] = "insufficient_endpoint_disjoint_queries; no redraw/padding/fallback"
        return output
    for stream in streams:
        output["inner"].append({"pos_ids": stream["positive"].sample(eligible_pos, inner_size),
                                "neg_ids": stream["negative"].sample(eligible_neg, inner_size)})
    return output


def outer_only_fallback(episode):
    """Describe an unadopted ordinary outer-only geometry for a rejected draw.

    This preserves the draw and remains infeasible for endpoint transfer. A
    runner may use it only after separate adoption of an ordinary-update rule.
    """
    if episode["feasible"] or episode["inner"]:
        raise ValueError("Fallback proposal applies only to a rejected endpoint draw")
    return {**episode, "outer_pos_ids": list(episode["outer_pos_ids"]),
            "outer_neg_ids": list(episode["outer_neg_ids"]),
            "outer_endpoints": list(episode["outer_endpoints"]), "inner": [],
            "feasible": False, "fallback_adopted": False,
            "proposal": "ordinary_outer_only_update_requires_separate_review",
            "support_rule": "mask_original_outer_positive_targets_only"}


def stratum(pair, graph):
    u, v = pair
    return (min(len(graph[u]), len(graph[v])), max(len(graph[u]), len(graph[v])),
            len(graph[u].intersection(graph[v])))


def matched_random_episode(train, negative, reference, streams, graph):
    """Match exact full-TRAIN degree/CN strata; allow endpoint overlap.

    Inner/outer equivalent query facts remain disjoint. Pair both arms with the
    union mask below. Post-mask strata are measured, not asserted to match.
    """
    _streams(streams)
    if not reference["feasible"]:
        return {**reference, "inner": [], "control_feasible": False}
    if len(reference["inner"]) != len(streams):
        raise ValueError("Reference route count differs from control stream count")
    outer_pos = set(reference["outer_pos_ids"])
    outer_neg = {canonical(negative[i]) for i in reference["outer_neg_ids"]}
    pools = {}
    for role, pairs, eligible in (
        ("positive", train, [i for i in range(len(train)) if i not in outer_pos]),
        ("negative", negative, [i for i, pair in enumerate(negative) if canonical(pair) not in outer_neg])):
        buckets = {}
        for i in eligible:
            buckets.setdefault(stratum(pairs[i], graph), []).append(i)
        pools[role] = buckets
    output = {key: reference[key] for key in ("outer_pos_ids", "outer_neg_ids", "outer_endpoints")}
    output.update(inner=[], feasible=True, control_feasible=True, match_scope="exact_full_TRAIN_degree_and_CN_counts")
    for route, stream in zip(reference["inner"], streams):
        selected = {}
        for role, pairs, key in (("positive", train, "pos_ids"), ("negative", negative, "neg_ids")):
            required = Counter(stratum(pairs[i], graph) for i in route[key])
            ids = []
            for group, count in sorted(required.items()):
                population = pools[role].get(group, [])
                if len(population) < count:
                    output.update(inner=[], feasible=False, control_feasible=False,
                                  reason="insufficient_matching_stratum; no substitute")
                    return output
                ids.extend(stream[role].sample(population, count))
            selected[key] = ids
        output["inner"].append(selected)
    return output


def mask_indices(*episodes):
    removed = set()
    for episode in episodes:
        removed.update(episode["outer_pos_ids"])
        for route in episode["inner"]:
            removed.update(route["pos_ids"])
    return removed


def support(train, nodes, removed):
    _ids(removed, len(train), "support mask")
    kept = [i for i in range(len(train)) if i not in removed]
    return kept, neighbors([train[i] for i in kept], nodes)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    middle = len(values)//2
    median = values[middle] if len(values) % 2 else (values[middle-1]+values[middle])/2
    return {"count": len(values), "minimum": values[0], "median": median,
            "maximum": values[-1], "mean": sum(values)/len(values)}


def strata_counts(pairs, graph):
    counts = Counter(stratum(pair, graph) for pair in pairs)
    return {":".join(map(str, key)): value for key, value in sorted(counts.items())}


def describe_episode(train, negative, episode, full_graph, masked_graph, removed):
    outer_nodes = set(episode["outer_endpoints"])
    inner_pos = [i for route in episode["inner"] for i in route["pos_ids"]]
    inner_neg = [i for route in episode["inner"] for i in route["neg_ids"]]
    roles = {"outer_positive": [train[i] for i in episode["outer_pos_ids"]],
             "outer_negative": [negative[i] for i in episode["outer_neg_ids"]],
             "inner_positive": [train[i] for i in inner_pos],
             "inner_negative": [negative[i] for i in inner_neg]}
    def describe_pairs(pairs):
        endpoints = {node for pair in pairs for node in pair}
        return {"queries": len(pairs), "unique_equivalent_facts": len({canonical(p) for p in pairs}),
            "unique_nodes": len(endpoints), "isolated_endpoints_after_mask": sum(not masked_graph[n] for n in endpoints),
            "endpoint_degree_before": distribution([len(full_graph[n]) for pair in pairs for n in pair]),
            "endpoint_degree_after": distribution([len(masked_graph[n]) for pair in pairs for n in pair]),
            "CN_before": distribution([len(full_graph[u].intersection(full_graph[v])) for u, v in pairs]),
            "CN_after": distribution([len(masked_graph[u].intersection(masked_graph[v])) for u, v in pairs]),
            "degree_CN_strata_before": strata_counts(pairs, full_graph),
            "degree_CN_strata_after": strata_counts(pairs, masked_graph)}
    descriptions = {role: describe_pairs(pairs) for role, pairs in roles.items()}
    all_inner = roles["inner_positive"]+roles["inner_negative"]
    return {"roles": descriptions,
            "inner_routes": [{"positive": describe_pairs([train[i] for i in route["pos_ids"]]),
                              "negative": describe_pairs([negative[i] for i in route["neg_ids"]])}
                             for route in episode["inner"]],
            "inner_queries_touching_outer_endpoints": sum(not outer_nodes.isdisjoint(p) for p in all_inner),
            "masked_positive_facts": len(removed), "support_undirected_edges": len(train)-len(removed),
            "support_symmetric_entries": 2*(len(train)-len(removed)),
            "support_nodes_with_edges": sum(bool(row) for row in masked_graph), "full_node_shape_retained": len(masked_graph),
            "outer_incident_context_edges_remaining": sum(i not in removed and not outer_nodes.isdisjoint(pair) for i, pair in enumerate(train)),
            "cross_route_positive_repeats": len(inner_pos)-len(set(inner_pos)),
            "cross_route_negative_equivalent_repeats": len(inner_neg)-len({canonical(negative[i]) for i in inner_neg})}


def exposure_summary(train, negative, episodes, nodes, members):
    outer_positive, outer_negative = Counter(), Counter()
    outer_negative_ids = Counter()
    inner_positive = [Counter() for _ in range(members)]
    inner_negative = [Counter() for _ in range(members)]
    inner_negative_ids = [Counter() for _ in range(members)]
    node_outer, node_inner = Counter(), [Counter() for _ in range(members)]
    for episode in episodes:
        for i in episode["outer_pos_ids"]:
            outer_positive[i] += 1
            node_outer.update(train[i])
        for i in episode["outer_neg_ids"]:
            outer_negative[canonical(negative[i])] += 1
            outer_negative_ids[i] += 1
            node_outer.update(negative[i])
        if not episode["feasible"]:
            continue
        if len(episode["inner"]) != members:
            raise ValueError("Feasible episode route count differs from exposure schedule")
        for member, route in enumerate(episode["inner"]):
            for i in route["pos_ids"]:
                inner_positive[member][i] += 1; node_inner[member].update(train[i])
            for i in route["neg_ids"]:
                inner_negative[member][canonical(negative[i])] += 1; node_inner[member].update(negative[i])
                inner_negative_ids[member][i] += 1
    return {"outer_exposure_scope": "candidate_draws_including_rejected_episodes; no_optimizer_supervision_claim",
            "inner_exposure_scope": "feasible_candidate_draws_only; no_optimizer_supervision_claim",
            "outer_positive_exposure_all_TRAIN_ids": distribution([outer_positive[i] for i in range(len(train))]),
            "outer_positive_ids_unseen": sum(outer_positive[i] == 0 for i in range(len(train))),
            "outer_positive_instances": sum(outer_positive.values()), "outer_negative_instances": sum(outer_negative.values()),
            "outer_negative_unique_equivalent_facts": len(outer_negative),
            "outer_negative_exposure_all_bank_ids": distribution([outer_negative_ids[i] for i in range(len(negative))]),
            "outer_node_exposure_all_nodes": distribution([node_outer[n] for n in range(nodes)]),
            "routes": [{"positive_instances": sum(inner_positive[m].values()),
                        "positive_ids_unseen": sum(inner_positive[m][i] == 0 for i in range(len(train))),
                        "positive_exposure_all_TRAIN_ids": distribution([inner_positive[m][i] for i in range(len(train))]),
                        "negative_instances": sum(inner_negative[m].values()),
                        "negative_unique_equivalent_facts": len(inner_negative[m]),
                        "negative_exposure_all_bank_ids": distribution([inner_negative_ids[m][i] for i in range(len(negative))]),
                        "node_exposure_all_nodes": distribution([node_inner[m][n] for n in range(nodes)]),
                        "planned_inner_loss_evaluation_instances": 2*(sum(inner_positive[m].values())+sum(inner_negative[m].values()))}
                       for m in range(members)],
            "inner_twice_computed_convention": "virtual update plus recomputation; one committed private update",
            "actual_optimizer_updates": 0}
