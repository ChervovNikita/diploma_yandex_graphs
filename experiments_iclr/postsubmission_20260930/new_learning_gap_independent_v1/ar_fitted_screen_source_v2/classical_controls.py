"""Unexecuted exact-ticket and finite empirical Gibbs controls.

All chain/burn-in/retention choices come from the future root freeze. These
controls have separate build/draw timings and never train or select a model.
"""
from bisect import bisect_right
from collections import Counter
import hashlib
import math
from random import Random


def seed(graph_seed, fugacity, method, chain_index, role):
    text = "|".join(map(str, ("ar_fitted_screen_v1", "classical", graph_seed, fugacity, method, chain_index, role)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact_build(spatial, instance, fugacity):
    states, partition = spatial.complete_integer_target(instance, fugacity=fugacity)
    cumulative, total = [], 0
    for mask, weight in states:
        total += weight
        cumulative.append(total)
    require(total == partition and len(states) > 0, "Exact integer ticket table failed")
    return states, partition, tuple(cumulative)


def exact_draws(instance, fugacity, table, count):
    states, partition, cumulative = table
    rng = Random(seed(instance.seed, fugacity, "exact_ticket", "none", "tickets"))
    tickets = [rng.randrange(partition) for _ in range(count)]
    masks = [states[bisect_right(cumulative, ticket)][0] for ticket in tickets]
    return {"sample_masks": masks, "sample_tickets": tickets, "randrange_calls": count,
            "sample_law": "Exact integer target-ticket law; no approximate floating CDF",
            "target_kl_q_p": 0, "target_total_variation": 0,
            "zero_metrics_scope": "Analytic law of the sampler, not an eight-draw empirical histogram",
            "rng_bits": "Random.randrange rejection work is included in time; exact random-bit count not measured"}


def gibbs_build(instance, fugacity, schedule, on_chain):
    require(len(instance.points) == 15 and fugacity in (1, 4), "Pinned spatial15 regimes required")
    chains, burn, retained, stride = (schedule[name] for name in
        ("chains", "burn_in_sweeps", "retained_states_per_chain", "sweeps_between_retained"))
    require(all(type(x) is int and x > 0 for x in (chains, burn, retained, stride)), "Positive root-frozen Gibbs schedule required")
    neighbors = [[] for _ in range(15)]
    for u, v in instance.edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    histogram, chain_records = Counter(), []
    for index in range(chains):
        site_rng = Random(seed(instance.seed, fugacity, "gibbs", index, "sites"))
        coin_rng = Random(seed(instance.seed, fugacity, "gibbs", index, "uniforms"))
        state, site_calls, uniform_calls, blocked_updates = [0] * 15, 0, 0, 0
        def sweep():
            nonlocal site_calls, uniform_calls, blocked_updates
            for _ in range(15):
                node = site_rng.randrange(15)
                uniform = coin_rng.random()  # Consume even when blocked.
                site_calls += 1
                uniform_calls += 1
                blocked = any(state[neighbor] for neighbor in neighbors[node])
                blocked_updates += int(blocked)
                state[node] = 0 if blocked else int(uniform < fugacity / (1 + fugacity))
        for _ in range(burn):
            sweep()
        for _ in range(retained):
            for _ in range(stride):
                sweep()
            mask = sum(label << node for node, label in enumerate(state))
            require(not any(state[u] and state[v] for u, v in instance.edges), "Gibbs retained infeasible state")
            histogram[mask] += 1
        record = {"chain": index, "initial_mask": 0, "retained_states": retained,
                  "site_randrange_calls": site_calls, "uniform_calls": uniform_calls,
                  "blocked_updates": blocked_updates, "final_mask": sum(label << node for node, label in enumerate(state))}
        chain_records.append(record)
        on_chain(record)
    require(sum(histogram.values()) == chains * retained, "Gibbs histogram omitted retained states")
    return histogram, {"chains": chain_records, "retained_total": chains * retained,
                       "random_scan_site_updates": sum(row["site_randrange_calls"] for row in chain_records),
                       "law": "Unsmoothed empirical frequency table of correlated retained states; no mixing guarantee",
                       "schedule": dict(schedule)}


def empirical_metrics(histogram, weighted_states, partition):
    total = sum(histogram.values())
    require(total > 0, "Complete retained empirical cache required")
    target = dict(weighted_states)
    require(set(histogram) <= set(target), "Cache contains an infeasible/out-of-target state")
    kl_q_p, kl_p_q, tv, missing = 0.0, 0.0, 0.0, 0
    target_marginals, cache_marginals = [0.0] * 15, [0.0] * 15
    for mask, weight in weighted_states:
        p, q = weight / partition, histogram.get(mask, 0) / total
        for node in range(15):
            if (mask >> node) & 1:
                target_marginals[node] += p
                cache_marginals[node] += q
        tv += 0.5 * abs(q - p)
        if q:
            kl_q_p += q * (math.log(q) - math.log(weight) + math.log(partition))
            kl_p_q += p * (math.log(weight) - math.log(partition) - math.log(q))
        else:
            missing += 1
    return {"complete_feasible_states": len(weighted_states), "complete_binary_states": 32768,
            "retained_states": total, "unique_cache_states": len(histogram), "missing_positive_target_states": missing,
            "kl_q_p": kl_q_p, "kl_p_q": "infinity" if missing else kl_p_q,
            "total_variation": tv, "smoothing": "none", "mass": 1.0,
            "target_node_marginals": target_marginals, "cache_node_marginals": cache_marginals,
            "node_marginal_errors_q_minus_p": [q - p for q, p in zip(cache_marginals, target_marginals)],
            "node_marginal_mean_absolute_error": sum(abs(q - p) for q, p in zip(cache_marginals, target_marginals)) / 15,
            "node_marginal_maximum_absolute_error": max(abs(q - p) for q, p in zip(cache_marginals, target_marginals))}


def cache_draws(instance, fugacity, histogram, count, method="gibbs_cache"):
    rows = sorted(histogram.items())
    cumulative, total = [], 0
    for mask, frequency in rows:
        total += frequency
        cumulative.append(total)
    rng = Random(seed(instance.seed, fugacity, method, "none", "tickets"))
    tickets = [rng.randrange(total) for _ in range(count)]
    masks = [rows[bisect_right(cumulative, ticket)][0] for ticket in tickets]
    return {"sample_masks": masks, "randrange_calls": count,
            "scope": "Independent ticket lookups from the finite empirical cache; chain/cache construction costs separate"}


def exact_iid_histogram(instance, fugacity, table, count):
    states, partition, cumulative = table
    require(type(count) is int and count > 0, "Root-frozen matched IID sample count required")
    rng = Random(seed(instance.seed, fugacity, "exact_iid_histogram", "none", "tickets"))
    histogram = Counter()
    for _ in range(count):
        ticket = rng.randrange(partition)
        histogram[states[bisect_right(cumulative, ticket)][0]] += 1
    require(sum(histogram.values()) == count, "Exact-IID empirical histogram lost a draw")
    return histogram, {"unbiased_integer_ticket_draws": count, "unique_states": len(histogram),
        "scope": "Finite-sample empirical law from exact IID draws; distinguishes sampling error from Gibbs-chain effects",
        "source_sampler_ideal_kl_q_p": 0, "source_sampler_ideal_total_variation": 0,
        "setup": "Reuses the separately timed cold exact integer table; histogram draw/cache work timed separately"}
