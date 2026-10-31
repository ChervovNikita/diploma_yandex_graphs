"""Unexecuted descriptive whole-cell summaries; no selection or new scores.

No incomplete cell is silently excluded. Raw KL values are retained even when
roundoff makes a near-zero value slightly negative. Root owns the decision gate.
"""


def complete_summary(progress, freeze):
    if progress["status"] != "completed_frozen_fitted_screen":
        return {"status": "incomplete_no_go", "decision": "No complete screen conclusion",
                "unfinished_fits": [row for row in progress["fits"] if row["status"] != "completed_final_checkpoint"],
                "unfinished_laws": [{k: row[k] for k in ("arm", "model_seed", "graph_seed", "fugacity", "status")}
                    for row in progress["evaluations"] if row["status"] != "completed"],
                "unfinished_classical": [row for row in progress["classical"] if row["status"] != "completed"]}
    metrics = ("kl_q_p", "kl_p_q", "total_variation", "node_marginal_mean_absolute_error")
    cells = {(row["arm"], row["model_seed"], row["graph_seed"], row["fugacity"]): row["whole_law"] for row in progress["evaluations"]}
    required = {(arm["name"], seed, graph, fugacity) for arm in freeze["arms_in_order"] for seed in freeze["model_seeds"]
                for graph in freeze["development_topology_seeds"] for fugacity in freeze["fugacities_in_order"]}
    if set(cells) != required:
        raise AssertionError("Complete frozen cell set required; no successful-subset summary")
    candidate = next(arm["name"] for arm in freeze["arms_in_order"] if arm["role"] == "candidate")
    means, pairs = [], []
    for arm in freeze["arms_in_order"]:
        for seed in freeze["model_seeds"]:
            for fugacity in freeze["fugacities_in_order"]:
                group = [cells[(arm["name"], seed, graph, fugacity)] for graph in freeze["development_topology_seeds"]]
                means.append({"arm": arm["name"], "model_seed": seed, "fugacity": fugacity,
                    "complete_topology_units": len(group), "means": {metric: sum(row[metric] for row in group) / len(group) for metric in metrics}})
                if arm["name"] != candidate:
                    for graph in freeze["development_topology_seeds"]:
                        control, fitted = cells[(arm["name"], seed, graph, fugacity)], cells[(candidate, seed, graph, fugacity)]
                        pairs.append({"control": arm["name"], "candidate": candidate, "model_seed": seed, "graph_seed": graph,
                            "fugacity": fugacity, "control_minus_candidate": {metric: control[metric] - fitted[metric] for metric in metrics}})
    controls = [arm for arm in freeze["arms_in_order"] if arm["name"] != candidate]
    identities = [arm["name"] for arm in controls if arm["role"] in ("all_factor_ablation", "recurrent_factor_ablation")]
    def mean_metric(arm, fugacity, metric, only_seed=None):
        values = [cells[(arm, seed, graph, fugacity)][metric] for seed in freeze["model_seeds"] if only_seed is None or seed == only_seed
                  for graph in freeze["development_topology_seeds"]]
        return sum(values) / len(values)
    strongest = min(controls, key=lambda arm: mean_metric(arm["name"], 4, "kl_q_p"))["name"]
    gate = freeze["scientific_gate"]
    checks = []
    for arm in controls:
        name = arm["name"]
        gain4 = mean_metric(name, 4, "kl_q_p") - mean_metric(candidate, 4, "kl_q_p")
        regression1 = mean_metric(candidate, 1, "kl_q_p") - mean_metric(name, 1, "kl_q_p")
        checks.extend([{"criterion": "lambda4_KL_gain", "control": name, "value": gain4,
                        "passed": gain4 >= gate["lambda4_minimum_mean_kl_control_minus_candidate_nats"]},
                       {"criterion": "lambda1_KL_nonregression", "control": name, "value": regression1,
                        "passed": regression1 <= gate["lambda1_maximum_mean_kl_candidate_regression_nats"]}])
        for fugacity in (4, 1):
            regression = mean_metric(candidate, fugacity, "total_variation") - mean_metric(name, fugacity, "total_variation")
            checks.append({"criterion": "TV_nonregression", "control": name, "fugacity": fugacity, "value": regression,
                           "passed": regression <= gate["tv_maximum_mean_candidate_regression_both_regimes"]})
    for name in identities:
        gain = mean_metric(name, 4, "total_variation") - mean_metric(candidate, 4, "total_variation")
        checks.append({"criterion": "lambda4_identity_TV_gain", "control": name, "value": gain, "passed": gain > 0})
    for name in dict.fromkeys(identities + [strongest]):
        for seed in freeze["model_seeds"]:
            gain = mean_metric(name, 4, "kl_q_p", seed) - mean_metric(candidate, 4, "kl_q_p", seed)
            checks.append({"criterion": "per_seed_lambda4_KL_gain", "control": name, "model_seed": seed, "value": gain, "passed": gain > 0})
    passed = all(check["passed"] for check in checks)
    return {"status": "complete_descriptive_screen", "root_primary_metric": freeze["primary_metric"],
            "per_model_seed_topology_means": means, "all_paired_cell_effects": pairs,
            "scientific_gate": {"passed": passed, "checks": checks, "strongest_pooled_KL_learned_control": strongest,
                "tie_break": "Declared control order", "scope": gate["scope"]},
            "decision": "exploratory_continuation_gate_pass" if passed else "no_go_at_this_screen",
            "independence": "Eight topology units and three model seeds; the 24 paired seed/topology cells are not 24 independent model runs",
            "scope": "N15 development adaptation only; no original-score recalculation, confirmation or scalability claim"}
