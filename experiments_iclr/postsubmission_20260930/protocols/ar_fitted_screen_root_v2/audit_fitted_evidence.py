"""Audit retained new-screen evidence and apply the frozen continuation gate.

This uses stored metrics, not fresh predictions. It never imports model code,
changes a score, selects a checkpoint or reads the original benchmark results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

PHASE = Path(__file__).resolve().parents[2]
PACKET = PHASE / "new_learning_gap_independent_v1/ar_fitted_screen_source_v2"
EXPECTED_FREEZE_SHA256 = "210ba66129493a973f8826e207ba0841826b63040bc670701a18004b8f7e6833"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit(freeze, progress, run):
    require(progress["status"] == "completed_frozen_fitted_screen", "Complete frozen screen required")
    require(progress["root_freeze_sha256"] == EXPECTED_FREEZE_SHA256, "Progress does not bind the frozen protocol")
    require(progress["all_final_checkpoints_complete_before_development"] is True, "Development draw gate not satisfied")
    fits = {(r["arm"], r["model_seed"]): r for r in progress["fits"]}
    expected_fits = {(r["arm"], r["model_seed"]) for r in freeze["fit_order"]}
    require(len(progress["fits"]) == len(fits) == 24 and set(fits) == expected_fits, "Complete unique24fit matrix required")
    require([r["seed"] for r in progress["train_instances"]] == freeze["training_topology_seeds"], "Training topology inventory changed")
    require([r["seed"] for r in progress["development_instances"]] == freeze["development_topology_seeds"], "Development topology inventory changed")
    arms = {r["name"]: r for r in freeze["arms_in_order"]}
    candidate = next(r["name"] for r in freeze["arms_in_order"] if r["role"] == "candidate")
    identities = [r["name"] for r in freeze["arms_in_order"] if r["role"] in ("all_factor_ablation", "recurrent_factor_ablation")]
    controls = [name for name in arms if name != candidate]
    for (name, seed), row in fits.items():
        require(row["status"] == "completed_final_checkpoint" and row["completed_updates"] == 64 and row["checkpoint_update"] == 64,
                "Missing final64update fit")
        require(row["complete_training_schedules"] == 512 and row["backward_calls_completed"] == row["optimizer_steps_completed"] == 64,
                "Completed training work differs from frozen budget")
        require(row["uniform_values_generated"] == 7680, "Uniform slot accounting changed")
        cp = run / "checkpoints" / (name + "_seed" + str(seed) + "_final.pt")
        require(cp.is_file() and sha(cp) == row["final_checkpoint"]["sha256"], "Retained checkpoint bytes mismatch")
        require(row["parameters"]["stored_elements"] == 2992 + arms[name]["expected_decoder_stored_elements"], "Stored element mismatch")
        require(row["parameters"]["trainable_elements"] == 2992 + arms[name]["expected_decoder_trainable_elements"], "Trainable element mismatch")
    expected = {(name, seed, graph, fugacity) for name in arms for seed in freeze["model_seeds"]
                for graph in freeze["development_topology_seeds"] for fugacity in [4, 1]}
    rows = {(r["arm"], r["model_seed"], r["graph_seed"], r["fugacity"]): r for r in progress["evaluations"]}
    require(len(progress["evaluations"]) == len(rows) == 384 and set(rows) == expected, "Complete unique384law matrix required")
    parity_count = 0
    cells = {}
    for key, row in rows.items():
        name, seed, graph, fugacity = key
        require(row["status"] == "completed" and row["checkpoint_update"] == 64, "Incomplete or wrong-checkpoint law")
        require(row["checkpoint_sha256"] == fits[(name, seed)]["final_checkpoint"]["sha256"], "Evaluation checkpoint mismatch")
        law = row["whole_law"]
        require(law["support_complete"] is True and law["complete_binary_states"] == 32768,
                "Full state-space accounting required")
        F = law["complete_feasible_states"]
        require(0 < F <= 32768 and law["structural_zero_states"] == 32768 - F, "Support counts inconsistent")
        require(len(law["member_mass"]) == arms[name]["members"] and all(abs(x - 1) <= 1e-10 for x in law["member_mass"]),
                "Member law normalization failed")
        require(abs(law["mixture_mass"] - 1) <= 1e-10, "Mixture normalization failed")
        for metric in ("kl_q_p", "kl_p_q", "total_variation", "member_information", "mean_branch_kl_q_p"):
            require(math.isfinite(law[metric]), "Nonfinite stored metric")
        require(min(law["kl_q_p"], law["kl_p_q"], law["member_information"]) >= -1e-10, "Divergence negative beyond roundoff")
        require(-1e-10 <= law["total_variation"] <= 1 + 1e-10, "Invalid TV")
        require(law["member_information"] <= math.log(arms[name]["members"]) + 1e-9, "Invalid mixture information")
        require(abs(law["mean_branch_kl_q_p"] - law["kl_q_p"] - law["member_information"]) <= 1e-9,
                "Mixture/branch KL identity inconsistent")
        if graph in freeze["evaluation"]["parity_development_topology_seeds"]:
            require(row["parity"]["maximum_absolute_error"] <= 1e-10, "Final actual-H parity failed")
            require(row["parity"]["all_members"] == arms[name]["members"] and
                    row["parity"]["selected_ordinals"] == sorted({j * (F - 1) // 15 for j in range(16)}),
                    "Final parity member/ordinal coverage changed")
            parity_count += 1
        cells[key] = law
    require(parity_count == 48, "Final parity inventory incomplete")
    classical = {(r["graph_seed"], r["fugacity"]): r for r in progress["classical"]}
    require(len(progress["classical"]) == len(classical) == 16 and set(classical) == {(s, f) for s in freeze["development_topology_seeds"] for f in [4, 1]},
            "Complete classical matrix required")
    require(all(r["status"] == "completed" for r in classical.values()), "Classical matrix incomplete")
    require(all(r["gibbs"]["retained_total"] == 8192 and r["gibbs"]["random_scan_site_updates"] == 130560 for r in classical.values()),
            "Gibbs work contract changed")
    require(all(r["exact_iid_record"]["unbiased_integer_ticket_draws"] == 8192 and
                sum(frequency for _, frequency in r["exact_iid_histogram"]) == 8192 and
                r["exact_iid_whole_cache_law"]["retained_states"] == 8192 for r in classical.values()),
            "Matched exact-IID empirical control omitted or changed")
    seeds, graphs = freeze["model_seeds"], freeze["development_topology_seeds"]
    def mean_value(name, fugacity, metric, seed=None):
        values = [cells[(name, s, g, fugacity)][metric] for s in ([seed] if seed is not None else seeds) for g in graphs]
        return sum(values) / len(values)
    strongest = min(controls, key=lambda name: (mean_value(name, 4, "kl_q_p"), list(arms).index(name)))
    effects = []
    checks = []
    def check(label, passed, value):
        checks.append({"condition": label, "passed": bool(passed), "observed": value})
    for name in controls:
        for fugacity in [4, 1]:
            contrasts = {metric: mean_value(name, fugacity, metric) - mean_value(candidate, fugacity, metric)
                         for metric in ("kl_q_p", "kl_p_q", "total_variation")}
            effects.append({"control": name, "fugacity": fugacity, "mean_control_minus_candidate": contrasts})
            if fugacity == 4:
                check("lambda4_KL_gain_at_least_0.01_vs_" + name, contrasts["kl_q_p"] >= .01, contrasts["kl_q_p"])
            else:
                check("lambda1_KL_regression_at_most_0.01_vs_" + name, contrasts["kl_q_p"] >= -.01, contrasts["kl_q_p"])
            check("TV_regression_at_most_0.01_lambda" + str(fugacity) + "_vs_" + name,
                  contrasts["total_variation"] >= -.01, contrasts["total_variation"])
            if name in identities and fugacity == 4:
                check("lambda4_TV_identity_gain_positive_vs_" + name, contrasts["total_variation"] > 0, contrasts["total_variation"])
    for name in dict.fromkeys([*identities, strongest]):
        for seed in seeds:
            effect = mean_value(name, 4, "kl_q_p", seed) - mean_value(candidate, 4, "kl_q_p", seed)
            check("lambda4_seed" + str(seed) + "_KL_gain_positive_vs_" + name, effect > 0, effect)
    summaries = [{"arm": name, "fugacity": f, "means": {m: mean_value(name, f, m) for m in ("kl_q_p", "kl_p_q", "total_variation")},
                  "seed_topology_means": [{"seed": s, "kl_q_p": mean_value(name, f, "kl_q_p", s),
                                           "total_variation": mean_value(name, f, "total_variation", s)} for s in seeds]}
                 for name in arms for f in [4, 1]]
    return {"status": "audited_complete_development_screen", "continuation_gate_passed": all(r["passed"] for r in checks),
            "strongest_learned_control_by_frozen_rule": strongest, "checks": checks,
            "summaries": summaries, "all_control_effects": effects,
            "scope": "SmallN15 development evidence only; not established novelty, confirmation, convergence, scalability or acceptance",
            "independence": "Three modelseeds crossed with eight topologies;24 cells are not24 independent modelruns",
            "no_score_recalculation_or_clamping": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(PHASE) and not args.output.exists(), "New phase-confined audit output required")
    freeze_path = PHASE / "protocols/ar_fitted_screen_root_v2/SCIENTIFIC_FREEZE_v2.json"
    require(sha(freeze_path) == EXPECTED_FREEZE_SHA256, "Root scientific freeze changed")
    run = PACKET / "runs/admitted_run_v1"
    progress_path = run / "progress.json"
    freeze, progress = json.loads(freeze_path.read_text()), json.loads(progress_path.read_text())
    result = {"UTC": datetime.now(timezone.utc).isoformat(), "freeze_sha256": sha(freeze_path),
              "progress_sha256": sha(progress_path), "audit_source_sha256": sha(Path(__file__)), **audit(freeze, progress, run)}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "continuation_gate_passed": result["continuation_gate_passed"]}))


if __name__ == "__main__":
    main()
