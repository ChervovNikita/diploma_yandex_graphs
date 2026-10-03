"""Adopt finished numerical evidence without changing fitted or original scores."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
UTC = datetime.now(timezone.utc).isoformat()


def read(relative):
    return json.loads((ROOT / relative).read_text())


def binding(relative):
    p = ROOT / relative
    b = p.read_bytes()
    return {"path": relative, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write(relative, value):
    p = ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


init_root = "graph_init_analysis_companion_v10_source_valid_execution_root_v1/run_v1/"
init = read(init_root + "SOURCE_VALID_AUDIT.json")
init_terminal_path = "graph_init_analysis_companion_v10_cpu_supervision_root_v1/run_v1/TERMINAL.json"
terminal = read(init_terminal_path)
freeze = read(init_root + "AUDIT_FREEZE.json")
assert terminal["completed"] and terminal["child_exit_code"] == 0
assert not terminal["timed_out"] and terminal["within_whole_cap"]
assert init["completed"] and freeze["completed"]
assert init["source_cells"] == 30 and init["registered_phases"] == 72
assert not init["heldout_labels_or_final_report_opened"]
assert not init["serialized_checkpoint_inference_replayed"]
rows = init["metrics"]
assert len(rows) == len(init["complete_array_guards"]) == 30
assert len(init["physical_cost_bindings"]) == 72
assert len({(r["graph"], r["seed"], r["arm"]) for r in rows}) == 30
assert all(r["full_array_finite"] for r in init["complete_array_guards"])
assert max(r["absolute_difference"] for r in rows) <= 1e-6
graphs = ("Photo", "Squirrel")
arms = ("graph", "common_only", "random_tangent", "topology_permuted", "warm_copy")
seeds = (17, 29, 43)
lookup = {(r["graph"], r["seed"], r["arm"]): r for r in rows}
nll_key = "recomputed_native_FP32_member_mean_stable_FP64_NLL"
acc_key = "recomputed_native_pool_accuracy"
means, paired = [], []
for graph in graphs:
    for arm in arms:
        rr = [lookup[graph, seed, arm] for seed in seeds]
        means.append({"graph": graph, "arm": arm, "blocks": 3,
                      "VALID_NLL_nats": statistics.mean(r[nll_key] for r in rr),
                      "VALID_accuracy_percent": 100 * statistics.mean(r[acc_key] for r in rr)})
    for comparator in arms[1:]:
        dn = [lookup[graph, s, "graph"][nll_key] - lookup[graph, s, comparator][nll_key] for s in seeds]
        da = [100 * (lookup[graph, s, "graph"][acc_key] - lookup[graph, s, comparator][acc_key]) for s in seeds]
        half = 4.302652729911275 * statistics.stdev(dn) / math.sqrt(3)
        paired.append({"graph": graph, "graph_minus": comparator, "seeds": list(seeds),
                       "NLL_differences_nats": dn, "accuracy_differences_pp": da,
                       "mean_NLL_difference_nats": statistics.mean(dn),
                       "mean_accuracy_difference_pp": statistics.mean(da),
                       "descriptive_df2_NLL_interval": [statistics.mean(dn)-half, statistics.mean(dn)+half],
                       "interval_is_not_confirmatory": True})
init_adoption = {"schema": "initializer-complete30-saved-array-root-adoption-v1", "UTC": UTC,
                 "audit": binding(init_root + "SOURCE_VALID_AUDIT.json"),
                 "freeze": binding(init_root + "AUDIT_FREEZE.json"),
                 "physical_terminal": binding(init_terminal_path),
                 "all30_cases_and72_phases_retained": True,
                 "max_saved_NLL_recomputation_difference": max(r["absolute_difference"] for r in rows),
                 "means": means, "paired_contrasts": paired,
                 "selection_role": "VALID also selected checkpoints. These are development scores on overlapping graph splits.",
                 "statistical_limit": "Three blocks. The df2 t interval is descriptive. Split independence and normality are unestablished. No familywise significance claim.",
                 "serialized_checkpoint_inference_replayed": False,
                 "historical_epoch_scores_independently_recomputed": False,
                 "outcome_aware_exploratory_cfg0": init["outcome_aware_exploratory_cfg0"],
                 "original_scores_changed": False, "TEST_scored": False,
                 "decision": "No consistent graph-specific advantage. Close this initialization variant without further tuning or heldout promotion. Finish serialized inference for reproducibility.",
                 "predictive_superiority_or_novelty_established": False}
write("graph_init_analysis_companion_v10_cpu_release_v1/NUMERICAL_AUDIT_ADOPTION_v1.json", init_adoption)
table = ["# Complete initialization comparison", "", "The saved-array audit completed all 30 cases and 72 phase bindings. It reproduces saved selected-state validation NLL within the original 1e-6 tolerance. The separate supervisor reaped the child with exit 0.", "", "| Graph | Initialization | Mean VALID NLL (nats) | Mean VALID accuracy (%) |", "|---|---|---:|---:|"]
for r in means:
    table.append(f"| {r['graph']} | {r['arm']} | {r['VALID_NLL_nats']:.6f} | {r['VALID_accuracy_percent']:.3f} |")
table += ["", "Graph initialization has no consistent advantage over the matched alternatives. On Photo its mean NLL is worse than unchanged warm copying. On Squirrel its contrasts with random or permuted initialization are very small. This variant is closed without further tuning or heldout promotion.", "", "These are development results. VALID selected checkpoints, there are three overlapping split blocks, and the configuration was explored after earlier outcomes. Descriptive paired intervals and all case bindings are retained in NUMERICAL_AUDIT_ADOPTION_v1.json. Reopening serialized checkpoints for independent inference remains outstanding. Original paper scores are unchanged.", ""]
out = ROOT / "graph_init_analysis_companion_v10_cpu_release_v1/RESULTS_SUMMARY_v1.md"
with out.open("x") as f:
    f.write("\n".join(table))

qroot = "forde_graph_small_real_B2_oracle_execution_root_20261003_v1/"
q = read(qroot + "run01/ORACLE_RECEIPT.json")
qt = read(qroot + "Q03_CONSOLE_TERMINAL.json")
assert qt["exit_code"] == 0
assert q["status"] == "BOTH_FIXED_B2_CPU_CACHE_CUDA_ENDPOINTS_QUALIFIED_NO_B128_OR_SCIENTIFIC_PROMOTION"
assert len(q["cases"]) == 2
assert q["tolerances_predeclared_before_execution"] == {"value_absolute": 3e-6, "value_relative": 3e-4, "gradient_absolute": 1e-4, "gradient_relative": 3e-3}


def checked_metrics(value):
    result = []
    if isinstance(value, dict):
        if "within_predeclared_tolerance" in value:
            assert value["within_predeclared_tolerance"] is True
            assert math.isfinite(value["max_scaled_error"]) and value["max_scaled_error"] <= 1
            result.append(value)
        for v in value.values():
            result.extend(checked_metrics(v))
    elif isinstance(value, list):
        for v in value:
            result.extend(checked_metrics(v))
    return result


qc = []
for c in q["cases"]:
    assert c["logical_B"] == 2 and c["members"] == 4 and all(c["custody"].values())
    mm = checked_metrics(c)
    assert mm
    qc.append({"recipe": c["recipe"], "elementwise_comparison_records": len(mm),
               "private_tensor_count": c["private_tensor_count"],
               "max_scaled_error_over_all_checks": max(v["max_scaled_error"] for v in mm),
               "peak_allocated_GiB": c["GPU_peak_allocated_bytes"] / 2**30,
               "peak_reserved_GiB": c["GPU_peak_reserved_bytes"] / 2**30,
               "case_wall_seconds": c["complete_wall_seconds_including_cleanup"],
               "custody_all_true": True})
q_adoption = {"schema": "forde-Q03-fixed-real-B2-root-adoption-v1", "UTC": UTC,
              "oracle": binding(qroot + "run01/ORACLE_RECEIPT.json"),
              "physical_terminal": binding(qroot + "Q03_CONSOLE_TERMINAL.json"), "cases": qc,
              "dispatch_to_exit_seconds": qt["dispatch_to_exit_seconds"],
              "scope": "Both source-fixed Amazon M4/B2 CPU-cache/CUDA endpoints. Full live 24492x300 input recurrence and all existing private R/S derivative coordinates compared with independent full-X and coefficient references.",
              "B128_numerical_equality_admitted": False,
              "CUDA_sparse_recurrence_equality_admitted": False,
              "trained_checkpoint_or_predictive_success": False,
              "optimizer_update": False,
              "upstream_default_equivalence": False,
              "epsilon_limit": "Graph adapter epsilon=1e-24 differs from upstream FoRDE default 1e-12.",
              "decision": "Adopt fixed B2 numerical qualification. Keep B128 feasibility separate. Comparator is ready for a concrete fitting integration decision, not automatically released for science."}
write(qroot + "Q03_RESULTS_ADOPTION_v1.json", q_adoption)
with (ROOT / (qroot + "RESULTS_SUMMARY_v1.md")).open("x") as f:
    f.write("# FoRDE full-input numerical check\n\nBoth fixed Amazon recipes passed every declared M4/B2 value and private-gradient comparison. The physical console exited 0 in %.3f seconds. The independent reference differentiates the selected raw scores through the complete 24,492 by 300 live input recurrence.\n\n" % qt["dispatch_to_exit_seconds"])
    f.write("| Recipe | Comparison records | Largest scaled error | Peak allocated GiB | Case seconds |\n|---|---:|---:|---:|---:|\n")
    for c in qc:
        f.write(f"| {c['recipe']} | {c['elementwise_comparison_records']} | {c['max_scaled_error_over_all_checks']:.6g} | {c['peak_allocated_GiB']:.3f} | {c['case_wall_seconds']:.3f} |\n")
    f.write("\nScaled error must be at most 1 under the declared elementwise tolerances. All custody checks passed. This proves the two fixed B2 endpoints only. The separate B128 run established feasibility, not B128 numerical equivalence. No optimizer update, trained-state result or predictive score was produced. The graph adapter uses epsilon 1e-24, so unchanged upstream-default FoRDE equivalence is not claimed.\n")

write("graph_init_analysis_companion_v10_cpu_release_v1/FETCH_FAILURE_NOTE_v1.json",
      {"UTC": UTC, "first_fetch": "Directory tar fetch refused an unsafe archive member before reaching numerical evidence. The wrapper terminal is deliberately hardlinked to its pending receipt.",
       "recovery": "Fetched six explicit regular-file paths with the unchanged authorized utility. No source or scientific process restarted.",
       "exact_success_receipt": binding("graph_init_analysis_companion_v10_cpu_release_v1/EXACT_TERMINAL_AUDIT_FETCH_v2.json")})
print(json.dumps({"initializer_means": means, "initializer_decision": init_adoption["decision"], "Q03_cases": qc}))
