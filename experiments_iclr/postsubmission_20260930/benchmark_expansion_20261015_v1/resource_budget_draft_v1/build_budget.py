#!/usr/bin/env python3
"""Rebuild a metadata-only budget. No scientific libraries or remote operations."""
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
E = OUT.parent
PHASE = E.parent
READS = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(rel, compressed=False, text=False):
    path = PHASE / rel
    raw = path.read_bytes()
    READS[rel] = {"path": rel, "bytes": len(raw), "sha256": sha(raw)}
    if compressed:
        raw = gzip.decompress(raw)
    return raw.decode() if text else json.loads(raw)


def erel(rel):
    return str(E.relative_to(PHASE) / rel)


def load(rel):
    return read(erel(rel))


def dump(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    (OUT / name).write_bytes(raw)
    return {"path": name, "bytes": len(raw), "sha256": sha(raw)}


def stats(values):
    values = list(values)
    assert values
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def fit_summary(rows):
    return {
        "fit_count": len(rows),
        "completed_epochs": sum(r["trained_epochs"] for r in rows),
        "completed_optimizer_maps": sum(sum(r["optimizer_calls"]["completed"]) for r in rows),
        "worker_invocation_wall_sum_seconds": sum(r["invocation_wall_seconds"] for r in rows),
        "epochs_per_fit": stats(r["trained_epochs"] for r in rows),
        "fit_invocation_wall_seconds": stats(r["invocation_wall_seconds"] for r in rows),
        "invocation_wall_per_completed_epoch_seconds": stats(r["invocation_wall_seconds"] / r["trained_epochs"] for r in rows),
        "resources": {
            key: stats(r["resources"][key] for r in rows)
            for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "own_peak_rss_bytes")
        },
        "convergence_claim": "Recorded patience stop and complete work only; no scientific convergence conclusion is inferred from an inherited field.",
    }


roster = load("expanded_study_design_v1/PROPOSED_ROSTER.json")
native = load("expanded_study_design_v1/NATIVE_SETTINGS.json")
readiness = load("expanded_readiness_coverage_v1/READINESS_COVERAGE.json")
remaining = load("expanded_readiness_coverage_v1/REMAINING_FINITE_PLAN.json")
settings = {r["preset"]: r for r in native["rows"]}
assert len(settings) == 27 and len(roster["cells"]) == 27
assert sum(c["total_fits"] for c in roster["cells"]) == 6460
assert all("gnnm_boundary_native" in c["shared_method_ids"] for c in roster["cells"])
mol = load("expanded_launch_preparation_v1/MOLBACE24_COST_INPUT.json")
catalogue = load("expanded_launch_preparation_v1/compact_generated_v1/COST_CATALOGUE.json")
bn = load("resource_budget_draft_v1/BN3_COST_METADATA.json")
smoke = read("reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/RESULTS.json.gz", compressed=True)
assert len(smoke["case_matrix"]) == 167
assert sum(r["status"] == "passed" for r in smoke["case_matrix"]) == 94
assert sum(r["status"] == "failed" for r in smoke["case_matrix"]) == 73

ncnc_gate = load("qualification/NCNC_RUNTIME_V6_QUALIFICATION_07.json")
ncnc = load("qualification/runtime_link_v6_native_lifecycle_driver_v1/fetched_attempt01/RESULT.json")
ncnc_cost = load("qualification/runtime_link_v6_native_lifecycle_driver_v1/fetched_attempt01/COST.json")
buddy_gate = load("qualification/BUDDY_PRODUCTION_SEED1_QUALIFICATION_07.json")
buddy = load("qualification/buddy_production_conditional_seed1_cpu_v2/fetched_attempt01/RESULT.json")
mol_gate = load("qualification/MOLECULAR_RUNTIME_V4_QUALIFICATION_09.json")
mol_lifecycle = load(mol_gate["local_result_path"])
for receipt in (ncnc_gate, buddy_gate):
    for authority in receipt["files"]:
        key = erel(authority["path"])
        if key in READS:
            assert READS[key]["bytes"] == authority["bytes"]
            assert READS[key]["sha256"] == authority["sha256"]
assert READS[erel(mol_gate["local_result_path"])]["sha256"] == mol_gate["remote_result"]["sha256"]

# Read frozen text for evidence pins; do not import the native implementations.
link_epoch = read(erel("link_native_integration_v1/epoch.py"), text=True)
ogb_adapter = read(erel("runtime_v4/adapters.py"), text=True)
qualifier = read(erel("qualification/runtime_link_v6_native_lifecycle_driver_v1/qualify.py"), text=True)
assert 'batch = 65536' in link_epoch and 'batch = 1024' in link_epoch
assert 'DataLoader(self.session.train_view, batch_size=32' in ogb_adapter
assert 'if batch.x.shape[0] == 1 or int(batch.batch[-1]) == 0:' in ogb_adapter
assert "'elapsed_seconds':time.perf_counter()-started" in qualifier

native_smokes = {
    r["recipe_id"]: r for r in smoke["attempts"]
    if r["selected_matrix_attempt"] and r["procedure"] == "native_single"
}
assert len(native_smokes) == 27
dataset_source_counts = {}
dataset_count_witnesses = {}
for dataset, count in (("ogbg-molbace", 1210), ("ogbg-molhiv", 32901)):
    rows = [r for r in native_smokes.values() if r["dataset"] == dataset]
    witnesses = [r for r in rows if isinstance(r.get("source_metadata"), dict)]
    assert len(rows) == 8 and witnesses
    assert all(r["source_metadata"]["TRAIN_graphs"] == count for r in witnesses)
    dataset_source_counts[dataset] = count
    dataset_count_witnesses[dataset] = {
        "TRAIN_graphs": count,
        "native_presets_with_structured_source_metadata": len(witnesses),
        "native_presets_with_legacy_string_metadata_not_used": len(rows) - len(witnesses),
        "authority_cases": [r["case_id"] for r in witnesses],
        "scope": "Same official dataset/split cardinality across frozen source presets; not a claim that every old source_metadata field was structured or that every current method executed.",
    }
assert ncnc["epoch_receipts"][0]["native_layout"]["optimizer_maps_per_epoch"] == 17
assert ncnc["epoch_receipts"][0]["native_layout"]["source_iterator_records_per_epoch"] + ncnc["epoch_receipts"][0]["native_layout"]["dropped_positive_tail"] == 1179052
assert buddy["native_TRAIN_positive_records"] == 811405
assert buddy["source_joint_batch"] == 1024

workloads = []
for cell in roster["cells"]:
    spec = settings[cell["id"]]
    matched = spec["matched_controller"]
    source = spec["source_reference_controller"]
    mf = cell["matched_native_body_fits"] + cell["shared_method_fits"]
    if cell["task_family"] == "node":
        steps, law = 1, "one complete native TRAIN full-graph optimizer map per epoch"
        source_fixed = False
    elif cell["dataset"] in ("ogbg-molbace", "ogbg-molhiv"):
        count = dataset_source_counts[cell["dataset"]]
        steps = (count + 31) // 32
        assert count % 32 > 1
        law = f"ceil({count}/32)={steps}; final batch {count % 32}; native singleton skip law retained"
        source_fixed = True
    elif cell["id"] == "collab-buddy":
        queries = 2 * buddy["native_TRAIN_positive_records"]
        steps = (queries + 1023) // 1024
        law = f"ceil(({buddy['native_TRAIN_positive_records']} positive + same negative)/1024)={steps}; partial tail retained"
        source_fixed = True
    else:
        steps = ncnc["epoch_receipts"][0]["native_layout"]["optimizer_maps_per_epoch"]
        law = "floor(1179052/65536)=17; native positive tail 64940 dropped, one sampled negative per included positive"
        source_fixed = True
    assert source_fixed == (source["native_stopping"].get("patience") is None)
    minimum = matched["stopping"]["minimum_units"]
    cap = matched["maximum"]
    src_cap = source["maximum"]
    workloads.append({
        "preset": cell["id"], "dataset": cell["dataset"], "task_family": cell["task_family"],
        "cohorts": cell["cohorts"], "source_reference_fits": cell["source_reference_fits"],
        "independent_native_body_fits": cell["matched_native_body_fits"],
        "shared_fits": cell["shared_method_fits"], "total_fits": cell["total_fits"],
        "shared_recipes_per_cohort": len(cell["shared_method_ids"]),
        "source_epoch_cap": src_cap, "source_cap_is_fixed_successful_horizon": source_fixed,
        "matched_epoch_minimum": minimum, "matched_epoch_cap": cap,
        "optimizer_maps_per_epoch": steps, "epoch_accounting": law,
        "matched_minimum_optimizer_map_obligation": mf * minimum * steps,
        "source_fixed_optimizer_maps": cell["source_reference_fits"] * src_cap * steps if source_fixed else None,
        "source_cap_plus_matched_minimum_optimizer_maps": (cell["source_reference_fits"] * src_cap + mf * minimum) * steps,
        "all_caps_optimizer_maps": (cell["source_reference_fits"] * src_cap + mf * cap) * steps,
        "measured_target_fit_seconds": None,
    })

aggregates = []
for dataset in sorted({r["dataset"] for r in workloads}):
    rows = [r for r in workloads if r["dataset"] == dataset]
    aggregates.append({"dataset": dataset, "presets": len(rows), **{
        key: sum(r[key] for r in rows) for key in (
            "total_fits", "source_reference_fits", "independent_native_body_fits", "shared_fits",
            "matched_minimum_optimizer_map_obligation", "source_cap_plus_matched_minimum_optimizer_maps", "all_caps_optimizer_maps")
    }})

pilot_groups = defaultdict(list)
for row in mol["fits"]:
    pilot_groups[row["procedure"]].append(row)
measured_fits = {key: fit_summary(rows) for key, rows in sorted(pilot_groups.items())}
bn_rows = [{"trained_epochs": r["scalars"]["completed_epochs"],
            "invocation_wall_seconds": r["scalars"]["invocation_wall_seconds"],
            "optimizer_calls": r["nested"]["optimizer_calls"], "resources": r["nested"]["resources"]} for r in bn["rows"]]
assert len(bn_rows) == 3
measured_fits["gnnm_private_bn_state_v3"] = fit_summary(bn_rows)
assert sum(r["trained_epochs"] for r in mol["fits"]) == 3680
assert sum(sum(r["optimizer_calls"]["completed"]) for r in mol["fits"]) == 139840
assert sum(r["trained_epochs"] for r in bn_rows) == 464
assert sum(sum(r["optimizer_calls"]["completed"]) for r in bn_rows) == 17632

intervals = []
for epoch in (1, 2):
    times = [r["elapsed_seconds"] for r in ncnc["update_receipts"] if r["native_epoch"] == epoch]
    assert len(times) == 17
    intervals.extend(b - a for a, b in zip(times, times[1:]))

def invocation_costs(raw):
    return [{"name": r["name"], **{
        k: r["result"].get(k) for k in (
            "invocation_wall_seconds", "construction_seconds", "optimizer_calls_this_invocation",
            "resume_requested", "finalization_only_resume")
    }} for r in raw["invocations"]]

engineering = {
    "NCNC_conditional_interaction_CUDA": {
        "receipt": "qualification/NCNC_RUNTIME_V6_QUALIFICATION_07.json",
        "supervisor_wall_seconds": ncnc_gate["wall_seconds"], "child_wall_seconds": ncnc["elapsed_seconds"],
        "complete_epochs": 2, "optimizer_maps": 34, "whole_VALID_calls": 4,
        "constructions": 3, "selected_mean_exports": 2,
        "cuda_peak_allocated_bytes": ncnc["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": ncnc["cuda_peak_reserved_bytes"],
        "sampled_child_peak_GPU_bytes": ncnc_cost["observed_child_peak_GPU_bytes"],
        "sampled_child_peak_RSS_bytes": ncnc_cost["observed_child_peak_RSS_bytes"],
        "child_own_RUsage_peak_RSS_KiB": ncnc["process_peak_RSS_KiB"],
        "invocations": invocation_costs(ncnc),
        "within_epoch_successive_update_end_intervals_seconds": stats(intervals),
        "interval_scope": "32 end-to-end intervals inside the two uninterrupted epochs; includes observer/instrumentation work. First updates, construction, validation/export and restore are excluded. This is not a pure kernel or mature target-epoch estimate.",
        "limitations": "Full two-epoch engineering lifecycle only; no convergence, scientific fit, arbitrary-method resource dominance or pair-throughput claim.",
    },
    "BUDDY_seed1_production_conditional_CPU": {
        "receipt": "qualification/BUDDY_PRODUCTION_SEED1_QUALIFICATION_07.json",
        "supervisor_wall_seconds": buddy_gate["wall_seconds"], "child_wall_seconds": buddy["elapsed_seconds"],
        "optimizer_maps": 2, "whole_VALID_calls": 3, "constructions": 2, "forwards": 32,
        "process_peak_RSS_KiB": buddy["process_peak_RSS_KiB"],
        "checkpoint_bytes": buddy["checkpoint"]["bytes"], "final_checkpoint_bytes": buddy["final_checkpoint"]["bytes"],
        "complete_epochs": 0, "GPU_time_seconds": None,
        "limitations": "CPU provider+model construction, hydration, checkpoint and full VALID overhead; wall/2 is not an epoch/update/GPU coefficient. No selected-runtime or whole-epoch evidence.",
    },
    "molecular_v4_conditional_interaction_CPU": {
        "receipt": "qualification/MOLECULAR_RUNTIME_V4_QUALIFICATION_09.json",
        "child_wall_seconds": mol_lifecycle["elapsed_seconds"],
        "complete_epochs": 2, "optimizer_maps": 76, "whole_VALID_calls": 4,
        "constructions": 3, "selected_mean_exports": 2,
        "process_peak_RSS_KiB": mol_lifecycle["process_peak_RSS_KiB"],
        "checkpoint_bytes": mol_lifecycle["selected_checkpoint"]["bytes"],
        "recovery_checkpoint_bytes": mol_lifecycle["recovery_checkpoint"]["bytes"],
        "selected_VALID_export_bytes": mol_lifecycle["selected_VALID_export"]["bytes"],
        "observed_own_output_disk_bytes_including_engineering_snapshots": mol_lifecycle["own_output_disk_bytes"],
        "resource_observations": mol_lifecycle["resource_observations"],
        "invocations": invocation_costs(mol_lifecycle),
        "limitations": "Fresh current v4 CPU lifecycle. Not GPU cost, MolHIV full epoch, mature campaign output-disk peak or concurrency evidence. Allocation-wide cache/headroom is not process-owned RAM.",
    },
}

smoke_rows = []
for preset in sorted(settings):
    rows = [r for r in smoke["case_matrix"] if r["recipe_id"] == preset]
    assert rows
    # Current I4 consists of separately trained physical bodies. Old simultaneous
    # independent4 allocation is kept separately and never a native-body bound.
    eligible = [r for r in rows if r["procedure"] != "independent4"]
    smoke_rows.append({
        "preset": preset, "selected_cases": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "non_independent4_cases": len(eligible),
        "non_independent4_maxima": {
            key: max((r[key] for r in eligible if r.get(key) is not None), default=None)
            for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "peak_rss_bytes", "checkpoint_bytes")
        },
        "selected_case_wall_seconds": stats(r["wall_seconds"] for r in rows),
        "cases": [{k: r.get(k) for k in (
            "case_id", "procedure", "status", "wall_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
            "peak_rss_bytes", "checkpoint_bytes", "actual_optimizer_completed", "functional_completion", "functional_complete_VALID")}
            for r in rows],
    })

partial = []
# These are the only 60 intentions with the pre-existing catalogue's exact-setting
# scheduling priors. Keep new BN evidence separate and old residual nonportable.
counts = {"native_single": 40, "gnnm_boundary_native": 10, "task_geometry_alignment_variance": 10}
assert set(counts) == {r["scientific_method_id"] for r in catalogue["case_classes"]}
for horizon in (100, 149, 1000):
    totals = {bound: sum(counts[r["scientific_method_id"]] * horizon *
                        r["measured_invocation_wall_per_completed_epoch_seconds"][bound]
                        for r in catalogue["case_classes"]) for bound in ("min", "median", "max")}
    partial.append({"common_assumed_completed_epochs_per_fit": horizon,
                    "conditional_worker_invocation_wall_sum_seconds": totals,
                    "conditional_worker_invocation_wall_sum_hours": {k: v / 3600 for k, v in totals.items()}})

scenarios = {
    "schema": "expanded-metadata-workload-scenarios-v1", "campaign_ETA_seconds": None,
    "full_campaign_GO": False, "no_new_execution_authorized": True,
    "work_count_unit": "one source-sized optimizer transition; not FLOPs, route forwards, GPU seconds or a convergence claim",
    "counts": {"physical_fits": 6460, "source_reference_fits": 270, "independent_native_bodies": 1080,
               "shared_fits": 5110, "I4_no_training_exports": 270, "extra_M1_fits": 0, "presets": 27},
    "roster_changed": False,
    "dataset_workloads": aggregates,
    "matched_minimum_optimizer_map_obligation": sum(r["matched_minimum_optimizer_map_obligation"] for r in workloads),
    "source_cap_plus_matched_minimum_optimizer_maps": sum(r["source_cap_plus_matched_minimum_optimizer_maps"] for r in workloads),
    "all_caps_optimizer_maps": sum(r["all_caps_optimizer_maps"] for r in workloads),
    "scenario_boundaries": [
        "Source-cap + matched-minimum is a declared work scenario, not an exact earliest-stop lower bound; node source references have native patience250 and may stop earlier.",
        "Matched minimum is an obligatory nominal horizon, not evidence that a fit stops there. Real molecular pilot fits stopped at149-210 epochs.",
        "All caps is a trajectory-policy work ceiling conditional on successful completion without repeats/recovery. Replays, partial-epoch restarts, engineering checks, construction, evaluation and export add cost.",
        "A shared optimizer map includes four routes and method-specific reverse collections; node activation checkpoint recomputation adds further work. Native maps and shared maps are not equal FLOP units.",
        "No TEST forward/scoring or source RNG diagnostic handling has been assigned a timing coefficient. These exact reporting branches remain distinct.",
    ],
    "MolHIV_to_MolBACE_optimizer_maps_per_epoch_ratio": 1029 / 38,
    "ratio_is_runtime_ratio": False,
    "MolBACE_60_fit_conditional_time_prior": {
        "preset": "ogb-ogbg-molbace-gin-drop0.5", "fit_counts": counts, "fit_count": 60,
        "source_reference_fits_included": 0, "target_fit_measurement": False,
        "assumptions": [
            "Each target fit has its stated common illustrative horizon, regardless of actual patience outcome.",
            "Per-epoch invocation-wall averages from the same preset/procedure pilot persist despite changed target runtime/reporting generation and seeds.",
            "Current hardware, contention and data-cache conditions match the observed pilot; the pilot shared its GPU with an older training worker.",
            "The min/median/max use empirical per-epoch extrema, not confidence intervals or guaranteed bounds; construction/selection are amortized as in observed completed fits.",
            "No new native tests or target measurements support this calculation. No transfer to other OGB architectures, dropout settings, MolHIV or BUDDY is made.",
        ],
        "scenarios": partial,
        "three_GPU_or_two_jobs_speedup_assumed": False,
    },
    "recommendation_only_staged_user_approval": {
        "changes_roster_or_heldout_protocol": False,
        "stage0": "Close only remaining finite source/controller/resource/provider/lifecycle obligations, reusing passed evidence; no per-seed qualification matrix.",
        "stage1": {"scope": "One frozen cohort for every native preset and every declared recipe; all settings represented before remaining cohorts.",
                   "physical_fits": 646, "source_references": 27, "independent_native_bodies": 108,
                   "shared_fits": 511, "I4_no_training_exports": 27,
                   "permission": "User review and explicit campaign-stage GO required; no launch from this recommendation."},
        "stage2": {"scope": "Remaining nine frozen cohorts for the same complete roster, as separately approved.",
                   "physical_fits": 5814, "I4_no_training_exports": 243},
        "interpretation": "Staging does not make incomplete coverage confirmatory or authorize outcome-driven pruning. Any later exploratory redesign must retain failed outcomes and require a revised frozen protocol before heldout scoring.",
    },
}
assert scenarios["source_cap_plus_matched_minimum_optimizer_maps"] == 247526000
assert scenarios["all_caps_optimizer_maps"] == 2364025000

coverage = {
    "schema": "expanded-resource-coverage-budget-v1", "campaign_ETA_seconds": None,
    "full_campaign_resource_admission": False, "three_GPU_portability_qualified": False,
    "two_workers_per_GPU_qualified": False,
    "readiness_obligations_snapshot": remaining["counts"],
    "later_passed_receipts_attached": ["NCNC_RUNTIME_V6_QUALIFICATION_07", "BUDDY_PRODUCTION_SEED1_QUALIFICATION_07", "MOLECULAR_RUNTIME_V4_QUALIFICATION_09"],
    "reusable_evidence": [
        {"class": "completed_MolBACE_scientific_fits", "coverage": "24 fits,3680epochs,139840maps and BN3 464epochs,17632maps under original pilot runtimes; no repeats required", "limits": "One native GIN/dropout setting; allocator/RUsage observations and scheduling priors, no mature process/cache/disk/pair envelope"},
        {"class": "node_native_shapes", "coverage": "Existing selected full-graph smoke on all8presets with original source/optimizer and recorded allocated/reserved/RSS/checkpoint sizes", "limits": "Short measured source/operator shape witnesses; attach current-source equivalence and required component branch cover. Do not rerun8presets by default or infer every method resource dominance."},
        {"class": "NCNC_current_CUDA_lifecycle", "coverage": "Passed2nativeepochs/34maps/4VALID/3constructions/2exports for input+conditional-J", "limits": "Reuse it. Other native/method branches require evidence attachment, not duplicate NCNC lifecycle."},
        {"class": "molecular_v4_current_CPU_lifecycle", "coverage": "Passed2MolBACEepochs/76maps/4VALID/3constructions/2exports, explicit disk and process observations", "limits": "Reuse it. This is not a MolHIV CUDA whole-epoch or mature resource profile."},
        {"class": "BUDDY_production_seed1_CPU", "coverage": "Passed2maps/3VALID/2constructions and authentic compact production provider", "limits": "Two maps, no complete native epoch or GPU selected-runtime lifecycle"},
    ],
    "smallest_remaining_resource_workload_recommendations": [
        {"id": "HIV_whole_epoch_current_GPU", "new_duplicate_scientific_fit": False,
         "workload": "Attach resource observation to the already-needed MolHIV current-source whole-epoch lifecycle on a documented heavy native branch (candidate GIN-VN).",
         "why": "32901TRAIN/4113VALID graphs and1029maps per epoch; only first-batch smoke and BACE completed-fit data exist here.",
         "dominance_required": "Bound padded/native graph batch nodes/edges and virtual-node/operator work against all molecular presets; dataset size alone does not prove peak batch-memory dominance. Retain any uncovered GCN operator/branch as explicit unknown before adding only a necessary witness."},
        {"id": "BUDDY_GPU_production_whole_epoch", "new_duplicate_scientific_fit": False,
         "workload": "Attach process/cache/checkpoint/export observations to the already-needed current-runtime compact-provider whole-epoch GPU lifecycle.",
         "why": "1585mixed-query maps per epoch. CPU2map gate and legacy GPU smoke do not measure current production whole-epoch runtime or late cache/provider growth."},
        {"id": "disk_process_and_cache_cover", "new_duplicate_scientific_fit": False,
         "workload": "Reuse existing file-size receipts and NCNC sampled VRAM plus current molecular lifecycle; close only missing mature process/cache/atomic-checkpoint/export peak branches during necessary gates.",
         "why": "Allocator reserve excludes CUDA context/operator allocations; own RSS excludes shared file cache. Snapshot bytes are engineering footprint, not campaign peak. Cgroup headroom is shared and must be rechecked at admission."},
        {"id": "optional_reciprocal_concurrency", "required_for_serial_release": False,
         "workload": "Only a finite compatible-pair cover after single-process resource envelopes and shared headroom are accepted; reciprocal measurement must preserve source work and record both throughput and process/cache/disk peaks.",
         "why": "Two jobs per GPU has no demonstrated2x speedup, and40GiB+40GiB allocator arithmetic is not safe process admission. Serial one-worker release remains valid."},
    ],
    "unknown_costs": [
        "Mature current-runtime time for most of6460physical fits and all270I4exports.",
        "MolHIV complete-epoch/per-fit GPU time, molecular native architecture/dropout transfer and target selection/reporting overhead.",
        "BUDDY compact production complete-epoch GPU time, cache/provider growth and checkpoint/export process peaks.",
        "Per-method backward/activation recomputation and native source TRAIN-diagnostic/RNG-emulation overhead on uncovered branches.",
        "Disk high-water marks across concurrent workers, atomic replacements, recovery histories, route exports and retained evidence.",
        "Actual completion horizons, failures, partial-epoch recovery costs and mature contention.",
        "18.77 hardware/runtime/dependencies/device resources and any cross-host or reciprocal pair throughput; no access attempted in this task.",
    ],
    "execution_authorized_by_this_package": False,
}

outputs = []
outputs.append(dump("MEASURED_COSTS.json", {
    "schema": "expanded-measured-cost-metadata-v1", "measurements_are_target_campaign_fits": False,
    "MolBACE24": {"fit_groups": {k: v for k, v in measured_fits.items() if k != "gnnm_private_bn_state_v3"},
                  "elapsed_pilot_seconds": mol["recorded_pilot_elapsed_seconds"],
                  "worker_invocation_wall_sum_seconds": catalogue["recorded_worker_wall_sum_seconds"],
                  "worker_wall_sum_is_not_elapsed": True,
                  "old_residual_and_combination_portability": "Original pilot algorithms; not portable_input_residual_v2 or new campaign interaction costs."},
    "BN3": {"fit_summary": measured_fits["gnnm_private_bn_state_v3"],
            "additional_to_MolBACE24": True, "target_campaign_fits_prepaid": False},
    "engineering_lifecycles": engineering,
    "dataset_count_witnesses": dataset_count_witnesses,
    "legacy_smoke": {"selected_cases": 167, "selected_passed": 94, "selected_failed": 73,
                     "actual_work_costs": smoke["actual_work_costs"],
                     "limitations": "Short full-check walls include construction/replay/VALID/checkpoint work.73floating-replay failures remain failures.Independent4simultaneous model peaks are not current separate-body requirements."},
    "TEST_or_tensor_files_read": False, "native_models_or_updates": False,
}))
outputs.append(dump("LEGACY_SMOKE_RESOURCE_SHAPES.json", {
    "schema": "expanded-legacy-resource-shapes-v1", "source": "reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/RESULTS.json.gz",
    "mature_resource_or_method_dominance": False, "rows": smoke_rows,
}))
outputs.append(dump("SCENARIOS.json", scenarios))
outputs.append(dump("RESOURCE_COVERAGE.json", coverage))
outputs.append(dump("PRESET_WORKLOADS.json", {"schema": "expanded-preset-workload-v1", "rows": workloads}))
stream = io.StringIO(newline="")
writer = csv.DictWriter(stream, fieldnames=list(workloads[0]))
writer.writeheader()
writer.writerows(workloads)
raw = stream.getvalue().encode()
(OUT / "PRESET_WORKLOADS.csv").write_bytes(raw)
outputs.append({"path": "PRESET_WORKLOADS.csv", "bytes": len(raw), "sha256": sha(raw)})
outputs.append(dump("INPUTS.json", {"schema": "resource-budget-input-pins-v1", "inputs": list(READS.values()),
                                  "BN3_remote_authorities_already_verified_in_extraction": [r["source"] for r in bn["rows"]]}))
dump("BUILD_RECEIPT.json", {
    "schema": "metadata-budget-build-receipt-v1", "builder": {"path": "build_budget.py", "sha256": sha(Path(__file__).read_bytes())},
    "human_report": {"path": "README.md", "sha256": sha((OUT / "README.md").read_bytes())},
    "outputs": outputs, "full_campaign_GO": False, "execution_or_remote_calls": 0,
    "native_or_tensor_library_imports": 0, "TEST_or_tensor_files_read": 0,
    "roster_fits": 6460, "presets": 27, "campaign_ETA_seconds": None,
})
print(json.dumps({"status": "built", "physical_fits": 6460,
                  "source_cap_plus_matched_minimum_maps": scenarios["source_cap_plus_matched_minimum_optimizer_maps"],
                  "all_caps_maps": scenarios["all_caps_optimizer_maps"],
                  "partial_60_fit_hours": [r["conditional_worker_invocation_wall_sum_hours"] for r in partial]}))
