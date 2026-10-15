#!/usr/bin/env python3
"""Reproducible conditional workload/storage arithmetic from saved metadata."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    E, pins = args.expansion, []
    def read(name):
        raw = (E / name).read_bytes()
        pins.append({"path_relative_to_expansion": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    def write(name, value):
        with (args.output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    workloads = read("resource_budget_draft_v1/PRESET_WORKLOADS.json")["rows"]
    assert len(workloads) == 27 and sum(r["total_fits"] for r in workloads) == 6460
    hiv = read("MOLHIV_RESOURCE_SUMMARY_17.json")
    hiv_raw = read("MOLHIV_RESOURCE_OBSERVATION_16.json")
    bace = read("expanded_launch_preparation_v1/compact_generated_v1/COST_CATALOGUE.json")
    sync = "planning/three_gpu_sync_20261015_v1/"
    queues = read(sync + "CAMPAIGN_DORMANT_ACTUAL_SUMMARY.json")
    old_disk = read(sync + "SERVER51_RESOURCE_ADMISSION.json")["snapshot"]
    native_storage = read(sync + "STORAGE_ALLOCATION_01.json")["storage_samples"][0]
    shapes = read("resource_budget_draft_v1/LEGACY_SMOKE_RESOURCE_SHAPES.json")
    buddy = read("BUDDY_LIFECYCLE_TERMINAL_22.json")
    ncnc = read("qualification/runtime_link_v6_native_lifecycle_driver_v1/fetched_attempt01/RESULT.json")
    epoch_seconds = hiv["phase_wall_seconds"]["whole_TRAIN_epoch"]
    hiv_rows = []
    for name, fits in [("one_measured_setting_shared_fit", 1), ("one_preset_one_cohort_19_shared_recipes", 19),
                       ("one_preset_full_190_shared_fits", 190), ("eight_presets_one_cohort_152_shared_fits", 152),
                       ("eight_presets_full_1520_shared_fits", 1520)]:
        for horizon in (100, 300, 1000):
            hours = fits * horizon * epoch_seconds / 3600
            hiv_rows.append({"scenario": name, "fits": fits, "epochs_each": horizon,
                             "TRAIN_device_hours": hours, "ideal_three_equal_GPU_days": hours / 72,
                             "scope": "same measured GIN-VN conditional-J four-route TRAIN seconds substituted for every included shared fit",
                             "architecture_and_method_substitution": fits > 1})
    bace_rates = {r["scientific_method_id"]: r["measured_invocation_wall_per_completed_epoch_seconds"]["median"] for r in bace["case_classes"]}
    bace_rows = []
    for horizon in (100, 300, 1000):
        # The native-reference track remains fixed at100.40 matched native body
        # fits and190 shared fits take the scenario's horizon, with no pruning.
        for method in ("gnnm_boundary_native", "task_geometry_alignment_variance"):
            hours = ((10 * 100 + 40 * horizon) * bace_rates["native_single"]
                     + 190 * horizon * bace_rates[method]) / 3600
            bace_rows.append({"scenario": "one_preset_240_fits", "matched_epochs": horizon,
                              "native_reference_epochs": 100, "shared_rate_proxy": method,
                              "worker_hours": hours, "ideal_three_equal_GPU_hours": hours / 3,
                              "one_cohort_24_fit_worker_hours": hours / 10,
                              "scope": "All shared recipes substituted with one observed old-pilot class; architecture/source/reporting equivalence not established"})
    node_rates = []
    for host in ("allocation", "server51"):
        before, after = read(sync + "RUNNING_" + host + "_02.json"), read(sync + "RUNNING_" + host + "_03.json")
        dt = (datetime.fromisoformat(after["UTC"]) - datetime.fromisoformat(before["UTC"])).total_seconds()
        old = {w["worker"]: sum(c["completed_epochs"] for c in w["curves"]) for w in before["workers"]}
        for worker in after["workers"]:
            delta = sum(c["completed_epochs"] for c in worker["curves"]) - old[worker["worker"]]
            seconds = dt / delta
            node_rates.append({"worker": worker["worker"], "window_seconds": dt, "completed_epoch_delta": delta,
                               "observed_seconds_per_native_epoch": seconds,
                               "conditional_2099_epoch_native_fit_minutes": 2099 * seconds / 60,
                               "conditional_10000_epoch_native_fit_minutes": 10000 * seconds / 60,
                               "scope": "PubMed-Cheb native-body pilot; server51 interval includes one fit boundary; current peer contention retained",
                               "shared_fit_or_other_dataset_transfer": False})
    server_fits = sum(w["fits"] for w in queues["workers"] if w["worker"].startswith("server51/"))
    raw_result = hiv_raw["terminal_RESULT"]
    files = {Path(r["receipt"]["path"]).name: r["receipt"]["bytes"] for r in raw_result["artifact_writes"]}
    retained_payload = files["selected.pt"] + files["recovery.pt"] + files["selected_VALID.pt"]
    native_files = dict(native_storage["files"])
    checkpoint_priors = []
    for r in shapes["rows"]:
        checkpoint_priors.append({"preset": r["preset"], "legacy_checkpoint_bytes": r["non_independent4_maxima"]["checkpoint_bytes"],
                                  "two_file_payload_bytes": 2 * r["non_independent4_maxima"]["checkpoint_bytes"],
                                  "source_and_method_equivalence_verified": False})
    link_rows = []
    for name, observed_seconds, observed_epochs, shared_fits in (
            ("BUDDY_current_combined_lifecycle", buddy["supervisor"]["child_wall_seconds"], buddy["qualification"]["TRAIN_epochs_completed"], 190),
            ("NCNC_v6_lifecycle_prior", ncnc["elapsed_seconds"], ncnc["TRAIN_epochs_completed"], 180)):
        for horizon in (100, 300, 1000):
            hours = observed_seconds / observed_epochs * horizon / 3600
            link_rows.append({"profile": name, "observed_lifecycle_seconds": observed_seconds,
                              "observed_complete_TRAIN_epochs": observed_epochs,
                              "conditional_epochs_each": horizon, "one_fit_worker_hours": hours,
                              "preset_shared_fits": shared_fits, "preset_shared_worker_hours": shared_fits * hours,
                              "ideal_three_equal_GPU_days": shared_fits * hours / 72,
                              "scope": "Repeat the observed two-epoch lifecycle cadence, including its extra constructors, VALID, restores and exports; not a mature production epoch measurement or all-method equivalence"})
    scenarios = {"schema": "conditional-campaign-workload-slices-v1", "campaign_fits": 6460,
                 "exports": 270, "server51_fits": server_fits, "full_campaign_ETA": None,
                 "optional_tranche": {"fits": 646, "cohorts_per_preset": 1, "presets": 27,
                     "status": "proposed only; no authorization, pruning or task selection introduced"},
                 "MolHIV_observed": {"TRAIN_epoch_seconds": epoch_seconds, "updates": hiv["updates_completed"],
                     "TRAIN_seconds_per_update": epoch_seconds / hiv["updates_completed"],
                     "whole_invocation_seconds": hiv["elapsed_seconds"],
                     "configuration": "GIN-VN/drop0.5 conditional input correction shared bank, width300/layers5, allocation CUDA,1029maps/4route forwards per map",
                     "limits": "One full epoch.100/300/1000-epoch costs are linear TRAIN-only scenarios; VALID, finalization, diagnostics, retries and contention excluded. Other methods/architectures and server51 speed are not measured."},
                 "MolHIV_scenarios": hiv_rows, "MolBACE_scenarios": bace_rows, "PubMed_current_rates": node_rates,
                 "link_lifecycle_cadence_scenarios": link_rows,
                 "unmeasured_families": ["Current complete-fit BUDDY/NCN/NCNC mature rates", "CiteSeer/CS/Physics mature rates", "Other molecular architectures and normalized method branches"],
                 "two_job_speedup_assumed": False,
                 "ideal_three_GPU_interpretation": "Arithmetic division by3 only under equal speed, full occupancy and no contention; not measured throughput or promised finish time"}
    storage = {"schema": "incremental-storage-budget-v1", "server51_fits": server_fits,
               "historical_free_bytes_20261015T202839Z": old_disk["free_disk_bytes"],
               "later_parent_observation": {"free_bytes_approx": 700886000000, "status": "parent reported user freed space; require fresh exact bytes at admission"},
               "core_dependencies_bytes": 16346414909,
               "checkpoint_semantics": "selected.pt and recovery.pt each include model, optimizer, RNG, selector, validation and runtime binding; atomic save may hold one extra full checkpoint",
               "dense_FP32_Adam_estimate": {"one_checkpoint_bytes_per_parameter": 12, "retained_two_bytes_per_parameter": 24,
                     "atomic_extra_bytes_per_parameter": 12, "exclude": "buffers, private bank state, non-FP32 tensors, aliases/duplicates, RNG, VALID outputs, logs; measure actual serialized files"},
               "observed_MolHIV_shared": {"checkpoint_bytes_each": files["selected.pt"], "selected_VALID_bytes": files["selected_VALID.pt"],
                     "two_checkpoint_and_export_bytes": retained_payload, "engineering_directory_bytes": raw_result["own_output_disk_bytes_final_observation"],
                     "one_extra_checkpoint_peak_increment_bytes": files["selected.pt"],
                     "all_server51_same_payload_scenario_bytes": retained_payload * server_fits,
                     "all_server51_same_engineering_directory_scenario_bytes": raw_result["own_output_disk_bytes_final_observation"] * server_fits,
                     "is_campaign_estimate": False},
               "observed_PubMed_native_in_progress": {"directory_bytes": native_storage["total_bytes"],
                     "checkpoint_bytes_each": native_files["selected.pt"], "curve_bytes": native_files["CURVE.jsonl"],
                     "selected_VALID_not_yet_observed": True},
               "legacy_checkpoint_priors": checkpoint_priors,
               "VALID_export_formula": "Stored dtype bytes times N times stored route/class outputs, plus IDs, labels, endpoints, optional same-bank means and metadata. Infer from shape/dtype or actual serialized bytes, never universal fixed logits bytes.",
               "policy": {"default_floor_bytes": 20 * 1024**3, "retained_cap_bytes": "explicit host budget based on fresh free space and peer growth; no fixed reservation for all4306fits",
                    "admission_formula": "free >= floor + not-yet-written transfer bytes + transfer temporary overlap + peer growth allowance + sum(active and next fit incremental peak growth)",
                    "retention_formula": "current campaign retained bytes + active/candidate remaining retained growth <= approved retained cap",
                    "checks": ["before transfer", "before each new claim", "at complete-epoch/checkpoint boundaries", "before export"],
                    "low_water_action": "Pause new claims/transfers, preserve active checkpoint headroom, notify root; never auto-delete checkpoints or peers",
                    "unknown_profile": "Measure/bound one profile before capacity pass; unknown is not zero",
                    "archive_action": "Separate explicit retention/verified-transfer policy required before reclaiming files",
                    "scope": "admission.py makes a stateless recommendation; frozen pilots and active dispatchers are not patched"}}
    args.output.mkdir(exist_ok=False)
    write("TIME_SLICES.json", scenarios)
    write("STORAGE_BUDGET.json", storage)
    write("INPUT_PINS.json", {"files": pins})
    print(json.dumps({"server51_fits": server_fits, "MolHIV_100epoch_TRAIN_hours": epoch_seconds / 36,
                      "MolHIV_1520shared_100epoch_TRAIN_hours": 1520 * epoch_seconds / 36,
                      "same_MolHIV_payload_4306fits_bytes": retained_payload * server_fits,
                      "PubMed_rates": node_rates}, sort_keys=True))


if __name__ == "__main__":
    main()
