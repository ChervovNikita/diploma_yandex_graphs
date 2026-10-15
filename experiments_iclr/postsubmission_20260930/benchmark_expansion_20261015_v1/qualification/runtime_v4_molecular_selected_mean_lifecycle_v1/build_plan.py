#!/usr/bin/env python3
"""Freeze only source and existing metadata. Never import the native qualifier."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
P = E.parent
ROOT = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
PHASE = ROOT + "/experiments_iclr/postsubmission_20260930"
REMOTE_E = PHASE + "/" + E.name


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def receipt(path, remote=False):
    path = Path(path)
    assert path.suffix in {".py", ".json", ".md"}, path
    raw = path.read_bytes()
    name = str(path.resolve()) if not remote else PHASE + "/" + str(path.relative_to(P))
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def metadata(name):
    return json.loads((E / name).read_text())


original = metadata("qualification/interaction_node_molecular_v1_cpu_v1/SOURCE_DESCRIPTORS.json")["molecular_job"]
native_inputs = metadata("qualification/runtime_v3_cpu_full_epoch_v1/INPUTS.json")
base = copy.deepcopy(original)
base.update(id="runtime_v4/molecular_selected_mean_lifecycle/interaction_input_conditionalJ/seed101/split0",
    cohort_id="runtime_v4_molecular_selected_mean_lifecycle_v1/engineering", procedure="interaction_input_conditionalJ",
    base_procedure="gnnm_boundary_native", seed=101, split=0, engineering_only=True, TEST_access=False,
    torch_threads=2, session_transform={"path": REMOTE_E + "/interaction_node_molecular_v1/extension.py", "function": "transform",
        "sha256": receipt(E / "interaction_node_molecular_v1/extension.py")["sha256"]},
    method_settings={"manifest_sha256": receipt(E / "interaction_node_molecular_v1/MANIFEST.json")["sha256"]})
base["recipe"]["_repository_root"] = ROOT
base["recipe"]["_seed"] = 101
base["recipe"]["GO"] = False
base["policy"] = {"track": "matched_pilot", "metric": "rocauc", "direction": "max", "ties": "first",
    "maximum_epochs": 2, "minimum_epochs": 2, "patience": 1, "evaluate_TRAIN_each_epoch": False}
for device, file in (("cpu", "JOB_CPU.json"), ("cuda:0", "JOB_CUDA.json")):
    job = copy.deepcopy(base)
    job.update(device=device, execution_mode="cpu_engineering_v1" if device == "cpu" else "production_gpu")
    write(file, job)
assert base["recipe"]["native_fixed_config"] == original["recipe"]["native_fixed_config"]
native_config = {"JK": "last", "batch_size": 32, "dataset": "ogbg-molbace", "drop_ratio": 0.5,
                 "emb_dim": 300, "feature": "full", "gnn": "gin", "graph_pooling": "mean", "lr": 0.001,
                 "num_layer": 5, "residual": False}
GIB = 1024 ** 3
limits = {"TRAIN_epochs": 2, "TRAIN_records_per_epoch": 1210, "raw_batches_per_epoch": 38,
          "batch_size": 32, "TRAIN_tail_graphs": 26, "source_skipped_batches": 0,
          "optimizer_attempts": 76, "optimizer_completed": 76, "native_backward_calls": 76,
          "native_update_calls_started_including_exhaustion": 78, "epoch_exhaustion_calls": 2,
          "native_constructions": 3, "whole_native_VALID_calls": 4, "VALID_records_per_call": 151,
          "same_call_mean_diagnostics": 4, "selected_mean_exports": 2, "adapter_state_restores": 4,
          "global_RNG_restores": 4, "TRAIN_diagnostic_calls": 0, "TEST_calls": 0,
          "duplicate_uninterrupted_trajectories": 0, "wall_watchdog_seconds": 600}
envelopes = {d: {"Torch_threads": 2, "Torch_interop_threads": 1, "declared_peak_RSS_reservation_bytes": 4 * GIB,
                "minimum_conservative_available_bytes": 6 * GIB, "disk_reservation_bytes": GIB // 2,
                "declared_GPU_reservation_bytes": 0 if d == "cpu" else 8 * GIB,
                "minimum_GPU_free_bytes": 0 if d == "cpu" else 10 * GIB,
                "measured_by_this_source_plan": False} for d in ("cpu", "cuda:0")}
redesign = P / "reference_wrapper_redesign_20261013_v1"
deps = [redesign / "runtime/training_rules.py", redesign / "runtime/boundary_core.py"]
deps += [redesign / "smoke_20261014/ogb_port" / n for n in ("session.py", "native_loader.py", "categorical_boundary.py")]
deps += [redesign / "native_sources/ogb" / n for n in ("gnn.py", "conv.py", "main_pyg.py")]
inputs = {"schema": "runtime-v4-molecular-selected-mean-lifecycle-inputs-v1", "repository_root": ROOT,
    "runtime_manifest_sha256": receipt(E / "runtime_v4/MANIFEST.json")["sha256"],
    "interaction_manifest_sha256": receipt(E / "interaction_node_molecular_v1/MANIFEST.json")["sha256"],
    "native_smoke_manifest": receipt(redesign / "smoke_20261014/MANIFEST.json", True),
    "data_files": native_inputs["data_files"], "data_receipts_inherited_not_refreshed": True,
    "data_tensors_opened_in_preparation": 0, "native_session_config": native_config,
    "native_dependencies": [receipt(p, True) for p in deps],
    "qualifier_sources": [receipt(HERE / name, True) for name in ("qualify.py", "resource_sampler.py")],
    "job_sha256": {d: receipt(HERE / f)["sha256"] for d, f in (("cpu", "JOB_CPU.json"), ("cuda:0", "JOB_CUDA.json"))},
    "exclusive_output": REMOTE_E + "/qualification/runtime_v4_molecular_selected_mean_lifecycle_attempt01",
    "exclusive_choice_claim": REMOTE_E + "/qualification/runtime_v4_molecular_selected_mean_lifecycle_choice_v1.json",
    "bounds": limits, "resource_envelopes": envelopes, "no_automatic_retry": True,
    "root_native_admission": "pending review; not launched by this source task"}
write("INPUTS.json", inputs)

pilot = E / "pilot_20261015_bn_geometry_correction_v1/session_extension.py"
pilot_tree = ast.parse(pilot.read_text())
pilot_literals = {n.targets[0].id: ast.literal_eval(n.value) for n in pilot_tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"METHODS", "PINS", "NATIVE_CONFIG"}}
notes = {"schema": "molecular-v4-upcoming-BN-pilot-reuse-source-audit-v1", "pilot_extension": receipt(pilot, True),
    "pilot_methods": pilot_literals["METHODS"], "native_config_matches": pilot_literals["NATIVE_CONFIG"] == native_config,
    "observed_frozen_pilot_SPEC_at_preparation": (pilot.parent / "SPEC.json").exists(),
    "source_findings": ["Composition wraps old completed molecular ingredient transform and private BN v3; native objective/BN ports remain unchanged.",
      "Its methods are geometry/correction/combined with private BN, not input+conditional-J.",
      "This composition source exposes no interaction_metadata or conditional_aggregation_metadata and no same_bank_mean_* VALID fields; runtime v4 cannot invent a gated mean sidecar from it."],
    "usable_future_evidence": ["If the pilot is frozen on exact runtime v4 and source/dependency bindings, its actual whole epochs/own selected exports can cover molecular training/finalization and these composed BN/ingredient state schemas within that source scope.",
      "Fresh selected exports can be reconstructed later from the exact saved payloads without new scientific fits; preserve full native and BN state, source/JOB/runtime/checkpoint/epoch/IDs/labels provenance.",
      "If it naturally performs a complete-epoch or stopped-finalization resume, that exact observed branch can be attached rather than duplicated; a fresh fit alone does not prove resume."],
    "cannot_replace_this_gate_without_new_frozen_protocol": ["Gated actual+same-call-mean selected exporter", "Input+conditional-J residual/scorer checkpoint schema", "Controlled completed-epoch reporting fault", "Four exact state/RNG restores and stopped0step finalizer unless actually exercised"],
    "avoid_duplicate_work": "Do not rerun completed BN or geometry/correction science for readiness. This gate targets the uncovered gated-mean finalizer only. If an upcoming necessary pilot is prospectively bound to the exact same runtime/protocol and lifecycle before freeze, root can supersede this unlaunched gate in a fresh plan; no source or serving rule is changed here.",
    "native_models": 0, "native_forwards": 0, "optimizer_updates": 0, "tensor_reads": 0}
write("PILOT_REUSE.json", notes)
plan = {"schema": "runtime-v4-molecular-selected-mean-finite-plan-v1", "status": "frozen source; native execution pending root review",
    "published_parent": "c58a59ffe00c75e81082973139234a41b907c09b", "preferred_device": "cpu",
    "alternative_device": "cuda:0", "alternatives_are_mutually_exclusive": True,
    "question": "Does unchanged runtime_v4 molecular full-epoch code authenticate its actual extension-selected same-call mean through completed-epoch resume and stopped finalization?",
    "method": "interaction_input_conditionalJ", "native_config": native_config,
    "only_schedule_change": "New engineering descriptor min=max2epochs,patience1; original native Adam/LR/WD/dropout/operators/batches/source reductions unchanged. Not a production/source-reference policy or scientific convergence check.",
    "sequence": [{"invocation": 1, "fresh_native_constructions": 1, "TRAIN_epochs": 1, "optimizer_updates": 38,
                  "whole_VALID_calls": 1, "selected_mean_exports": 0, "outcome": "Expected controlled failure after durable epoch1recovery/CURVE/PROGRESS; reporting must retain38steps/1epoch"},
                 {"invocation": 2, "fresh_native_constructions": 1, "TRAIN_epochs": 1, "optimizer_updates": 38,
                  "whole_VALID_calls": 2, "selected_mean_exports": 1, "outcome": "Resume exact epoch1JOB/policy/recovery; train epoch2; actual first strict best selects epoch1or2; load own selected payload and export actual+same-call mean"},
                 {"invocation": 3, "fresh_native_constructions": 1, "TRAIN_epochs": 0, "optimizer_updates": 0,
                  "whole_VALID_calls": 1, "selected_mean_exports": 1, "outcome": "Resume stopped epoch2recovery;0steps; load own same selected payload; repeat actual+mean export; checkpoint bytes unchanged"}],
    "exact_bounds": limits, "resource_envelopes": envelopes,
    "floating_policy": "All saved state/Adam/loader/wrapper/int/RNG/source authority exact on both alternatives. CPU repeated inference bit equal. CUDA one reexecution only: outputs/loss fixedrtol1e-4/atol1e-5, bit/RMS/maxabs reported, native rocauc exactly unchanged; no tolerance search, reselect or retry.",
    "resource_recording": ["Own process RSS high water and sampled current RSS", "Sampled process GPU memory plus allocator allocated/reserved peaks across invocation resets if CUDA chosen", "Allocation-wide cgroup current/peak/events and anon/file cache with attribution caveat", "Native epoch/update/VALID/finalization wall costs and all output/checkpoint/snapshot disk bytes"],
    "limits": ["One actual method/preset/seed/device trajectory; no duplicate uninterrupted trajectory", "No TEST provider or scoring", "No production schedule, source reference, arbitrary state schema or all-device qualification", "Sampled resource peaks are observations, not mature all-method or reciprocal pair envelopes", "No old checkpoint migration; each resume uses this exact ownJOB/hash/policy"],
    "ancestors": [receipt(E / "qualification/runtime_v3_cpu_full_epoch_v1/qualify.py", True),
                  receipt(E / "qualification/runtime_v4_node_selected_mean_lifecycle_v1/qualify.py", True),
                  receipt(E / "qualification/runtime_v4_node_selected_mean_lifecycle_attempt02_local/READINESS.json", True)],
    "native_work_in_preparation": 0, "TEST_accessed": False, "new_optional_fake_checks": 0,
    "source_schedule_modified": False, "Git_operations": 0, "remote_access": False}
write("PLAN.json", plan)

# Source/metadata inspection only: parse syntax, compare frozen settings, and
# confirm hard limits are present. Do not import native modules or this driver.
source = (HERE / "qualify.py").read_text()
tree = ast.parse(source)
ast.parse((HERE / "resource_sampler.py").read_text())
cpu, gpu = (json.loads((HERE / n).read_text()) for n in ("JOB_CPU.json", "JOB_CUDA.json"))
device_fields = {"device", "execution_mode"}
assert {k: v for k, v in cpu.items() if k not in device_fields} == {k: v for k, v in gpu.items() if k not in device_fields}
assert cpu["recipe"]["native_fixed_config"] == original["recipe"]["native_fixed_config"]
assert all(fragment in source for fragment in ["os.O_EXCL", "report[\"TRAIN_epochs_started\"] >= 2", "report[\"native_constructions_attempted\"] >= 3",
    "report[\"optimizer_step_attempts_observed\"] >= 76", "report[\"native_VALID_calls_started\"] >= 4", "report[\"selected_mean_exports\"] >= 2",
    "(76, 76, 78, 76, 2, 76, 4, 4)", "signal.alarm(600)", "original_restore_rng", "first_strict_best_epoch"])
assert len([n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run" and isinstance(n.func.value, ast.Name) and n.func.value.id == "run_fit"]) == 1
checks = {"schema": "runtime-v4-molecular-source-metadata-checks-v1", "status": "source/metadata only passed; native behavior unexecuted",
    "syntax_AST_parsed": ["qualify.py", "resource_sampler.py"], "CPU_GPU_jobs_differ_only_in_device_mode": True,
    "native_fixed_config_exact_to_existing_completed_MolBACE_source_descriptor": True,
    "same_policy_across_actual_resume_invocations": True, "bounded_run_callsite": 1,
    "hard_constructor_epoch_step_VALID_export_limits_source_checked": True,
    "independent_actual_metric_selection_audit_present": True, "state_RNG_exact_restore_observers_present": True,
    "resource_sampler_source_parsed_not_executed": True, "sources_and_data_hashes_not_refreshed_by_native_execution": True,
    "native_models": 0, "native_forwards": 0, "optimizer_updates": 0, "tensor_reads": 0,
    "new_optional_fake_checks": 0, "remote_access": False, "Git_operations": 0}
write("SOURCE_CHECKS.json", checks)
files = ["qualify.py", "resource_sampler.py", "build_plan.py", "JOB_CPU.json", "JOB_CUDA.json", "INPUTS.json",
         "PLAN.json", "PLAN.md", "PILOT_REUSE.json", "SOURCE_CHECKS.json"]
write("MANIFEST.json", {"schema": "runtime-v4-molecular-selected-mean-source-manifest-v1", "status": "source prepared; root native admission pending",
    "published_parent": plan["published_parent"], "files": [{**receipt(HERE / f), "path": f} for f in files],
    "native_work": 0, "source_or_runtime_packages_modified": False, "TEST_accessed": False,
    "CPU_GPU_alternatives_both_executed": False, "scientific_fits": 0, "campaign_GO": False})
print(json.dumps({"status": "source-plan-frozen", "bounds": limits, "native_work": 0, "manifest_sha256": receipt(HERE / "MANIFEST.json")["sha256"]}))
