#!/usr/bin/env python3
"""Build a source/metadata-only readiness map. Never import study code or tensors."""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent
P = E.parent
SNAPSHOT = "2026-10-15; evidence pins below define the snapshot, not live gate status"


def read(rel):
    return json.loads((P / rel).read_text())


def pin(rel):
    path = P / rel
    assert path.suffix in {".json", ".md", ".py", ".csv", ".log", ".gz"}, path
    raw = path.read_bytes()
    return {"path_relative_to_phase": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def erel(rel):
    return "benchmark_expansion_20261015_v1/" + rel


evidence = []


def ev(id, rel, status, exercised, proves, excludes, kind="native_bounded", counts=None):
    evidence.append({"id": id, "receipt": pin(rel), "status_at_receipt": status,
                     "kind": kind, "exercised": exercised, "reusable_claim": proves,
                     "not_qualified_by_this_receipt": excludes, "work_counts": counts})


ev("EV01_smoke_matrix", "reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/REPORT.md",
   "94 passed / 73 failed of 167 latest eligible cases", "27 presets; short actual-source GPU engineering matrix; legacy and newer replay criteria coexist",
   "Use per-case operator, optimizer ownership, finite-work, VALID, immediate load and replay columns separately; preserve all 200 attempts and 1208 optimizer calls.",
   ["A passing legacy node row is not exact-restoration qualification", "73 failed replay rows remain failed", "No complete epoch, later phase, current finalizer, mature process memory or all-pair admission"],
   counts={"presets": 27, "selected_cases": 167, "selected_passed": 94, "selected_failed": 73,
           "all_terminal_attempts": 200, "all_observed_optimizer_calls": 1208, "pair_records": 1})
ev("EV02_node_provider", erel("qualification/node_all_split_provider_v1/TERMINAL.json"), "passed",
   "CPU role-only audit: 40 unique dataset/split roles, 80 preset/split bindings",
   "Recorded node provider role/materialization audit; pair with its source manifest and child scalar receipt.",
   ["No models, forwards, optimizer updates or TEST scoring", "Not a fresh data recheck at eventual dispatch"], "provider_only",
   {"unique_dataset_split_roles": 40, "native_setting_split_bindings": 80, "native_models": 0})
ev("EV03_node_geometry", erel("qualification/task_geometry_node_port_v1_READINESS.json"), "passed bounded CPU",
   "CiteSeer/Cheb; alignment and alignment+variance; 1929 TRAIN / 665 VALID",
   "Task geometry objective, active gradient, native owner and fresh restore semantics in the exercised node source.",
   ["No other node shape/operator or CUDA claim", "No production selector/finalizer claim"], counts={"optimizer_updates": 4, "native_constructions": 4})
ev("EV04_molecular_alignment", erel("qualification/task_geometry_molecular_alignment_control_v1_READINESS_AUDIT03.json"), "passed including separately admitted active-support successor",
   "MolBACE GIN/drop0.5; first one-class inactive check plus active two-class check",
   "Active alignment support was resolved by active_support_CPU_check; do not reuse the initial inactive result as active evidence or preserve it as an unresolved gap.",
   ["Engineering active batch is not a source scientific trajectory", "No other molecular operator/shape, CUDA or selected finalizer claim"],
   counts={"optimizer_updates_all_checks": 4, "native_constructions_all_checks": 4, "whole_VALID_calls_all_checks": 6})
ev("EV05_node_responsibility", erel("qualification/structural_class_responsibility_node_port_v2_READINESS.json"), "passed bounded CPU",
   "CiteSeer/Cheb structured and within-class permuted responsibility",
   "Exact TRAIN table/normalization, shared F/private W ownership, native optimizer and fresh restore.",
   ["No full runtime finalizer or target CUDA qualification", "Other datasets need their own input/table authority"], counts={"optimizer_updates": 4})
ev("EV06_molecular_responsibility", erel("qualification/structural_class_responsibility_v1_READINESS.json"), "passed bounded CPU molecular; node status in this ancestor is superseded by EV05",
   "MolBACE GIN/drop0.5; structured and permuted tables",
   "Native reductions/ownership and exact fresh VALID restore for two molecular responsibility branches.",
   ["No all-molecular architecture/GPU/full-worker claim"], counts={"optimizer_updates": 4, "native_constructions": 4, "whole_VALID_calls": 6})
ev("EV07_node_context", erel("qualification/context_affinity_node_port_v1_READINESS.json"), "passed bounded CPU",
   "CiteSeer/Cheb common/split factual context; selected TRAIN donor support",
   "Meaningful target support, active shared/private gradients, native reductions and checkpoint authority.",
   ["No other node shape/operator, CUDA or selected runtime qualification"], counts={"optimizer_updates": 4})
ev("EV08_molecular_context", erel("qualification/context_affinity_molecular_port_v1_READINESS.json"), "passed bounded CPU",
   "MolBACE GIN/drop0.5 common/split context",
   "Active target/gradient paths and fresh restore for molecular context source.",
   ["No other molecular operator/shape, CUDA or selected runtime qualification"], counts={"optimizer_updates": 4, "native_constructions": 4})
ev("EV09_portable_input", erel("qualification/portable_private_input_v2_READINESS.json"), "passed bounded CPU",
   "CiteSeer/Cheb and MolBACE GIN/drop0.5; BE+residual and residual-only",
   "Zero-start adapter construction, native ownership, removal of BE in residual-only and fresh restore for exercised bridges.",
   ["Old pilot residual is a different source; no numerical cost or lifecycle substitution", "Link input sites and nonzero-seed BUDDY successor remain separate"],
   counts={"optimizer_updates": 8, "native_constructions": 8})
ev("EV10_node_aggregation", erel("qualification/conditional_aggregation_node_port_v1_cpu_bridge_v1/READINESS.json"), "qualified exact CPU cell",
   "CiteSeer/Cheb mean/global/conditional J controls",
   "Same first F/L/J, uniform start, live scorer/native owner, same-call actual/mean and exact fresh restore.",
   ["Direct served tensors are not selected exports", "No other operator/shape/device qualification"],
   counts={"optimizer_updates": 6, "native_constructions": 6, "whole_VALID_calls": 15})
ev("EV11_molecular_aggregation", erel("qualification/conditional_aggregation_v1_cpu_bridge_v1/READINESS.json"), "qualified exact CPU cell",
   "MolBACE GIN/drop0.5 mean/global/conditional J controls",
   "Uniform start, native optimizer ownership and actual/mean VALID restore for three pooling branches.",
   ["Direct served tensors are not selected exports", "No other molecular shape/operator/device qualification"],
   counts={"optimizer_updates": 6, "native_constructions": 6, "whole_VALID_calls": 9})
ev("EV12_node_molecular_interactions", erel("qualification/interaction_node_molecular_v1_cpu_v1/READINESS.json"), "8 CPU cases passed; production finalizer separate",
   "Four explicit interactions on CiteSeer/Cheb and MolBACE GIN/drop0.5; active molecular class support",
   "Ingredient composition, shared/private ownership, native Adam, live residual/scorer, exact fresh actual/mean restore.",
   ["Does not execute production selector/export", "No all-preset or CUDA claim"],
   counts={"optimizer_updates": 16, "native_constructions": 16, "whole_VALID_calls": 16})
ev("EV13_NCN_base", erel("link_expansion_ports_v1/NCN_CPU_QUALIFICATION.json"), "8 methods passed bounded CPU",
   "NCN, Collab source batches and whole VALID; seed0",
   "Base link responsibility, residual and aggregation short-state branches with source BCE, sampler/RNG and native ownership.",
   ["No complete TRAIN epoch/current runtime selection/export", "CPU RSS is not GPU process memory"],
   counts={"optimizer_updates": 16, "native_constructions": 16, "whole_VALID_calls": 24})
ev("EV14_NCNC_base", erel("link_expansion_ports_NCNC_v1/NCNC_CPU_QUALIFICATION.json"), "8 methods passed bounded CPU",
   "NCNC, actual recursion and task-only output scope; seed0",
   "Recursive completion is excluded from task rows; exact native BCE, sampler and fresh restore for short state paths.",
   ["No complete TRAIN epoch/current runtime export", "Concurrent CPU costs are not isolated performance or GPU admission"],
   counts={"optimizer_updates": 16, "native_constructions": 16, "whole_VALID_calls": 24})
ev("EV15_BUDDY_base_seed0", erel("link_expansion_ports_BUDDY_v2/BUDDY_CPU_QUALIFICATION.json"), "passed bounded seed0 oracle CPU",
   "BUDDY eight methods using frozen seed0 context authority",
   "Preserve method objective/gradient/Adam/state evidence for unchanged functions; source equivalence can carry only those semantics to successors.",
   ["Not evidence that old callbacks work for seeds1-9", "No production-provider state nesting, full epoch, current runtime or CUDA claim"],
   counts={"optimizer_updates": 16, "native_constructions": 16, "whole_VALID_calls": 24})
ev("EV16_link_geometry", erel("qualification/link_C1_native_cpu_v1/READINESS.json"), "3 methods passed bounded CPU",
   "NCN same-query cosine/global/endpoint controls",
   "Authenticated query targets/support, separate source positive/negative reductions and fresh served restore.",
   ["No all-link architecture or current runtime qualification"],
   counts={"optimizer_updates": 6, "native_constructions": 6, "whole_VALID_calls": 9})
ev("EV17_NCN_context_CPU", erel("qualification/context_affinity_link_port_v1_READINESS.json"), "passed bounded CPU; CUDA defect preserved in v1",
   "NCN common/split context v1",
   "Active source context targets and exact CPU restore; v2 device correction needs its own source pin and EV18.",
   ["Never launch v1 on CUDA", "No NCNC/BUDDY integration claim"],
   counts={"optimizer_updates": 4, "native_constructions": 4, "whole_VALID_calls": 6})
ev("EV18_NCN_context_CUDA", erel("qualification/context_affinity_link_v2_cuda_v2_run01/child/RESULT.json"), "passed common-only CUDA lifecycle; zero numerical warnings",
   "NCN common v2; A100; 3 updates including one replay / 4 whole VALID",
   "Exact saved state, sampler/integer/global-private RNG and fixed predeclared CUDA replay diagnostics for this source branch.",
   ["Split is source-equivalence-only unless separately witnessed", "No complete epoch/selected finalizer or mature total-process/pair memory"],
   counts={"optimizer_updates": 3, "native_constructions": 2, "whole_VALID_calls": 4,
           "peak_CUDA_allocated_bytes": 8450892288, "peak_CUDA_reserved_bytes": 8728346624})
ev("EV19_NCNC_context", erel("qualification/context_affinity_ncnc_native_cpu_v1/READINESS.json"), "2 methods passed bounded CPU",
   "NCNC common/split; source65536 positive +65536 negative; recursion scope retained",
   "Exact short task/recursive assignment, active targets, sampler/RNG and served/mean restore.",
   ["No complete epoch/current runtime/CUDA"], counts={"optimizer_updates": 4, "native_constructions": 4, "whole_VALID_calls": 6})
ev("EV20_NCN_interactions", erel("qualification/interaction_link_ncn_v2_READINESS_ACTUAL_ASSIGNMENT.json"), "4 cases passed actual-assignment CPU successor",
   "NCN four explicit interactions; failed earlier harness preserved",
   "Actual task/auxiliary gradient assignments, native source BCE/optimizer and fresh actual/mean restore.",
   ["No current runtime, full epoch or CUDA claim"],
   counts={"qualified_optimizer_updates": 8, "qualified_native_constructions": 8, "qualified_whole_VALID_calls": 12,
           "qualified_reverse_collections": 12})
ev("EV21_NCNC_interactions", erel("qualification/interaction_link_ncnc_v1_cpu_v1/READINESS.json"), "4 cases passed bounded CPU",
   "NCNC four interactions, encoder32/top-level64/recursive128 existing source calls",
   "Actual source recursive/task assignment, BCE/backward/Adam and exact fresh served/mean restore.",
   ["No complete native epoch/runtime selected export/CUDA", "13.55GiB CPU peak is not CUDA or isolated resource admission"],
   counts={"optimizer_updates": 8, "native_constructions": 8, "whole_VALID_calls": 12, "reverse_collections": 12})
ev("EV22_BUDDY_context_seed0", erel("qualification/context_affinity_BUDDY_port_v1_CPU_QUALIFICATION.json"), "passed bounded seed0 CPU",
   "BUDDY common/split old context callback",
   "Original factual target/objective/ownership semantics under exact old source; state and provider deltas remain explicit.",
   ["No nonzero seed production-provider/current runtime/GPU"], counts={"optimizer_updates": 4, "native_constructions": 4, "whole_VALID_calls": 6})
ev("EV23_BUDDY_BN_seed0", erel("qualification/buddy_private_bn_state_v1_CPU_QUALIFICATION.json"), "passed bounded seed0 CPU",
   "BUDDY route-private running buffers with shared original affine/native weights",
   "Native TRAIN loss, private BN counters/banks and exact fresh served/mean restore.",
   ["No nonzero seed production-provider nesting/current finalizer/full epoch/CUDA"],
   counts={"optimizer_updates": 2, "native_constructions": 3, "whole_VALID_calls": 3})
ev("EV24_BUDDY_interactions_seed0", erel("qualification/interaction_BUDDY_v1_CPU_QUALIFICATION.json"), "passed bounded seed0 CPU",
   "BUDDY four interactions, old seed0 context callback",
   "Original interaction objective, real backward assignment, optimizer and direct restore evidence.",
   ["No production-provider nesting, nonzero seed/current selected runtime/full epoch/CUDA"],
   counts={"optimizer_updates": 8, "native_constructions": 8, "whole_VALID_calls": 12})
ev("EV25_molecular_runtime_v3", erel("qualification/runtime_v3_cpu_full_epoch_v1/READINESS.json"), "passed with bookkeeping caveat",
   "MolBACE GIN/drop0.5 native M1 + boundary; CPU matched engineering policy; five complete native epochs",
   "Complete loader traversal, epoch-boundary exact restore and stopped export under runtime_v3. v4 unchanged common/bookkeeping/controller/source_rng files are reusable only as unchanged components.",
   ["v3 failed-invocation summary counter stale; durable receipts are authoritative", "No automatic v4 finalizer transfer or checkpoint migration", "No node/link/CUDA/abrupt partial-epoch kill"],
   counts={"complete_TRAIN_epochs": 5, "optimizer_updates": 190, "native_constructions": 6, "whole_VALID_calls": 10})
ev("EV26_runtime_v4_saved_contracts", erel("qualification/runtime_v4_saved_tensor_regression_v1/READINESS.json"), "source/saved-tensor contracts passed",
   "Five native/C6/interaction node/molecular saved-output cases; simulated selected context",
   "76 authority mutation refusals, exact same-call mean and source reporting AST correction; no new native execution.",
   ["No selector or native lifecycle executed", "Does not qualify molecular v4 epoch/selection/resume"], "saved_output_contract",
   {"cases": 5, "mutation_refusals": 76, "recorded_callbacks": 41, "native_constructions": 0, "optimizer_updates": 0})
ev("EV27_node_runtime_v4", erel("qualification/runtime_v4_node_selected_mean_lifecycle_attempt02_local/READINESS.json"), "passed bounded native CPU node lifecycle",
   "CiteSeer/Cheb input+conditionalJ, engineering seed101/split0; own actual selected checkpoint",
   "Two complete node epochs, completed-epoch interruption/resume, reporting fault, stopped finalization0steps, exact persisted actual/mean and bound job/checkpoint/source/IDs/labels.",
   ["Initial launcher failure remains separately preserved", "No source-reference policy, molecular/link/CUDA or other state-schema pass"],
   counts={"complete_TRAIN_epochs": 2, "optimizer_updates": 2, "native_constructions": 3, "whole_VALID_calls": 4, "selected_mean_exports": 2})
ev("EV28_selected_reconstruction", erel("qualification/selected_valid_reconstruction_v1_cpu_v2/READINESS.json"), "passed with explicit limits",
   "Four existing MolBACE runtime_v1 selected exports; fourteen short engineering states reconstructed fresh/fresh",
   "Existing exact native/base/geometry/old-residual production selections reconstructed under frozen tolerances; short C6/portable/C2 state reconstruction is engineering evidence.",
   ["Fresh/fresh engineering states are not selected exports", "Old residual not portable/interaction source", "No new C1 controls, persisted BNv3 selection, link or CUDA"],
   counts={"production_selected_cases": 4, "engineering_state_cases": 14, "native_constructions": 32, "whole_VALID_calls": 32, "optimizer_updates": 0})
ev("EV29_link_runtime_v6", erel("qualification/runtime_link_v6_native_lifecycle_v1/COMPILER_HANDOFF.json"), "source prepared; native full-epoch gate pending with owner",
   "Explicit NCNC conditionalJ engineering policy; production link policy unchanged",
   "Source/export authority design and narrow v5 annotation repair. Only explicit interaction_metadata plus selected_mean_sidecar currently supported.",
   ["No native lifecycle execution at this pinned handoff", "Engineering2epoch policy cannot replace production source/matched clocks", "Other callback protocols need integration/equivalence review"], "source_only")
ev("EV30_BUDDY_provider", erel("qualification/buddy_native_provider_v1_INPUT_QUALIFICATION.json"), "corrected seed1 input parity passed; first unseeded harness failed",
   "Production seed_all before constructors; original vs compact native provider; traversal1585batches of native1024 including tail794",
   "Exact seed1 source inputs and sampler traversal; compact ownership without seed0 oracle injection or ten duplicate context tensors.",
   ["3170 sampler calls are not trained epochs", "0 forwards/updates; no candidate optimization/state/selection qualification", "Do not lift90context blocks"], "provider_only",
   {"corrected_gate_native_constructions": 2, "corrected_gate_sampler_calls": 3170, "model_forwards": 0, "optimizer_updates": 0})
ev("EV31_BUDDY_successors", erel("qualification/BUDDY_PRODUCTION_SUCCESSORS_HANDOFF_v1.json"), "source successors frozen; representative conditional seed1 gate pending with owner",
   "Production expansion/privateBN/interactionv2 provider ownership and checkpoint nesting;11prepared runtime IDs",
   "Recorded AST/byte equivalence of original objective/credit/BCE/backward/Adam/BN/factor functions; provider, native mapping and checkpoint deltas remain explicitly unqualified by that equivalence.",
   ["No individual native optimization/runtime pass for other methods or seeds", "90contexts stay blocked until accepted successor qualification and fresh compiler overlay"], "source_only")
ev("EV32_NCNC_old_CUDA_resource", erel("qualification/runtime_link_v6_NCNC_CUDA_RESOURCE_EVIDENCE/APPLICABILITY.json"), "authenticated old native boundary resource context only",
   "NCNC base short CUDA smoke; not new conditionalJ measurement",
   "Recorded allocator peak12.26GB /reserved20.20GB supports a transparent root-admitted conservative gate reservation proposal.",
   ["40GiB recommendation is an envelope, not observed conditionalJ or mature full-epoch demand", "No process VRAM, pair admission, or extrapolation to all methods"], "resource_context")
ev("EV33_cost_catalogue", erel("expanded_launch_preparation_v1/compact_generated_v1/COST_CATALOGUE.json"), "completed24-fit pilot descriptive ledger",
   "MolBACE GIN/drop0.5; 3680epochs;139840updates;27912.868worker seconds",
   "Same-source native/base/geometry case priors describe60matched target intentions; every failed/excluded cost retained.",
   ["No target-seed or other-preset ETA", "Source-reference costsnull because diagnostic/controller differ", "Old residual/combination cannot transfer numerical costs to portable/interactions", "No mature resource/pair envelope"], "cost_only")
ev("EV34_I4_fixed_readouts", erel("analysis/complete_route_diagnostic_v1/RESULT.json"), "complete saved-export analysis; no new execution requested here",
   "Three existing MolBACE development seeds101/202/303, all24fits; native I4 four selected native exports per seed",
   "Already computes exact raw-logit mean and fixed mean-sigmoid probabilities, AUC/BCE and repair/harm diagnostics without fits/forwards; preserves both rules and all tradeoffs.",
   ["Selected VALID is not independent confirmation", "No learned combiner or causal/oracle performance claim", "Does not change frozen I4 serving or expanded roster"], "posthoc_existing",
   {"new_models": 0, "new_forwards": 0, "optimizer_updates": 0, "graphs": 151, "development_seeds": 3})
ev("EV35_compiler_v2", erel("expanded_job_binding_v1/compact_generated_v2/READINESS.json"), "6370source descriptors /90blocked contexts",
   "6460intentions;270source;1080matchednative;5110shared;270I4groups;5920logicalbanks",
   "Metadata validates exact fit accounting, native settings, independent I4 ownership and M1body0 reuse; source vs matched remain distinct.",
   ["0runtime/resource attachments or GO", "Inherited data receipts not refreshed", "Metadata validation is not execution qualification"], "metadata_only")
ev("EV36_dispatcher", erel("expanded_dispatcher_v1/MANIFEST.json"), "frozen actual guarded Popen source;38bounded fake checks passed",
   "Allocation-only startup-through-reap cap2, durable claims/leases, skips, resume/continuation/adoption and collector failure handling",
   "Source/fake-subprocess custody and admission logic; collector root-review fix is part of the frozen generation.",
   ["Fake text .pt files are not native state evidence", "No production qualification/resource/runtime bindings or humanGO", "Hardcodedallocationroot/UUID and77rejection mean this is not a portable3GPUlauncher"], "fake_subprocess")
ev("EV37_completed_molecular_pilot", erel("pilot_20261015_molbace_v1/complete_analysis_v1/REPORT.json"), "all24frozen development fits complete;0failed fits",
   "Three paired MolBACE/GIN/drop0.5 development seeds101/202/303; native12bodies,base3,geometry3,OLDresidual3,OLDcombination3; original runtime",
   "Actual full native training/validation/selection and final selected exports under the pinned original runtime;3680complete epochs and139840optimizer calls. Reuse unchanged native/operator/loader/selector/export components and exact original method evidence; do not repeat fits because of old short replay tolerances.",
   ["Not current v4changed adapters/mean finalizer qualification", "Old residual/combination are not portablev2/interactions", "Development fits do not substitute for prospective fresh target cohorts or source-reference runs", "No device-wide mature total-process/pair envelope"], "completed_scientific_pilot",
   {"completed_fits": 24, "complete_TRAIN_epochs": 3680, "optimizer_updates": 139840})
ev("EV38_completed_molecular_BN_pilot", erel("pilot_20261015_bn_state_v1/complete_analysis_v1/RESULT.json"), "all3privateBNv3development fits complete",
   "Same MolBACE/GIN/drop0.5 development cohorts; exact private_bn_state_v3 source;149/149/166epochs and selected92/54/116;original runtime",
   "PrivateBNv3has already trained464complete native epochs and produced three own selected payloads; exact existing controls reused. Prefer those persisted states/source equivalence when checking selected-state support; no new BN scientific fit is needed for readiness.",
   ["This analysis0models/0steps records previous fit outcomes, not fresh state reconstruction", "Current v4state reconstruction/selected-finalizer source delta still needs narrow proof or bounded witness", "No independent confirmation or all-architecture/resource claim"], "completed_scientific_pilot",
   {"completed_fits": 3, "complete_TRAIN_epochs": 464, "new_analysis_optimizer_updates": 0})


roster = read(erel("expanded_study_design_v1/PROPOSED_ROSTER.json"))
settings = read(erel("expanded_study_design_v1/NATIVE_SETTINGS.json"))
registry = read(erel("expanded_job_binding_v1/REGISTRY_v1.json"))
overlay = read(erel("expanded_job_binding_v1/OVERLAY_v2.json"))
entries = {(r["cell_id"], r["method_id"]): r for r in registry["entries"]}
for row in overlay["entries"]:
    entries[(row["cell_id"], row["method_id"])] = row
assert len(entries) == 511
assert all(row["callback"] is not None for row in entries.values())
groups = {}
for key, row in sorted(entries.items()):
    signature = json.dumps({"callback": row["callback"], "session_transform": row.get("session_transform")}, sort_keys=True)
    value = groups.setdefault(signature, {"callback": row["callback"], "session_transform": row.get("session_transform"),
                                          "bindings": [], "methods": set(), "presets": set()})
    value["bindings"].append({"preset": key[0], "method": key[1], "runtime_procedure": row["runtime_procedure"],
                              "supported_seeds": row["supported_seeds"]})
    value["methods"].add(key[1]); value["presets"].add(key[0])
source_classes = []
for i, (_, group) in enumerate(sorted(groups.items()), 1):
    binding_rows = group.pop("bindings")
    source_classes.append({"id": f"SRC{i:02d}", **group, "binding_count": len(binding_rows),
                           "method_procedures": sorted({(r["method"], r["runtime_procedure"]) for r in binding_rows}),
                           "supported_seed_contracts": sorted({json.dumps(r["supported_seeds"], sort_keys=True) for r in binding_rows}),
                           "methods": sorted(group["methods"]),
                           "presets": sorted(group["presets"]), "qualification_status": "binding equivalence candidate only",
                           "rule": "Same source pin does not collapse different method branches, state schemas, operators, inputs, tracks or devices."})
assert len(source_classes) == 28

smoke_path = P / "reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/CASE_CHECKS.csv"
smokes = list(csv.DictReader(smoke_path.open()))
smoke_summary = {"cases": len(smokes), "passed": sum(r["status"] == "passed" for r in smokes),
                 "failed": sum(r["status"] == "failed" for r in smokes),
                 "functional_completion_true": sum(r["functional_completion"] == "True" for r in smokes),
                 "exact_loaded_restoration_column_true": sum(r["exact_restoration_qualified"] == "True" for r in smokes),
                 "caution": "Immediate loaded-state and replay columns are different obligations; a true load column never changes a failed replay status.",
                 "by_preset": []}
for cell in roster["cells"]:
    rows = [r for r in smokes if r["recipe_id"] == cell["id"]]
    smoke_summary["by_preset"].append({"preset": cell["id"], "cases": len(rows),
      "passed": sum(r["status"] == "passed" for r in rows), "failed": sum(r["status"] == "failed" for r in rows),
      "functional_completion_true": sum(r["functional_completion"] == "True" for r in rows),
      "replay_criteria": dict(collections.Counter(r["replay_criterion_version"] for r in rows))})

# The compressed file contains JSON scalar/source/check receipts, not tensors.
# Classify the recorded failure, never change the original status or tolerance.
smoke_raw_rel = "reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/RESULTS.json.gz"
smoke_raw = json.loads(gzip.decompress((P / smoke_raw_rel).read_bytes()))
failed = [r for r in smoke_raw["attempts"] if r["selected_matrix_attempt"] and r["status"] == "failed"]
failure_rows = []
for row in failed:
    resume = row["resume_checks"]
    immediate = row["immediate_loaded_state_checks"]
    mismatches = [m for v in resume.values() if isinstance(v, dict) for m in v.get("mismatches", [])]
    exact_loaded = all(v.get("passed") is True for v in immediate.values())
    numeric = (row["failure"]["message"] == "Checkpoint third-update replay outside declared engineering tolerance"
               and exact_loaded and resume["global_RNG"]["passed"] is True
               and all(row["checks"].get(k) is True for k in ["finite_train_losses", "finite_validation_losses", "gradients", "parameter_identity"])
               and all(m.get("reason") == "outside declared engineering tolerance" for m in mismatches))
    failure_rows.append({"case_id": row["case_id"], "attempt_sha256": row["attempt_sha256"], "original_status": row["status"],
      "classification": "post-update numerical tolerance only in recorded evidence" if numeric else "requires nonnumerical/unknown review",
      "exact_immediate_state_modes_requires_grad_RNG": exact_loaded, "post_update_global_RNG_exact": resume["global_RNG"]["passed"],
      "failed_checks": [k for k, v in resume.items() if isinstance(v, dict) and v.get("passed") is False],
      "outputs_max_abs": resume["outputs"].get("maximum_tensor_absolute_difference"),
      "outputs_max_RMS": resume["outputs"].get("maximum_tensor_RMS_difference"),
      "state_max_abs": resume["model_optimizer_and_declared_private_state"].get("maximum_tensor_absolute_difference"),
      "state_max_RMS": resume["model_optimizer_and_declared_private_state"].get("maximum_tensor_RMS_difference"),
      "failed_state_tensor_max_abs": max((m.get("max_abs_diff", 0) for m in resume["model_optimizer_and_declared_private_state"].get("mismatches", [])), default=0),
      "loss_abs_difference": abs(resume["loss"]["resumed"] - resume["loss"]["uninterrupted"]),
      "rtol": row["replay_criterion"]["rtol"], "atol": row["replay_criterion"]["atol"]})
assert len(failure_rows) == 73
assert all(r["classification"].startswith("post-update numerical") for r in failure_rows)
failure_summary = {"source": pin(smoke_raw_rel), "selected_failed_cases": len(failure_rows),
  "recorded_numerical_only_failures": len(failure_rows), "recorded_nonnumerical_failures_in_these_cases": 0,
  "exact_immediate_restores": 73, "exact_post_update_global_RNG": 73,
  "failed_check_counts": dict(collections.Counter(k for r in failure_rows for k in r["failed_checks"])),
  "recorded_mismatch_reason_counts": dict(collections.Counter(m["reason"] for r in failed for v in r["resume_checks"].values() if isinstance(v, dict) for m in v.get("mismatches", []))),
  "magnitude_ranges": {k: {"min": min(r[k] for r in failure_rows), "max": max(r[k] for r in failure_rows)}
                       for k in ["outputs_max_abs", "outputs_max_RMS", "state_max_abs", "state_max_RMS", "failed_state_tensor_max_abs", "loss_abs_difference"]},
  "interpretation": "All73preserved failures are post-update numerical tolerance failures in the recorded receipts, after exact loaded state and RNG. This does not prove the CUDA kernel cause. Some maxima are larger than a few ulps; all magnitudes remain visible. No source/shape/nonfinite/state/RNG failure is recorded in these73cases; other excluded/preparation histories are not reclassified.",
  "minimal_action": "Retain failed statuses and frozen thresholds, reuse valid construction/update/identity/checkpoint/VALID columns plus completed pilots. No mass reruns or tolerance tightening. Only investigate a shared branch if a concrete current authority/state/nonfinite/semantic risk remains.",
  "complete_VALID_caveat": "21of167matrix functional-completion fields are unknown; missing complete-VALID receipts are not inferred from terminal pass/fail or retroactively supplied by the analysis.",
  "cases": failure_rows}
dump("SMOKE_NUMERICAL_FAILURES.json", failure_summary)

method_classes = [
 ("MC01", ["gnnm_boundary_native", "boundary_private_half_pool", "boundary_private_fixed_responsibility", "boundary_own_within_route_supcon"],
  ["EV01_smoke_matrix", "EV25_molecular_runtime_v3"], "Original own/private pooled/fixed responsibility/SupCon branches are distinct; use per-case old evidence, not one base pass."),
 ("MC02", ["task_geometry_alignment", "task_geometry_alignment_variance"], ["EV03_node_geometry", "EV04_molecular_alignment", "EV28_selected_reconstruction"], "Binary/multiclass and active/two-class versus inactive support; variance and alignment controls remain distinct."),
 ("MC03", ["link_same_query_cosine", "link_same_query_global", "link_same_query_endpoint"], ["EV16_link_geometry"], "Cosine/global/endpoint support, no-support handling, source positive/negative reductions and NCNC recursion need explicit source coverage."),
 ("MC04", ["structural_class_quartile_private_v1", "structural_class_quartile_permuted_private_v1"], ["EV05_node_responsibility", "EV06_molecular_responsibility"], "Node incoming degree versus molecular atom count; exact tie/permutation normalization/table identity differs from generic bootstrap."),
 ("MC05", ["link_endpoint_responsibility_v2", "link_query_responsibility_v2"], ["EV13_NCN_base", "EV14_NCNC_base", "EV15_BUDDY_base_seed0"], "Corrected endpoint/query raw laws, loops/nonloops, positive/negative group normalization; BUDDY provider successor differs."),
 ("MC06", ["context_affinity_common", "context_affinity_split"], ["EV07_node_context", "EV08_molecular_context", "EV17_NCN_context_CPU", "EV18_NCN_context_CUDA", "EV19_NCNC_context", "EV22_BUDDY_context_seed0"], "Five task/native target bindings; common/split share work but target assignment is a real branch; NCN v1 CUDA is refused."),
 ("MC07", ["portable_input_residual_be_v1", "portable_input_residual_only_v1"], ["EV09_portable_input", "EV13_NCN_base", "EV14_NCNC_base", "EV15_BUDDY_base_seed0"], "Native encoded input sites and BE removal differ by operator; old pilot residual has no source equivalence to v2."),
 ("MC08", ["aggregation_mean_globalJ_v1", "aggregation_global_globalJ_v1", "aggregation_conditional_globalJ_v1"], ["EV10_node_aggregation", "EV11_molecular_aggregation", "EV13_NCN_base", "EV14_NCNC_base", "EV15_BUDDY_base_seed0"], "Uniform/global/conditional state and serving; mean is predecessor alias, one physical fit; actual serving alone selects."),
 ("MC09", ["interaction_geometry_input", "interaction_responsibility_private_pool", "interaction_input_meanJ", "interaction_input_conditionalJ"], ["EV12_node_molecular_interactions", "EV20_NCN_interactions", "EV21_NCNC_interactions", "EV24_BUDDY_interactions_seed0", "EV27_node_runtime_v4"], "Four declared compositions have real ownership/backward/state branches; input+conditional is a broad witness, not proof of all four assignments."),
 ("MC10", ["gnnm_private_bn_state_v3"], ["EV23_BUDDY_BN_seed0", "EV38_completed_molecular_BN_pilot"], "Tracked native running buffers/counters route-private; affine/native weights shared; no BN inserted in node/NCN/NCNC. Three OGBv3selected pilot states exist; current selected reconstruction/finalizer delta requires narrow inventory, not new scientific fits."),
]
assert set(m for _, methods, _, _ in method_classes for m in methods) == {r["id"] for r in roster["methods"]}

runtime_coverage = [
 {"unit": "node matched runtime_v4", "evidence": ["EV27_node_runtime_v4"], "status": "exact CPU CiteSeer/Cheb input+conditionalJ lifecycle covered", "remaining": "Attach exact pins; reuse unchanged finalizer functions for other methods only after protocol/state equivalence review. Target CUDA/operator/input branches remain separate."},
 {"unit": "molecule matched ancestor runtimes", "evidence": ["EV25_molecular_runtime_v3", "EV37_completed_molecular_pilot", "EV38_completed_molecular_BN_pilot"], "status": "27completed actual development fits plus exact native/base CPU v3complete-epoch lifecycle covered", "remaining": "Current runtime_v4 molecular changed adapters/finalizer/mean plus representative extension require current evidence. Reuse three existing BNv3selected states first; do not rerun completed science or unchanged ancestor components as seed tests."},
 {"unit": "link matched runtime_link_v6 NCNC interaction", "evidence": ["EV21_NCNC_interactions", "EV29_link_runtime_v6"], "status": "method short lifecycle covered; current full-epoch lifecycle pending owner", "remaining": "Consume owner result after acceptance; do not duplicate planned gate or transfer engineering policy to production."},
 {"unit": "link NCN and BUDDY runtime", "evidence": ["EV13_NCN_base", "EV18_NCN_context_CUDA", "EV31_BUDDY_successors"], "status": "short method/provider/source evidence; no current complete-epoch selected lifecycle", "remaining": "Native-family full traversal/state/finalizer branches need one representative each unless exact shared implementation and native-family equivalence proves a component reusable."},
 {"unit": "source-reference policy branches", "evidence": ["EV01_smoke_matrix", "EV35_compiler_v2"], "status": "native settings and early path only; not full current source-track qualification", "remaining": "Witness actual source TRAIN diagnostic/loader RNG and skipped TEST iterator audit for OGB, source node stopping, explicit native link selector/policy; long clocks can be source/pure-controller checked without100/2000training epochs."},
 {"unit": "new state schemas and selected exports", "evidence": ["EV26_runtime_v4_saved_contracts", "EV28_selected_reconstruction"], "status": "partial saved/direct-state coverage", "remaining": "Map original, persistent table/context, residual/noBE, global/conditional, interaction, privateBN and production-provider nesting to authenticated own-selected exports; a fresh/fresh state is not a production selected export."},
 {"unit": "target device and resources", "evidence": ["EV01_smoke_matrix", "EV18_NCN_context_CUDA", "EV32_NCNC_old_CUDA_resource", "EV33_cost_catalogue"], "status": "short GPU/state samples and measured pilot costs only", "remaining": "GPU runtime family/changed-device branches plus mature process VRAM/RAM/cache/disk profiles; measured reciprocal pairs only if using concurrency2."},
]

track_contracts = []
for family in ("node", "ogb", "link"):
    cells = {c["id"] for c in roster["cells"] if c["task_family"] == family}
    rows = [r for r in settings["rows"] if r["preset"] in cells]
    policies = {}
    for row in rows:
        for key in ("source_reference_controller", "matched_controller"):
            val = row[key]
            contract = {"maximum": val["maximum"], "stopping": val.get("stopping"), "native_stopping": val.get("native_stopping"),
              "selection": val["selection"], "native_schedule": val["native_schedule"], "native_TEST_diagnostics": val.get("native_TEST_diagnostics")}
            signature = json.dumps(contract, sort_keys=True)
            item = policies.setdefault(key, {}).setdefault(signature, {"presets": [], "contract": contract})
            item["presets"].append(row["preset"])
    track_contracts.append({"family": family, "contracts": {k: list(v.values()) for k, v in policies.items()},
      "reuse_boundary": "Only unchanged initial operator/optimizer/schedule components may share evidence; source diagnostic RNG, horizon, selector and stopping are separately qualified. Seeded-repeat/source omissions are disclosed; no whole-paper replication claim."})

units = []


def unit(id, title, kind, status, action, acceptance, reuse, depends=(), mandatory=True):
    units.append({"id": id, "title": title, "kind": kind, "status_at_snapshot": status,
                  "remaining_action": action, "acceptance": acceptance, "reuse_evidence": reuse,
                  "depends_on": list(depends), "mandatory_for_allocation_campaign_readiness": mandatory,
                  "native_invocation_count": None,
                  "native_count_reason": "Acceptance obligation, not a new run. Freeze explicit bounded plans only after equivalence/branch set cover and root admission; no duplicate seed matrix."})


unit("RP01", "Pin the coverage and reuse ledger", "source_metadata", "missing attachment work",
     "Resolve exact callbacks, transforms, runtime/policy, provider/dependency receipts and state schemas for511method/preset bindings plus27native presets; record each changed function/branch.",
     "Each claimed component points to an accepted evidence pin and explicit compatible dimensions; failed/unknown claims remain failed/unknown. No self-promotion from shared hashes.",
     ["EV01_smoke_matrix", "EV35_compiler_v2"])
unit("RP02", "Finish provider/data bindings", "source_metadata_and_input_audit", "partly reusable",
     "Attach node40role/80binding audit, source OGB/link split/evaluator/source-cache authority and compact BUDDY input audit. Refresh actual source/data receipts at the eventual authorized admission if unchanged cannot be demonstrated.",
     "All270cohorts and native source roles have literal immutable bindings; no TEST scoring; provider-only evidence never becomes optimization/runtime evidence.",
     ["EV02_node_provider", "EV30_BUDDY_provider", "EV35_compiler_v2"], ["RP01"])
unit("RP03", "Close BUDDY production-provider deltas", "owned_pending_native_gate_and_source_review", "successors and conditional seed1 gate prepared by BUDDY owner",
     "Consume accepted representative conditional seed1 production gate; use recorded objective/Adam/BN AST equivalence for unchanged functions. Audit remaining provider/checkpoint state branches and run only an uncovered branch witness if needed.",
     "Provider seed_all precedes construction; native1024/tail794 inputs, sampler replay, nesting and actual/mean fresh restore authenticated. Other methods are described as equivalent components, not individual passes. Keep90contexts blocked until qualified successors and RP15 fresh overlay.",
     ["EV15_BUDDY_base_seed0", "EV22_BUDDY_context_seed0", "EV23_BUDDY_BN_seed0", "EV24_BUDDY_interactions_seed0", "EV30_BUDDY_provider", "EV31_BUDDY_successors"], ["RP01", "RP02"])
unit("RP04", "Qualify production controller tracks", "source_policy_and_representative_native_branch", "source metadata exists; runtime branch evidence incomplete",
     "Review three family source/matched controller classes against frozen settings. Use pure/source controller witnesses for long min/cap/patience, schedules, first ties and completed-epoch resume. Use native branch witnesses for source OGB TRAIN-eval/global-loader-RNG/omitted-TEST-iterator bookkeeping and each non-equivalent node/link source path.",
     "Source-reference node2000/min0/patience250; OGB100/no early stopping with TRAIN diagnostic; link100/no early stopping. Matched node10000/min2000/patience100; OGB/link1000/min100/patience50. No engineering policy is production proof. No need to train100or2000epochs merely to test a counter.",
     ["EV01_smoke_matrix", "EV25_molecular_runtime_v3", "EV27_node_runtime_v4", "EV29_link_runtime_v6", "EV35_compiler_v2"], ["RP01", "RP02"])
unit("RP05", "Reuse the passed node lifecycle", "evidence_attachment", "native CPU node runtime_v4 lifecycle already covered",
     "Attach EV27 exact source/runtime/job pins and source-only equivalence for unchanged finalizer paths; preserve the launcher failure and successful attempt as separate records.",
     "No duplicate CiteSeer/Cheb conditionalJ CPU run. Additional native work only for a differing operator, state protocol, policy or device identified by RP01/RP04/RP10/RP11.",
     ["EV12_node_molecular_interactions", "EV26_runtime_v4_saved_contracts", "EV27_node_runtime_v4"], ["RP01"])
unit("RP06", "Qualify the current molecular lifecycle", "representative_native_runtime", "current v4 extension selected lifecycle missing",
     "Freeze one current molecular input+conditionalJ complete-epoch/selection/epoch-resume/stopped-finalization representative using existing active MolBACE source, covering changed v4 adapters/finalizer and loader state. Reuse unchanged v3 components and all short ingredient semantics.",
     "Actual complete source TRAIN traversal, exact own selected job/checkpoint/epoch and actual+mean authority; finite source loader tail/skips accounted. This is current molecular runtime evidence, not sixteen presets or ten seeds individually executed.",
     ["EV04_molecular_alignment", "EV11_molecular_aggregation", "EV12_node_molecular_interactions", "EV25_molecular_runtime_v3", "EV26_runtime_v4_saved_contracts", "EV37_completed_molecular_pilot", "EV38_completed_molecular_BN_pilot"], ["RP01", "RP02"])
unit("RP07", "Consume the NCNC v6 full-epoch gate", "owned_pending_native_runtime", "prepared by NCNC owner; pending at pinned source handoff",
     "Use the separately prepared runtime_link_v6 NCNC conditionalJ full-epoch/selected/epoch-resume/stopped-finalization result after root acceptance; do not launch a duplicate.",
     "Native full source query traversal, correct recursive/task scope, sampler/global-private RNG/Adam and authentic annotation-bearing receipts; selected actual+mean bound to own checkpoint. Engineering clock remains distinct from production link_policy_v1.",
     ["EV14_NCNC_base", "EV19_NCNC_context", "EV21_NCNC_interactions", "EV29_link_runtime_v6", "EV32_NCNC_old_CUDA_resource"], ["RP01", "RP02"])
unit("RP08", "Cover the NCN whole-epoch runtime branch", "representative_native_runtime", "short CPU/CUDA state evidence reusable; current whole-epoch runtime missing",
     "Review source equivalence to the v6 NCNC lifecycle; choose one NCN actual whole-epoch/state/selection representative only for native-family traversal and sampler/finalizer deltas left uncovered.",
     "Native NCN pos/neg reductions, complete source order/tail, selected actual/mean export and resume verified; no repetition of8base+4interaction objective checks when unchanged.",
     ["EV13_NCN_base", "EV16_link_geometry", "EV18_NCN_context_CUDA", "EV20_NCN_interactions"], ["RP01", "RP02", "RP07"])
unit("RP09", "Cover the BUDDY whole-epoch runtime branch", "representative_native_runtime", "provider traversal is not a trained epoch",
     "After production successor acceptance, freeze one BUDDY full native TRAIN epoch/selected/resume representative. Reuse input traversal, original method semantics and shared finalizer where exact; witness unique cached-query/provider/state nesting.",
     "1585native mixed-query batches including794tail actually trained, native BN/source sampler ownership and own selected export authenticated. Nonzero-seed2update gate never substitutes for this.",
     ["EV23_BUDDY_BN_seed0", "EV30_BUDDY_provider", "EV31_BUDDY_successors"], ["RP03", "RP07"])
unit("RP10", "Cover selected-state and callback protocols", "source_integration_and_state_witnesses", "partly missing integration, partly reusable direct state",
     "Enumerate original, persistent table/context, residual/noBE, global/conditional, four interactions, privateBN and production-provider checkpoint schemas. Extend link runtime beyond its current interaction_metadata/selected_mean_sidecar protocol in fresh source generations as needed; extend selected reconstruction registry for actual new selected exports. First reuse existing completed molecular/BNselected payloads and source equivalence; only residual state/finalizer branches require bounded witnesses, never new149epochscientific BNfits.",
     "Every bound protocol has one authentic own-selected mapping with exact source/dependency/runtime/JOB/checkpoint/epoch/role/IDs/labels or a narrow source-equivalence proof. Existing BNv3selected payload running buffers/counters support reconstruction checks; fresh/fresh or direct tensors do not fill selected-export gaps. TEST provider execution remains separately gated.",
     ["EV23_BUDDY_BN_seed0", "EV26_runtime_v4_saved_contracts", "EV27_node_runtime_v4", "EV28_selected_reconstruction", "EV29_link_runtime_v6", "EV31_BUDDY_successors", "EV37_completed_molecular_pilot", "EV38_completed_molecular_BN_pilot"], ["RP01", "RP03", "RP06", "RP07"])
unit("RP11", "Cover native operator and device deltas", "representative_native_and_source_equivalence", "27preset short evidence exists; current device/runtime coverage partial",
     "Build a set cover over Mono/Cheb; GCN/GIN and virtual-node branches; dropout0/.5; four node datasets/splits, two molecular graph populations, NCN/NCNC/BUDDY. Reuse exact input/provider and operator checks. Add a representative only if no accepted evidence covers the changed operator/input/state/device branch; combine it with RP12resource measurement where possible.",
     "GPU execution and replay diagnostics use frozen tolerances; state/integer/source authority remains exact. Original73failures are recorded numerical-only after exact restore; preserve their magnitudes and statuses, reuse valid construction/update/load evidence and completed pilots. Do not rerun73cases or tighten tiny differences. Escalate only a concrete current source/state/RNG/shape/nonfinite risk. CPU seed101 evidence does not become seed0-9 or target-GPU native passes.",
     ["EV01_smoke_matrix", "EV02_node_provider", "EV03_node_geometry", "EV12_node_molecular_interactions", "EV18_NCN_context_CUDA", "EV27_node_runtime_v4"], ["RP01", "RP04", "RP10"])
unit("RP12", "Measure serial resource envelopes and costs", "representative_target_resource_profiles", "mature profiles missing; pilot costs reusable only within declared source",
     "Partition by actual graph/query population, native operator/width, virtual-node/input/cache behavior, wrapper persistent state, activation/retained-graph/backward work and actual runtime/device. Select real roster methods that cover each distinct memory/work maximum; prove any dominance used to reuse a profile. Measure construction, warmed Adam/BN/cache, full TRAIN/VALID, selected export/reload and checkpoint write peaks.",
     "Pin process-level GPU VRAM plus allocator peaks, CPU RSS/cgroup/cache, disk/checkpoint/export demand, wall/update/epoch/finalization costs and observed peer baseline. An allocator peak or arbitrary cap is not a mature envelope. No numeric ETA for unmeasured signatures. Serial-only admission1 is a valid first safe release.",
     ["EV01_smoke_matrix", "EV18_NCN_context_CUDA", "EV32_NCNC_old_CUDA_resource", "EV33_cost_catalogue"], ["RP06", "RP07", "RP08", "RP09", "RP10", "RP11"])
unit("RP13", "Qualify a restricted reciprocal pair graph", "optional_concurrency_resource", "one old short pair sample; no mature reciprocal profiles attached",
     "If concurrency2 is desired, choose a small useful set of resource-class pairs, measure each reciprocal startup/warmed TRAIN/VALID/checkpoint/export overlap and pin the compatible pair edge. Leave all other combinations serial.",
     "Both sides accept the same measured pair signature and per-GPU RAM/VRAM/disk/claim limits; no third own worker from startup through reap. Max2 is a ceiling, not an obligation to qualify all pairs. May be deferred by enforcing serial1.",
     ["EV01_smoke_matrix", "EV36_dispatcher"], ["RP12"], False)
unit("RP14", "Check I4 assembly and fair fixed-readout reporting", "source_metadata_and_saved_exports", "contract and prior analysis reusable; expanded reporting attachment missing",
     "Keep four separately charged independent native bodies, offsets0/100003/200003/300007, own optimizers/RNG/selectors/checkpoints and exact M1body0 reuse. Verify aligned IDs/labels/query roles, source policies and four selected payloads before arithmetic raw-logit mean. Attach both fixed mean-logit and mean-probability diagnostics to saved native exports without refitting, inference or tuning.",
     "Frozen I4 serving remains raw-logit mean. Binary mean probability is mean(sigmoid(z_m)); multiclass is mean(softmax(z_m)), with native metric and appropriate proper-loss reporting. Learned shared aggregator claims include ordinary I4 under the same fixed diagnostic rules and same-selected-bank fixed mean, not only an arbitrary weak readout. No claim of fair learned-on-both-sides aggregator comparison without a separately frozen TRAIN-only matched design; do not add one here.",
     ["EV34_I4_fixed_readouts", "EV35_compiler_v2", "EV36_dispatcher"], ["RP01", "RP10"])
unit("RP15", "Create a new compiler/readiness overlay", "source_metadata_release", "v2 frozen;90BUDDYcontexts blocked",
     "Root creates a fresh generation after accepted qualified successors/runtime protocols/resources exist. Attach narrowly scoped accepted evidence to descriptors by equivalence class; retain all old IDs/contracts and ancestors.",
     "6460physical intentions accounted,270I4groups/M1body0/aliases preserved.90BUDDYcontexts close only if explicit production successor, accepted gate and new compiler entries cover their seeds. Metadata validation alone never marks runtime/resource/GO complete.",
     ["EV31_BUDDY_successors", "EV35_compiler_v2", "EV36_dispatcher"], ["RP01", "RP02", "RP03", "RP04", "RP05", "RP06", "RP07", "RP08", "RP09", "RP10", "RP11", "RP12", "RP14"])
unit("RP16", "Freeze expanded scientific reporting and decisions", "scientific_plan_metadata", "expanded decisions remain pending",
     "Freeze exploration/confirmation roles, primary paired contrasts, multiplicity/heldout access, all failure/cap/nonconvergence reporting, member quality, proper losses, repair/harm and actual cost reporting for expanded25recipe roster. Keep development24+3pilot fits distinct from prospective fresh cohorts.",
     "Old540statistical/evaluate jobs remain historical unless explicitly superseded; several presets are not independent datasets. Source references are separate measurements, not extra paired method-effect observations; fixed readout diagnostics cannot silently select the serving rule.",
     ["EV33_cost_catalogue", "EV34_I4_fixed_readouts", "EV35_compiler_v2"], ["RP14"])
unit("RP17", "Review allocation release and obtain campaign GO", "release_admission", "no campaignGO; current preparedrelease6730tasks blocked",
     "Root assembles exact source/runtime/provider/qualification/resource/statistical bindings, one-command allocation preflight and dry-run summary; user reviews the concrete finite roster, measured costs, gaps and entrypoint before any giant campaign.",
     "Explicit humanGO attached to exact generation. Every admission rechecks source/GO/readiness/resource gates and retains failure/stop claims; no automatic scientific retries or repeated selected scoring. Scope remains allocation-only unless future authorization changes.",
     ["EV35_compiler_v2", "EV36_dispatcher"], ["RP15", "RP16"])
unit("RP18", "Implement future portable host profiles", "future_implementation", "missing for eventual threeGPU package; not current authorization",
     "In a future reviewed source generation, externalize explicit allowlisted host/GPU profiles, repository/phase/cache/output roots, source-relative resolution and dependency/provider receipt remapping. Design global physical-fit/I4 claims, startup-through-reap leases, orphan/STOP adoption and result custody across shared-storage and disconnected-root cases.",
     "Synthetic/source tests prove each profile fails closed on host/UUID/path/hash mismatch and prevents cross-host duplicates, including migration/resume with exact source/dependency provenance. Current hardcodedallocationroot/UUID and77rejection remain unchanged here. No credential bundle or host probing.",
     ["EV36_dispatcher"], ["RP15"], False)
unit("RP19", "Deploy and qualify future threeGPU profiles", "future_deployment", "requires separate user route authorization; no77access now",
     "After route authorization and RP18implementation, verify each literal route/host/UUID/root/dependency/data pin, shared claim visibility/locking and permissions; stage exact package, run per-device lifecycle/resource checks and selected reciprocal concurrency on that hardware, then bind assignments/entrypoints to a reviewed release.",
     "Future threeGPU one-command package has reproducible config/path bindings, globally unique fit claims, measured target-host resources and explicitGO. Allocation evidence is not automatically portable device or deployment evidence. No77monitoring/execution until reauthorized.",
     ["EV36_dispatcher"], ["RP18", "RP12"], False)

coverage = {
 "schema": "expanded-readiness-coverage-source-only-v1", "snapshot": SNAPSHOT,
 "status": "coverage map and finite acceptance obligations only; not execution readiness",
 "scope": {"native_tests_launched": 0, "scientific_fits_launched": 0, "data_tensors_checkpoints_logits_opened": 0,
           "TEST_accessed": False, "remote_access": False, "dispatcher_compiler_existing_sources_modified": False,
           "Git_operations": 0, "campaign_GO": False},
 "counts": {"physical_fit_intentions": 6460, "source_reference": 270, "matched_native_bodies": 1080,
            "shared_fits": 5110, "presets": 27, "task_presets": {"node": 8, "ogb": 16, "link": 3},
            "scientific_method_definitions": 25, "method_component_classes": len(method_classes),
            "shared_method_preset_bindings": len(entries), "exact_callback_transform_signatures": len(source_classes),
            "evidence_units": len(evidence), "current_concrete_descriptors": 6370, "BUDDY_blocked_contexts": 90,
            "logical_banks": 5920, "I4_groups": 270},
 "inputs": [pin(erel(f)) for f in ["expanded_study_design_v1/PROPOSED_ROSTER.json", "expanded_study_design_v1/NATIVE_SETTINGS.json",
             "expanded_job_binding_v1/REGISTRY_v1.json", "expanded_job_binding_v1/OVERLAY_v2.json", "runtime_v4/MANIFEST.json",
             "runtime_link_v6/MANIFEST.json", "node_all_split_provider_audit_v1/MANIFEST.json"]]
            + [pin("reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/CASE_CHECKS.csv"),
               pin(erel("qualification/node_all_split_provider_v1/child.log"))],
 "reuse_rules": [
   "Qualification attaches to an unchanged component and its explicit operating assumptions, not to6460individual fits.",
   "Same callback/transform hash is a candidate source equivalence class; runtime procedure, objective branch, state schema, operator, input population, policy track, dependencies and device still matter.",
   "Function AST/byte equivalence preserves only unchanged semantics. Provider ownership, checkpoint nesting, new exports, annotation receipts and device paths require their own coverage.",
   "Representative gates may combine branches only if source instrumentation proves every claimed branch actually executed; source equivalence is never renamed as an individual native run.",
   "Node one native update is a whole epoch. Molecular/link two batches are not a whole epoch. Provider query traversal with0forwards is not a trained epoch.",
   "Short state checks, complete epochs, selection, own-selected export, stopped finalization, epoch-boundary resume, source fidelity, resource admission and scientificGO are separate obligations.",
   "Abrupt partial-epoch crash recovery is not claimed. A release may explicitly support only durable completed-epoch continuation; never call lost in-flight work free or implement automatic retry by implication.",
   "Seed independence/data binding can be metadata audited across all declared seeds; CPU seed101 or seed0 optimization evidence cannot become ten actual native seed passes.",
   "Do not lift90BUDDYcontext blocks until accepted successors and a fresh compiler overlay; input parity alone does not qualify method optimization.",
   "Per-case failures, omitted source TEST diagnostics, seeded-repeat adaptations, caps, unknown costs and unsupported schemas remain explicit."],
 "evidence": evidence, "original_smoke_case_summary": smoke_summary,
 "original_smoke_failure_classification": {k: v for k, v in failure_summary.items() if k != "cases"},
 "original_smoke_failure_details": pin(erel("expanded_readiness_coverage_v1/SMOKE_NUMERICAL_FAILURES.json")),
 "source_binding_equivalence_candidates": source_classes,
 "method_component_classes": [{"id": i, "methods": m, "reusable_evidence": evs, "branch_boundary": b}
                              for i, m, evs, b in method_classes],
 "runtime_coverage": runtime_coverage, "track_contracts": track_contracts,
 "I4_reporting_contract": {
   "serving": "Arithmetic mean of native raw logits from four independently selected bodies; own optimizers/RNG/selectors/checkpoints; offsets0/100003/200003/300007; M1 exactlybody0.",
   "saved_export_derivation": "If each member export includes aligned native raw outputs, native labels/query roles and source identity, both fixed raw-logit mean and mean probabilities need no fitting or forward. Sigmoid for binary, softmax for multiclass; link score/probability transformation and official evaluator semantics must be declared.",
   "missing_information_boundary": "If only a pooled mean survives, member mean probabilities cannot generally be recovered. Reconstructing missing route logits would require separately authorized forward work, not a new fit; none is requested here.",
   "existing_witness": "EV34 already includes the standard independent I4 both fixed rules with AUC and BCE tradeoffs for all3completeddevelopment seeds.",
   "fairness": "Keep declared I4 primary serving unchanged, report the same fixed readout diagnostics for independent and shared banks, preserve actual gated versus same-selected-bank fixed mean. A shared learned gate versus only an arbitrarily weak I4 readout would not support a broad aggregation superiority claim.",
   "no_new_tuning": True, "no_roster_or_serving_change": True},
 "resource_class_contract": {
   "default": "Distinct signatures until a source/measurement dominance argument explicitly proves reuse.27presets are a finite upper indexing grid, not6460seed-level resource tests.",
   "native_shape_branches": "8node(dataset×Mono/Cheb);16molecular(dataset×GCN/GIN×virtualnode×dropout);3linknative families. Dropout0/.5 may merge only with a documented maximal allocation/work witness; dimension alone is insufficient.",
   "method_work_axes": ["single native versus four-route persistent weights/factors/Adam", "native checkpointed activations and live retained graphs", "one versus split objective reverse collections", "class/query contrast and context donor matrices/support", "residual/scorer/BN/table/context persistent state", "NCNC recursive/task source views", "BUDDY cached query/input provider and checkpoint nesting", "TRAIN/VALID/selected-export/reload/cache/write phase overlap"],
   "selection": "Use real roster methods as representatives in a finite set cover over these axes; an invented union method is not a resource witness. If dominance cannot be proved, keep the signature separate and serialize until measured.",
   "concurrency": "Initial max1 is valid. Only measured reciprocal pair edges permit2; unmeasured pairs stay serial. No all-method-pair matrix required.",
   "cost": "Descriptive same-source pilot priors only; no campaign ETA from two-update smokes or CPU RSS/allocator peaks."},
 "portability": {"current": "Allocation-only guarded dispatcher; exact host/UUID/root;77hard-rejected. This correctly implements current authorization.",
                 "not_complete": "Not yet a portable3GPUone-command package.",
                 "future_implementation": ["explicit host/GPU profiles and root/path binding", "global fit/I4 claims/leases across host/storage layouts", "receipt remapping, migration/resume/adoption and result custody"],
                 "future_deployment": ["separate route authorization", "literal guard and dependency/data/locking checks per host", "target-device lifecycle/resources and reviewed assignments/GO"],
                 "guards_loosened_here": False, "server77_accessed": False}
}
plan = {"schema": "remaining-finite-readiness-plan-source-only-v1", "snapshot": SNAPSHOT,
        "status": "proposed acceptance plan; no native gate execution or campaign authorization",
        "count_convention": "19 finite acceptance obligations, not19native runs; RP05 reuses a passed gate; RP03/RP07 consume other owners' pending gates. Native run count is determined only by explicit branch/equivalence set cover and pre-execution bounded plans.",
        "counts": {"obligations": len(units), "mandatory_allocation_obligations": sum(u["mandatory_for_allocation_campaign_readiness"] for u in units),
                   "optional_concurrency_obligations": 1, "future_portability_obligations": 2, "new_native_runs_authorized_by_this_document": 0},
        "sequence": ["RP01-RP02source/provider ledger", "RP03andRP07consume owner gates; RP04source/matched track audits", "RP05reuse; RP06/RP08/RP09current family lifecycle residuals", "RP10-RP11protocol/operator/device branch cover", "RP12serialresource envelopes; optionalRP13pair edges", "RP14I4/diagnostics; RP15fresh overlay; RP16scientificfreeze; RP17concreteGO", "Future-onlyRP18implementation then authorizedRP19deployment"],
        "minimal_native_scope": {
          "already_covered_no_repeat": "Node runtime_v4 CiteSeer/Cheb input+conditionalJ CPU complete epoch/selected/resume/stopped finalizer. Original27molecular development fits and short method gates remain usable within unchanged-source scope.",
          "consume_existing_owner_plans": ["BUDDY production conditional seed1 bounded optimization/restore gate", "runtime_link_v6 NCNC whole-epoch/selected/resume gate"],
          "remaining_family_case_classes": ["Current molecular runtime_v4 input+conditionalJ complete-epoch/selected/resume representative", "NCN native-family whole-epoch/sampler/selected branch after shared v6equivalence review", "BUDDY production-provider whole-epoch/BN/sampler/selected branch after successor acceptance"],
          "controller_branch": "Use pure/source policy checks for long horizons; add a short actual source OGB TRAIN-diagnostic/global-loader-RNG witness only if no pinned existing exact evidence covers it. Combine compatible branches/resources in already necessary representatives.",
          "state_operator_device_residuals": "No blanket all-method rerun. Saved authentic selected payloads and source equivalence first; only uncovered state/native-operator/device branches get bounded witnesses. Target-GPU resources remain required per proven resource class.",
          "serial_first": "Qualify process memory/cost signatures and release serial1 first; reciprocal-pair graph is optional. No all-pair tests, extra fake-subprocess expansion, or new scientific fits follow from this document.",
          "invocation_count": None,
          "reason_count_is_not_fixed_yet": "Case classes are finite; exact invocation/step/epoch/construction/VALID limits must be frozen by root after source equivalence and target-resource cover are resolved. Counting each case class as one invocation would invent work bounds."},
        "stopping_rule": "Stop adding engineering runs when every required component/branch has accepted exact evidence or a documented narrow equivalence proof, source/reference adaptations are disclosed, all serialized resources fit and release gates attach. Broaden only for a differing source/branch/shape/device or a concrete failure; never for seed-count symmetry.",
        "units": units}
assert len(units) == 19 and len(evidence) == 38
assert sum(u["mandatory_for_allocation_campaign_readiness"] for u in units) == 16
assert {d for u in units for d in u["depends_on"]} <= {u["id"] for u in units}
assert {e for u in units for e in u["reuse_evidence"]} <= {e["id"] for e in evidence}
dump("READINESS_COVERAGE.json", coverage)
dump("REMAINING_FINITE_PLAN.json", plan)
dump("BUILD_RECEIPT.json", {"schema": "readiness-coverage-source-only-build-receipt-v1",
     "builder": pin(erel("expanded_readiness_coverage_v1/build_coverage.py")),
     "outputs": [pin(erel("expanded_readiness_coverage_v1/" + name)) for name in ["READINESS_COVERAGE.json", "REMAINING_FINITE_PLAN.json", "SMOKE_NUMERICAL_FAILURES.json", "README.md"]],
     "checks": {"evidence_pins_exist": len(evidence), "shared_bindings": len(entries), "source_signatures": len(source_classes),
                "method_ids_covered": 25, "finite_obligations": len(units), "dependency_and_evidence_references_closed": True,
                "all_work_source_and_metadata_only": True},
     "native_models": 0, "native_forwards": 0, "optimizer_updates": 0, "tensor_files_opened": 0,
     "remote_access": False, "Git_operations": 0, "campaign_GO": False})
dump("MANIFEST.json", {"schema": "expanded-readiness-coverage-package-v1", "status": "source and existing metadata only",
     "files": [pin(erel("expanded_readiness_coverage_v1/" + name)) for name in ["build_coverage.py", "README.md",
               "READINESS_COVERAGE.json", "REMAINING_FINITE_PLAN.json", "SMOKE_NUMERICAL_FAILURES.json", "BUILD_RECEIPT.json"]],
     "native_work": 0, "new_fake_subprocess_checks": 0, "remote_access": False, "server77_accessed": False,
     "frozen_dispatcher_or_compiler_modified": False, "scientific_campaign_authorized": False})
print(json.dumps({"status": "built-source-only", "evidence_units": len(evidence), "source_signatures": len(source_classes),
                  "obligations": len(units), "mandatory_allocation_obligations": 16, "native_work": 0}))
