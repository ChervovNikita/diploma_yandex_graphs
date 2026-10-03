"""Prepare/seal this source-only successor using stdlib metadata; never launch it."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import difflib
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OLD = PHASE / "graph_ncNC_structural_pattern_pilot_preparation_20261003_v3"
UTC = datetime.now(timezone.utc).isoformat()
TEXT = {".py", ".json", ".md", ".txt", ".log", ".sh", ".patch", ".diff"}


def descriptor(path):
    assert path.suffix in TEXT and path.resolve().is_relative_to(PHASE)
    raw = path.read_bytes()
    raw.decode("utf8")
    return {"path": str(path.relative_to(PHASE)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


assert descriptor(OLD / "MANIFEST.json")["sha256"] == "fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89"
old_rows = json.loads((OLD / "MANIFEST.json").read_text())["files"]
for row in old_rows:
    got = descriptor(OLD / row["path"])
    assert got["sha256"] == row["sha256"] and got["bytes"] == row.get("bytes", row.get("size"))

if len(sys.argv) == 2 and sys.argv[1] == "--seal":
    check = json.loads((HERE / "STATIC_SOURCE_CHECK.json").read_text())
    assert check["status"] == "PASS_SOURCE_ONLY" and len(check["python_files"]) == 20
    for row in check["python_files"]:
        got = descriptor(HERE / row["path"])
        assert got["sha256"] == row["sha256"] and got["bytes"] == row["bytes"]
    payload = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "SEAL.json"}:
            continue
        got = descriptor(path)
        if path.suffix == ".py":
            compile(ast.parse(path.read_text()), str(path), "exec")
        elif path.suffix == ".json":
            json.loads(path.read_text())
        payload.append({"path": str(path.relative_to(HERE)), "bytes": got["bytes"], "sha256": got["sha256"]})
    # Nested preserved historical manifests/seals are payloads too.
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and path.parent != HERE and path.name in {"MANIFEST.json", "SEAL.json"}:
            got = descriptor(path)
            payload.append({"path": str(path.relative_to(HERE)), "bytes": got["bytes"], "sha256": got["sha256"]})
    payload.sort(key=lambda row: row["path"])
    save("MANIFEST.json", {"schema": "ncnc-pattern-activation-checkpoint-source-manifest-v1", "UTC": UTC,
                           "files": payload, "status": "SOURCE_ONLY_INDEPENDENT_REVIEW_AND_EXACT_QUALIFICATION_REQUIRED",
                           "previous_V3_manifest": descriptor(OLD / "MANIFEST.json"),
                           "scientific_recipe_or_native_work_changed": False, "execution_authorized": False})
    save("SEAL.json", {"schema": "ncnc-pattern-activation-checkpoint-source-seal-v1", "UTC": UTC,
                       "manifest_path": "MANIFEST.json", "manifest_sha256": descriptor(HERE / "MANIFEST.json")["sha256"],
                       "manifest_bytes": descriptor(HERE / "MANIFEST.json")["bytes"],
                       "payload_count": len(payload), "payload_bytes": sum(row["bytes"] for row in payload),
                       "source_check": descriptor(HERE / "STATIC_SOURCE_CHECK.json"),
                       "Python_files_AST_compiled": len(check["python_files"]),
                       "numerical_or_project_or_model_or_fixture_execution": False,
                       "GPU_SSH_upload_or_launch": False, "execution_authorized": False,
                       "independent_successor_review_required": True, "caps_raised_again": False,
                       "both_full_graph_cap_failures_and_costs_preserved": True})
    for row in payload:
        got = descriptor(HERE / row["path"])
        assert got["bytes"] == row["bytes"] and got["sha256"] == row["sha256"]
    for path in HERE.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({"manifest": descriptor(HERE / "MANIFEST.json"), "seal": descriptor(HERE / "SEAL.json"),
                      "payload_count": len(payload), "payload_bytes": sum(row["bytes"] for row in payload)}))
    raise SystemExit(0)

assert len(sys.argv) == 1
for name in ("README.md", "PREPARATION_PROVENANCE.json", "PREPARATION_RESULT.json", "STATIC_SOURCE_CHECK.json", "ROOT_RELEASE_EXAMPLE.json"):
    (HERE / ("PRESERVED_V3_" + name)).write_bytes((OLD / name).read_bytes())
failures = []
for version in (1, 2):
    root = PHASE / ("graph_ncNC_structural_pattern_full_graph_execution_root_20261003_v" + str(version)) / "remote_receipts_observation01"
    evidence = []
    for relative in ("supervision/run01/SUPERVISOR_TERMINAL.json", "supervision/run01/CHILD_CUDA_PEAKS.json", "full_graph/run01/ATTEMPTS.json", "supervision/run01/CHILD.stderr.log", "ROOT_RELEASE_FULL_GRAPH.json"):
        source = root / relative
        original = descriptor(source)
        target = HERE / "PRESERVED_FULL_GRAPH_CAP_FAILURES" / ("v" + str(version)) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        preserved = descriptor(target)
        assert preserved["sha256"] == original["sha256"] and preserved["bytes"] == original["bytes"]
        evidence.append({"original": original, "preserved_copy": preserved})
    terminal = json.loads((root / "supervision/run01/SUPERVISOR_TERMINAL.json").read_text())
    attempts_document = json.loads((root / "full_graph/run01/ATTEMPTS.json").read_text())
    attempt = attempts_document["attempts"][0]
    assert terminal["status"] == "FAILED" and terminal["child_exit_code"] == 88 and terminal["qualification_receipt"] is None
    assert attempt["work"]["arm"] == "J" and attempt["work"]["completed_batches"] == 0
    failures.append({"attempt_version": version, "original_identity": attempts_document["identity"], "status": terminal["status"],
                     "child_exit_code": terminal["child_exit_code"], "cap_violation": terminal["cap_violation"],
                     "paid_supervisor_inclusive_wall_seconds": terminal["supervisor_inclusive_wall_seconds"],
                     "kernel_user_CPU_seconds": terminal["kernel_child_user_CPU_seconds"],
                     "kernel_system_CPU_seconds": terminal["kernel_child_system_CPU_seconds"],
                     "peak_observed_session_RSS_bytes": terminal["peak_observed_session_RSS_bytes"],
                     "kernel_child_peak_RSS_bytes": terminal["kernel_child_peak_RSS_bytes"],
                     "CUDA_allocated_peak_bytes": terminal["CUDA_peak_observation"]["cuda_peak_allocated_bytes"],
                     "CUDA_reserved_peak_bytes": terminal["CUDA_peak_observation"]["cuda_peak_reserved_bytes"],
                     "caps": terminal["CUDA_peak_observation"]["caps"], "completed_first_J_updates": 0,
                     "graph_metric_or_predictive_result": False, "evidence": evidence})
save("PRESERVED_FULL_GRAPH_FAILURES.json", {"schema": "ncnc-activation-checkpoint-predecessor-paid-failures-v1", "UTC": UTC,
    "failures": failures, "paid_wall_sum_seconds": sum(row["paid_supervisor_inclusive_wall_seconds"] for row in failures),
    "cost_authority": "Exact physical SUPERVISOR_TERMINAL.json fields; no approximate successor annotation substituted",
    "V1_annotation_discrepancy": "Earlier annotation26.045865029096603 differs from terminal26.04586512222886 by about9.3e-8seconds; original annotation is preserved in the old cap source packet.",
    "original_sources_and_receipts_edited": False, "caps_raised_again": False, "need_for_more_GPUs_established": False})
save("ACTIVATION_CHECKPOINT_POLICY.json", {
    "schema": "ncnc-auxiliary-depth-zero-checkpoint-policy-v1", "UTC": UTC,
    "policy_id": "complete_auxiliary_depth_zero_nonreentrant_rng_preserved_v1",
    "scope": "Captured differentiable auxiliary scorer calls only; complete native depth_zero bodies unchanged",
    "use_reentrant": False, "preserve_rng_state": True, "determinism_check": "default", "early_stop": False,
    "arguments": ["bound self.depth_zero callable", "transformed tensor", "immutable graph", "query tensor", "member integer"],
    "no_late_bound_member_closure": True, "no_new_chunking": True, "no_empty_query_shortcut": True,
    "in_place_dropout_analysis": "Inherited dropout overwrites newly created linear or normalization results, never the checkpoint transformed/query inputs or parameters. Entire depth_zero is replayed; its original dropout sequence and RNG snapshot are preserved.",
    "graph_analysis": "Frozen Graph fields and enumerate_neighbors/feature_sum are read-only and have no enumeration cache or stochastic mutation. Backward re-enumeration uses the same graph/query and exact ordering; extra enumeration work is charged.",
    "shared_parameters_analysis": "The existing scorer and target retain the same shared parameters/member factors. Bound callable and explicit member fix recomputation identity. One total.backward finishes all recomputation before unchanged Adam.step; module training flags and parameters remain stable during that backward.",
    "RNG_analysis": "The qualified depth_zero uses only Torch randomness and a single already initialized CUDA device exposed by transformed/query arguments; it makes no device moves. preserve_rng_state restores Torch CPU/CUDA caller state after recomputation; exact all-RNG probes independently require it.",
    "unchanged": ["inherited target forward order/values", "target score detach boundary", "full graph/support/teacher", "original native batch65536", "candidate_split_size=-1", "J/F losses/lambda1", "Adam/initialization/seeds", "100epoch/scientific selectors", "existing34native+6replay full-graph qualification and two full five-route VALID traversals"],
    "expected_effect_not_measured": "Retain fewer scorer activations and pay recomputation; full-call temporaries, native outer activations, output scores and allocator reservations may still exceed physical caps.",
    "physical_caps_ceiling_retained_from_prior_candidate": failures[1]["caps"],
    "numerical_parity_or_memory_feasibility_verified": False, "source_or_execution_authorized": False,
})
save("V4_ENGINEERING_QUALIFICATION_PLAN.json", {
    "schema": "ncnc-activation-checkpoint-engineering-qualification-plan-v1", "UTC": UTC,
    "status": "PROSPECTIVELY_FIXED_NOT_EXECUTED", "reference": {"manifest": descriptor(OLD / "MANIFEST.json"), "model": descriptor(OLD / "pattern_model.py")},
    "sequential_GPU_variants": ["saved exact V3", "current V4"], "arms": ["J", "F"],
    "fixed_probe_record_ids": [0, 2, 3, 5], "fixed_negative_pairs": [[7,9],[8,10],[9,11],[10,12]],
    "fabricated_data": "Existing exact V3 fixture:16nodes,128features,10records including duplicates and isolated nodes",
    "bounded_real_graph": "Complete235868nodes/1179052TRAINrecords/128features; four fixed query records only for parity, no reduced graph/support",
    "actual_Adam_updates_each_variant_per_arm": 2, "new_audit_optimizer_updates_per_stage": 8,
    "new_audit_member_trajectory_updates_per_stage": 32,
    "additional_J_arm_probes": ["auxiliary_grad=False full positive/negative forward and RNG", "empty captured query full native forward/backward and RNG"],
    "checks": ["exact initial model/Adam/RNG/flags", "complete forward logits/raw scores/t/clamped scores/J/F/counts/rho/support/order", "main-target detach boundary", "all named parameter gradients", "actual next model/Adam after each of two updates", "exact forward/backward/next-step all-RNG and flags", "caller RNG restored; no engineering state donation"],
    "arithmetic_tolerance": "unchanged atol=rtol=128*finfo(dtype).eps; exact integer/metadata/RNG checks",
    "existing_numerical_serialized_replay_updates": 6, "new_numerical_total_engineering_updates": 14,
    "existing_full_graph_native_updates": 34, "existing_full_graph_serialized_replay_updates": 6,
    "new_full_graph_total_engineering_updates": 48, "unchanged_complete_five_route_VALID_traversals": 2,
    "cost_scope": "All baseline/candidate updates, CPU snapshots/comparisons, empty/disabled forward/backward probes and checkpoint re-enumeration/recomputation are in measured stage phases and cumulative pre-reset CUDA peaks. No cost-free audit or full native work reduction.",
    "admission": "Exact current numerical PASS with parity must precede full_graph; exact full_graph PASS with parity plus complete original work must precede scientific fit. Saved V3 qualification is historical, not a current substitute.",
    "cap_increase_retry_or_budget_shrink_authorized": False, "numerical_or_full_graph_or_fit_executed": False,
})
changed = [row["path"] for row in old_rows if (HERE / row["path"]).read_bytes() != (OLD / row["path"]).read_bytes()]
python_changes = [name for name in changed if name.endswith(".py")]
assert set(python_changes) == {"pattern_model.py", "pattern_qualification.py", "pilot_common.py"}
diff = ""
for name in python_changes:
    diff += "".join(difflib.unified_diff((OLD/name).read_text().splitlines(keepends=True), (HERE/name).read_text().splitlines(keepends=True), fromfile=str((OLD/name).relative_to(PHASE)), tofile=str((HERE/name).relative_to(PHASE))))
(HERE / "V3_TO_V4.diff").write_text(diff)
save("V4_SOURCE_DIFF_EVIDENCE.json", {"schema": "ncnc-V4-checkpoint-source-diff-v1", "UTC": UTC,
    "changed_existing_Python_files": python_changes, "new_runtime_Python": ["checkpoint_parity.py"],
    "new_stdlib_metadata_utility": "prepare_successor_metadata.py", "exact_diff": descriptor(HERE / "V3_TO_V4.diff"),
    "scientific_and_shared_byte_identical": [name for name in ["PILOT_PLAN.json", "pattern_train.py", "pattern_objective.py", "pattern_checks.py", "pattern_teacher.py", "pattern_fit.py", "pattern_diagnostics.py", "pattern_close.py", "pattern_run.py", "pilot_model.py", "pilot_data.py", "pilot_state.py", "pilot_evaluate.py", "pilot_accounting.py", "DEPENDENCIES.json"] if (HERE/name).read_bytes() == (OLD/name).read_bytes()],
    "all_other_original_Python_bodies_byte_identical": True, "source_check_byte_identical": True,
    "scientific_recipe_and_native_batch_and_full_work_changed": False, "source_or_numerical_execution": False,
})
example = json.loads((OLD / "ROOT_RELEASE_EXAMPLE.json").read_text())
example = json.loads(json.dumps(example).replace("graph_ncNC_structural_pattern_execution_root_20261003_v2", "graph_ncNC_structural_pattern_execution_root_20261003_v4").replace("graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2", "graph_ncNC_structural_pattern_numerical_execution_root_20261003_v4"))
example["preserved_predecessor_failure_custody"] = "Preserved V2 numerical failure plus both V3-driver full-graph cap failures in PRESERVED_FULL_GRAPH_FAILURES.json. Root carries original identities/costs separately."
example["activation_checkpoint_source_review_required"] = True
example["current_V4_numerical_and_full_graph_parity_receipts_required"] = True
assert example["authorized_stages"] == [] and example["authorized_invocations"] == [] and example["root_authorization_reference"] is None
save("ROOT_RELEASE_EXAMPLE.json", example)
save("PREPARATION_PROVENANCE.json", {"schema": "ncnc-V4-activation-checkpoint-preparation-provenance-v1", "UTC": UTC,
    "scope": "Root-authorized minimal engineering successor after two paid full-graph cap failures of exact V3",
    "previous_source": {"manifest": descriptor(OLD / "MANIFEST.json"), "seal": descriptor(OLD / "SEAL.json")},
    "implementation": "Only auxiliary scorer activation retention changes; complete nonreentrant depth_zero checkpoint with preserved RNG and disabled early stop",
    "new_primary_reads_or_public_retrievals": 0, "project_or_numerical_or_model_or_fixture_execution": False,
    "Torch_NumPy_or_runtime_imports_executed": False, "arrays_archives_checkpoints_labels_runtime_binaries_opened": False,
    "GPU_SSH_upload_or_launch": False, "subagents": False, "canonical_or_existing_source_runtime_or_running_family_edits": False,
    "caps_raised_again": False, "more_GPU_need_established": False, "memory_feasibility_established": False,
    "independent_review_and_fresh_current_qualification_before_launch": True})
save("PREPARATION_RESULT.json", {"schema": "ncnc-V4-activation-checkpoint-preparation-result-v1", "UTC": UTC,
    "status": "SOURCE_ONLY_PENDING_INDEPENDENT_REVIEW_AND_RUNTIME_PARITY", "entrypoint": "pattern_run.py",
    "modified_existing_Python": python_changes, "new_runtime_Python": "checkpoint_parity.py",
    "scientific_recipe_and_native_complete_work_preserved": True, "paid_failures_preserved": "PRESERVED_FULL_GRAPH_FAILURES.json",
    "parity_plan": "V4_ENGINEERING_QUALIFICATION_PLAN.json", "policy": "ACTIVATION_CHECKPOINT_POLICY.json",
    "source_check": "STATIC_SOURCE_CHECK.json", "numerical_parity_verified": False, "full_graph_feasibility_verified": False,
    "execution_authorized": False, "need_for_more_GPU_established": False})
(HERE / "README.md").write_text("""# V4 auxiliary scorer activation-checkpoint source successor

Status: source preparation only. Independent exact source review, fresh numerical parity and unchanged complete full-graph resource qualification are required. No upload, runtime, fit, release or canonical edit occurred.

The exact saved V3 pilot remains immutable. This successor checkpoints only captured differentiable auxiliary `depth_zero` calls using explicit `use_reentrant=False`, `preserve_rng_state=True`, `determinism_check='default'` and `set_checkpoint_early_stop(False)`. Each complete scorer body and original member/left/right order remain. Native target scores remain detached; uncaptured target/inference and auxiliary-disabled paths remain direct. No new chunking or empty-query shortcut is introduced.

Inherited in-place dropout changes fresh linear/normalization results, not checkpoint inputs. Graph enumeration has no mutable cache and is repeated from the same graph/query. Explicit callable/input/member bindings avoid delayed closure identities. Parameters and training flags remain stable until all backward work completes, followed by the unchanged Adam step. Recomputation pays real time and memory; feasibility is unproved.

The prospective parity code loads exact saved V3 source and compares sequential V3/V4 J/F trajectories on the existing fabricated fixture and the complete real graph with four fixed query records. It checks full forward/support, target detach boundary, all gradients, exact RNG/flags and two actual Adam updates per variant/arm. Disabled auxiliary and empty-query forward/backward probes are fixed too. Existing arithmetic tolerances are unchanged. No runtime parity or fabricated test was run during preparation.

Parity adds eight paid audit optimizer updates per stage (32 member trajectories). It preserves numerical serialized replay's six updates and full-graph qualification's 34 native + 6 replay updates plus two complete five-route VALID traversals. Full-graph total including added parity is 48 engineering updates. The native batch remains 65,536 and full scientific schedule/selectors remain unchanged. All additional probes, CPU comparisons and recomputation are measured inside stage accounting and cumulative pre-reset peaks. No engineering state or RNG donates into a fit.

Both previous full-graph child-exit-88 failures are retained with exact physical terminal authority: 26.04586512222886 seconds at 46,954,556,416 allocated / 59,624,128,512 reserved bytes, and 29.33481625840068 seconds at 73,043,022,336 / 83,332,431,872 bytes. Both failed before the first completed J update and produced no graph metric. No cap increase or more-GPU conclusion is made. The prior 70/75 GiB caps are the prospective ceiling, subject to root admission; they are not a feasibility certificate.

`ROOT_RELEASE_EXAMPLE.json` remains disabled. Prepared command paths select fresh V4 roots; launch support must be separately reviewed/bound to exact V4 with fresh physical resource admission. The prior V3 numerical PASS is historical and cannot replace V4 parity. Details: ACTIVATION_CHECKPOINT_POLICY.json, V4_ENGINEERING_QUALIFICATION_PLAN.json, V4_SOURCE_DIFF_EVIDENCE.json, V3_TO_V4.diff and PRESERVED_FULL_GRAPH_FAILURES.json.
""")
print(json.dumps({"metadata_prepared": True, "old_payloads_preserved": len(old_rows), "changed_python": python_changes,
                  "failure_paid_wall_seconds": [row["paid_supervisor_inclusive_wall_seconds"] for row in failures],
                  "runtime_executed": False}))
