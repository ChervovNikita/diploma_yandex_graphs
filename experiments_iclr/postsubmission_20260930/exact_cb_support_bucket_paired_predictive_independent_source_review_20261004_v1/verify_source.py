"""Independent stdlib-only byte/JSON/AST verification; never imports project code."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import json
import stat
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CANDIDATE = PHASE / "exact_cb_support_bucket_paired_predictive_preparation_20261004_v1"
MANIFEST_SHA = "0d0899d94f0a20d317f7e30491f3ed5b35c293d462f28a0baf3f4ea350e9d229"
SEAL_SHA = "00cfa256ce00a9dc19ae32f27b8535f5ea9b069ff92b48377a5244efe6f33e7d"
ALLOWED_KINDS = {".py", ".json", ".md", ".txt", ".patch", ".diff", ".gitignore"}
checks = []


def check(name, value, detail=None):
    checks.append({"check": name, "passed": bool(value), "detail": detail})
    if not value:
        raise AssertionError(name)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def pin(root, row, label):
    path = root / row["path"]
    check(label + ": confined text path", not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
          and path.suffix in ALLOWED_KINDS or (not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
                                             and path.name == ".gitignore"), row["path"])
    check(label + ": bytes and SHA256", path.stat().st_size == row.get("bytes", row.get("size"))
          and digest(path) == row["sha256"], row)
    return path


def definition(path, name):
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name)


def ast_same(new, old, names, label):
    for name in names:
        check(label + ": " + name, ast.dump(definition(new, name), include_attributes=False)
              == ast.dump(definition(old, name), include_attributes=False))


def main():
    check("candidate MANIFEST exact requested SHA256", digest(CANDIDATE / "MANIFEST.json") == MANIFEST_SHA)
    check("candidate SEAL exact requested SHA256", digest(CANDIDATE / "SEAL.json") == SEAL_SHA)
    manifest, seal = read(CANDIDATE / "MANIFEST.json"), read(CANDIDATE / "SEAL.json")
    check("seal binds candidate manifest", seal["manifest_sha256"] == seal["manifest"]["sha256"] == MANIFEST_SHA
          and seal["manifest"]["bytes"] == (CANDIDATE / "MANIFEST.json").stat().st_size)
    check("source packet immutable directory mode", stat.S_IMODE(CANDIDATE.stat().st_mode) == 0o555)
    for row in manifest["files"]:
        path = pin(CANDIDATE, row, "candidate payload")
        check("candidate payload 0444: " + row["path"], stat.S_IMODE(path.stat().st_mode) == 0o444)
    check("manifest/seal modes 0444", all(stat.S_IMODE((CANDIDATE / name).stat().st_mode) == 0o444
                                        for name in ("MANIFEST.json", "SEAL.json")))
    check("candidate file inventory exact", {p.name for p in CANDIDATE.iterdir()}
          == {r["path"] for r in manifest["files"]} | {"MANIFEST.json", "SEAL.json"})
    ast_files = []
    for path in sorted(CANDIDATE.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
        ast_files.append(path.name)
    check("15 candidate Python files AST parse only", len(ast_files) == 15, ast_files)
    plan, bindings = read(CANDIDATE / "PILOT_PLAN.json"), read(CANDIDATE / "INPUT_BINDINGS.json")
    for row in plan["source_pins"]:
        pin(PHASE, row, "dependency source/receipt")
    dependency_manifests = []
    for row in plan["source_pins"]:
        if row["path"].endswith("/MANIFEST.json"):
            path = PHASE / row["path"]
            data = read(path)
            for entry in data["files"]:
                pin(path.parent, entry, "dependency manifest payload")
            dependency_manifests.append({"path": row["path"], "sha256": row["sha256"], "payloads": len(data["files"])})
    source_by_path = {r["path"]: r for r in plan["source_pins"]}
    for directory, expected in bindings["dependency_manifest_sha256"].items():
        check("declared dependency manifest exact: " + directory,
              source_by_path[directory + "/MANIFEST.json"]["sha256"] == expected)
    native = PHASE / "graph_ncNC_structural_pattern_pilot_preparation_20261004_v5"
    unchanged = ("pilot_data.py", "pilot_state.py", "pilot_model.py", "pilot_accounting.py", "pilot_evaluate.py",
                 "pattern_model.py", "pattern_teacher.py")
    for name in unchanged:
        check("unchanged V5 helper bytes: " + name, (CANDIDATE / name).read_bytes() == (native / name).read_bytes())
    conditional = PHASE / "exact_cb_support_bucket_implementation_hypothesis_preparation_20261004_v2" / "conditional_loss.py"
    check("conditional_loss exact optimized source bytes", (CANDIDATE / "conditional_loss.py").read_bytes() == conditional.read_bytes())
    ast_same(CANDIDATE / "pilot_common.py", native / "pilot_common.py",
             ("require", "file_sha", "read_bound_json", "verify_manifest", "utc", "atomic_json", "require_profile_receipt",
              "runtime_stdlib", "fresh_output", "lock_output"), "unchanged common AST")
    ast_same(CANDIDATE / "paired_train.py", native / "pattern_train.py", ("finite",), "unchanged finite helper AST")
    old_supervision = PHASE / "graph_ncNC_structural_pattern_fit_normal_supervision_preparation_20261004_v1"
    ast_same(CANDIDATE / "supervise_fit.py", old_supervision / "supervise_fit.py",
             ("require", "sha", "utc", "write", "verify_packet", "physical", "session_rss"), "unchanged supervisor helper AST")
    ast_same(CANDIDATE / "fit_child.py", old_supervision / "fit_child.py", ("require", "write", "peaks", "merge"),
             "unchanged child helper AST")
    check("fixed seeds arms budget coefficient", plan["base_seeds"] == [0, 1, 2]
          and plan["arms"] == ["target_only", "joint", "separate"] and plan["epochs"] == 100
          and plan["full_batches_per_epoch"] == 17 and plan["train_batch_size"] == 65536
          and plan["lambda"] == 1 and plan["members"] == 4 and plan["total_parameters"] == 43790)
    invocations = plan["root_release_invocations_NOT_AUTHORIZATION"]
    expected_cells = {(s, a) for s in (0, 1, 2) for a in ("target_only", "joint", "separate")}
    check("exact nine prospective cells", len(invocations) == 9
          and {(r["base_seed"], r["unit"]) for r in invocations} == expected_cells
          and len({r["output_directory"] for r in invocations}) == 9)
    server_phase = str(Path(plan["repository"]) / "experiments_iclr" / "postsubmission_20260930")
    for row in invocations:
        expected = str(Path(server_phase) / plan["execution_directory"]
                       / ("fit_" + row["unit"] + "_seed" + str(row["base_seed"])) / "run01")
        check("exact cell path: " + row["unit"] + str(row["base_seed"]), row["stage"] == "fit"
              and row["output_directory"] == expected)
    check("total work exact", plan["optimizer_updates_per_fit"] == 1700
          and plan["scientific_optimizer_updates"] == 15300 and plan["scientific_fits"] == 9
          and plan["selector_candidates_per_fit"] == 100 and plan["scientific_selector_candidates"] == 900
          and plan["complete_VALID_traversals_per_fit"] == 102 and plan["scientific_complete_VALID_traversals"] == 918)
    caps = {"wall_seconds": 86400, "host_RSS_bytes": 32 * 1024**3,
            "cuda_peak_allocated_bytes": 70 * 1024**3, "cuda_peak_reserved_bytes": 75 * 1024**3}
    check("exact existing caps GPU1 and arithmetic", plan["caps"] == caps
          and plan["GPU_UUID"] == "GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced"
          and plan["arithmetic_rule"] == {"atol": 128 * 2**-23, "rtol": 128 * 2**-23})
    release = read(CANDIDATE / "ROOT_RELEASE_TEMPLATE.json")
    check("release template disabled and unbound", release["execution_enabled"] is False
          and release["authorized_invocations"] == [] and release["independent_source_review"] is None
          and release["root_authorization_reference"] is None and release["driver_manifest_sha256"] is None
          and release["fit_supervision_manifest_sha256"] is None and release["TEST_supported"] is False)
    check("plan no execution TEST retries donors fallback grid", plan["execution_authorized"] is False
          and plan["TEST_supported"] is False and plan["Collab_TEST_history_consumed"] is True
          and plan["automatic_retry_or_resume"] is False and plan["no_state_donors"] is True
          and plan["no_precision_batch_budget_normalization_fallback"] is True and plan["no_hyperparameter_grid"] is True)
    mapping = {}
    for entry in bindings["server_receipt_to_local_mirror_mapping"]:
        local, canonical = entry["local_source_pin"], entry["server_canonical_receipt"]
        check("local/canonical bytes/hash equality: " + local["path"], local["bytes"] == canonical["bytes"]
              and local["sha256"] == canonical["sha256"] and source_by_path[local["path"]] == local
              and entry["staged_local_mirror_is_not_a_substitute_for_server_canonical_receipt"] is True)
        mapping[canonical["path"]] = read(PHASE / local["path"])
    for name, qpin in plan["native_qualification"].items():
        value, terminal = mapping[qpin["path"]], mapping[plan["native_qualification_terminals"][name]["path"]]
        transition, final = value["runtime_profile_transition"], value["runtime_profile_final"]
        before, after = transition["actual_profile_before"], transition["actual_profile_after"]
        check("actual native qualification: " + name, value["status"] == "PASS"
              and value["identity"]["driver_manifest_sha256"] == plan["native_driver_manifest_sha256"]
              and value["identity"]["prototype_manifest_sha256"] == bindings["dependency_manifest_sha256"]["graph_ncNC_member_completion_qualification_preparation_20261003_v2"]
              and value["state_donor"] is False and value["test_file_opened"] is False)
        check("actual native profile transition: " + name, transition["status"] == "TRANSITION_VERIFIED"
              and transition["declared_transition"] == plan["runtime_profile_transition"]
              and transition["ordinary_runtime_source_binary_admission_completed"] is True
              and before["deterministic_algorithms"] is False and before["deterministic_warn_only"] is False
              and after == dict(before, deterministic_algorithms=True, deterministic_warn_only=False)
              and transition["RNG_before_sha256"] == transition["RNG_after_sha256"]
              and transition["RNG_components_before_sha256"] == transition["RNG_components_after_sha256"]
              and transition["RNG_exactly_unchanged"] is True and final["actual_runtime_profile"] == after
              and final["runtime_profile_exact"] is True)
        check("actual native physical terminal: " + name, terminal["status"] == "COMPLETE"
              and terminal["child_exit_code"] == 0 and terminal["child_signal"] is None
              and terminal["cap_violation"] is None and terminal["ordinary_host_execution"] is True
              and terminal["qualification_receipt"] == qpin)
    bp = plan["native_bucket_prerequisites"]
    diagnostic, terminal, final, custody = (mapping[bp[k]["path"]] for k in ("diagnostic", "terminal", "final", "supervisor_custody"))
    check("actual exact bucket equivalence receipt", diagnostic["status"] == "COMPLETE_DIAGNOSTIC_ONLY"
          and diagnostic["equivalence_verdict"] == "PASS_FIXED_ORIGINAL_RULE"
          and diagnostic["source_manifest_sha256"] == plan["native_bucket_source_manifest_sha256"]
          and diagnostic["candidate_manifest_sha256"] == plan["conditional_manifest_sha256"]
          and diagnostic["core_manifest_sha256"] == plan["core_manifest_sha256"]
          and diagnostic["optimizer_updates"] == 0 and diagnostic["VALID_TEST_reads"] is False)
    check("actual exact bucket owned terminal/custody", terminal["status"] == "COMPLETE_DIAGNOSTIC_ONLY"
          and terminal["physical_session_closed"] is True and terminal["direct_child_reaped"] is True
          and terminal["physical_exit_code"] == 0 and terminal["stop"] is None
          and terminal["linked"]["status"] == "COLLECTED_COMPLETE_DIAGNOSTIC_ONLY"
          and terminal["linked"]["result_sha256"] == bp["diagnostic"]["sha256"]
          and terminal["linked"]["final_custody_sha256"] == bp["final"]["sha256"]
          and custody["terminal_sha256"] == bp["terminal"]["sha256"] and final["completed"] is True)
    check("no project or numerical modules imported", not any(name in sys.modules for name
          in ("torch", "numpy", "pandas", "scipy", "pilot_common", "prototype", "graph_ops", "paired_train", "paired_fit")))
    result = {"schema": "ncnc-exact-CB-paired-predictive-independent-static-verification-v1",
              "status": "PASS", "candidate_manifest_sha256": MANIFEST_SHA, "candidate_seal_sha256": SEAL_SHA,
              "UTC": datetime.now(timezone.utc).isoformat(), "checks_passed": len(checks), "checks": checks,
              "candidate_payload_files": len(manifest["files"]), "candidate_AST_parsed_files": ast_files,
              "dependency_source_receipt_pins_verified": len(plan["source_pins"]), "dependency_manifests": dependency_manifests,
              "local_canonical_receipt_mappings_verified": len(mapping), "unchanged_native_helpers": list(unchanged),
              "project_or_numerical_module_imports": False, "project_code_execution": False,
              "datasets_or_checkpoints_read": False, "server_access_staging_or_launch": False,
              "candidate_source_mutation": False, "review_scope": "local text, byte hashes, JSON and AST only"}
    (HERE / "VERIFICATION.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_passed", "candidate_payload_files",
          "dependency_source_receipt_pins_verified", "local_canonical_receipt_mappings_verified")}))


if __name__ == "__main__":
    main()
