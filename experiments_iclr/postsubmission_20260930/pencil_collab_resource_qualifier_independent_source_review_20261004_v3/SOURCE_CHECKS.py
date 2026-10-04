"""Bounded independent v2-to-v3 text/JSON/hash/AST checks, no target imports."""
from pathlib import Path
from hashlib import sha256, sha1
from datetime import datetime, timezone
import ast
import copy
import difflib
import json
import stat
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
V2 = PHASE / "pencil_collab_resource_qualifier_preparation_20261004_v2"
V3 = PHASE / "pencil_collab_resource_qualifier_preparation_20261004_v3"
PRIOR_REVIEW = PHASE / "pencil_collab_resource_qualifier_independent_source_review_20261004_v2"
SOURCE_SHA = "211c140f7aa95e9af6395d841206160bb50aa8b20980b4b00d75a5a79a8291e6"
SEAL_SHA = "29789295827a485a0702854e2040a4abf112e0cc8846ff1adeeeac6b49b9e55e"
PLAN_SHA = "6c423bc7a28f7acdab622c75472413716230db0a081bdc84e0a5929f0e229aa7"
checks = []


def check(name, value, evidence=None):
    checks.append(dict(check=name, passed=bool(value), evidence=evidence))
    if not value:
        raise AssertionError(name)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def bound(root, row, label):
    path = root / row["path"]
    check(label + ": path/bytes/hash", path.is_file() and not path.is_symlink()
          and path.resolve().is_relative_to(root.resolve())
          and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], row)
    return path


def main():
    check("exact v3 requested manifest/seal/plan", digest(V3 / "MANIFEST.json") == SOURCE_SHA
          and digest(V3 / "SEAL.json") == SEAL_SHA and digest(V3 / "PLAN.json") == PLAN_SHA)
    manifest, seal, plan, provenance = (read(V3 / name) for name
                                      in ("MANIFEST.json", "SEAL.json", "PLAN.json", "V2_TO_V3_PROVENANCE.json"))
    check("v3 seal binds exact source", seal["manifest_sha256"] == SOURCE_SHA)
    for row in manifest["files"]:
        bound(V3, row, "v3 payload")
    check("v3 exact file inventory", {str(p.relative_to(V3)) for p in V3.rglob("*") if p.is_file()}
          == {r["path"] for r in manifest["files"]} | {"MANIFEST.json", "SEAL.json"})
    check("v3 immutable modes", all(stat.S_IMODE(p.stat().st_mode) == (0o555 if p.is_dir() else 0o444)
                                   for p in (V3, *V3.rglob("*"))))
    check("exact preserved v2 source manifest/seal", digest(V2 / "MANIFEST.json") == provenance["parent_manifest_sha256"]
          == "1285a03bcc4a790162f6cebf1d49548c2c6ac18788b2008776e12038acbbf984"
          and digest(V2 / "SEAL.json") == provenance["parent_seal_sha256"]
          == "dbc31991be21d07e20a2b14363df0bd0701ff55ff376492a5dc4fd7dc4e0f890")
    prior_manifest = read(V2 / "MANIFEST.json")
    for row in prior_manifest["files"]:
        bound(V2, row, "preserved v2 payload")
    prior_review_manifest, prior_review_seal, prior_review = (read(PRIOR_REVIEW / name) for name
                                                           in ("MANIFEST.json", "SEAL.json", "REVIEW.json"))
    check("preserved v2 independent review manifest/seal", digest(PRIOR_REVIEW / "MANIFEST.json")
          == prior_review_seal["manifest_sha256"] == "c32a4b5e5e93fd1ce74df06b1bf1caa2f3f8f540500d7b472b52f651dc3fef97")
    for row in prior_review_manifest["files"]:
        bound(PRIOR_REVIEW, row, "preserved v2 independent review payload")
    check("v2 PASS scope reused explicitly", prior_review["status"] == "PASS"
          and prior_review["blocking_findings"] == [] and prior_review["execution_authorized"] is False
          and prior_review["candidate_manifest_sha256"] == provenance["parent_manifest_sha256"]
          and prior_review["reviewer_context"]["fresh_zero_context_review"] is False)
    bindings = read(V3 / "INPUT_BINDINGS.json")
    for row in bindings["inputs"]:
        bound(PHASE, row, "declared external text/metadata input")
    old_bindings = read(V2 / "INPUT_BINDINGS.json")
    check("all v2 external input pins retained", bindings["inputs"][:len(old_bindings["inputs"])] == old_bindings["inputs"])
    old_rows, new_rows = ({r["path"]: r for r in m["files"]} for m in (prior_manifest, manifest))
    excluded = set(provenance["derived_control_files_excluded_from_self_diff"])
    changed = {name for name in set(old_rows) | set(new_rows) if name not in excluded
               and (name not in old_rows or name not in new_rows or old_rows[name]["sha256"] != new_rows[name]["sha256"])}
    check("complete declared nonderived delta", changed == {r["path"] for r in provenance["changed_or_added_payloads"]})
    unchanged = {name for name in set(old_rows) & set(new_rows) if name not in excluded
                 and old_rows[name]["sha256"] == new_rows[name]["sha256"]}
    check("complete declared unchanged payloads", unchanged == set(provenance["unchanged_payloads"]))
    parts = []
    for row in provenance["changed_or_added_payloads"]:
        name = row["path"]
        before = (V2 / name).read_bytes() if (V2 / name).exists() else None
        after = (V3 / name).read_bytes()
        check("declared delta before/after hashes: " + name,
              row["before"] == (None if before is None else {"bytes": len(before), "sha256": sha256(before).hexdigest()})
              and row["after"] == {"bytes": len(after), "sha256": sha256(after).hexdigest()})
        parts.extend(difflib.unified_diff([] if before is None else before.decode().splitlines(keepends=True),
                     after.decode().splitlines(keepends=True), fromfile="/dev/null" if before is None else "v2/" + name,
                     tofile="v3/" + name))
    diff = "".join(parts).encode()
    check("exact V2_TO_V3.diff reproduced from bound originals", diff == (V3 / "V2_TO_V3.diff").read_bytes()
          and sha256(diff).hexdigest() == provenance["diff_sha256"])
    native = read(V3 / "NATIVE_SOURCE_BINDINGS.json")
    check("native binding metadata unchanged", (V3 / "NATIVE_SOURCE_BINDINGS.json").read_bytes()
          == (V2 / "NATIVE_SOURCE_BINDINGS.json").read_bytes()
          and native["commit"] == "2d32e29dbed533288d9d758138e07547a0a7d8a9" and len(native["files"]) == 20)
    for row in native["files"]:
        data = (V3 / row["path"]).read_bytes()
        check("native byte/origin/Git identity: " + row["path"], data == (V2 / row["path"]).read_bytes()
              == (PHASE / row["origin"]).read_bytes() and len(data) == row["bytes"]
              and sha256(data).hexdigest() == row["sha256"]
              and sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == row["git_blob_sha1"])
    check("supervisor common data config authorities dependency closures unchanged", all((V3 / name).read_bytes()
          == (V2 / name).read_bytes() for name in ("supervise.py", "common.py", "data_adapter.py", "metadata/config.json",
          "metadata/RUNTIME_AUTHORITY.json", "metadata/DATA_AUTHORITY.json", "OFFICIAL_CONFIG_REFERENCE.yaml",
          "DEPENDENCY_ADMISSION_TEMPLATE.json", "DEPENDENCY_INSTALL_PLAN.json", "ACTUAL_DEPENDENCY_BINDING.json",
          "dependency_requirements.txt", "dependency_existing_constraints.txt")))
    old_plan = read(V2 / "PLAN.json")
    policy = dict(native_constructor_pin_memory=True, TRAIN=False, VALID=False, harness_override_before_iterator=True)
    check("exact plan transfer policy", plan["loader_pin_memory_policy"] == policy)
    restored = copy.deepcopy(plan)
    restored.pop("loader_pin_memory_policy")
    restored["execution_directory"] = old_plan["execution_directory"]
    check("one appended declared protocol deviation", restored["protocol_differences"][:-1] == old_plan["protocol_differences"])
    restored["protocol_differences"] = restored["protocol_differences"][:-1]
    check("all remaining plan fields unchanged", restored == old_plan)
    old_worker, new_worker = (ast.parse((v / "worker.py").read_text()) for v in (V2, V3))
    new_main = next(n for n in new_worker.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignments = [n for n in ast.walk(new_main) if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == "pin_memory"]
    check("only two pin_memory assignments exactly False", len(assignments) == 2
          and {ast.unparse(n.targets[0]) for n in assignments} == {"train_loader.pin_memory", "valid_loader.pin_memory"}
          and all(isinstance(n.value, ast.Constant) and n.value.value is False for n in assignments))
    pin_requires = [n for n in new_main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "require" and ".pin_memory" in ast.unparse(n.value.args[0])]
    check("native/effective pin policy assertions", len(pin_requires) == 2
          and ast.unparse(pin_requires[0].value.args[0]) == "train_loader.pin_memory is True and valid_loader.pin_memory is True"
          and ast.unparse(pin_requires[1].value.args[0]) == "train_loader.pin_memory is False and valid_loader.pin_memory is False")
    calls = [n for n in ast.walk(new_main) if isinstance(n, ast.Call)]
    call = lambda name: next(n for n in calls if ast.unparse(n.func) == name)
    build, train, evaluate = (call("run_lp." + name) for name in ("build_loaders", "train_loop", "evaluate_loop"))
    check("override ordering before train/evaluation", build.end_lineno < pin_requires[0].lineno
          < assignments[0].lineno < assignments[1].lineno < pin_requires[1].lineno < train.lineno < evaluate.lineno)
    native_ast = ast.parse((V3 / "native/run_lp.py").read_text())
    native_build = next(n for n in native_ast.body if isinstance(n, ast.FunctionDef) and n.name == "build_loaders")
    check("native build_loaders creates no loader iterator", not any(isinstance(n, ast.Call)
          and ast.unparse(n.func) in ("iter", "next", "enumerate") for n in ast.walk(native_build)))
    new_main.body = [n for n in new_main.body if n not in assignments and n not in pin_requires]
    result_call = next(n for n in ast.walk(new_main) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)).value
    disclosure = [k for k in result_call.keywords if k.arg == "loader_pin_memory_policy"]
    check("exact resource policy disclosure", len(disclosure) == 1
          and ast.unparse(disclosure[0].value) == "plan['loader_pin_memory_policy']")
    result_call.keywords = [k for k in result_call.keywords if k.arg != "loader_pin_memory_policy"]
    check("entire worker AST unchanged except four policy statements and receipt keyword",
          ast.dump(new_worker, include_attributes=False) == ast.dump(old_worker, include_attributes=False))
    failure = read(V3 / "V2_FAILURE_OBSERVATION.json")
    for row in failure["snapshots"]:
        data = (V3 / row["snapshot"]).read_bytes()
        check("preserved actual failure snapshot: " + row["snapshot"], data == (PHASE / row["path"]).read_bytes()
              and len(data) == row["bytes"] and sha256(data).hexdigest() == row["sha256"])
    progress = read(V3 / "failure_snapshot_v2/run01/PROGRESS.json")
    physical, terminal, custody = (read(V3 / "failure_snapshot_v2/supervision/run01" / name)
                                  for name in ("PHYSICAL_TERMINAL.json", "TERMINAL.json", "SUPERVISOR_CUSTODY.json"))
    stderr = (V3 / "failure_snapshot_v2/supervision/run01/STDERR.txt").read_text()
    check("actual seven-batch zero-update no-VALID failure preserved", failure["progress"] == progress
          and progress["TRAIN_batches"] == 7 and progress["TRAIN_queries"] == 7168
          and progress["optimizer_updates"] == progress["VALID_batches"] == progress["VALID_queries"] == 0
          and progress["TEST_reads"] is False and progress["scores_saved"] is False)
    check("reported pinning API and asynchronous caveat", all(s in stderr for s in
          ("Caught RuntimeError in pin memory thread for device 0", "data.pin_memory(device)",
           "CUDA error: invalid argument", "CUDA kernel errors might be asynchronously reported"))
          and failure["root_cause_established"] is False and failure["resource_adoption"] is False)
    check("actual failed physical and custody links", physical["physical_exit_code"] == terminal["physical_exit_code"] == 1
          and physical["physical_session_closed"] is True and physical["direct_child_reaped"] is True
          and physical["errors"] == [] and physical["unresolved_cleanup"] is False
          and terminal["status"] == custody["status"] == "FAILED_NO_RESOURCE_ADOPTION"
          and terminal["physical_terminal_sha256"] == digest(V3 / "failure_snapshot_v2/supervision/run01/PHYSICAL_TERMINAL.json")
          and custody["terminal_sha256"] == digest(V3 / "failure_snapshot_v2/supervision/run01/TERMINAL.json"))
    template = read(V3 / "ROOT_RELEASE_TEMPLATE.json")
    check("v3 template disabled exact plan bound", template["status"] == "DISABLED_TEMPLATE_NOT_AUTHORIZATION"
          and template["root_authorization_reference"] is None and template["plan_sha256"] == PLAN_SHA
          and template["caps"] == plan["caps"] and template["workload"] == plan["workload"])
    ast_files = []
    for path in sorted(V3.rglob("*.py")):
        ast.parse(path.read_text(), filename=str(path)); ast_files.append(str(path.relative_to(V3)))
    check("no project/numerical imports", not any(name in sys.modules for name
          in ("torch", "numpy", "pandas", "transformers", "worker", "common", "run_lp", "supervise")))
    result = dict(schema="pencil-resource-v3-independent-source-verification-v1", UTC=datetime.now(timezone.utc).isoformat(),
                  status="PASS", candidate_manifest_sha256=SOURCE_SHA, source_seal_sha256=SEAL_SHA,
                  plan_sha256=PLAN_SHA, checks_passed=len(checks), checks=checks,
                  payload_count=len(manifest["files"]), payload_bytes=sum(r["bytes"] for r in manifest["files"]),
                  declared_external_inputs=len(bindings["inputs"]), AST_parsed_files=len(ast_files),
                  native_copied_and_git_blob_bindings=20, preserved_failure_snapshots=6,
                  exact_delta_reproduced=True, entire_worker_AST_delta_confined=True,
                  inherited_v2_review=dict(path=str(PRIOR_REVIEW / "REVIEW.json"), sha256=digest(PRIOR_REVIEW / "REVIEW.json"),
                                           candidate_manifest_sha256=provenance["parent_manifest_sha256"]),
                  fresh_zero_context_review=False, reused_agent_context=True, source_only=True,
                  numerical_execution=False, target_import_compile_execute=False,
                  network_server_access=False, installation_staging_launch=False,
                  scientific_payloads_read=False, candidate_source_modified=False,
                  scope="Independent bounded v3 delta/byte/AST/failure verification with authenticated v2 assessment reuse; prior QA not rebuilt.")
    (HERE / "SOURCE_VERIFICATION.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_passed", "payload_count", "declared_external_inputs",
                                          "AST_parsed_files", "native_copied_and_git_blob_bindings")}))


if __name__ == "__main__":
    main()
