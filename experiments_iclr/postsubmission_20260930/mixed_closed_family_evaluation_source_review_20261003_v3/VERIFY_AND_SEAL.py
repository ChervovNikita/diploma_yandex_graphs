"""Independent one-line source successor review; no evaluator imports/execution."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import difflib
import hashlib
import json

OUT = Path(__file__).resolve().parent
PHASE = OUT.parent
OLD = PHASE / "graph_mixed_block_closed_family_evaluation_preparation_20261003_v2"
NEW = PHASE / "graph_mixed_block_closed_family_evaluation_preparation_20261003_v3"
UTC = datetime.now(timezone.utc).isoformat()
TEXT = {".json", ".py", ".md", ".txt", ".log", ".diff", ".patch", ".sh"}


def desc(path):
    assert path.suffix in TEXT and path.resolve().is_relative_to(PHASE)
    raw = path.read_bytes()
    raw.decode("utf8")
    return {"path": str(path.relative_to(PHASE)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def verify(path, row):
    got = desc(path)
    assert got["sha256"] == row["sha256"] and got["bytes"] == row["bytes"]
    if path.suffix == ".py":
        ast.parse(path.read_text())
    elif path.suffix == ".json":
        json.loads(path.read_text())
    return got


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


assert desc(NEW / "MANIFEST.json")["sha256"] == "3d2c3549afba16ee2b1689aa82e168b3251d8d3bb544f2f38dd8883372dcfb59"
assert desc(NEW / "MANIFEST.json")["bytes"] == 2447
assert desc(NEW / "SEAL.json")["sha256"] == "08a522ec1f022365ceeecd0fe6b32f50ccfc054597ea455969c7c9c1e125cc5a"
manifest = json.loads((NEW / "MANIFEST.json").read_text())
seal = json.loads((NEW / "SEAL.json").read_text())
assert seal["manifest_sha256"] == desc(NEW / "MANIFEST.json")["sha256"] and seal["manifest_bytes"] == 2447
assert seal["payload_count"] == len(manifest["payload"]) == 13
payloads = [verify(NEW / r["path"], r) for r in manifest["payload"]]
assert {p.name for p in NEW.iterdir() if p.is_file()} == {r["path"] for r in manifest["payload"]} | set(manifest["excluded"])
correction = json.loads((NEW / "V3_CORRECTION.json").read_text())
old_man = verify(PHASE / correction["predecessor_source"]["manifest"]["path"], correction["predecessor_source"]["manifest"])
old_seal = verify(PHASE / correction["predecessor_source"]["seal"]["path"], correction["predecessor_source"]["seal"])
old_payloads = [verify(OLD / r["path"], r) for r in json.loads((OLD / "MANIFEST.json").read_text())["payload"]]
for key in ["evaluator_before", "evaluator_after", "predecessor_source_review", "diagnosis", "diagnosis_seal", "metadata_fixture", "diff"]:
    verify(PHASE / correction[key]["path"], correction[key])
transport_rows = correction["both_failed_launch_transports_preserved"] + correction["both_readonly_diagnostic_transports_preserved"]
transports = [verify(PHASE / r["path"], r) for r in transport_rows]
before = (OLD / "evaluate.py").read_text()
after = (NEW / "evaluate.py").read_text()
oldline = "require(json.loads(verify(graph_schema).read_text()) == schema, 'Reconstructed full graph schema differs')"
newline = "require(json.loads(verify(graph_schema).read_text()) == json.loads(json.dumps(schema)), 'Reconstructed full graph schema differs')"
assert before.count(oldline) == 1 and after == before.replace(oldline, newline)
assert after.splitlines()[340].strip() == newline
declared_diff = (NEW / "V2_TO_V3.diff").read_text()
actual_diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=str((OLD / "evaluate.py").relative_to(PHASE)), tofile=str((NEW / "evaluate.py").relative_to(PHASE))))
assert actual_diff == declared_diff
diff_evidence = json.loads((NEW / "SOURCE_DIFF_EVIDENCE.json").read_text())
unchanged = diff_evidence["inherited_V2_payloads_byte_identical_except_evaluator"]
for name in unchanged:
    assert (NEW / name).read_bytes() == (OLD / name).read_bytes()
assert (NEW / "PRESERVED_V2_STATIC_CHECK_RECEIPT.json").read_bytes() == (OLD / "STATIC_CHECK_RECEIPT.json").read_bytes()
for name in ["EVALUATION_RELEASE_TEMPLATE.json", "EVALUATION_FREEZE_TEMPLATE.json"]:
    json.loads((NEW / name).read_text())
assert json.loads((NEW / "EVALUATION_RELEASE_TEMPLATE.json").read_text())["execution_authorized"] is False
static = desc(NEW / "STATIC_CHECK_RECEIPT.json")
assert static["sha256"] == "609414976490ea2e40734c31641364757e4c5ffe572527766b42b9d008a8b64d"
diagnosis = json.loads((PHASE / correction["diagnosis"]["path"]).read_text())
assert diagnosis["actual_V2_failure"]["process_exit_code"] == 1
assert diagnosis["actual_V2_failure"]["elapsed_seconds"] == 8.606621995568275
assert diagnosis["actual_V2_failure"]["evaluation_output_created"] is False
captured_failure = [verify(PHASE / r["local_copy"]["path"], r["local_copy"]) for r in diagnosis["captured_v2_output_metadata_logs"]]
fixture = json.loads((PHASE / correction["metadata_fixture"]["path"]).read_text())
ast_before = ast.parse(before)
ast_after = ast.parse(after)
before_run = next(n for n in ast_before.body if isinstance(n, ast.FunctionDef) and n.name == "run")
after_run = next(n for n in ast_after.body if isinstance(n, ast.FunctionDef) and n.name == "run")
for left, right in zip(ast_before.body, ast_after.body):
    if left is before_run:
        continue
    assert ast.dump(left) == ast.dump(right)
assert "graph_schema=schema" in after and "features, schema)" in after
checks = {
    "schema": "independent-mixed40-V3-one-line-source-check-v1", "UTC": UTC, "status": "PASS_SOURCE_ONLY",
    "payloads": payloads, "predecessor_manifest": old_man, "predecessor_seal": old_seal, "predecessor_payloads": old_payloads,
    "whole_evaluator_exactly_one_expected_substitution": True, "independent_diff_matches_declared": True,
    "changed_line": 341, "unchanged_inherited_payloads": unchanged,
    "all_non_run_top_level_AST_bodies_unchanged": True, "live_schema_checkpoint_replay_unmodified": True,
    "author_static_receipt": static, "author_metadata_fixture_inspected_not_rerun": correction["metadata_fixture"],
    "failed_transports_and_diagnostics_verified": transports, "captured_V2_failure_metadata_and_logs": captured_failure,
    "project_evaluator_numeric_or_fixture_execution": False,
    "arrays_archives_checkpoints_labels_runtime_binaries_GPU_SSH_upload_or_launch": False, "execution_authorized": False,
}
save("STATIC_VERIFICATION.json", checks)
save("READ_SCOPES.json", {
    "schema": "independent-mixed40-V3-read-scopes-v1", "UTC": UTC,
    "source_scope": [{"path": str((NEW / "evaluate.py").relative_to(PHASE)), "lines": "319-411", "scope": "changed JSON comparison and unchanged scoring/replay downstream"}, {"path": str((NEW / "evaluate.py").relative_to(PHASE)), "lines": "175-207", "scope": "live integer-key schema binding remains by exact whole-source substitution/AST inspection"}],
    "metadata_scope": "V3 manifests/seal/payloads/correction/diff/static receipt; V2 manifests/seal/payloads; earlier review; preserved diagnosis, fabricated fixture, captured failure metadata/logs and transports.",
    "earlier_review_scope_inherited_not_repeated": True,
    "whole_paper_certifications": 0, "primary_literature_reads": 0,
    "numeric_evaluator_or_fixture_execution": False, "new_scientific_tests": False,
})
save("REVIEW.json", {
    "schema": "independent-mixed40-V3-JSON-representation-source-review-v1", "UTC": UTC,
    "status": "passed", "review_outcome": "PASS_SOURCE_ONLY", "blocking_findings": [],
    "source": {"manifest": desc(NEW / "MANIFEST.json"), "seal": desc(NEW / "SEAL.json")},
    "evaluator": desc(NEW / "evaluate.py"), "diff": desc(NEW / "V2_TO_V3.diff"),
    "predecessor_review": correction["predecessor_source_review"], "preserved_diagnosis": correction["diagnosis"],
    "review_type": "Independent minimal one-line successor check, not a repeated full implementation audit",
    "checks": [
        {"id": "exact_integrity_and_diff", "status": "passed", "finding": "Exact requested V3 manifest, seal, evaluator and diff verify. All 13 payloads and preserved V2 payloads verify. Evaluator equals V2 after exactly the declared one-line substitution at line 341; independently generated diff matches. Templates, provenance and every other Python source remain byte-identical; all non-run top-level AST bodies match."},
        {"id": "representation_only_schema_compare", "status": "passed_source_reasoning", "finding": "The saved schema is decoded JSON with string object keys. Immutable graph constructors expose integer raw relation IDs; the writer stringified those keys when saving GRAPH_SCHEMA.json. JSON roundtrip at this one comparison matches saved representation while keeping full schema equality. Relation geometry/order/rows, node counts and other schema fields remain checked; the guard is not removed."},
        {"id": "live_checkpoint_binding", "status": "passed", "finding": "The original live schema is never assigned its JSON roundtrip. It remains passed directly to model_replay and exact checkpoint graph_schema binding, preserving integer-key trained-checkpoint identity. Graph construction, saved state replay, full 40 primary plus 15 native controls, metrics, calibration, contrasts and release gates remain unchanged by exact source comparison."},
        {"id": "failure_and_fixture_custody", "status": "passed_metadata_only", "finding": "Pinned V2 failure is exit 1 at the schema guard after 8.606621995568275 seconds, with no evaluation output. Failed launch transports, diagnostic transports and captured metadata/logs verify and remain unchanged. Author's metadata fixture covers DBLP/ACM representation equivalence and rejects changed node counts, relation order and parameter rows; source/receipt inspected, not independently rerun."},
        {"id": "remaining_execution_scope", "status": "pending_root", "finding": "This one-line change addresses the demonstrated representation failure in source; it does not establish evaluator completion, numerical inference replay, custody closure or scientific gates. Fresh V3-bound release/output and root-authorized scoring plus result verification remain required. Prior review limits and full denominator/gates persist."},
    ],
    "scientific_design_sha256": manifest["scientific_design_sha256"],
    "execution_authorized": False, "training_authorized": False, "scoring_authorized": False,
    "numerical_evaluator_qualification": False, "predictive_or_acceptance_verdict": False,
    "static_verification": desc(OUT / "STATIC_VERIFICATION.json"), "read_scopes": desc(OUT / "READ_SCOPES.json"),
    "read_accounting": {"source_AST_JSON_text_metadata_only": True, "numeric_or_project_modules_imported": False, "evaluator_or_fixture_run": False, "arrays_archives_checkpoints_labels_runtime_binaries_read_or_hashed": False, "remote_GPU_SSH_upload_or_launch": False, "subagents": False, "existing_source_or_canonical_or_prior_review_edits": False, "new_output_directory_only": OUT.name},
})
(OUT / "REVIEW.md").write_text("""# Mixed40 V3 one-line successor source review

**PASS_SOURCE_ONLY; no blocking source findings.** The exact V3 source passes the minimal independent successor review. All 13 payloads, exact evaluator/diff and preserved V2 source/failure metadata verify.

Line 341 compares decoded saved JSON with the JSON representation of the reconstructed schema. This repairs the demonstrated integer-key/string-key mismatch while preserving full schema equality. The live integer-key schema remains unchanged for graph construction and checkpoint replay. Full 40 primary and 15 native-control scope, metrics, calibration, contrasts and gates remain unchanged.

The pinned V2 failure remains exit 1 after 8.606621995568275 seconds with no evaluation output. Metadata fixtures were inspected, not rerun. This review grants no scoring or execution authority and establishes no numerical evaluator completion, predictive advantage or scientific verdict. Root still needs a fresh V3-bound release/output, authorized scoring and result/custody verification.
""")
sealed_payloads = [desc(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}]
save("MANIFEST.json", {"schema": "independent-source-review-manifest-v1", "UTC": UTC, "payload": sealed_payloads, "execution_authorized": False})
save("SEAL.json", {"schema": "independent-source-review-seal-v1", "UTC": UTC, "manifest": desc(OUT / "MANIFEST.json"), "review": desc(OUT / "REVIEW.json"), "execution_authorized": False})
for row in sealed_payloads:
    assert desc(PHASE / row["path"]) == row
for path in OUT.iterdir():
    if path.is_file():
        path.chmod(0o444)
print(json.dumps({"status": "PASS_SOURCE_ONLY", "review": desc(OUT / "REVIEW.json"), "manifest": desc(OUT / "MANIFEST.json"), "seal": desc(OUT / "SEAL.json")}))
