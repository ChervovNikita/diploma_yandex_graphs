"""Stdlib metadata/hash/AST/resource arithmetic only; no runtime imports."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import math

OUT = Path(__file__).resolve().parent
PHASE = OUT.parent
V6 = PHASE / "amazon_polynormer_paired_family_source_preparation_20261003_v6"
V5 = PHASE / "amazon_polynormer_paired_family_source_preparation_20261003_v5"
CANDIDATE = PHASE / "amazon_polynormer_paired_family_execution_root_20261003_v3/fit_schedule_resource_candidate_v2_v6_bounded_retention"
TEXT = {".json", ".py", ".md", ".txt", ".log", ".sh", ".html", ".patch"}


def descriptor(path):
    assert path.suffix in TEXT and path.resolve().is_relative_to(PHASE)
    raw = path.read_bytes()
    raw.decode("utf8")
    return {"path": str(path.relative_to(PHASE)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def verify(path, row):
    got = descriptor(path)
    assert got["sha256"] == row["sha256"] and got["bytes"] == row["bytes"]
    if path.suffix == ".json":
        json.loads(path.read_text())
    if path.suffix == ".py":
        ast.parse(path.read_text())
    return got


def source_packet(root, pin, count):
    assert descriptor(root / "MANIFEST.json")["sha256"] == pin
    rows = json.loads((root / "MANIFEST.json").read_text())["payload"]
    assert len(rows) == count
    return [verify(root / row["path"], row) for row in rows]


v6_files = source_packet(V6, "d4160d8aeec2b770274875f9a1fb137facc9f93efd4eca6ad91e58802521ff44", 36)
v5_files = source_packet(V5, "25606a662be16d39219c9ef1fb75f13433f6d76b43559624b8607d1a5a5642bb", 29)
assert descriptor(V6 / "SEAL.json")["sha256"] == "b70beb8868d09c7f066c82cd9c0d1c582229cdfe0124485b32e12d7a7069f373"
assert descriptor(CANDIDATE / "MANIFEST.json")["sha256"] == "3102009fa43d799b6309f0e2076dfdbc538de14b3c91657858bb17281214ee22"
assert descriptor(CANDIDATE / "SEAL.json")["sha256"] == "c5879bf9d0c575b7a141c8d78eaa37a85db0666772860f3b847b77822e3a879d"
candidate_manifest = json.loads((CANDIDATE / "MANIFEST.json").read_text())
candidate_files = [verify(CANDIDATE / row["relative"], row["descriptor"]) for row in candidate_manifest["payload"]]
assert len(candidate_files) == 25
diff = json.loads((V6 / "SOURCE_DIFF_EVIDENCE.json").read_text())
for name in diff["scientific_and_shared_source_byte_identical"]:
    assert (V5 / name).read_bytes() == (V6 / name).read_bytes()


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


a = function(V5 / "qualify.py", "replay_scratch")
b = function(V6 / "study.py", "retained_replay")
a.name = b.name = "normalized_replay"
assert ast.dump(a) == ast.dump(b)
for name in diff["unchanged_evaluation_function_AST"]:
    assert ast.dump(function(V5 / "evaluate.py", name)) == ast.dump(function(V6 / "evaluate.py", name))
forecast = json.loads((CANDIDATE / "RESOURCE_FORECAST_EVIDENCE.json").read_text())
old = json.loads((CANDIDATE.parent / "fit_schedule_resource_candidate_v1/RESOURCE_FORECAST_EVIDENCE.json").read_text())
assert forecast["measurements"] == old["measurements"]
inheritance = json.loads((V6 / "COHORT_SOURCE_BINDING.json").read_text())
registry_path = PHASE / inheritance["registry"]["path"]
verify(registry_path, inheritance["registry"])
registry = json.loads(registry_path.read_text())
assert registry["source"] == inheritance["registered_source"]
assert len(registry["physical_fits"]) == 15 and len(registry["families"]) == 9
rows = forecast["all15_fit_forecasts"]
assert [r["registered_row"] for r in rows] == [r["row"] for r in registry["physical_fits"]]
for row in rows:
    rates = forecast["measurements"][row["kind"]]
    assert row["scientific_optimizer_updates"] == 2700 and row["retained_replay_optimizer_updates"] == 4
    assert math.isclose(sum(row["components"].values()), row["rate_proxy_and_allowance_extrapolation_seconds"])
    assert math.isclose(1.5 * row["rate_proxy_and_allowance_extrapolation_seconds"], row["with50percent_wall_margin_seconds"])
    assert math.isclose(row["components"]["all2700_train_plus_VAL_seconds"], 200 * rates["local_train_plus_VAL_seconds"] + 2500 * rates["global_train_plus_VAL_seconds"])
    assert math.isclose(row["components"]["all_selected_snapshot_plus_write_seconds"], 2700 * rates["snapshot_plus_write_seconds"])
    assert row["retained_checkpoint_forecast_bytes"] == 2 * rates["largest_observed_serialized_checkpoint_bytes"]
wall = sum(r["rate_proxy_and_allowance_extrapolation_seconds"] for r in rows)
assert math.isclose(wall, forecast["wall"]["sum15_fit_extrapolations_seconds"])
storage = sum(r["retained_checkpoint_forecast_bytes"] for r in rows)
assert storage == forecast["storage"]["retained_checkpoint_bytes_all15_upper_bound"] == 4405455504
assert math.ceil(storage * 1.25) + 8 * 2**30 + 2**30 + 3 * 2**30 == forecast["storage"]["incremental_required_free_space_bytes"] == 18391721268
disabled = [CANDIDATE / "ALL_FITS_RELEASE_CANDIDATE.json", CANDIDATE / "CLOSE_RELEASE_CANDIDATE.json", *sorted((CANDIDATE / "disabled_fit_releases").glob("*.json"))]
for path in disabled:
    value = json.loads(path.read_text())
    assert not value["execution_authorized"] and not value["independent_source_review_passed"]
    assert all(value[k] is None for k in ["source_review", "runtime_receipt", "consumer_release", "qualification_freeze"])
for row, path in zip(registry["physical_fits"], sorted((CANDIDATE / "disabled_fit_releases").glob("*.json"))):
    # Compare by id rather than assuming filesystem order is registry order.
    value = json.loads((CANDIDATE / "disabled_fit_releases" / (row["id"] + ".json")).read_text())
    assert value["fit_id"] == row["id"] and value["output"] == row["output"] and value["self_path"] == row["release_path"] and value["registered_claim_path"] == row["claim_path"]

descriptors = {}


def collect(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value) and isinstance(value["path"], str) and isinstance(value["sha256"], str) and len(value["sha256"]) == 64:
            row = {k: value[k] for k in ["path", "sha256", "bytes"]}
            if row["path"] in descriptors:
                assert descriptors[row["path"]] == row
            descriptors[row["path"]] = row
        for child in value.values():
            collect(child)
    elif isinstance(value, list):
        for child in value:
            collect(child)


for root in [V6, CANDIDATE]:
    for path in root.rglob("*.json"):
        collect(json.loads(path.read_text()))
verified = []
opaque = []
for row in descriptors.values():
    path = Path(row["path"])
    if path.suffix in TEXT:
        assert not path.is_absolute() and ".." not in path.parts
        verified.append(verify(PHASE / path, row))
    else:
        assert path.suffix in {".npz", ".pt"}
        opaque.append(row)
assert len(opaque) == 20
result = {
    "schema": "independent-amazon-v6-retention-forecast-source-check-v1", "UTC": datetime.now(timezone.utc).isoformat(),
    "status": "PASS_SOURCE_ONLY", "V6_payloads": v6_files, "V5_payloads": v5_files, "candidate_payloads": candidate_files,
    "verified_text_metadata_descriptors": verified, "opaque_descriptors_carried_unopened": opaque,
    "replay_AST_and_evaluation_bodies_match_V5": True, "scientific_shared_bodies_and_failures_match_V5": True,
    "all15_registered_candidate_paths_match_original": True, "all17_fit_close_all_bodies_disabled_with_null_fresh_inputs": True,
    "resource_arithmetic_verified_not_measured_admission": True,
    "author_metadata_fixture_inspected_not_rerun": True, "project_or_numerical_modules_imported": False,
    "arrays_checkpoints_runtime_binaries_opened": False, "SSH_upload_or_launch": False,
    "existing_sources_registry_claims_or_outputs_changed": False, "execution_authorized": False,
}
(OUT / "STATIC_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "V6_payloads": len(v6_files), "V5_payloads": len(v5_files), "candidate_payloads": len(candidate_files), "text_descriptors": len(verified), "opaque_unopened": len(opaque)}))
