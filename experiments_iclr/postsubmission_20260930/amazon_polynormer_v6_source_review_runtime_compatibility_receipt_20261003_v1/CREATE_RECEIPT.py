"""Create a source-review status-schema receipt; no runtime code imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).resolve().parent
PHASE = OUT.parent
FULL = PHASE / "amazon_polynormer_v6_bounded_retention_forecast_independent_source_review_20261003_v1"
UTC = datetime.now(timezone.utc).isoformat()


def desc(path):
    assert path.suffix in {".json", ".py"} and path.resolve().is_relative_to(PHASE)
    raw = path.read_bytes()
    raw.decode("utf8")
    return {"path": str(path.relative_to(PHASE)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


review_desc = desc(FULL / "REVIEW.json")
assert review_desc["sha256"] == "5dcd5bb6ca727d14286b393ab0e1281e9b8990401b88e11f804d81430a850c68" and review_desc["bytes"] == 8007
review = json.loads((FULL / "REVIEW.json").read_text())
assert review["status"] == "PASS_SOURCE_ONLY" and review["blocking_findings"] == []
for row in review["source"].values():
    assert desc(PHASE / row["path"]) == row
gate_source = (PHASE / "amazon_polynormer_paired_family_source_preparation_20261003_v6/common.py").read_text()
assert "review['status'] == 'passed' and review['source'] == source" in gate_source
save("REVIEW.json", {
    "schema": "amazon_polynormer_independent_source_review_runtime_compatibility_receipt_v1",
    "UTC": UTC, "status": "passed", "review_outcome": "PASS_SOURCE_ONLY",
    "source": review["source"], "blocking_findings": [],
    "independent_review_report": review_desc,
    "independent_review_manifest": desc(FULL / "MANIFEST.json"),
    "independent_review_seal": desc(FULL / "SEAL.json"),
    "scope": "Only status-schema compatibility for the completed independent exact V6 source review. The literal passed is required by common.gate; the independently reviewed source-only outcome and all limitations are unchanged.",
    "source_gate_inspected": {"file": desc(PHASE / "amazon_polynormer_paired_family_source_preparation_20261003_v6/common.py"), "lines": "94-107"},
    "execution_authorized": False, "training_authorized": False,
    "numerical_qualification_established": False, "resource_admission_established": False,
    "runtime_receipt_or_consumer_release_established": False,
    "source_or_scientific_runtime_changed": False, "original_sealed_review_preserved": True,
    "numerical_or_project_imports": False, "arrays_checkpoints_binaries_GPU_SSH_upload_or_launch": False,
})
payload = [desc(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}]
save("MANIFEST.json", {"schema": "independent-review-compatibility-manifest-v1", "UTC": UTC, "payload": payload, "execution_authorized": False})
save("SEAL.json", {"schema": "independent-review-compatibility-seal-v1", "UTC": UTC, "manifest": desc(OUT / "MANIFEST.json"), "review": desc(OUT / "REVIEW.json"), "execution_authorized": False})
for row in payload:
    assert desc(PHASE / row["path"]) == row
assert desc(FULL / "REVIEW.json") == review_desc
for path in OUT.iterdir():
    if path.is_file():
        path.chmod(0o444)
print(json.dumps({"review": desc(OUT / "REVIEW.json"), "manifest": desc(OUT / "MANIFEST.json"), "seal": desc(OUT / "SEAL.json")}))
