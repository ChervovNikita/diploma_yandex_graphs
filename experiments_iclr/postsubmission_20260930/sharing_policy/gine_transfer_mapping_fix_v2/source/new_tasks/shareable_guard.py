"""Strict development-manifest boundary; stdlib only, no dataset access."""
import hashlib
import json


def require_development_manifest(data, dataset):
    if (data.get("dataset") != dataset or data.get("scoring_performed") is not False
            or data.get("confirmation_data_accessed") is not False):
        raise ValueError("Require an unscored development freeze with confirmation_data_accessed=false")


def require_shareable_manifest(data, dataset):
    require_development_manifest(data, dataset)
    amendment = data.get("manifest_amendment", {})
    if (amendment.get("schema") != "gnnm-shareable-freeze-v2"
            or amendment.get("input_data_changed") is not False
            or amendment.get("predictive_scores_read") is not False):
        raise ValueError("Require the unchanged, unscored shareable-v2 derivative")
    for key in ("parent_manifest_sha256", "amendment_source_sha256"):
        value = amendment.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid shareable provenance digest: " + key)
    test = data.get("split_summaries", {}).get("test", {})
    fingerprints = test.get("graph_tensor_fingerprints")
    if (not isinstance(fingerprints, list) or type(test.get("graphs")) is not int
            or len(fingerprints) != test["graphs"]
            or type(amendment.get("individual_test_target_hashes_removed")) is not int
            or amendment["individual_test_target_hashes_removed"] != test["graphs"]):
        raise ValueError("Shareable test fingerprint/removal counts differ")
    forbidden = {"y", "label_summary", "class_counts", "labels", "targets", "target", "target_sha256"}

    def audit(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("Shareable test metadata contains targets or label summaries")
            for child in value.values():
                audit(child)
        elif isinstance(value, list):
            for child in value:
                audit(child)

    audit(test)
    digest = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if test.get("canonical_split_sha256") != digest:
        raise ValueError("Shareable test input fingerprint hash differs")
    return data
