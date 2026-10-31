"""Derive a shareable data identity without individual test-target hashes.

Only existing manifest JSON is read. No graph tensors, labels, predictions,
or scores are loaded. The original immutable author manifest is preserved.
This is publication hygiene, not cryptographic concealment of public data.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(original, parent_hash, source_hash):
    result = copy.deepcopy(original)
    test = result["split_summaries"]["test"]
    removed = sum("y" in graph for graph in test["graph_tensor_fingerprints"])
    for graph in test["graph_tensor_fingerprints"]:
        graph.pop("y", None)
    for key in ("label_summary", "class_counts"):
        test.pop(key, None)
    payload = json.dumps(test["graph_tensor_fingerprints"], sort_keys=True, separators=(",", ":")).encode()
    test["canonical_split_sha256"] = hashlib.sha256(payload).hexdigest()
    result["manifest_amendment"] = {"schema": "gnnm-shareable-freeze-v2",
        "parent_manifest_sha256": parent_hash, "amendment_source_sha256": source_hash,
        "individual_test_target_hashes_removed": removed,
        "input_data_changed": False, "predictive_scores_read": False,
        "test_fingerprint_rule": "Per-graph input/topology/official-index identities only; complete provider-file hashes still bind data bytes",
        "boundary": "Original freezer fingerprinted test targets. Public provider cache contains test labels. Development runners do not instantiate test loaders or compute predictive test scores. This manifest does not cryptographically seal public labels."}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    allowed = Path(__file__).resolve().parent
    original, output = args.manifest.resolve(), args.output.resolve()
    if allowed not in original.parents or allowed not in output.parents or original.parent != output.parent:
        raise ValueError("Both manifests must stay in the same permitted new_tasks cache folder")
    if output.exists():
        raise ValueError("Refuse to overwrite a manifest")
    document = json.loads(original.read_text())
    if document["scoring_performed"] or document["confirmation_data_accessed"]:
        raise ValueError("Expected the existing unscored development manifest")
    result = project(document, sha(original), sha(Path(__file__)))
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "shareable_manifest_created", "sha256": sha(output),
                      "parent_sha256": sha(original), "removed_test_target_hashes": result["manifest_amendment"]["individual_test_target_hashes_removed"],
                      "data_or_predictive_scores_loaded": False}))


if __name__ == "__main__":
    main()
