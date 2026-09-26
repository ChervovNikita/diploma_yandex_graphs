"""Independent exact decision/float comparison of audited source and compact projection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    a = p.parse_args()
    source, stage = a.source.resolve(), a.stage.resolve()
    manifest = json.loads((stage / "MANIFEST.json").read_text())
    output = stage / "FULL_TO_COMPACT_ARRAY_AUDIT.json"
    need(not output.exists(), "Refusing to replace exact array audit")
    roots = {"narrow72": source / "narrow_untied72_results",
             "wide18": source / "wikics_wide_untied18_results"}
    validation, scores = {}, {}
    for identity, row in manifest["validation"].items():
        kind, cell = identity.split("/", 1)
        original = roots[kind] / "results" / cell / "validation_companion.npz"
        projected = stage / row["path"]
        need(sha(original) == row["source_companion_sha256"] and
             sha(projected) == row["sha256"], f"Validation SHA differs: {identity}")
        with np.load(original, allow_pickle=False) as raw, np.load(projected, allow_pickle=False) as compact:
            need(np.array_equal(raw["valid_pooled_logits"].argmax(-1).astype(np.uint8),
                                compact["pooled_class"]) and
                 np.array_equal(raw["valid_member_logits"].argmax(-1).astype(np.uint8),
                                compact["member_class"]),
                 f"Exact validation array projection differs: {identity}")
        validation[identity] = {"source_sha256": sha(original),
                                "projection_sha256": sha(projected),
                                "exact_pooled_decisions": True,
                                "exact_member_decisions": True}
    for identity, row in manifest["scores"].items():
        kind, cell = identity.split("/", 1)
        original = roots[kind] / "scores" / cell / "predictions.npz"
        projected = stage / row["path"]
        need(sha(original) == row["source_predictions_sha256"] and
             sha(projected) == row["sha256"], f"Test SHA differs: {identity}")
        with np.load(original, allow_pickle=False) as raw, np.load(projected, allow_pickle=False) as compact:
            need(np.array_equal(raw["test_pooled_logits"], compact["pooled_logits"]) and
                 np.array_equal(raw["test_member_logits"].argmax(-1).astype(np.uint8),
                                compact["member_class"]),
                 f"Exact test array projection differs: {identity}")
        scores[identity] = {"source_sha256": sha(original),
                            "projection_sha256": sha(projected),
                            "exact_pooled_floats": True,
                            "exact_member_decisions": True}
    need(len(validation) == 90 and len(scores) == len(manifest["scores"]),
         "Exact array audit coverage differs")
    output.write_text(json.dumps({"protocol": "narrow_wide_compact_exact_array_audit_v2",
                                  "manifest_sha256": sha(stage / "MANIFEST.json"),
                                  "source_projection_audit_sha256": sha(Path(__file__)),
                                  "validation": validation, "scores": scores,
                                  "status": "PASS"}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit_sha256": sha(output),
                      "validation_cells": len(validation), "test_scores": len(scores),
                      "status": "PASS"}))


if __name__ == "__main__":
    main()
