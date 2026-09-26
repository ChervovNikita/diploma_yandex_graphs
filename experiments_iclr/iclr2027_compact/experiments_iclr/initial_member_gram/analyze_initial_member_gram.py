"""Post hoc train-gradient Gram diagnostic; no model training or test access.

For mean-member loss and an infinitesimal SGD graph-parameter step,
delta L_m / eta = -(1/4) sum_n g_m dot g_n. This is only an initial,
graph-parameter, training-loss identity, not an AdamW or generalization claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "initial_update_raw_gradients"
OUT = ROOT / "INITIAL_MEMBER_GRADIENT_GRAM.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite existing gradient Gram analysis")
    manifest_path = INPUT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == "postfreeze_initial_update_raw_gradient_export_v1"
    rows = []
    for source in manifest["rows"]:
        path = INPUT / source["file"]
        assert sha(path) == source["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            assert set(saved.files) == {"member_graph_gradients",
                                        "actual_tied_graph_update",
                                        "actual_sync_graph_update"}
            gradients = saved["member_graph_gradients"].astype(np.float64)
        assert gradients.shape == tuple(source["shape"]) and gradients.shape[0] == 4
        assert np.isfinite(gradients).all()
        gram = gradients @ gradients.T
        assert np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12)
        norms = np.sqrt(np.diag(gram))
        assert np.all(norms > 0)
        cosine = gram / np.outer(norms, norms)
        off = cosine[np.triu_indices(4, 1)]
        self_term = np.diag(gram)
        cross_term = gram.sum(axis=1) - self_term
        row_total = gram.sum(axis=1)
        mean_gradient = gradients.mean(axis=0)
        assert np.isclose(row_total.mean() / 4, np.dot(mean_gradient, mean_gradient),
                          rtol=1e-9, atol=1e-9)
        rows.append({
            "dataset": source["dataset"], "seed": source["seed"],
            "source_file": source["file"], "source_sha256": source["sha256"],
            "gram": gram.tolist(), "pairwise_cosine": cosine.tolist(),
            "mean_offdiagonal_cosine": float(off.mean()),
            "negative_pair_count": int((off < 0).sum()),
            "self_dot": self_term.tolist(),
            "other_member_dot_sum": cross_term.tolist(),
            "cross_to_self_ratio": (cross_term / self_term).tolist(),
            "member_total_dot": row_total.tolist(),
            "members_with_adverse_graph_only_sgd_first_order_loss_change":
                int((row_total < 0).sum()),
            "mean_gradient_squared_norm": float(np.dot(mean_gradient, mean_gradient)),
        })
    assert len(rows) == 12 and {(r["dataset"], r["seed"]) for r in rows} == {
        (dataset, seed) for dataset in ("cora", "wikics", "actor", "chameleon_filtered")
        for seed in range(3)}
    report = {"protocol": "posthoc_initial_graph_gradient_gram_v1",
              "input_manifest_sha256": sha(manifest_path),
              "source_sha256": sha(Path(__file__)),
              "scope": "train labels, initial graph-gradient vectors, four members, 12 graph/seed cases",
              "interpretation": "infinitesimal SGD graph-parameter first-order training-loss identity only",
              "rows": rows}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report_sha256": sha(OUT),
                      "negative_pairs": sum(r["negative_pair_count"] for r in rows),
                      "pairs": len(rows) * 6,
                      "adverse_member_cases": sum(r["members_with_adverse_graph_only_sgd_first_order_loss_change"] for r in rows),
                      "member_cases": len(rows) * 4,
                      "graph_means": {dataset: float(np.mean([
                          r["mean_offdiagonal_cosine"] for r in rows
                          if r["dataset"] == dataset]))
                          for dataset in ("cora", "wikics", "actor", "chameleon_filtered")}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
