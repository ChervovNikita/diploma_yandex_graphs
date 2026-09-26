"""Recompute all strict Hits@50 metrics from complete saved official-pool scores.

This compact check verifies score arithmetic and provenance bindings. It does
not independently reconstruct graph intersections without the official data.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "experiments_iclr"
OUT = SRC / "ogbl_collab_topology_baselines_results"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((ROOT / "PUBLIC_MANIFEST.json").read_text())
for name, digest in manifest["files"].items():
    assert sha(ROOT / name) == digest, name
freeze = json.loads((SRC / "OGBL_COLLAB_TOPOLOGY_BASELINES_FREEZE.json").read_text())
result = json.loads((OUT / "results.json").read_text())
audit = json.loads((OUT / "INDEPENDENT_AUDIT.json").read_text())
assert sha(SRC / "OGBL_COLLAB_TOPOLOGY_BASELINES_FREEZE.json") == result["freeze_sha256"] == audit["freeze_sha256"]
assert sha(SRC / "ogbl_collab_topology_baselines.py") == freeze["source_sha256"] == result["source_sha256"]
assert sha(SRC / "verify_ogbl_collab_topology_baselines.py") == freeze["verifier_sha256"]
assert sha(SRC / "OGBL_COLLAB_TOPOLOGY_BASELINES_PROTOCOL.md") == freeze["protocol_sha256"]
assert sha(SRC / "ogbl_collab_frozen.py") == freeze["learned_runner_sha256"]
assert freeze["expected_data_fingerprints"] == result["data_fingerprints"]
assert freeze["official_zip_sha256"] == result["official_zip_sha256"]
assert sha(OUT / "results.json") == audit["result_sha256"]
assert sha(OUT / "official_pool_scores.npz") == result["score_archive_sha256"] == audit["raw_score_sha256"]
assert freeze["hits_at_k"] == 50
metrics = {}
with np.load(OUT / "official_pool_scores.npz", allow_pickle=False) as arrays:
    expected = {f"{split}_{arm}_{kind}" for split in ("valid", "test")
                for arm in ("common_neighbors", "adamic_adar") for kind in ("pos", "neg")}
    assert set(arrays.files) == expected
    for split in ("valid", "test"):
        for arm in ("common_neighbors", "adamic_adar"):
            pos, neg = (arrays[f"{split}_{arm}_{kind}"] for kind in ("pos", "neg"))
            assert pos.ndim == neg.ndim == 1 and len(neg) == 100000
            assert np.isfinite(pos).all() and np.isfinite(neg).all()
            if arm == "common_neighbors":
                assert np.issubdtype(pos.dtype, np.integer) and np.issubdtype(neg.dtype, np.integer)
            assert (pos >= 0).all() and (neg >= 0).all()
            threshold = sorted(neg.tolist(), reverse=True)[49]
            replay = {
                "hits50": sum(float(x) > threshold for x in pos) / len(pos),
                "negative_threshold": float(threshold),
                "positive_ties_at_threshold": sum(float(x) == threshold for x in pos),
                "positive_count": len(pos), "negative_count": len(neg),
            }
            assert replay == result["metrics"][arm][split] == audit["verified_metrics"][f"{split}/{arm}"]
            metrics[f"{split}/{arm}"] = replay
print(json.dumps({"status": "TOPOLOGY_PUBLIC_ALL_SCORES_PASS", "arrays": 8,
                  "metrics": metrics, "scope": "All official-pool score arithmetic; graph regeneration requires official OGB data"}, sort_keys=True))
