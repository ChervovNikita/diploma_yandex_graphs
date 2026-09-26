"""Independent full-score replay of frozen CN and Adamic--Adar OGBL-Collab context."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import ogbl_collab_topology_baselines as study


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def pair_scores(pairs: np.ndarray, graph, degree: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recompute with searchsorted, independent of runner's intersect1d."""
    ptr, idx = graph.indptr, graph.indices
    cn = np.empty(len(pairs), dtype=np.int64)
    aa = np.empty(len(pairs), dtype=np.float64)
    weight = np.zeros(len(degree), dtype=np.float64)
    mask = degree > 1
    weight[mask] = 1.0 / np.log(degree[mask].astype(np.float64))
    for i, (a, b) in enumerate(pairs):
        a, b = int(a), int(b)
        left = idx[ptr[a]:ptr[a + 1]]
        right = idx[ptr[b]:ptr[b + 1]]
        if len(left) > len(right):
            left, right = right, left
        if len(left) == 0 or len(right) == 0:
            cn[i], aa[i] = 0, 0.0
            continue
        positions = np.searchsorted(right, left)
        common = left[(positions < len(right)) &
                      (right[np.minimum(positions, len(right) - 1)] == left)]
        cn[i] = len(common)
        aa[i] = weight[common].sum(dtype=np.float64)
    return cn, aa


def strict_hits(pos: np.ndarray, neg: np.ndarray) -> dict:
    threshold = np.sort(neg)[-study.K]
    return {"hits50": float(np.count_nonzero(pos > threshold) / len(pos)),
            "negative_threshold": float(threshold),
            "positive_ties_at_threshold": int(np.count_nonzero(pos == threshold)),
            "positive_count": int(len(pos)), "negative_count": int(len(neg))}


def audit() -> None:
    freeze = study.check_freeze()
    result_path = study.OUT / "results.json"
    raw_path = study.OUT / "official_pool_scores.npz"
    audit_path = study.OUT / "INDEPENDENT_AUDIT.json"
    require(result_path.is_file() and raw_path.is_file() and not audit_path.exists(),
            "Missing baseline artifact or existing audit")
    result = json.loads(result_path.read_text())
    require(result["protocol"] == freeze["protocol"] and
            result["freeze_sha256"] == study.sha(study.FREEZE) and
            result["source_sha256"] == study.sha(study.HERE / "ogbl_collab_topology_baselines.py") and
            result["score_archive_sha256"] == study.sha(raw_path) and
            result["data_fingerprints"] == study.FINGERPRINTS,
            "Result/source/data identity mismatch")
    bundle = study.load_bundle()
    graph, degree = study.adjacency(bundle)
    expected = set()
    checked = {}
    with np.load(raw_path, allow_pickle=False) as raw:
        for split in ("valid", "test"):
            pools = {}
            for kind, edge_key in (("pos", "edge"), ("neg", "edge_neg")):
                pairs = bundle.split[split][edge_key].cpu().numpy()
                cn, aa = pair_scores(pairs, graph, degree)
                for baseline, replay in (("common_neighbors", cn), ("adamic_adar", aa)):
                    name = f"{split}_{baseline}_{kind}"
                    expected.add(name)
                    require(name in raw and raw[name].shape == replay.shape and
                            np.allclose(raw[name], replay, rtol=0, atol=1e-12),
                            f"Raw-score replay differs: {name}")
                    pools[name] = replay
            for baseline in ("common_neighbors", "adamic_adar"):
                metrics = strict_hits(pools[f"{split}_{baseline}_pos"],
                                      pools[f"{split}_{baseline}_neg"])
                require(metrics == result["metrics"][baseline][split],
                        f"Strict Hits@50 replay differs: {split}/{baseline}")
                checked[f"{split}/{baseline}"] = metrics
        require(set(raw.files) == expected, "Unexpected or missing raw-score array")
    audit_result = {"protocol": freeze["protocol"],
                    "freeze_sha256": study.sha(study.FREEZE),
                    "result_sha256": study.sha(result_path),
                    "raw_score_sha256": study.sha(raw_path),
                    "verified_arrays": len(expected), "verified_metrics": checked,
                    "replay": "all official pairs; independent searchsorted intersections and sorted negative threshold"}
    audit_path.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit_sha256": study.sha(audit_path),
                      "verified_arrays": len(expected)}, sort_keys=True))


if __name__ == "__main__":
    audit()
