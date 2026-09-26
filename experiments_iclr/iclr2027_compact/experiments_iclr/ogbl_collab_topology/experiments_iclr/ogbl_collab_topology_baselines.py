"""Frozen CPU-only CN and Adamic--Adar context for the existing OGBL-Collab study.

Install beside ogbl_collab_frozen.py in experiments_iclr. The study uses only
the official pre-2018 training graph and official positive/negative pools.
Neither validation nor test edges are added to the message graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

import ogbl_collab_frozen as learned


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "OGBL_COLLAB_TOPOLOGY_BASELINES_PROTOCOL.md"
FREEZE = HERE / "OGBL_COLLAB_TOPOLOGY_BASELINES_FREEZE.json"
VERIFIER = HERE / "verify_ogbl_collab_topology_baselines.py"
OUT = HERE / "ogbl_collab_topology_baselines_results"
RUNNER_SHA = "5ba77c00311e465fd0cadcbe51d0dd66f34690be72a48138a33cb0b502caf4ac"
ZIP_SHA = "c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc"
FINGERPRINTS = {
    "message_edge_year": "b62fec411f6d350e84df55bf3610413c6936774a1dda28df770d27668072892e",
    "message_edges": "b56a93b89f573a39c39e360c5c28dad050104c6f69021214487eb76fd9632dd9",
    "test_neg": "a74669777c6fb33cf847a6689ca7007fc99c96f931f6eaecf228a0b870c362f8",
    "test_pos": "3a7599a6d8dccce93837bbea0f7bf5a66feb0a231ce24ba6e6ca73a00f4fcf60",
    "train_pos": "882d3f11e79ee6854f98c033bfb95247d2ea371f81f214a8562686ebeef7bf8e",
    "valid_neg": "3c9ed8ff47008dc68afa7461e47023ed3003fe1de81c2d0c877bac5c52a7f348",
    "valid_pos": "ab19528e8607fb7f7f714293ee842c06f73b4c95f539eef6b92b7f0a005a1a21",
    "x": "694436f87163c1fd403ef6e1f401cf546069e475ae0b353dd25cef57b8b65496",
}
K = 50


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_freeze() -> dict:
    require(FREEZE.is_file(), "Missing reviewed source/protocol freeze")
    freeze = json.loads(FREEZE.read_text())
    require(freeze["protocol"] == "ogbl_collab_topology_baselines_v1" and
            freeze["source_sha256"] == sha(Path(__file__)) and
            freeze["verifier_sha256"] == sha(VERIFIER) and
            freeze["protocol_sha256"] == sha(PROTOCOL) and
            freeze["learned_runner_sha256"] == sha(HERE / "ogbl_collab_frozen.py") == RUNNER_SHA and
            freeze["official_zip_sha256"] == sha(ROOT / "data/ogb/collab.zip") == ZIP_SHA and
            freeze["expected_data_fingerprints"] == FINGERPRINTS and
            freeze["baselines"] == ["common_neighbors", "adamic_adar"] and
            freeze["splits"] == ["valid", "test"] and freeze["hits_at_k"] == K and
            freeze["message_graph"] == "official pre-2018 training edges only",
            "Source, data, or decision protocol differs from reviewed freeze")
    return freeze


def load_bundle() -> learned.DataBundle:
    torch.set_num_threads(1)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' for CPU-only baseline")
    bundle = learned.DataBundle(learned.repo_path("data/ogb"), torch.device("cpu"))
    require(bundle.fingerprints == FINGERPRINTS, "Official graph/split fingerprints differ")
    return bundle


def adjacency(bundle: learned.DataBundle) -> tuple[sparse.csr_matrix, np.ndarray]:
    edges = bundle.graph.edge_index.cpu().numpy().astype(np.int64, copy=False)
    require(edges.ndim == 2 and edges.shape[0] == 2, "Unexpected message edge shape")
    require(np.all(edges[0] != edges[1]), "Unexpected self-edge in official training graph")
    n = bundle.n_nodes
    matrix = sparse.coo_matrix((np.ones(edges.shape[1], dtype=np.uint8),
                                (edges[0], edges[1])), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1
    matrix.sort_indices()
    require((matrix != matrix.T).nnz == 0, "Training adjacency is not undirected")
    degrees = np.diff(matrix.indptr).astype(np.int64)
    require(matrix.nnz >= 1 and degrees.max() >= 2, "Degenerate official graph")
    return matrix, degrees


def two_scores(pairs: np.ndarray, graph: sparse.csr_matrix,
               aa_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(pairs.ndim == 2 and pairs.shape[1] == 2, "Bad official edge pool")
    require(np.issubdtype(pairs.dtype, np.integer), "Nonintegral edge endpoints")
    require(np.all((pairs >= 0) & (pairs < graph.shape[0])), "Out-of-range edge endpoint")
    indptr, indices = graph.indptr, graph.indices
    cn = np.empty(len(pairs), dtype=np.int64)
    aa = np.empty(len(pairs), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        shared = np.intersect1d(indices[indptr[u]:indptr[u + 1]],
                                 indices[indptr[v]:indptr[v + 1]], assume_unique=True)
        cn[i] = len(shared)
        aa[i] = aa_weight[shared].sum(dtype=np.float64)
    require(np.isfinite(aa).all(), "Nonfinite Adamic--Adar score")
    return cn, aa


def hits_at_50(pos: np.ndarray, neg: np.ndarray) -> dict:
    require(pos.ndim == neg.ndim == 1 and len(pos) > 0 and len(neg) >= K,
            "Incomplete official Hits@50 pools")
    threshold = np.partition(neg, len(neg) - K)[len(neg) - K]
    return {"hits50": float(np.count_nonzero(pos > threshold) / len(pos)),
            "negative_threshold": float(threshold),
            "positive_ties_at_threshold": int(np.count_nonzero(pos == threshold)),
            "positive_count": int(len(pos)), "negative_count": int(len(neg))}


def run() -> None:
    freeze = check_freeze()
    require(not OUT.exists(), "Refusing to overwrite baseline results")
    bundle = load_bundle()
    graph, degree = adjacency(bundle)
    weight = np.zeros(bundle.n_nodes, dtype=np.float64)
    mask = degree > 1
    weight[mask] = 1.0 / np.log(degree[mask].astype(np.float64))
    scores: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for split in ("valid", "test"):
        positive = bundle.split[split]["edge"].cpu().numpy()
        negative = bundle.split[split]["edge_neg"].cpu().numpy()
        p_cn, p_aa = two_scores(positive, graph, weight)
        n_cn, n_aa = two_scores(negative, graph, weight)
        for name, p, n in (("common_neighbors", p_cn, n_cn),
                           ("adamic_adar", p_aa, n_aa)):
            scores[f"{split}_{name}_pos"] = p
            scores[f"{split}_{name}_neg"] = n
            metrics.setdefault(name, {})[split] = hits_at_50(p, n)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "official_pool_scores.npz", **scores)
    result = {"protocol": freeze["protocol"], "freeze_sha256": sha(FREEZE),
              "source_sha256": sha(Path(__file__)), "official_zip_sha256": ZIP_SHA,
              "data_fingerprints": bundle.fingerprints,
              "train_undirected_edges": int(graph.nnz // 2),
              "train_degree_min": int(degree.min()), "train_degree_max": int(degree.max()),
              "score_archive_sha256": sha(OUT / "official_pool_scores.npz"),
              "metrics": metrics,
              "interpretation": "post hoc deterministic topology-only context; no fitting or validation selection"}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result_sha256": sha(OUT / "results.json"), "metrics": metrics}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check-freeze", "run"))
    args = parser.parse_args()
    if args.command == "check-freeze":
        print(json.dumps({"freeze_sha256": sha(FREEZE), "protocol": check_freeze()["protocol"]}))
    else:
        run()


if __name__ == "__main__":
    main()
