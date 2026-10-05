"""Disabled TRAIN custodian and public+B reader; no scientific fitting/scoring.

The compact source labels array contains both B and A. The custodian decodes
that whole array once, without metrics or label-dependent role choices, and
projects A into a separate evaluator artifact. A is held from new fitting and
selection, not physically unread by the custodian or the historical producer.
The public+B reader never opens the compact source or evaluator A artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE_RELEASED = False
DATA_MANIFEST = {
    "path": "amazon_ratings_native_warm_execution_root_20261003_v3/data/DATA_MANIFEST.json",
    "sha256": "8d565c20c6d669c2577e9fb8c5405575a6ed97a9964f91818b052e43cd0379bf",
}
PUBLIC_GRAPH = {
    "path": "amazon_ratings_native_warm_execution_root_20261003_v3/data/public_graph.npz",
    "sha256": "19757299bcfd9e493e9ceae9e73248753ab1e773ccc1f6b57c6fadf8c843310f",
    "bytes": 33104032,
}
TRAIN0 = {
    "path": "amazon_ratings_native_warm_execution_root_20261003_v3/data/split0_train.npz",
    "sha256": "9d700d1897bbf73ec70a4c3158e07d6b42909603e28ac54716800f3d4883f749",
    "bytes": 196440,
}
ROLE_METHOD = "amazon-response-G0|split=0|seed=17"
EXPECTED_COUNTS = {"TRAIN": 12246, "A": 2449, "B": 9797,
                   "W": 4898, "innerS": 2449, "outerR": 2450}
VISIBILITY = {
    "historical_raw_all_node_labels_decoded": True,
    "custodian_compact_all_TRAIN0_labels_decoded_including_A": True,
    "custodian_predictive_scoring_or_label_dependent_role_choices": False,
    "A_values_excluded_from_new_acquisition_probes_losses_and_selection": True,
    "VAL_TEST_label_arrays_decoded_by_this_custodian": False,
    "VAL_TEST_mask_arrays_decoded_by_this_custodian": False,
    "custodian_W_S_R_labels_decoded": True,
    "warm_loader_receives_S_R_label_values": False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "Only relative in-root artifact paths are accepted")
    path = root / relative
    _require(not path.is_symlink() and path.resolve().is_relative_to(root),
             "Artifact symlink or path escape")
    return path


def _verify(root, row, *, frozen=False):
    path = _inside(root, row["path"])
    _require(path.is_file(), "Missing artifact: " + row["path"])
    _require(_sha256(path) == row["sha256"], "Artifact hash differs: " + row["path"])
    if "bytes" in row:
        _require(path.stat().st_size == row["bytes"], "Artifact size differs")
    if frozen:
        _require(path.stat().st_mode & 0o222 == 0, "Artifact must be frozen before use")
    return path


def _descriptor(path, relative):
    path = Path(path)
    return {"path": str(relative), "sha256": _sha256(path),
            "bytes": path.stat().st_size}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o444)


def _roles(train_ids):
    """Hash-only membership; all stored tensor row orders are increasing IDs."""
    ids = list(train_ids)
    _require(len(ids) == EXPECTED_COUNTS["TRAIN"] and
             all(type(i) is int and 0 <= i < 24492 for i in ids) and
             ids == sorted(set(ids)), "Exact increasing official TRAIN0 IDs required")

    def key(tag, node):
        raw = f"{ROLE_METHOD}|{tag}|{node}".encode("utf-8")
        return hashlib.sha256(raw).digest(), node

    ordered = sorted(ids, key=lambda i: key("A", i))
    a = sorted(ordered[:len(ids) // 5])
    a_set = set(a)
    b = [i for i in ids if i not in a_set]
    ordered_b = sorted(b, key=lambda i: key("WSR", i))
    n_w, n_s = len(b) // 2, len(b) // 4
    w = sorted(ordered_b[:n_w])
    s = sorted(ordered_b[n_w:n_w + n_s])
    r = sorted(ordered_b[n_w + n_s:])
    result = {"TRAIN": ids, "A": a, "B": b, "W": w, "innerS": s, "outerR": r}
    _require({k: len(v) for k, v in result.items()} == EXPECTED_COUNTS,
             "Source-declared role counts differ; no alternative partition")
    return result


def _read_compact(np, path, expected_ids):
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == {"ids", "labels"}, "Compact integer schema differs")
        ids, labels = archive["ids"], archive["labels"]
    _require(ids.dtype == labels.dtype == np.int64 and ids.ndim == labels.ndim == 1
             and ids.shape == labels.shape and ids.tolist() == expected_ids,
             "Compact IDs/labels alignment differs")
    _require(bool(((labels >= 0) & (labels < 5)).all()), "Invalid class code")
    return labels


def prepare_train_roles(project_root, destination):
    """Custodian only. Does decode A in the shared TRAIN0 label array; no scores.

    Writes destination/public_b (roles plus B-only labels) and evaluator_a
    (A-only labels), then a custody manifest. Existing destinations are refused.
    This callable is disabled before NumPy imports, artifact reads or writes.
    """
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only TRAIN custodian")
    import numpy as np

    metadata = _read_json(_verify(project_root, DATA_MANIFEST))
    _require(metadata["schema"] == "amazon_native_train_validation_projection_v1"
             and metadata["public_graph"] == PUBLIC_GRAPH
             and metadata["train_labels"]["0"] == TRAIN0
             and metadata["raw_all_node_label_payload_decoded"] is True
             and metadata["test_label_artifacts"] == [], "Pinned producer metadata differs")
    public_path = _verify(project_root, PUBLIC_GRAPH)
    with np.load(public_path, allow_pickle=False) as archive:
        _require(set(archive.files) == {"features", "edge_index", "train_mask",
                                      "val_mask", "test_mask"}, "Public-only schema differs")
        # NPZ members are decoded separately. Only TRAIN mask is opened here.
        train_mask = archive["train_mask"]
    _require(train_mask.dtype == np.bool_ and train_mask.shape == (24492, 10),
             "Native official TRAIN mask shape differs")
    roles = _roles(np.flatnonzero(train_mask[:, 0]).tolist())
    # Role membership is complete before the labels member is decoded.
    labels = _read_compact(np, _verify(project_root, TRAIN0), roles["TRAIN"])
    train_ids = np.asarray(roles["TRAIN"], dtype=np.int64)
    b_labels = labels[np.searchsorted(train_ids, roles["B"])].copy()
    s_labels = b_labels[np.searchsorted(roles["B"], roles["innerS"])]
    _require(bool((np.bincount(s_labels, minlength=5) > 0).all()),
             "Fixed innerS misses a class; fail without reseeding or rebalancing")
    a_labels = labels[np.searchsorted(train_ids, roles["A"])].copy()
    w_labels = labels[np.searchsorted(train_ids, roles["W"])].copy()
    destination = Path(destination).resolve()
    destination.mkdir(parents=False, exist_ok=False)
    public_b = destination / "public_b"
    evaluator_a = destination / "evaluator_a"
    public_b.mkdir()
    evaluator_a.mkdir()
    role_file = public_b / "TRAIN_ROLES.json"
    _write_json(role_file, {"schema": "amazon_G0_hash_roles_v2", "method": ROLE_METHOD,
                            "row_order": "increasing_node_ID", "roles": roles})
    b_file = public_b / "B_LABELS.npz"
    a_file = evaluator_a / "A_LABELS.npz"
    w_file = public_b / "W_LABELS.npz"
    with b_file.open("xb") as stream:
        np.savez(stream, ids=np.asarray(roles["B"], dtype=np.int64), labels=b_labels)
    with a_file.open("xb") as stream:
        np.savez(stream, ids=np.asarray(roles["A"], dtype=np.int64), labels=a_labels)
    with w_file.open("xb") as stream:
        np.savez(stream, ids=np.asarray(roles["W"], dtype=np.int64), labels=w_labels)
    w_file.chmod(0o444)
    b_file.chmod(0o444)
    a_file.chmod(0o444)
    manifest = {
        "schema": "amazon_G0_public_B_projection_v2", "split": 0, "seed": 17,
        "source_metadata": DATA_MANIFEST, "public_graph": PUBLIC_GRAPH,
        "roles": _descriptor(role_file, role_file.name),
        "B_labels": _descriptor(b_file, b_file.name),
        "W_labels": _descriptor(w_file, w_file.name), "visibility": VISIBILITY,
        "custodian_source_sha256": _sha256(Path(__file__)),
    }
    manifest_path = public_b / "PUBLIC_B_MANIFEST.json"
    _write_json(manifest_path, manifest)
    custody = {"schema": "amazon_G0_TRAIN_label_custody_v2",
               "public_b_manifest": _descriptor(manifest_path, "public_b/" + manifest_path.name),
               "A_labels": _descriptor(a_file, "evaluator_a/" + a_file.name),
               "visibility": VISIBILITY}
    custody_path = destination / "CUSTODY.json"
    _write_json(custody_path, custody)
    # Returning custody metadata exposes no predictive result or A label values.
    return _descriptor(custody_path, custody_path.name)


def load_public_b(project_root, public_b_dir, *, device="cpu"):
    """Return complete public FP32 graph context, B labels and fixed role IDs.

    Output keys: features, edge_index, B_ids, B_labels, W_ids, W_labels, inner_indices,
    inner_labels, query_indices, query_labels, A_ids (IDs ONLY), provenance.
    Numeric entries are Torch tensors; provenance is JSON-compatible metadata.
    No masks, compact all-TRAIN label source, A label artifact or old state read.
    """
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only public+B accessor")
    import numpy as np
    import torch

    public_b_dir = Path(public_b_dir).resolve()
    manifest_path, manifest, roles = _projection(public_b_dir)
    b_labels = _read_compact(np, _verify(public_b_dir, manifest["B_labels"], frozen=True),
                             roles["B"])
    s_labels = b_labels[np.searchsorted(roles["B"], roles["innerS"])].copy()
    r_labels = b_labels[np.searchsorted(roles["B"], roles["outerR"])].copy()
    _require(bool((np.bincount(s_labels, minlength=5) > 0).all()),
             "Fixed innerS misses a class; no alternative partition")
    features, edges = _public_context(np, project_root)

    def ids(value):
        return torch.tensor(value, dtype=torch.long, device=device)

    return {"features": torch.from_numpy(features.copy()).to(device),
            "edge_index": torch.from_numpy(edges.copy()).to(device),
            "B_ids": ids(roles["B"]), "B_labels": ids(b_labels.tolist()),
            "W_ids": ids(roles["W"]),
            "W_labels": ids(b_labels[np.searchsorted(roles["B"], roles["W"])].tolist()),
            "inner_indices": ids(roles["innerS"]), "inner_labels": ids(s_labels.tolist()),
            "query_indices": ids(roles["outerR"]), "query_labels": ids(r_labels.tolist()),
            "A_ids": ids(roles["A"]),
            "provenance": {"public_b_manifest": _descriptor(manifest_path, str(manifest_path)),
                           "source_metadata": DATA_MANIFEST, "public_graph": PUBLIC_GRAPH,
                           "roles": manifest["roles"], "B_labels": manifest["B_labels"],
                           "W_labels": manifest["W_labels"],
                           "visibility": VISIBILITY}}


def _projection(public_b_dir):
    public_b_dir = Path(public_b_dir).resolve()
    manifest_path = public_b_dir / "PUBLIC_B_MANIFEST.json"
    _require(not manifest_path.is_symlink() and manifest_path.stat().st_mode & 0o222 == 0,
             "Frozen B-only manifest required")
    manifest = _read_json(manifest_path)
    _require(set(manifest) == {"schema", "split", "seed", "source_metadata", "public_graph",
                               "roles", "B_labels", "W_labels", "visibility", "custodian_source_sha256"}
             and manifest["schema"] == "amazon_G0_public_B_projection_v2"
             and manifest["split"] == 0 and manifest["seed"] == 17
             and manifest["source_metadata"] == DATA_MANIFEST
             and manifest["public_graph"] == PUBLIC_GRAPH and manifest["visibility"] == VISIBILITY
             and manifest["custodian_source_sha256"] == _sha256(Path(__file__))
             and manifest["roles"]["path"] == "TRAIN_ROLES.json"
             and manifest["B_labels"]["path"] == "B_LABELS.npz"
             and manifest["W_labels"]["path"] == "W_LABELS.npz", "B-only manifest differs")
    role_record = _read_json(_verify(public_b_dir, manifest["roles"], frozen=True))
    _require(set(role_record) == {"schema", "method", "row_order", "roles"}
             and role_record["schema"] == "amazon_G0_hash_roles_v2"
             and role_record["method"] == ROLE_METHOD
             and role_record["row_order"] == "increasing_node_ID", "Role schema differs")
    roles = role_record["roles"]
    _require(roles == _roles(roles["TRAIN"]), "ID-only hash roles differ")
    return manifest_path, manifest, roles


def _public_context(np, project_root):
    with np.load(_verify(project_root, PUBLIC_GRAPH), allow_pickle=False) as archive:
        _require(set(archive.files) == {"features", "edge_index", "train_mask",
                                      "val_mask", "test_mask"}, "Public-only schema differs")
        features, edges = archive["features"], archive["edge_index"]
    _require(features.dtype == np.float32 and features.shape == (24492, 300)
             and bool(np.isfinite(features).all()) and edges.dtype == np.int64
             and edges.ndim == 2 and edges.shape[0] == 2
             and bool(((edges >= 0) & (edges < 24492)).all()), "Public context schema differs")

    return features, edges


def load_public_w(project_root, public_b_dir, *, device="cpu"):
    """Warm-only reader: W labels plus full public features/graph, no S/R values.

    Custody already projected the joint source TRAIN array. This reader opens
    W_LABELS only and never opens B_LABELS, original TRAIN labels, or A labels.
    """
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only public+W acquisition accessor")
    import numpy as np
    import torch
    public_b_dir = Path(public_b_dir).resolve()
    manifest_path, manifest, roles = _projection(public_b_dir)
    w_labels = _read_compact(np, _verify(public_b_dir, manifest["W_labels"], frozen=True), roles["W"])
    features, edges = _public_context(np, project_root)
    return {"features": torch.from_numpy(features.copy()).to(device),
            "edge_index": torch.from_numpy(edges.copy()).to(device),
            "W_ids": torch.tensor(roles["W"], dtype=torch.long, device=device),
            "W_labels": torch.tensor(w_labels.tolist(), dtype=torch.long, device=device),
            "provenance": {"public_b_manifest": _descriptor(manifest_path, str(manifest_path)),
                           "source_metadata": DATA_MANIFEST, "public_graph": PUBLIC_GRAPH,
                           "roles": manifest["roles"], "B_labels": manifest["B_labels"],
                           "W_labels": manifest["W_labels"], "visibility": VISIBILITY}}
