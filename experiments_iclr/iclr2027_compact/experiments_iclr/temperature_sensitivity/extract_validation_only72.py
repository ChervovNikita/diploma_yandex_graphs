"""Extract the 72 original validation pooled-logit arrays without test arrays."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np


GRAPHS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
AUDIT_SHA = "63da49d3d797fa7856717d4b21a530c74419bfbdfa92614a03b036aa9950bdde"
ARCHIVE_SHA = "f872087c79abdc20ce36d088a974c96335e02beeac90b395141c17aa237c3410"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def validation_indices(graph: str, data: Path) -> np.ndarray:
    if graph == "cora":
        # PyG Planetoid public Cora split: 140 training, next 500 validation.
        return np.arange(140, 640, dtype=np.int64)
    if graph == "wikics":
        source = json.loads((data / "wiki_cs" / "data.json").read_text())
        return np.flatnonzero(np.asarray(source["val_masks"][0], dtype=bool)).astype(np.int64)
    if graph == "actor":
        with np.load(data / "actor" / "film_split_0.6_0.2_0.npz", allow_pickle=False) as source:
            return np.flatnonzero(source["val_mask"]).astype(np.int64)
    with np.load(data / "chameleon_filtered.npz", allow_pickle=False) as source:
        return np.flatnonzero(source["val_masks"][0]).astype(np.int64)


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    x = logits.astype(np.float64)
    peak = x.max(axis=1)
    return float(np.mean(peak + np.log(np.exp(x - peak[:, None]).sum(axis=1))
                         - x[np.arange(len(labels)), labels]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="Frozen study root containing the public data/ directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Refusing overwrite"
    assert sha_file(args.archive) == ARCHIVE_SHA
    assert sha_file(args.study / "VALIDATION_SELECTION_LOCK.json") == LOCK_SHA
    assert sha_file(args.study / "FINAL_SCORE_AUDIT.json") == AUDIT_SHA
    lock = json.loads((args.study / "VALIDATION_SELECTION_LOCK.json").read_text())
    audit = json.loads((args.study / "FINAL_SCORE_AUDIT.json").read_text())
    frozen = json.loads((args.study / "FROZEN_STUDY.json").read_text())
    hard = json.loads((args.study / "HARD_DECISION_EXPORT_MANIFEST.json").read_text())
    assert len(lock["cells"]) == 432 and len(audit["scores"]) == 141
    args.out.mkdir(parents=True)
    references, cells = {}, {}
    with tarfile.open(args.archive, "r:gz") as archive:
        original_manifest_raw = archive.extractfile("SELECTED_HPO72_LOGITS_MANIFEST.json").read()
        original_manifest = json.loads(original_manifest_raw)
        assert len(original_manifest["cells"]) == 72
        for graph in GRAPHS:
            descriptor = frozen["graphs"][graph]["descriptor"]
            for name, expected in frozen["graphs"][graph]["raw_sha256"].items():
                assert sha_file(args.data / name) == expected
            ref_record = hard["graph_references"][graph]
            ref_path = args.study / ref_record["reference_path"]
            assert sha_file(ref_path) == ref_record["reference_sha256"]
            with np.load(ref_path, allow_pickle=False) as ref:
                full_labels = ref["full_labels"]
            assert tensor_sha(full_labels) == descriptor["tensor_sha256"]["labels"]
            indices = validation_indices(graph, args.data / "data")
            assert tensor_sha(indices) == descriptor["tensor_sha256"]["valid_indices"]
            labels = full_labels[indices]
            assert labels.dtype == indices.dtype == np.int64
            assert len(labels) == descriptor["split_sizes"]["valid"]
            ref_out = args.out / "references" / f"{graph}.npz"
            ref_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ref_out, valid_indices=indices, valid_labels=labels)
            references[graph] = {"path": ref_out.relative_to(args.out).as_posix(),
                                 "sha256": sha_file(ref_out),
                                 "indices_tensor_sha256": tensor_sha(indices),
                                 "full_labels_tensor_sha256": tensor_sha(full_labels),
                                 "valid_nodes": len(labels)}
            for arm in ARMS:
                lr, wd = lock["selections"][graph][arm]["selected_candidate"]
                for seed in range(3):
                    key = f"{graph}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"
                    original = archive.extractfile(f"scores/{key}/predictions.npz").read()
                    original_sha = sha_bytes(original)
                    assert original_sha == original_manifest["cells"][key]["predictions_sha256"]
                    assert original_sha == audit["scores"][key]["predictions_sha256"]
                    with np.load(io.BytesIO(original), allow_pickle=False) as arrays:
                        # Accesses only the validation member of the source NPZ.
                        valid = arrays["valid_pooled_logits"]
                    assert valid.dtype == np.float32 and valid.shape == (len(labels), descriptor["classes"])
                    result = json.loads((args.study / "results" / key / "result.json").read_text())
                    accuracy = float(np.mean(valid.argmax(axis=1) == labels))
                    assert abs(accuracy - result["selected_valid_accuracy"]) < 1e-7
                    assert abs(ce(valid, labels) - result["selected_valid_ce"]) < 1e-5
                    out = args.out / "logits" / f"{key}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out, valid_pooled_logits=valid)
                    cells[key] = {
                        "path": out.relative_to(args.out).as_posix(),
                        "sha256": sha_file(out),
                        "valid_logits_bytes_sha256": sha_bytes(valid.tobytes(order="C")),
                        "source_predictions_sha256": original_sha,
                        "shape": list(valid.shape),
                        "selected_valid_accuracy": accuracy,
                        "selected_valid_ce": ce(valid, labels),
                    }
    assert len(cells) == 72
    manifest = {
        "protocol": "selected_hpo72_validation_only_logits_transport_v1",
        "selection_lock_sha256": LOCK_SHA,
        "independent_final_score_audit_sha256": AUDIT_SHA,
        "original_archive_sha256": ARCHIVE_SHA,
        "original_archive_manifest_sha256": sha_bytes(original_manifest_raw),
        "frozen_study_sha256": sha_file(args.study / "FROZEN_STUDY.json"),
        "source_extractor_sha256": sha_file(Path(__file__)),
        "scope": "all 72 validation-selected cells; validation labels/indices and pooled logits only; no test arrays",
        "references": references,
        "cells": cells,
        "cell_count": len(cells),
    }
    (args.out / "VALIDATION_ONLY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "VALIDATION_ONLY72_EXTRACTED", "cells": len(cells),
                      "manifest_sha256": sha_file(args.out / "VALIDATION_ONLY_MANIFEST.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
