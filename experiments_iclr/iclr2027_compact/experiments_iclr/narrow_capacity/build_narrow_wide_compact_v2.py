"""Project audited narrow72 and WikiCS wide18 into a small public record.

The projection omits checkpoints and member floats. Its manifest preserves the
SHA-256 of each source array, and the full source archive remains author evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

import tuning as primary


STUDIES = {
    "narrow72": ("narrow_untied72_results", "FROZEN_NARROW_UNTIED72_STUDY.json",
                 ("cora", "wikics", "actor", "chameleon_filtered")),
    "wide18": ("wikics_wide_untied18_results", "FROZEN_WIKICS_WIDE_UNTIED18_STUDY.json",
               ("wikics",)),
}
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
              (0.001, 0.01), (0.003, 0.0), (0.003, 0.01))
SEEDS = (0, 1, 2)
DEFAULT = (0.001, 0.0)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key(graph, lr, wd, seed):
    return f"{graph}/untied/lr{lr:g}_wd{wd:g}/seed{seed}"


def cross_entropy(logits, labels):
    x = logits.astype(np.float64)
    maximum = x.max(axis=1)
    logsum = maximum + np.log(np.exp(x - maximum[:, None]).sum(axis=1))
    return float(np.mean(logsum - x[np.arange(len(labels)), labels]))


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return sha(target)


def archive_file(archive, source, name, hashes):
    info = archive.gettarinfo(str(source), arcname=name)
    need(info.isfile(), f"Missing record: {source}")
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    with source.open("rb") as stream:
        archive.addfile(info, stream)
    hashes[name] = sha(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    need(not out.exists(), "Refusing to replace compact projection")
    work = out.with_name(out.name + ".inprogress")
    need(not work.exists(), "Interrupted compact projection needs inspection")
    primary_freeze_sha = primary.check_freeze()
    frozen_primary = json.loads((source / "FROZEN_STUDY.json").read_text())
    studies = {}
    for kind, (folder, freeze_name, graphs) in STUDIES.items():
        root = source / folder
        freeze_path = root / freeze_name
        lock_path = root / "VALIDATION_SELECTION_LOCK.json"
        audit_path = root / "FINAL_SCORE_AUDIT.json"
        need(all(p.is_file() for p in (freeze_path, lock_path, audit_path)),
             f"Incomplete {kind} freeze/validation/score audit")
        freeze = json.loads(freeze_path.read_text())
        lock = json.loads(lock_path.read_text())
        audit = json.loads(audit_path.read_text())
        need(len(lock["cells"]) == 18 * len(graphs) and
             lock["freeze_sha256"] == sha(freeze_path) and
             audit["validation_selection_lock_sha256"] == sha(lock_path),
             f"Incomplete {kind} lock/audit chain")
        for name, digest in freeze["source_sha256"].items():
            need(sha(source / name) == digest, f"Frozen {kind} source changed: {name}")
        studies[kind] = (root, freeze_path, lock_path, audit_path, lock, audit, graphs)
    need(studies["wide18"][4]["narrow_validation_lock_sha256"] ==
         sha(studies["narrow72"][2]), "Wide lock does not bind narrow lock")
    work.mkdir(parents=True)
    top_hashes = {}
    top_hashes["build_narrow_wide_compact_v2.py"] = copy(Path(__file__), work / "build_narrow_wide_compact_v2.py")
    verifier = Path(__file__).with_name("verify_narrow_wide_compact_v2.py")
    top_hashes["verify_narrow_wide_compact_v2.py"] = copy(verifier, work / "verify_narrow_wide_compact_v2.py")
    for kind, (_, freeze_path, lock_path, audit_path, _, _, _) in studies.items():
        for path in (freeze_path, lock_path, audit_path):
            target = work / kind / path.name
            top_hashes[target.relative_to(work).as_posix()] = copy(path, target)
        root = studies[kind][0]
        for extra in ("PREFLIGHT_CUDA.json", "CAPACITY_SENSITIVITY_COMPARISON.json",
                      "WIKICS_WIDTH_CONTRAST.json"):
            path = root / extra
            if path.is_file():
                target = work / kind / extra
                top_hashes[target.relative_to(work).as_posix()] = copy(path, target)
        for name in json.loads(freeze_path.read_text())["source_sha256"]:
            path = source / name
            target = work / "source" / name
            top_hashes[target.relative_to(work).as_posix()] = copy(path, target)
    for name in ("tuning.py", "models.py", "verify_tuning.py", "FROZEN_STUDY.json",
                 "NARROW_UNTIED72_PROTOCOL.md", "WIKICS_WIDE_UNTIED18_PROTOCOL.md",
                 "verify_narrow_untied72.py", "verify_wikics_wide_untied18.py"):
        path = source / name
        if path.is_file():
            target = work / "source" / name
            top_hashes[target.relative_to(work).as_posix()] = copy(path, target)
    graph_references = {}
    loaders = {"cora": primary.load_cora, "wikics": primary.load_wikics,
               "actor": primary.load_actor, "chameleon_filtered": primary.load_filtered}
    for graph, loader in loaders.items():
        _, labels_t, _, _, _, valid_mask, test_mask = loader()
        valid_t = valid_mask.nonzero(as_tuple=False).flatten().long()
        test_t = test_mask.nonzero(as_tuple=False).flatten().long()
        descriptor = frozen_primary["graphs"][graph]["descriptor"]
        need(primary.tensor_sha(labels_t) == descriptor["tensor_sha256"]["labels"] and
             primary.tensor_sha(valid_t) == descriptor["tensor_sha256"]["valid_indices"] and
             primary.tensor_sha(test_t) == descriptor["tensor_sha256"]["test_indices"],
             f"Frozen graph labels/splits differ: {graph}")
        path = work / "references" / f"{graph}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, all_labels=labels_t.numpy(),
                            valid_indices=valid_t.numpy(), test_indices=test_t.numpy())
        graph_references[graph] = {"path": path.relative_to(work).as_posix(),
                                   "sha256": sha(path),
                                   "frozen_label_tensor_sha256": descriptor["tensor_sha256"]["labels"],
                                   "frozen_valid_indices_tensor_sha256": descriptor["tensor_sha256"]["valid_indices"],
                                   "frozen_test_indices_tensor_sha256": descriptor["tensor_sha256"]["test_indices"]}
    references, validations, scores, record_hashes = {}, {}, {}, {}
    with tarfile.open(work / "CELL_RECORDS.tar.xz", "w:xz", preset=6) as archive:
        for kind, (root, _, _, _, lock, audit, graphs) in studies.items():
            expected_cells = {key(g, lr, wd, s) for g in graphs
                              for lr, wd in CANDIDATES for s in SEEDS}
            need(set(lock["cells"]) == expected_cells, f"Incomplete {kind} grid")
            expected_scores = set()
            for graph in graphs:
                selected = (tuple(lock["selections"][graph]["selected_candidate"])
                            if kind == "narrow72" else
                            tuple(lock["selection"]["selected_candidate"]))
                for lr, wd in dict.fromkeys((selected, DEFAULT)):
                    expected_scores.update(key(graph, lr, wd, s) for s in SEEDS)
            need(set(audit["scores"]) == expected_scores, f"Unauthorized/missing {kind} score")
            for identity in sorted(expected_cells):
                cell = root / "results" / identity
                locked = lock["cells"][identity]
                result = cell / "result.json"
                trace = cell / "validation_trace.csv"
                meta = cell / "validation_companion.json"
                companion = cell / "validation_companion.npz"
                need(sha(result) == locked["result_sha256"] and
                     sha(trace) == locked["trace_sha256"] and
                     sha(meta) == locked["validation_companion_manifest_sha256"] and
                     sha(companion) == locked["validation_companion_sha256"],
                     f"Locked {kind} cell differs: {identity}")
                row = json.loads(result.read_text())
                need(row["failure"] is None and row["epochs_completed"] == 1000,
                     f"Incomplete/nonfinite {kind} cell: {identity}")
                for path in (result, trace, meta):
                    archive_file(archive, path,
                                 f"{kind}/results/{identity}/{path.name}", record_hashes)
                with np.load(companion, allow_pickle=False) as arr:
                    need(set(arr.files) == {"valid_indices", "valid_labels",
                                            "valid_pooled_logits", "valid_member_logits"},
                         f"Companion schema {kind}/{identity}")
                    indices, labels = arr["valid_indices"], arr["valid_labels"]
                    pooled, members = arr["valid_pooled_logits"], arr["valid_member_logits"]
                need(pooled.dtype == np.float32 and members.dtype == np.float32 and
                     members.shape == (4, *pooled.shape) and
                     np.isfinite(pooled).all() and np.isfinite(members).all(),
                     f"Companion values {kind}/{identity}")
                graph = identity.split("/")[0]
                with np.load(work / graph_references[graph]["path"], allow_pickle=False) as ref:
                    expected_indices = ref["valid_indices"]
                    expected_labels = ref["all_labels"][expected_indices]
                    need(np.array_equal(indices, expected_indices) and
                         np.array_equal(labels, expected_labels),
                         f"Frozen validation reference mismatch {kind}/{identity}")
                references[graph] = graph_references[graph]
                pooled_class = pooled.argmax(-1).astype(np.uint8)
                member_class = members.argmax(-1).astype(np.uint8)
                need(members.shape[-1] <= 256 and
                     abs(float(np.mean(pooled.argmax(-1) == labels)) -
                         row["selected_valid_accuracy"]) <= 1e-7 and
                     abs(cross_entropy(pooled, labels) - row["selected_valid_ce"]) <= 1e-5,
                     f"Validation decisions/metric {kind}/{identity}")
                projected = work / "validation" / kind / f"{identity}.npz"
                projected.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(projected, pooled_class=pooled_class,
                                    member_class=member_class)
                validations[f"{kind}/{identity}"] = {
                    "source_companion_sha256": sha(companion),
                    "checkpoint_sha256": locked["checkpoint_sha256"],
                    "path": projected.relative_to(work).as_posix(),
                    "sha256": sha(projected),
                    "selected_valid_accuracy": row["selected_valid_accuracy"],
                    "selected_valid_ce": row["selected_valid_ce"],
                }
            for identity in sorted(expected_scores):
                folder = root / "scores" / identity
                score, prediction = folder / "score.json", folder / "predictions.npz"
                audited = audit["scores"][identity]
                need(sha(score) == audited["score_sha256"] and
                     sha(prediction) == audited["predictions_sha256"],
                     f"Audited score differs {kind}/{identity}")
                archive_file(archive, score, f"{kind}/scores/{identity}/score.json", record_hashes)
                row = json.loads(score.read_text())
                need(row["checkpoint_sha256"] == lock["cells"][identity]["checkpoint_sha256"] and
                     row["test_predictions_sha256"] == sha(prediction),
                     f"Score/checkpoint differs {kind}/{identity}")
                with np.load(prediction, allow_pickle=False) as arr:
                    pooled, members = arr["test_pooled_logits"], arr["test_member_logits"]
                need(pooled.dtype == np.float32 and members.dtype == np.float32 and
                     members.shape == (4, *pooled.shape) and np.isfinite(pooled).all() and
                     np.isfinite(members).all(), f"Test floats {kind}/{identity}")
                projected = work / "test" / kind / f"{identity}.npz"
                projected.parent.mkdir(parents=True, exist_ok=True)
                graph = identity.split("/")[0]
                with np.load(work / graph_references[graph]["path"], allow_pickle=False) as ref:
                    test_labels = ref["all_labels"][ref["test_indices"]]
                member_class = members.argmax(-1).astype(np.uint8)
                member_accuracy = [float(np.mean(pred == test_labels)) for pred in member_class]
                need(abs(float(np.mean(pooled.argmax(-1) == test_labels)) -
                         row["test_accuracy"]) <= 1e-7 and
                     abs(cross_entropy(pooled, test_labels) - row["test_ce"]) <= 1e-5,
                     f"Test labels/metrics differ {kind}/{identity}")
                np.savez_compressed(projected, pooled_logits=pooled,
                                    member_class=member_class)
                scores[f"{kind}/{identity}"] = {
                    "source_predictions_sha256": sha(prediction),
                    "path": projected.relative_to(work).as_posix(),
                    "sha256": sha(projected),
                    "test_accuracy": row["test_accuracy"],
                    "test_ce": row["test_ce"],
                    "member_accuracy": member_accuracy,
                }
    manifest = {"protocol": "narrow72_wikics_wide18_compact_v2",
                "scope": "All 90 validation cells and full traces plus all 27 selected/default test scores. Validation pooled/member hard classes and test pooled float32 logits/member hard classes are exact projections. Public validation CE is trace-only because validation float logits, source checkpoints and member float logits are retained only in author evidence.",
                "builder_sha256": sha(Path(__file__)),
                "verifier_sha256": sha(verifier),
                "primary_freeze_sha256": primary_freeze_sha,
                "cell_records_sha256": sha(work / "CELL_RECORDS.tar.xz"),
                "top_file_sha256": top_hashes, "record_sha256": record_hashes,
                "references": references, "validation": validations, "scores": scores,
                "counts": {"validation_cells": len(validations), "test_scores": len(scores),
                           "complete_traces": len(validations)}}
    need(len(validations) == 90 and len(references) == 4,
         "Compact projection coverage incomplete")
    (work / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    work.rename(out)
    print(json.dumps({"manifest_sha256": sha(out / "MANIFEST.json"),
                      "counts": manifest["counts"],
                      "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
