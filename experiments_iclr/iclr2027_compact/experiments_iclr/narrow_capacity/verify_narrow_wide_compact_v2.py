"""Verify public narrow72/wide18 hard-class projection and test floats."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def tensor_sha(array):
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(str(x.dtype).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def cross_entropy(logits, labels):
    x = logits.astype(np.float64)
    maximum = x.max(axis=1)
    logsum = maximum + np.log(np.exp(x - maximum[:, None]).sum(axis=1))
    return float(np.mean(logsum - x[np.arange(len(labels)), labels]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    stage = args.stage.resolve()
    manifest = json.loads((stage / "MANIFEST.json").read_text())
    need(manifest["protocol"] == "narrow72_wikics_wide18_compact_v2" and
         len(manifest["validation"]) == 90 and len(manifest["references"]) == 4 and
         manifest["counts"]["complete_traces"] == 90,
         "Projection manifest scope differs")
    need(sha(stage / "build_narrow_wide_compact_v2.py") == manifest["builder_sha256"],
         "Builder source differs")
    need(sha(stage / "verify_narrow_wide_compact_v2.py") == manifest["verifier_sha256"],
         "Verifier source differs")
    need(sha(stage / "source/FROZEN_STUDY.json") == manifest["primary_freeze_sha256"],
         "Original frozen graph descriptor differs")
    primary_freeze = json.loads((stage / "source/FROZEN_STUDY.json").read_text())
    for name, digest in manifest["top_file_sha256"].items():
        need(sha(stage / name) == digest, f"Top-level source/lock differs: {name}")
    locks, audits = {}, {}
    for kind, n in (("narrow72", 72), ("wide18", 18)):
        root = stage / kind
        freeze = next(root.glob("FROZEN_*_STUDY.json"))
        lock_path = root / "VALIDATION_SELECTION_LOCK.json"
        audit_path = root / "FINAL_SCORE_AUDIT.json"
        lock, audit = json.loads(lock_path.read_text()), json.loads(audit_path.read_text())
        locks[kind], audits[kind] = lock, audit
        need(len(lock["cells"]) == n and lock["freeze_sha256"] == sha(freeze) and
             audit["validation_selection_lock_sha256"] == sha(lock_path),
             f"{kind} lock/audit chain differs")
        frozen = json.loads(freeze.read_text())
        for name, digest in frozen["source_sha256"].items():
            need(sha(stage / "source" / name) == digest,
                 f"Frozen {kind} source differs: {name}")
    archive_path = stage / "CELL_RECORDS.tar.xz"
    need(sha(archive_path) == manifest["cell_records_sha256"],
         "Cell-record archive digest differs")
    records = {}
    with tarfile.open(archive_path, "r:xz") as archive:
        for item in archive:
            need(item.isfile() and not item.name.startswith("/") and
                 ".." not in Path(item.name).parts and
                 item.name in manifest["record_sha256"] and item.name not in records,
                 f"Unauthorized archive record: {item.name}")
            data = archive.extractfile(item).read()
            need(hash_bytes(data) == manifest["record_sha256"][item.name],
                 f"Archive record digest differs: {item.name}")
            records[item.name] = data
    need(set(records) == set(manifest["record_sha256"]),
         "Missing archive records")
    refs = {}
    for graph, row in manifest["references"].items():
        path = stage / row["path"]
        need(sha(path) == row["sha256"], f"Reference differs: {graph}")
        with np.load(path, allow_pickle=False) as arr:
            need(set(arr.files) == {"all_labels", "valid_indices", "test_indices"},
                 f"Reference schema differs: {graph}")
            labels, valid, test = (arr[name].copy() for name in
                                   ("all_labels", "valid_indices", "test_indices"))
        descriptor = primary_freeze["graphs"][graph]["descriptor"]
        need(tensor_sha(labels) == row["frozen_label_tensor_sha256"] ==
             descriptor["tensor_sha256"]["labels"] and
             tensor_sha(valid) == row["frozen_valid_indices_tensor_sha256"] ==
             descriptor["tensor_sha256"]["valid_indices"] and
             tensor_sha(test) == row["frozen_test_indices_tensor_sha256"] ==
             descriptor["tensor_sha256"]["test_indices"] and
             len(valid) == descriptor["split_sizes"]["valid"] and
             len(test) == descriptor["split_sizes"]["test"],
             f"Frozen graph labels/splits differ: {graph}")
        refs[graph] = {"valid_labels": labels[valid], "test_labels": labels[test]}
    for identity, row in manifest["validation"].items():
        kind, graph, *_ = identity.split("/")
        cell = identity[len(kind) + 1:]
        root = f"{kind}/results/{cell}"
        result = json.loads(records[root + "/result.json"])
        meta = json.loads(records[root + "/validation_companion.json"])
        locked = locks[kind]["cells"][cell]
        need(hash_bytes(records[root + "/result.json"]) == locked["result_sha256"] and
             hash_bytes(records[root + "/validation_trace.csv"]) == locked["trace_sha256"] and
             hash_bytes(records[root + "/validation_companion.json"]) ==
             locked["validation_companion_manifest_sha256"] and
             row["source_companion_sha256"] == locked["validation_companion_sha256"],
             f"Frozen cell record differs: {identity}")
        trace = list(csv.DictReader(io.StringIO(records[root + "/validation_trace.csv"].decode())))
        need(len(trace) == 1000 and result["failure"] is None and
             result["selected_valid_accuracy"] == row["selected_valid_accuracy"] and
             result["selected_valid_ce"] == row["selected_valid_ce"] and
             result["checkpoint_sha256"] == row["checkpoint_sha256"] and
             meta["predictions_sha256"] == row["source_companion_sha256"],
             f"Validation record differs: {identity}")
        need([int(t["epoch"]) for t in trace] == list(range(1, 1001)),
             f"Epoch sequence differs: {identity}")
        best_acc, best_ce = -float("inf"), float("inf")
        for entry in trace:
            acc, ce = float(entry["valid_accuracy"]), float(entry["valid_ce"])
            improved = acc > best_acc or (acc == best_acc and ce < best_ce)
            need(int(entry["selected_now"]) == int(improved),
                 f"Selection flag differs: {identity}")
            if improved:
                best_acc, best_ce = acc, ce
        best = max(trace, key=lambda t: (float(t["valid_accuracy"]),
                                         -float(t["valid_ce"]), -int(t["epoch"])))
        need(int(best["epoch"]) == result["selected_epoch"] and
             float(best["valid_accuracy"]) == result["selected_valid_accuracy"] and
             float(best["valid_ce"]) == result["selected_valid_ce"],
             f"Trace checkpoint selection differs: {identity}")
        path = stage / row["path"]
        need(sha(path) == row["sha256"], f"Validation projection differs: {identity}")
        with np.load(path, allow_pickle=False) as arr:
            need(set(arr.files) == {"pooled_class", "member_class"},
                 f"Projection schema differs: {identity}")
            pooled, member = arr["pooled_class"], arr["member_class"]
            need(pooled.dtype == np.uint8 and member.dtype == np.uint8 and
                 member.shape == (4, len(refs[graph]["valid_labels"])) and
                 pooled.shape == refs[graph]["valid_labels"].shape and
                 int(pooled.max()) < primary_freeze["graphs"][graph]["descriptor"]["classes"] and
                 int(member.max()) < primary_freeze["graphs"][graph]["descriptor"]["classes"] and
                 abs(float(np.mean(pooled == refs[graph]["valid_labels"])) -
                     row["selected_valid_accuracy"]) <= 1e-7,
                 f"Validation decisions differ: {identity}")
    for identity, row in manifest["scores"].items():
        kind, cell = identity.split("/", 1)
        score = json.loads(records[f"{kind}/scores/{cell}/score.json"])
        need(hash_bytes(records[f"{kind}/scores/{cell}/score.json"]) ==
             audits[kind]["scores"][cell]["score_sha256"] and
             row["source_predictions_sha256"] ==
             audits[kind]["scores"][cell]["predictions_sha256"] and
             score["test_predictions_sha256"] == row["source_predictions_sha256"] and
             score["test_accuracy"] == row["test_accuracy"] and
             score["test_ce"] == row["test_ce"],
             f"Test score record differs: {identity}")
        path = stage / row["path"]
        need(sha(path) == row["sha256"], f"Test projection differs: {identity}")
        with np.load(path, allow_pickle=False) as arr:
            need(set(arr.files) == {"pooled_logits", "member_class"},
                 f"Test projection schema differs: {identity}")
            pooled, member = arr["pooled_logits"], arr["member_class"]
            graph = cell.split("/")[0]
            test_labels = refs[graph]["test_labels"]
            need(pooled.dtype == np.float32 and member.dtype == np.uint8 and
                 member.shape == (4, pooled.shape[0]) and
                 len(test_labels) == pooled.shape[0] and
                 np.isfinite(pooled).all() and
                 abs(float(np.mean(pooled.argmax(-1) == test_labels)) -
                     row["test_accuracy"]) <= 1e-7 and
                 abs(cross_entropy(pooled, test_labels) - row["test_ce"]) <= 1e-5 and
                 np.allclose([float(np.mean(pred == test_labels)) for pred in member],
                             row["member_accuracy"], rtol=0, atol=1e-12),
                 f"Test projection floats/classes differ: {identity}")
    for kind in locks:
        need({item.split("/", 1)[1] for item in manifest["validation"] if item.startswith(kind + "/")} ==
             set(locks[kind]["cells"]), f"{kind} validation coverage differs")
        need({item.split("/", 1)[1] for item in manifest["scores"] if item.startswith(kind + "/")} ==
             set(audits[kind]["scores"]), f"{kind} score coverage differs")
        for graph in (("cora", "wikics", "actor", "chameleon_filtered")
                      if kind == "narrow72" else ("wikics",)):
            rows = []
            for lr, wd in ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
                           (0.001, 0.01), (0.003, 0.0), (0.003, 0.01)):
                records_for_candidate = [json.loads(records[
                    f"{kind}/results/{graph}/untied/lr{lr:g}_wd{wd:g}/seed{s}/result.json"])
                    for s in (0, 1, 2)]
                rows.append((float(np.mean([r["selected_valid_accuracy"] for r in records_for_candidate])),
                             -float(np.mean([r["selected_valid_ce"] for r in records_for_candidate])),
                             -lr, -wd, lr, wd))
            chosen = max(rows)
            locked = (locks[kind]["selections"][graph]["selected_candidate"]
                      if kind == "narrow72" else
                      locks[kind]["selection"]["selected_candidate"])
            need(tuple(locked) == (chosen[4], chosen[5]),
                 f"Validation candidate selection differs: {kind}/{graph}")
    print(json.dumps({"status": "PASS", "validation_cells": len(manifest["validation"]),
                      "test_scores": len(manifest["scores"]),
                      "manifest_sha256": sha(stage / "MANIFEST.json")}))


if __name__ == "__main__":
    main()
