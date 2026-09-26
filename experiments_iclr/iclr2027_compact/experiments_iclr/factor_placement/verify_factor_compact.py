"""CPU-only independent public verifier for all 108 factor/control cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np


GRAPHS = {"all_layer": ("cora", "wikics", "actor", "chameleon_filtered"),
          "tied36": ("cora", "wikics")}
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
              (0.001, 0.01), (0.003, 0.0), (0.003, 0.01))
SEEDS = (0, 1, 2)
DEFAULT = (0.001, 0.0)
FREEZES = {"all_layer": "FROZEN_ALL_LAYER_STUDY.json", "tied36": "FROZEN_STUDY.json"}
PRIMARY_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
ORIGINAL_DESIGN_FREEZE_SHA = "f0cdf4ee2ac7a48a8ca80093aca53c05240ba402d01640c1cf9a2ca34727a294"
ORIGINAL_BUILDER_SHA = "1d6069c67c71c6fce95350bcd12493fea56bddeda97aa37c14abd3e6aaabc121"
ORIGINAL_VERIFIER_SHA = "a049e94dce338d18089b524e64fc97b51aef9b81ea610f98b4c334defb04b57b"


def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def tensor_sha(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return sha_bytes(str(arr.shape).encode() + str(arr.dtype).encode() + arr.tobytes())


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def key(kind: str, graph: str, lr: float, wd: float, seed: int) -> str:
    prefix = f"{graph}/tied" if kind == "tied36" else graph
    return f"{prefix}/lr{lr:g}_wd{wd:g}/seed{seed}"


def allowed(lock: dict, kind: str) -> set[str]:
    return {key(kind, graph, lr, wd, seed) for graph in GRAPHS[kind]
            for lr, wd in dict.fromkeys((tuple(lock["selections"][graph]["selected_candidate"]), DEFAULT))
            for seed in SEEDS}


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    x = logits.astype(np.float64)
    top = x.max(axis=1)
    return float(np.mean(top + np.log(np.exp(x - top[:, None]).sum(axis=1))
                         - x[np.arange(len(labels)), labels]))


def trace_best(records: list[dict]) -> tuple[int, float, float]:
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    for epoch, row in enumerate(records, 1):
        need(int(row["epoch"]) == epoch, "Trace epoch order")
        acc, cross_entropy = float(row["valid_accuracy"]), float(row["valid_ce"])
        need(np.isfinite(acc) and np.isfinite(cross_entropy) and
             0 <= acc <= 1 and cross_entropy >= 0, "Trace finite metrics")
        better = acc > best_acc or (acc == best_acc and cross_entropy < best_ce)
        need(int(row["selected_now"]) == int(better), "Trace selection flag")
        if better:
            best_acc, best_ce, best_epoch = acc, cross_entropy, epoch
    return best_epoch, best_acc, best_ce


def candidate_choice(rows: list[dict]) -> tuple[float, float]:
    valid = [row for row in rows if row["valid"]]
    need(bool(valid), "No valid candidate")
    chosen = max(valid, key=lambda row: (row["mean_valid_accuracy"],
                                         -row["mean_valid_ce"], -row["lr"],
                                         -row["weight_decay"]))
    return chosen["lr"], chosen["weight_decay"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--primary", type=Path,
                        help="Primary public validation_tuning/ folder")
    args = parser.parse_args()
    root = args.stage
    primary = args.primary or root.parent / "validation_tuning"
    manifest = read(root / "MANIFEST.json")
    need(manifest["protocol"] == "factor_placement_compact_public_all108_v1" and
         manifest["primary_selection_lock_sha256"] == PRIMARY_LOCK_SHA and
         manifest["counts"]["validation_cells"] == 108 and
         manifest["counts"]["result_json"] == 108 and
         manifest["counts"]["full_validation_traces"] == 108,
         "Compact stage manifest/counts")
    need(sha(primary / "VALIDATION_SELECTION_LOCK.json") == PRIMARY_LOCK_SHA and
         sha(root / "FACTOR_COMPACT_PROTOCOL.md") == manifest["builder_protocol_sha256"],
         "Primary lock/builder protocol")
    need(sha(root / "build_factor_compact.py") == manifest["builder_source_sha256"] and
         sha(root / "verify_factor_compact.py") == manifest["public_verifier_sha256"],
         "Transport builder/verifier source")
    design_path = root / "FACTOR_COMPACT_DESIGN_FREEZE.json"
    amendment_path = root / "FACTOR_COMPACT_DESIGN_AMENDMENT_V2.json"
    need(sha(design_path) == manifest["original_design_freeze_sha256"] ==
         ORIGINAL_DESIGN_FREEZE_SHA and
         sha(amendment_path) == manifest["design_amendment_sha256"],
         "Pre-execution transport design history")
    design, amendment = read(design_path), read(amendment_path)
    need(design["builder_sha256"] == amendment["original_builder_sha256"] ==
         ORIGINAL_BUILDER_SHA and
         design["verifier_sha256"] == amendment["original_verifier_sha256"] ==
         ORIGINAL_VERIFIER_SHA and
         amendment["original_design_freeze_sha256"] == ORIGINAL_DESIGN_FREEZE_SHA and
         amendment["amended_builder_sha256"] == manifest["builder_source_sha256"] and
         amendment["amended_verifier_sha256"] == manifest["public_verifier_sha256"] and
         design["protocol_sha256"] == amendment["protocol_sha256"] ==
         manifest["builder_protocol_sha256"], "Versioned transport source binding")
    primary_frozen = read(primary / "FROZEN_STUDY.json")
    hard = read(primary / "HARD_DECISION_EXPORT_MANIFEST.json")
    need(hard["selection_lock_sha256"] == PRIMARY_LOCK_SHA,
         "Primary test reference provenance")
    full_labels = {}
    for graph in GRAPHS["all_layer"]:
        reference = hard["graph_references"][graph]
        path = primary / reference["reference_path"]
        need(sha(path) == reference["reference_sha256"], f"Primary graph reference {graph}")
        with np.load(path, allow_pickle=False) as loaded:
            y, test_idx = loaded["full_labels"], loaded["test_indices"]
        need(tensor_sha(y) == primary_frozen["graphs"][graph]["descriptor"]["tensor_sha256"]["labels"] and
             tensor_sha(test_idx) == primary_frozen["graphs"][graph]["descriptor"]["tensor_sha256"]["test_indices"],
             f"Frozen graph labels/split {graph}")
        full_labels[graph] = (y, test_idx)
    need(sha(root / "CELL_RECORDS.tar.xz") == manifest["cell_records_tar_xz_sha256"],
         "Full record archive hash")
    archive_data = {}
    with tarfile.open(root / "CELL_RECORDS.tar.xz", "r:xz") as archive:
        for entry in archive:
            need(entry.isfile() and not entry.name.startswith("/") and
                 ".." not in Path(entry.name).parts and entry.name not in archive_data,
                 "Archive member schema")
            archive_data[entry.name] = archive.extractfile(entry).read()
    need(set(archive_data) == set(manifest["cell_record_sha256"]),
         "Complete result/trace/score record set")
    for name, digest in manifest["cell_record_sha256"].items():
        need(sha_bytes(archive_data[name]) == digest, f"Original record hash {name}")
    studies = {}
    for kind in ("all_layer", "tied36"):
        frozen_path = root / kind / FREEZES[kind]
        lock_path = root / kind / "VALIDATION_SELECTION_LOCK.json"
        audit_path = root / kind / "FINAL_SCORE_AUDIT.json"
        frozen, lock, audit = read(frozen_path), read(lock_path), read(audit_path)
        need(sha(frozen_path) == manifest[f"{kind}_freeze_sha256"] and
             sha(lock_path) == manifest[f"{kind}_lock_sha256"] and
             sha(audit_path) == manifest[f"{kind}_final_audit_sha256"] and
             lock["freeze_sha256"] == sha(frozen_path) and
             lock["selection_uses_test_labels"] is False,
             f"{kind} freeze/lock/audit hashes")
        if kind == "all_layer":
            need(lock["primary_selection_lock_sha256"] == PRIMARY_LOCK_SHA and
                 frozen["primary_selection_lock_sha256"] == PRIMARY_LOCK_SHA and
                 audit["validation_selection_lock_sha256"] == sha(lock_path),
                 "All-layer original chain")
            primary_sources = frozen["primary_source_sha256"]
        else:
            need(lock["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA and
                 frozen["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA and
                 audit["validation_selection_lock_sha256"] == sha(lock_path) and
                 audit["all_layer_validation_lock_sha256"] == manifest["all_layer_lock_sha256"],
                 "TIED36 original chain")
            primary_sources = frozen["original_source_sha256"]
        for name, digest in frozen["source_sha256"].items():
            need(sha(root / "source" / name) == digest, f"Frozen source {kind}/{name}")
        for name, digest in primary_sources.items():
            need(sha(primary / name) == digest, f"Primary frozen source {name}")
        expected_cells = {key(kind, graph, lr, wd, seed) for graph in GRAPHS[kind]
                          for lr, wd in CANDIDATES for seed in SEEDS}
        need(set(lock["cells"]) == expected_cells and
             set(lock["selections"]) == set(GRAPHS[kind]) and
             set(audit["scores"]) == allowed(lock, kind),
             f"{kind} matrix and score allowlist")
        studies[kind] = {"frozen": frozen, "lock": lock, "audit": audit}
    need(set(manifest["validation_references"]) == set(GRAPHS["all_layer"]),
         "Shared validation reference set")
    valid_ref = {}
    for graph, row in manifest["validation_references"].items():
        path = root / row["path"]
        need(sha(path) == row["sha256"], f"Validation reference hash {graph}")
        with np.load(path, allow_pickle=False) as loaded:
            need(set(loaded.files) == {"valid_indices", "valid_labels"},
                 f"Validation reference schema {graph}")
            ids, labels = loaded["valid_indices"], loaded["valid_labels"]
        y, _ = full_labels[graph]
        need(ids.dtype == labels.dtype == np.int64 and
             len(ids) == row["nodes"] == primary_frozen["graphs"][graph]["descriptor"]["split_sizes"]["valid"] and
             tensor_sha(ids) == primary_frozen["graphs"][graph]["descriptor"]["tensor_sha256"]["valid_indices"] and
             np.array_equal(labels, y[ids]), f"Frozen validation reference {graph}")
        valid_ref[graph] = labels
    expected_validation, expected_scores = set(), set()
    for kind, study in studies.items():
        lock, audit = study["lock"], study["audit"]
        for graph in GRAPHS[kind]:
            candidate_rows = []
            for lr, wd in CANDIDATES:
                seed_records = []
                for seed in SEEDS:
                    identity = key(kind, graph, lr, wd, seed)
                    expected_validation.add(f"{kind}/{identity}")
                    recorded = lock["cells"][identity]
                    result_name = f"{kind}/results/{identity}/result.json"
                    trace_name = f"{kind}/results/{identity}/validation_trace.csv"
                    need(sha_bytes(archive_data[result_name]) == recorded["result_sha256"] and
                         sha_bytes(archive_data[trace_name]) == recorded["trace_sha256"],
                         f"Locked result/trace {kind}/{identity}")
                    result = json.loads(archive_data[result_name])
                    if kind == "all_layer":
                        need(float(result["initialization"]["initial_member_logits_max_abs_diff"]) <= 1e-5 and
                             int(result["initialization"]["factor_parameter_count"]) > 0,
                             f"All-layer identity initialization {identity}")
                    elif (lr, wd) == DEFAULT:
                        anchor = read(root / "source" / "SAME_RUNTIME_TIED36_INIT_ANCHOR.json")
                        expected_init = anchor["rows"][f"{graph}/seed{seed}"]
                        need(all(result["initialization"][field] == value
                                 for field, value in expected_init.items()
                                 if field != "initial_member_logits_sha256"),
                             f"Same-runtime TIED default initialization anchor {identity}")
                    trace = list(csv.DictReader(io.StringIO(archive_data[trace_name].decode())))
                    need(len(trace) == result["epochs_completed"] == 1000 and
                         result["failure"] is None and
                         result["checkpoint_sha256"] == recorded["checkpoint_sha256"],
                         f"Full finite trace {kind}/{identity}")
                    selected_epoch, val_acc, val_ce = trace_best(trace)
                    need((result["selected_epoch"], result["selected_valid_accuracy"], result["selected_valid_ce"]) ==
                         (selected_epoch, val_acc, val_ce) and
                         (recorded["selected_epoch"], recorded["selected_valid_accuracy"], recorded["selected_valid_ce"]) ==
                         (selected_epoch, val_acc, val_ce),
                         f"Validation checkpoint selection {kind}/{identity}")
                    if kind == "tied36":
                        meta_name = f"{kind}/results/{identity}/validation_companion.json"
                        need(sha_bytes(archive_data[meta_name]) ==
                             recorded["validation_companion_manifest_sha256"],
                             f"TIED36 companion record {identity}")
                    projection = manifest["validation_cells"][f"{kind}/{identity}"]
                    source_sha = (recorded["validation_predictions_sha256"] if kind == "all_layer"
                                  else recorded["validation_companion_predictions_sha256"])
                    need(projection["source_predictions_sha256"] == source_sha and
                         projection["original_checkpoint_sha256"] == recorded["checkpoint_sha256"] and
                         sha(root / projection["projection_path"]) == projection["projection_sha256"],
                         f"Validation projection binding {kind}/{identity}")
                    with np.load(root / projection["projection_path"], allow_pickle=False) as loaded:
                        need(set(loaded.files) == {"pooled_class", "member_class"},
                             f"Validation decision schema {kind}/{identity}")
                        pool, member = loaded["pooled_class"], loaded["member_class"]
                    labels = valid_ref[graph]
                    need(pool.dtype == member.dtype == np.uint8 and
                         pool.shape == (len(labels),) and member.shape == (4, len(labels)) and
                         sha_bytes(pool.tobytes(order="C")) == projection["pooled_class_bytes_sha256"] and
                         sha_bytes(member.tobytes(order="C")) == projection["member_class_bytes_sha256"],
                         f"Validation decision array hash {kind}/{identity}")
                    accuracy = float(np.mean(pool == labels))
                    need(abs(accuracy - val_acc) < 1e-7 and
                         abs(accuracy - projection["selected_valid_accuracy"]) < 1e-12,
                         f"Validation decision accuracy {kind}/{identity}")
                    seed_records.append((val_acc, val_ce))
                candidate_rows.append({"lr": lr, "weight_decay": wd, "valid": True,
                                       "mean_valid_accuracy": sum(x[0] for x in seed_records) / 3,
                                       "mean_valid_ce": sum(x[1] for x in seed_records) / 3,
                                       "seed_valid_accuracy": [x[0] for x in seed_records],
                                       "seed_selected_epoch": [
                                           int(lock["cells"][key(kind, graph, lr, wd, seed)]["selected_epoch"])
                                           for seed in SEEDS]})
            selected = candidate_choice(candidate_rows)
            need(tuple(lock["selections"][graph]["selected_candidate"]) == selected,
                 f"Validation candidate choice {kind}/{graph}")
            need(lock["selections"][graph]["candidate_table"] == candidate_rows,
                 f"Complete validation candidate table {kind}/{graph}")
        for identity in sorted(allowed(lock, kind)):
            expected_scores.add(f"{kind}/{identity}")
            graph = identity.split("/")[0]
            score_name = f"{kind}/scores/{identity}/score.json"
            audit_row = audit["scores"][identity]
            projection = manifest["test_scores"][f"{kind}/{identity}"]
            need(sha_bytes(archive_data[score_name]) ==
                 audit_row["score_sha256"] == projection["source_score_sha256"] and
                 audit_row["predictions_sha256"] == projection["source_predictions_sha256"] and
                 sha(root / projection["projection_path"]) == projection["projection_sha256"],
                 f"Final score projection binding {kind}/{identity}")
            row = json.loads(archive_data[score_name])
            need(row["test_predictions_sha256"] == audit_row["predictions_sha256"] and
                 row["checkpoint_sha256"] == lock["cells"][identity]["checkpoint_sha256"] and
                 abs(row["valid_accuracy"] -
                     lock["cells"][identity]["selected_valid_accuracy"]) < 1e-7 and
                 abs(row["valid_ce"] -
                     lock["cells"][identity]["selected_valid_ce"]) < 1e-5 and
                 row["selected_candidate"] ==
                 (tuple((row["lr"], row["weight_decay"])) ==
                  tuple(lock["selections"][graph]["selected_candidate"])) and
                 row["predeclared_default"] ==
                 (tuple((row["lr"], row["weight_decay"])) == DEFAULT),
                 f"Original score checkpoint {kind}/{identity}")
            with np.load(root / projection["projection_path"], allow_pickle=False) as loaded:
                need(set(loaded.files) == {"pooled_logits", "member_class"},
                     f"Test projection schema {kind}/{identity}")
                pooled, member = loaded["pooled_logits"], loaded["member_class"]
            y, test_idx = full_labels[graph]
            labels = y[test_idx]
            need(pooled.dtype == np.float32 and member.dtype == np.uint8 and
                 pooled.shape == (len(labels), primary_frozen["graphs"][graph]["descriptor"]["classes"]) and
                 member.shape == (4, len(labels)) and np.isfinite(pooled).all() and
                 sha_bytes(pooled.tobytes(order="C")) == projection["pooled_logits_bytes_sha256"] and
                 sha_bytes(member.tobytes(order="C")) == projection["member_class_bytes_sha256"],
                 f"Exact projected test arrays {kind}/{identity}")
            accuracy = float(np.mean(pooled.argmax(axis=1) == labels))
            cross_entropy = ce(pooled, labels)
            need(abs(accuracy - row["test_accuracy"]) < 1e-7 and
                 abs(accuracy - audit_row["test_accuracy"]) < 1e-7 and
                 abs(accuracy - projection["test_accuracy"]) < 1e-7 and
                 abs(cross_entropy - row["test_ce"]) < 1e-5 and
                 abs(cross_entropy - audit_row["test_ce"]) < 1e-5 and
                 abs(cross_entropy - projection["test_ce"]) < 1e-5,
                 f"Test score arithmetic {kind}/{identity}")
    need(set(manifest["validation_cells"]) == expected_validation and
         set(manifest["test_scores"]) == expected_scores and
         manifest["counts"]["test_scores"] == len(expected_scores),
         "Complete projected cell/score sets")
    print(json.dumps({"status": "FACTOR_COMPACT_PUBLIC_ALL108_PASS",
                      "validation_cells": len(expected_validation),
                      "test_scores": len(expected_scores),
                      "manifest_sha256": sha(root / "MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
