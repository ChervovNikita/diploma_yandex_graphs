"""Build an anonymous compact public transport for all 108 factor/control cells.

Requires the complete original GPU study trees and independent final score
audits. It never changes source artifacts. Do not run before both audits exist.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DESIGN_FREEZE = HERE / "FACTOR_COMPACT_DESIGN_FREEZE.json"
DESIGN_AMENDMENT = HERE / "FACTOR_COMPACT_DESIGN_AMENDMENT_V2.json"
ORIGINAL_DESIGN_FREEZE_SHA = "f0cdf4ee2ac7a48a8ca80093aca53c05240ba402d01640c1cf9a2ca34727a294"
ORIGINAL_BUILDER_SHA = "1d6069c67c71c6fce95350bcd12493fea56bddeda97aa37c14abd3e6aaabc121"
ORIGINAL_VERIFIER_SHA = "a049e94dce338d18089b524e64fc97b51aef9b81ea610f98b4c334defb04b57b"
PRIMARY_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
GRAPHS = {"all_layer": ("cora", "wikics", "actor", "chameleon_filtered"),
          "tied36": ("cora", "wikics")}
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
              (0.001, 0.01), (0.003, 0.0), (0.003, 0.01))
SEEDS = (0, 1, 2)
DEFAULT = (0.001, 0.0)
FOLDERS = {"all_layer": "all_layer_factor_results",
           "tied36": "same_runtime_tied36_results"}
FREEZES = {"all_layer": "FROZEN_ALL_LAYER_STUDY.json", "tied36": "FROZEN_STUDY.json"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def key(kind: str, graph: str, lr: float, wd: float, seed: int) -> str:
    prefix = f"{graph}/tied" if kind == "tied36" else graph
    return f"{prefix}/lr{lr:g}_wd{wd:g}/seed{seed}"


def allowed(lock: dict, kind: str) -> set[str]:
    out = set()
    for graph in GRAPHS[kind]:
        chosen = tuple(lock["selections"][graph]["selected_candidate"])
        for lr, wd in dict.fromkeys((chosen, DEFAULT)):
            for seed in SEEDS:
                out.add(key(kind, graph, lr, wd, seed))
    return out


def add_tar_record(archive: tarfile.TarFile, source: Path, name: str,
                   record_hashes: dict[str, str]) -> None:
    info = archive.gettarinfo(str(source), arcname=name)
    need(info.isfile(), f"Expected regular file: {source}")
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    with source.open("rb") as stream:
        archive.addfile(info, stream)
    record_hashes[name] = sha(source)


def load_valid_prediction(kind: str, folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    original = folder / ("validation_predictions.npz" if kind == "all_layer"
                         else "validation_companion.npz")
    with np.load(original, allow_pickle=False) as arrays:
        if kind == "all_layer":
            need(set(arrays.files) == {"pooled_logits", "member_logits",
                                      "valid_indices", "valid_labels"}, "All-layer validation schema")
            pooled, member = arrays["pooled_logits"], arrays["member_logits"]
        else:
            need(set(arrays.files) == {"valid_pooled_logits", "valid_member_logits",
                                      "valid_indices", "valid_labels"}, "TIED36 validation schema")
            pooled, member = arrays["valid_pooled_logits"], arrays["valid_member_logits"]
        indices, labels = arrays["valid_indices"], arrays["valid_labels"]
    need(pooled.dtype == member.dtype == np.float32 and
         indices.dtype == labels.dtype == np.int64 and
         pooled.shape[0] == len(labels) and member.shape == (4, *pooled.shape) and
         np.isfinite(pooled).all() and np.isfinite(member).all(),
         f"Validation prediction values: {original}")
    return pooled, member, indices, labels, original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Frozen study root with both complete result trees")
    parser.add_argument("--primary", type=Path, required=True,
                        help="Sibling public primary validation_tuning stage")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    need(sha(DESIGN_FREEZE) == ORIGINAL_DESIGN_FREEZE_SHA,
         "Original pre-execution compact transport freeze")
    original_design = read(DESIGN_FREEZE)
    amended_design = read(DESIGN_AMENDMENT)
    need(original_design["builder_sha256"] == ORIGINAL_BUILDER_SHA and
         original_design["verifier_sha256"] == ORIGINAL_VERIFIER_SHA and
         original_design["protocol_sha256"] == sha(HERE / "FACTOR_COMPACT_PROTOCOL.md") and
         amended_design["status"] == "PRE_EXECUTION_COMPACT_TRANSPORT_HARDENING_V2" and
         amended_design["original_design_freeze_sha256"] == ORIGINAL_DESIGN_FREEZE_SHA and
         amended_design["original_builder_sha256"] == ORIGINAL_BUILDER_SHA and
         amended_design["original_verifier_sha256"] == ORIGINAL_VERIFIER_SHA and
         amended_design["amended_builder_sha256"] == sha(Path(__file__)) and
         amended_design["amended_verifier_sha256"] == sha(HERE / "verify_factor_compact.py") and
         amended_design["protocol_sha256"] == sha(HERE / "FACTOR_COMPACT_PROTOCOL.md"),
         "Versioned compact transport amendment")
    need(not args.out.exists(), "Refusing to replace compact evidence")
    work = args.out.with_name(args.out.name + ".inprogress")
    need(not work.exists(), "Prior interrupted build requires inspection")
    need(sha(args.primary / "VALIDATION_SELECTION_LOCK.json") == PRIMARY_LOCK_SHA,
         "Primary 432-cell lock differs")
    primary_frozen = read(args.primary / "FROZEN_STUDY.json")
    hard = read(args.primary / "HARD_DECISION_EXPORT_MANIFEST.json")
    studies = {}
    for kind in ("all_layer", "tied36"):
        root = args.source / FOLDERS[kind]
        paths = {name: root / name for name in
                 (FREEZES[kind], "VALIDATION_SELECTION_LOCK.json", "FINAL_SCORE_AUDIT.json")}
        need(all(p.is_file() for p in paths.values()), f"Incomplete {kind} lock/audit")
        frozen = read(paths[FREEZES[kind]])
        lock = read(paths["VALIDATION_SELECTION_LOCK.json"])
        audit = read(paths["FINAL_SCORE_AUDIT.json"])
        n = 72 if kind == "all_layer" else 36
        need(len(lock["cells"]) == n and set(lock["selections"]) == set(GRAPHS[kind]) and
             lock["selection_uses_test_labels"] is False and
             lock["freeze_sha256"] == sha(paths[FREEZES[kind]]),
             f"Incomplete {kind} validation lock")
        if kind == "all_layer":
            need(frozen["primary_selection_lock_sha256"] ==
                 lock["primary_selection_lock_sha256"] == PRIMARY_LOCK_SHA and
                 audit["validation_selection_lock_sha256"] == sha(paths["VALIDATION_SELECTION_LOCK.json"]),
                 "All-layer lock/audit chain")
        else:
            need(frozen["original_validation_lock_sha256"] ==
                 lock["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA and
                 audit["validation_selection_lock_sha256"] == sha(paths["VALIDATION_SELECTION_LOCK.json"]),
                 "TIED36 lock/audit chain")
        expected_all = {key(kind, g, lr, wd, s) for g in GRAPHS[kind]
                        for lr, wd in CANDIDATES for s in SEEDS}
        need(set(lock["cells"]) == expected_all, f"Incomplete {kind} matrix")
        permitted = allowed(lock, kind)
        need(set(audit["scores"]) == permitted and
             {str(p.parent.relative_to(root / "scores"))
              for p in (root / "scores").rglob("score.json")} == permitted,
             f"Missing or unauthorized {kind} test scores")
        for name, digest in frozen["source_sha256"].items():
            need(sha(args.source / name) == digest, f"Frozen source mismatch {name}")
        studies[kind] = {"root": root, "paths": paths, "frozen": frozen,
                         "lock": lock, "audit": audit, "permitted": permitted}
    need(studies["all_layer"]["audit"]["validation_selection_lock_sha256"] ==
         sha(studies["all_layer"]["paths"]["VALIDATION_SELECTION_LOCK.json"]),
         "All-layer final audit lock")
    need(studies["tied36"]["audit"]["all_layer_validation_lock_sha256"] ==
         sha(studies["all_layer"]["paths"]["VALIDATION_SELECTION_LOCK.json"]),
         "TIED36 final audit all-layer lock")
    work.mkdir(parents=True)
    (work / "build_factor_compact.py").write_bytes(Path(__file__).read_bytes())
    (work / "verify_factor_compact.py").write_bytes((HERE / "verify_factor_compact.py").read_bytes())
    (work / "FACTOR_COMPACT_DESIGN_FREEZE.json").write_bytes(DESIGN_FREEZE.read_bytes())
    (work / "FACTOR_COMPACT_DESIGN_AMENDMENT_V2.json").write_bytes(DESIGN_AMENDMENT.read_bytes())
    for kind, study in studies.items():
        for name, path in study["paths"].items():
            target = work / kind / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        preflight = study["root"] / "PREFLIGHT_CUDA.json"
        if preflight.is_file():
            (work / kind / "PREFLIGHT_CUDA.json").write_bytes(preflight.read_bytes())
        for name in study["frozen"]["source_sha256"]:
            target = work / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((args.source / name).read_bytes())
    protocol_path = HERE / "FACTOR_COMPACT_PROTOCOL.md"
    (work / "FACTOR_COMPACT_PROTOCOL.md").write_bytes(protocol_path.read_bytes())
    references, validation, scores, record_hashes = {}, {}, {}, {}
    with tarfile.open(work / "CELL_RECORDS.tar.xz", "w:xz", preset=6) as archive:
        for kind, study in studies.items():
            root, lock, audit = study["root"], study["lock"], study["audit"]
            for graph in GRAPHS[kind]:
                for lr, wd in CANDIDATES:
                    for seed in SEEDS:
                        identity = key(kind, graph, lr, wd, seed)
                        folder = root / "results" / identity
                        result_path = folder / "result.json"
                        trace_path = folder / "validation_trace.csv"
                        locked = lock["cells"][identity]
                        need(result_path.is_file() and trace_path.is_file() and
                             sha(result_path) == locked["result_sha256"] and
                             sha(trace_path) == locked["trace_sha256"],
                             f"Result/trace lock {kind}/{identity}")
                        result = read(result_path)
                        need(result["failure"] is None and result["epochs_completed"] == 1000 and
                             result["selected_epoch"] == locked["selected_epoch"] and
                             result["checkpoint_sha256"] == locked["checkpoint_sha256"],
                             f"Incomplete trained cell {kind}/{identity}")
                        add_tar_record(archive, result_path,
                                       f"{kind}/results/{identity}/result.json", record_hashes)
                        add_tar_record(archive, trace_path,
                                       f"{kind}/results/{identity}/validation_trace.csv", record_hashes)
                        if kind == "tied36":
                            companion_meta = folder / "validation_companion.json"
                            need(sha(companion_meta) == locked["validation_companion_manifest_sha256"],
                                 f"TIED36 validation companion manifest {identity}")
                            add_tar_record(archive, companion_meta,
                                           f"{kind}/results/{identity}/validation_companion.json",
                                           record_hashes)
                        pooled, member, indices, labels, original = load_valid_prediction(kind, folder)
                        source_sha = (locked["validation_predictions_sha256"] if kind == "all_layer"
                                      else locked["validation_companion_predictions_sha256"])
                        need(sha(original) == source_sha and
                             abs(float(np.mean(pooled.argmax(axis=1) == labels)) -
                                 result["selected_valid_accuracy"]) < 1e-7,
                             f"Validation prediction source/accuracy {kind}/{identity}")
                        if kind == "tied36":
                            need(read(companion_meta)["predictions_sha256"] == source_sha,
                                 f"TIED36 companion source hash {identity}")
                        if graph not in references:
                            ref_path = work / "validation_references" / f"{graph}.npz"
                            ref_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(ref_path, valid_indices=indices, valid_labels=labels)
                            references[graph] = {"path": ref_path.relative_to(work).as_posix(),
                                                 "sha256": sha(ref_path),
                                                 "nodes": len(labels)}
                        else:
                            with np.load(work / references[graph]["path"], allow_pickle=False) as ref:
                                need(np.array_equal(ref["valid_indices"], indices) and
                                     np.array_equal(ref["valid_labels"], labels),
                                     f"Validation reference differs {kind}/{identity}")
                        pooled_class = pooled.argmax(axis=1).astype(np.uint8)
                        member_class = member.argmax(axis=2).astype(np.uint8)
                        need(int(pooled_class.max()) < 256 and int(member_class.max()) < 256,
                             "Class index exceeds uint8")
                        derivative = work / "validation_decisions" / kind / f"{identity}.npz"
                        derivative.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(derivative, pooled_class=pooled_class,
                                            member_class=member_class)
                        validation[f"{kind}/{identity}"] = {
                            "source_predictions_sha256": source_sha,
                            "original_checkpoint_sha256": locked["checkpoint_sha256"],
                            "projection_path": derivative.relative_to(work).as_posix(),
                            "projection_sha256": sha(derivative),
                            "pooled_class_bytes_sha256": sha_array(pooled_class),
                            "member_class_bytes_sha256": sha_array(member_class),
                            "selected_valid_accuracy": float(np.mean(pooled_class == labels)),
                        }
                for identity in sorted(study["permitted"]):
                    if not identity.startswith(graph + "/"):
                        continue
                    score_dir = root / "scores" / identity
                    score_path = score_dir / "score.json"
                    prediction_path = score_dir / "predictions.npz"
                    audit_row = audit["scores"][identity]
                    need(sha(score_path) == audit_row["score_sha256"] and
                         sha(prediction_path) == audit_row["predictions_sha256"],
                         f"Final score audit {kind}/{identity}")
                    row = read(score_path)
                    need(row["test_predictions_sha256"] == sha(prediction_path) and
                         row["checkpoint_sha256"] == lock["cells"][identity]["checkpoint_sha256"] and
                         row["selected_candidate"] ==
                         (tuple((row["lr"], row["weight_decay"])) ==
                          tuple(lock["selections"][graph]["selected_candidate"])) and
                         row["predeclared_default"] ==
                         (tuple((row["lr"], row["weight_decay"])) == DEFAULT),
                         f"Score checkpoint/source {kind}/{identity}")
                    add_tar_record(archive, score_path,
                                   f"{kind}/scores/{identity}/score.json", record_hashes)
                    with np.load(prediction_path, allow_pickle=False) as original:
                        need(set(original.files) == {"valid_pooled_logits", "test_pooled_logits",
                                                     "test_member_logits"},
                             f"Score predictions schema {kind}/{identity}")
                        pooled = original["test_pooled_logits"]
                        member = original["test_member_logits"]
                    need(pooled.dtype == member.dtype == np.float32 and
                         member.shape == (4, *pooled.shape) and np.isfinite(pooled).all() and
                         np.isfinite(member).all(), f"Test prediction values {kind}/{identity}")
                    classes = member.argmax(axis=2).astype(np.uint8)
                    derivative = work / "test_predictions" / kind / f"{identity}.npz"
                    derivative.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(derivative, pooled_logits=pooled, member_class=classes)
                    scores[f"{kind}/{identity}"] = {
                        "source_predictions_sha256": sha(prediction_path),
                        "source_score_sha256": sha(score_path),
                        "projection_path": derivative.relative_to(work).as_posix(),
                        "projection_sha256": sha(derivative),
                        "pooled_logits_bytes_sha256": sha_array(pooled),
                        "member_class_bytes_sha256": sha_array(classes),
                        "test_accuracy": row["test_accuracy"],
                        "test_ce": row["test_ce"],
                    }
    need(len(validation) == 108 and len(references) == 4, "Validation projection coverage")
    need(len(scores) == sum(len(s["permitted"]) for s in studies.values()),
         "Test projection coverage")
    manifest = {
        "protocol": "factor_placement_compact_public_all108_v1",
        "primary_selection_lock_sha256": PRIMARY_LOCK_SHA,
        "builder_source_sha256": sha(Path(__file__)),
        "public_verifier_sha256": sha(HERE / "verify_factor_compact.py"),
        "original_design_freeze_sha256": sha(DESIGN_FREEZE),
        "design_amendment_sha256": sha(DESIGN_AMENDMENT),
        "builder_protocol_sha256": sha(protocol_path),
        "all_layer_freeze_sha256": sha(studies["all_layer"]["paths"][FREEZES["all_layer"]]),
        "all_layer_lock_sha256": sha(studies["all_layer"]["paths"]["VALIDATION_SELECTION_LOCK.json"]),
        "all_layer_final_audit_sha256": sha(studies["all_layer"]["paths"]["FINAL_SCORE_AUDIT.json"]),
        "tied36_freeze_sha256": sha(studies["tied36"]["paths"][FREEZES["tied36"]]),
        "tied36_lock_sha256": sha(studies["tied36"]["paths"]["VALIDATION_SELECTION_LOCK.json"]),
        "tied36_final_audit_sha256": sha(studies["tied36"]["paths"]["FINAL_SCORE_AUDIT.json"]),
        "cell_records_tar_xz_sha256": sha(work / "CELL_RECORDS.tar.xz"),
        "cell_record_sha256": record_hashes,
        "validation_references": references,
        "validation_cells": validation,
        "test_scores": scores,
        "counts": {"validation_cells": len(validation), "test_scores": len(scores),
                   "result_json": 108, "full_validation_traces": 108},
    }
    (work / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    work.rename(args.out)
    print(json.dumps({"status": "FACTOR_COMPACT_ALL108_BUILT",
                      "manifest_sha256": sha(args.out / "MANIFEST.json"),
                      "validation_cells": len(validation), "test_scores": len(scores),
                      "bytes": sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())},
                     sort_keys=True))


if __name__ == "__main__":
    main()
