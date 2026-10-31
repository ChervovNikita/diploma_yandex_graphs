"""Descriptive Roman Empire SAGE decision strata for GNNM and pooled ENS.

The four strata and all reported statistics are fixed in
decision_strata_protocol.md. Run only after the complete 10-variant x 5-mask
control matrix and split-0 verification audit exist. No test label enters
training, checkpoint selection, or inference through this script.

This analysis is post hoc on one graph. It reports five official masks as
paired descriptions, without treating masks or nodes as independent samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments_iclr"
RESULTS = EXP / "results"
RERUN = EXP / "gnnm_split0_verification"
DATA = REPO / "data" / "roman_empire.npz"
INPUT = RESULTS / "projector_controls.csv"
AUDIT = RESULTS / "gnnm_split0_verification_audit.json"
OUT_SPLIT = RESULTS / "decision_strata_per_split.csv"
OUT_SUMMARY = RESULTS / "decision_strata_five_mask_summary.csv"
OUT_AUDIT = RESULTS / "decision_strata_audit.json"

VARIANTS = (
    "gnnm", "independent_projectors", "heads_only", "input_only",
    "output_only", "base", "gnnm_m1", "untied_backbone",
    "freeze_output_factors", "ens_pooled",
)
SPLITS = tuple(range(5))
KEY_FIELDS = ("dataset", "model", "variant", "split")
PRED_FIELDS = (
    "node_index", "y_true", "ensemble_pred", "member_pred",
    "member_logits", "degree", "local_homophily",
)
METRICS = (
    "n_nodes", "gnnm_pooled_acc_percent", "ens_pooled_acc_percent",
    "gnnm_minus_ens_acc_pp", "gnnm_mean_member_acc_percent",
    "ens_mean_member_acc_percent", "gnnm_pooling_gain_pp",
    "ens_pooling_gain_pp", "gnnm_member_disagreement_percent",
    "ens_member_disagreement_percent", "pooled_decision_disagreement_percent",
    "both_correct_nodes", "gnnm_only_correct_nodes",
    "ens_only_correct_nodes", "neither_correct_nodes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def row_sha256(row: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def finite_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {field} in {row['variant']} split {row['split']}")
    return value


def load_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as data:
        labels = np.asarray(data["node_labels"], dtype=np.int64)
        edges = np.asarray(data["edges"], dtype=np.int64)
        masks = np.asarray(data["test_masks"], dtype=bool)
    n = len(labels)
    if labels.ndim != 1 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Unexpected Roman Empire labels or edge shape")
    if masks.shape == (n, 10):
        masks = masks.T
    if masks.shape != (10, n):
        raise ValueError(f"Expected ten official test masks, got {masks.shape}")
    if np.any(edges < 0) or np.any(edges >= n) or np.any(labels < 0):
        raise ValueError("Out-of-range graph edge or class label")

    # datasets.load_dataset calls DGL to_bidirected before adding self loops.
    # Unique ordered pairs reproduce that graph's non-self-loop neighbors.
    pairs = np.concatenate((edges, edges[:, ::-1]), axis=0)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    degree = np.bincount(pairs[:, 1], minlength=n).astype(np.int64)
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    same_count = np.bincount(pairs[:, 1], weights=same.astype(np.int64),
                             minlength=n).astype(np.int64)
    if np.any(same_count > degree):
        raise ValueError("Same-label neighbor count exceeds degree")
    return labels, masks, degree, same_count


def selected_path(row: dict[str, str], field: str, folder: str) -> Path:
    value = Path(row[field])
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe selected {field} path: {value}")
    split = int(row["split"])
    variant = row["variant"]
    rerun_path = value.parts[:2] == ("experiments_iclr", "gnnm_split0_verification")
    root = RERUN if variant == "gnnm" and split == 0 and rerun_path else RESULTS
    extension = "npz" if folder == "predictions" else "pt"
    expected = root / folder / f"roman-empire_SAGE_{variant}_split{split}.{extension}"
    path = REPO / value
    if path != expected or not path.is_file() or path.resolve() != expected.resolve():
        raise ValueError(f"Missing or unexpected selected {field}: {value}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Selected {field} leaves the result directory: {value}")
    return path


def load_matrix(masks: np.ndarray) -> dict[tuple[str, int], dict[str, str]]:
    with INPUT.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    expected = {("roman-empire", "SAGE", v, str(s))
                for v in VARIANTS for s in SPLITS}
    keys = [tuple(row[k] for k in KEY_FIELDS) for row in rows]
    if len(rows) != 50 or len(set(keys)) != 50 or set(keys) != expected:
        raise ValueError("Decision strata require exactly 50 rows: ten fixed variants on five masks")
    by_key = {(row["variant"], int(row["split"])): row for row in rows}
    parameters: dict[str, int] = {}
    for row in rows:
        variant = row["variant"]
        split = int(row["split"])
        m = 1 if variant in ("base", "gnnm_m1") else 4
        if (int(row["seed"]) != split or int(row["num_layers"]) != 5
                or int(row["hidden_dim"]) != 512
                or not math.isclose(finite_float(row, "lr"), 3e-5,
                                    rel_tol=0, abs_tol=1e-12)
                or int(row["m"]) != m or int(row["num_steps"]) != 5000
                or not 1 <= int(row["best_step"]) <= 5000
                or int(row["best_step"]) != 1 and int(row["best_step"]) % 10 != 0
                or int(row["n_test"]) != int(masks[split].sum())):
            raise ValueError(f"Mixed or incomplete protocol in {variant} split {split}")
        for field in ("val_metric", "test_metric", "test_acc", "test_loss",
                      "mean_member_acc", "pair_disagreement"):
            finite_float(row, field)
        num_params = int(row["num_params"])
        if num_params < 1 or variant in parameters and num_params != parameters[variant]:
            raise ValueError(f"Inconsistent parameter count for {variant}")
        parameters[variant] = num_params
        checkpoint = selected_path(row, "checkpoint", "checkpoints")
        prediction = selected_path(row, "prediction_file", "predictions")
        if checkpoint.parent.parent != prediction.parent.parent:
            raise ValueError(f"Checkpoint and prediction roots differ for {variant} split {split}")
    if parameters["gnnm"] != 6_737_728 or parameters["base"] != 6_730_770:
        raise ValueError("GNNM or ordinary SAGE parameter count changed")
    if parameters["ens_pooled"] != 4 * parameters["base"]:
        raise ValueError("Pooled ENS is not four independent width-512 SAGE models")

    audit = json.loads(AUDIT.read_text())
    if sha256(EXP / "projector_controls.py") != audit["runner_sha256"]:
        raise ValueError("Control runner differs from split-0 verification source")
    if sha256(EXP / "verify_gnnm_split0.py") != audit["verifier_sha256"]:
        raise ValueError("Split-0 verifier differs from its audited source")
    selected = by_key[("gnnm", 0)]
    adopted = audit["adopted"]
    if adopted not in ("pilot", "fresh"):
        raise ValueError("Unknown split-0 verification decision")
    if row_sha256(selected) != audit[f"{adopted}_row_sha256"]:
        raise ValueError("Selected GNNM split-0 row disagrees with verification audit")
    expected_root = RERUN if adopted == "fresh" else RESULTS
    if selected_path(selected, "prediction_file", "predictions").parent.parent != expected_root:
        raise ValueError("Selected GNNM split-0 artifact disagrees with adoption decision")
    if sha256(selected_path(selected, "prediction_file", "predictions")) != audit[f"{adopted}_prediction_sha256"]:
        raise ValueError("Selected GNNM split-0 prediction changed after verification")
    return by_key


def load_prediction(row: dict[str, str], labels: np.ndarray,
                    masks: np.ndarray, degree: np.ndarray,
                    same_count: np.ndarray) -> dict[str, np.ndarray]:
    split = int(row["split"])
    variant = row["variant"]
    path = selected_path(row, "prediction_file", "predictions")
    with np.load(path, allow_pickle=False) as data:
        if not set(PRED_FIELDS).issubset(data.files):
            raise ValueError(f"Incomplete selected prediction: {path}")
        node_index = np.asarray(data["node_index"])
        y_true = np.asarray(data["y_true"])
        degree_saved = np.asarray(data["degree"])
        homophily_saved = np.asarray(data["local_homophily"])
        pooled = np.asarray(data["ensemble_pred"])
        member = np.asarray(data["member_pred"])
        logits = np.asarray(data["member_logits"])
        expected_index = np.flatnonzero(masks[split])
        n = len(expected_index)
        m = int(row["m"])
        c = int(labels.max()) + 1
        if (node_index.shape != (n,) or y_true.shape != (n,)
                or degree_saved.shape != (n,) or homophily_saved.shape != (n,)
                or pooled.shape != (n,) or member.shape != (m, n)
                or logits.shape != (m, n, c)):
            raise ValueError(f"Prediction shape mismatch for {variant} split {split}")
        if (not np.array_equal(node_index, expected_index)
                or not np.array_equal(y_true, labels[expected_index])):
            raise ValueError(f"Node IDs or true labels mismatch for {variant} split {split}")
        if not np.all(np.isfinite(degree_saved)) or not np.array_equal(degree_saved, degree[expected_index]):
            raise ValueError(f"Graph degree mismatch for {variant} split {split}")
        true_homophily = same_count[expected_index] / np.maximum(degree[expected_index], 1)
        if (not np.all(np.isfinite(homophily_saved))
                or not np.allclose(homophily_saved, true_homophily, rtol=0, atol=1e-6)):
            raise ValueError(f"Local label homophily mismatch for {variant} split {split}")
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"Nonfinite logits for {variant} split {split}")
        if (not np.array_equal(member, logits.argmax(axis=-1))
                or not np.array_equal(pooled, logits.mean(axis=0).argmax(axis=-1))):
            raise ValueError(f"Saved decisions disagree with logits for {variant} split {split}")
        if np.any(pooled < 0) or np.any(pooled >= c):
            raise ValueError(f"Out-of-range decision in {variant} split {split}")
    pooled_acc = np.mean(pooled == y_true)
    member_acc = np.mean(member == y_true[None, :])
    pairs = list(combinations(range(m), 2))
    disagreement = float(np.mean([np.mean(member[i] != member[j]) for i, j in pairs])) if pairs else 0.0
    if (abs(pooled_acc - finite_float(row, "test_acc")) > 1e-6
            or abs(member_acc - finite_float(row, "mean_member_acc")) > 1e-6
            or abs(disagreement - finite_float(row, "pair_disagreement")) > 1e-6):
        raise ValueError(f"Selected prediction metrics disagree with CSV for {variant} split {split}")
    return {"node_index": node_index.copy(), "y_true": y_true.copy(),
            "pooled": pooled.copy(), "member": member.copy()}


def fixed_strata(degree: np.ndarray, same_count: np.ndarray) -> tuple[tuple[str, str, np.ndarray], ...]:
    if np.any(degree < 2):
        raise ValueError("Degree below two has no predeclared stratum")
    if np.any(same_count < 0) or np.any(same_count > degree):
        raise ValueError("Invalid local same-label neighbor count")
    return (
        ("degree", "degree_2", degree == 2),
        ("degree", "degree_ge3", degree >= 3),
        ("label_homophily", "same_label_neighbors_0", same_count == 0),
        ("label_homophily", "same_label_neighbors_positive", same_count > 0),
    )


def member_disagreement(member: np.ndarray, selection: np.ndarray) -> float:
    pairs = list(combinations(range(member.shape[0]), 2))
    return float(np.mean([np.mean(member[i, selection] != member[j, selection])
                          for i, j in pairs]))


def describe_stratum(split: int, dimension: str, stratum: str,
                     selection: np.ndarray, gnnm: dict[str, np.ndarray],
                     ens: dict[str, np.ndarray]) -> dict[str, float | int | str]:
    if not np.array_equal(gnnm["node_index"], ens["node_index"]) or not np.array_equal(gnnm["y_true"], ens["y_true"]):
        raise ValueError(f"GNNM and ENS nodes or labels differ on split {split}")
    n = int(selection.sum())
    if n == 0:
        raise ValueError(f"Empty predeclared stratum {stratum} on split {split}")
    y = gnnm["y_true"][selection]
    gp = gnnm["pooled"][selection]
    ep = ens["pooled"][selection]
    gc = gp == y
    ec = ep == y
    g_acc = float(gc.mean())
    e_acc = float(ec.mean())
    g_member_acc = float(np.mean(gnnm["member"][:, selection] == y[None, :]))
    e_member_acc = float(np.mean(ens["member"][:, selection] == y[None, :]))
    g_only = int((gc & ~ec).sum())
    e_only = int((~gc & ec).sum())
    paired_pp = 100 * (g_acc - e_acc)
    if abs(paired_pp - 100 * (g_only - e_only) / n) > 1e-10:
        raise AssertionError("Paired correctness identity failed")
    return {
        "dataset": "roman-empire", "model": "SAGE", "split": split,
        "dimension": dimension, "stratum": stratum, "n_nodes": n,
        "gnnm_pooled_acc_percent": 100 * g_acc,
        "ens_pooled_acc_percent": 100 * e_acc,
        "gnnm_minus_ens_acc_pp": paired_pp,
        "gnnm_mean_member_acc_percent": 100 * g_member_acc,
        "ens_mean_member_acc_percent": 100 * e_member_acc,
        "gnnm_pooling_gain_pp": 100 * (g_acc - g_member_acc),
        "ens_pooling_gain_pp": 100 * (e_acc - e_member_acc),
        "gnnm_member_disagreement_percent": 100 * member_disagreement(gnnm["member"], selection),
        "ens_member_disagreement_percent": 100 * member_disagreement(ens["member"], selection),
        "pooled_decision_disagreement_percent": 100 * float(np.mean(gp != ep)),
        "both_correct_nodes": int((gc & ec).sum()),
        "gnnm_only_correct_nodes": g_only,
        "ens_only_correct_nodes": e_only,
        "neither_correct_nodes": int((~gc & ~ec).sum()),
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    result = []
    groups = sorted({(row["dimension"], row["stratum"]) for row in rows})
    for dimension, stratum in groups:
        group = sorted((row for row in rows if (row["dimension"], row["stratum"]) == (dimension, stratum)),
                       key=lambda row: int(row["split"]))
        if [int(row["split"]) for row in group] != list(SPLITS):
            raise ValueError(f"Missing split in {stratum} summary")
        summary: dict[str, float | int | str] = {
            "dataset": "roman-empire", "model": "SAGE",
            "dimension": dimension, "stratum": stratum,
            "n_masks": 5, "splits": "0 1 2 3 4",
        }
        for field in METRICS:
            values = np.asarray([float(row[field]) for row in group])
            summary[f"mean_{field}"] = float(values.mean())
            summary[f"min_{field}"] = float(values.min())
            summary[f"max_{field}"] = float(values.max())
        result.append(summary)
    return result


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run() -> None:
    labels, masks, degree, same_count = load_graph()
    matrix = load_matrix(masks)
    selected: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for split in SPLITS:
        expected_index = np.flatnonzero(masks[split])
        fixed_strata(degree[expected_index], same_count[expected_index])
        for variant in VARIANTS:
            prediction = load_prediction(matrix[(variant, split)], labels, masks,
                                         degree, same_count)
            if variant in ("gnnm", "ens_pooled"):
                selected[(variant, split)] = prediction
    rows = []
    for split in SPLITS:
        gp = selected[("gnnm", split)]
        ep = selected[("ens_pooled", split)]
        index = gp["node_index"]
        for dimension, stratum, selection in fixed_strata(degree[index], same_count[index]):
            rows.append(describe_stratum(split, dimension, stratum, selection, gp, ep))
    summaries = summarize(rows)
    provenance = {
        "analysis": "Descriptive post hoc decision strata on five overlapping official masks of one graph",
        "source_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(EXP / "decision_strata_protocol.md"),
        "data_sha256": sha256(DATA),
        "matrix_sha256": sha256(INPUT),
        "split0_verification_audit_sha256": sha256(AUDIT),
        "selected_predictions": [
            {"variant": variant, "split": split,
             "path": matrix[(variant, split)]["prediction_file"],
             "sha256": sha256(selected_path(matrix[(variant, split)], "prediction_file", "predictions"))}
            for split in SPLITS for variant in ("gnnm", "ens_pooled")
        ],
        "output_rows_per_split": len(rows),
        "output_rows_five_mask_summary": len(summaries),
    }
    write_csv_atomic(OUT_SPLIT, rows)
    write_csv_atomic(OUT_SUMMARY, summaries)
    temp = OUT_AUDIT.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, OUT_AUDIT)
    print(f"Wrote {len(rows)} paired stratum rows and {len(summaries)} five-mask summaries")


def self_test() -> None:
    d = np.array([2, 2, 3, 5])
    same = np.array([0, 1, 0, 2])
    groups = fixed_strata(d, same)
    assert [mask.tolist() for _, _, mask in groups] == [
        [True, True, False, False], [False, False, True, True],
        [True, False, True, False], [False, True, False, True],
    ]
    y = np.array([0, 1, 0, 1])
    g = {"node_index": np.arange(4), "y_true": y,
         "pooled": np.array([0, 0, 0, 1]),
         "member": np.array([[0, 1, 0, 1], [0, 0, 0, 1],
                             [0, 0, 1, 1], [0, 1, 0, 1]])}
    e = {"node_index": np.arange(4), "y_true": y,
         "pooled": np.array([0, 1, 1, 1]),
         "member": np.array([[0, 1, 1, 1], [0, 1, 0, 1],
                             [0, 1, 1, 1], [0, 1, 1, 1]])}
    low = describe_stratum(0, "degree", "degree_2", groups[0][2], g, e)
    high = describe_stratum(0, "degree", "degree_ge3", groups[1][2], g, e)
    assert low["n_nodes"] == high["n_nodes"] == 2
    assert low["gnnm_minus_ens_acc_pp"] == -50
    assert high["gnnm_minus_ens_acc_pp"] == 50
    assert low["ens_only_correct_nodes"] == 1
    assert high["gnnm_only_correct_nodes"] == 1
    print("Synthetic decision-strata smoke passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="Use tiny synthetic arrays; read no real results")
    args = parser.parse_args()
    self_test() if args.self_test else run()
