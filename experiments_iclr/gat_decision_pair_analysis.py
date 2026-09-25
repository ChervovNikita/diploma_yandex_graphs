"""Post hoc paired decision analysis for the Roman Empire GAT tying diagnostic.

Requires verify_gat_failure_pair.py --complete before reading any result.
All quantities come from saved held-out logits. The two variants must use
identical official test nodes and labels on each of the five masks.

The comparison is descriptive. The failure case and depths were chosen after
examining archived scores, so this script does not provide confirmatory
evidence or estimate performance on new graphs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments_iclr" / "gat_failure_pair_results"
INPUT = ROOT / "projector_controls.csv"
OUTPUT = ROOT / "gat_decision_pair_analysis.csv"
AUDIT = ROOT / "gat_decision_pair_analysis_audit.json"
DATA = REPO / "data" / "roman_empire.npz"
VERIFY = REPO / "experiments_iclr" / "verify_gat_failure_pair.py"
VARIANTS = ("gnnm", "untied_backbone")
SPLITS = tuple(range(5))
METRICS = (
    "pooled_accuracy_percent",
    "mean_member_accuracy_percent",
    "pooling_gain_pp",
    "pair_disagreement_percent",
    "rescue_member_node_pairs",
    "harm_member_node_pairs",
    "member_ce_nats",
    "pooled_ce_nats",
    "ce_gap_nats",
    "reverse_kl_nats",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def logsumexp(x: np.ndarray) -> np.ndarray:
    largest = np.max(x, axis=-1, keepdims=True)
    return np.squeeze(largest, -1) + np.log(
        np.sum(np.exp(x - largest), axis=-1)
    )


def calculate(logits: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Calculate metrics from an M by N by C logit array in float64."""
    native = np.asarray(logits)
    z = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(y)
    if z.ndim != 3 or z.shape[0] < 2 or z.shape[1] < 1 or z.shape[2] < 2:
        raise ValueError("Expected at least two members, nodes, and classes")
    m, n, c = z.shape
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be one integer per node")
    if np.any(labels < 0) or np.any(labels >= c) or not np.isfinite(z).all():
        raise ValueError("Invalid labels or nonfinite logits")
    member = native.argmax(axis=-1)
    pooled_logits = z.mean(axis=0)
    pooled = native.mean(axis=0).argmax(axis=-1)
    member_correct = member == labels[None, :]
    pooled_correct = pooled == labels
    k_correct = member_correct.sum(axis=0)
    pairs = list(combinations(range(m), 2))
    disagreement = np.mean(
        [np.mean(member[i] != member[j]) for i, j in pairs]
    )
    rescue = int(np.where(pooled_correct, m - k_correct, 0).sum())
    harm = int(np.where(~pooled_correct, k_correct, 0).sum())
    pooled_acc = float(pooled_correct.mean())
    member_acc = float(member_correct.mean())
    gain_pp = 100.0 * (pooled_acc - member_acc)
    if not math.isclose(gain_pp, 100.0 * (rescue - harm) / (m * n),
                        rel_tol=0, abs_tol=1e-10):
        raise AssertionError("Rescue/harm and pooling-gain identities disagree")

    member_lse = logsumexp(z)
    pooled_lse = logsumexp(pooled_logits)
    member_ce = member_lse - np.take_along_axis(
        z, np.broadcast_to(labels[None, :, None], (m, n, 1)), axis=-1
    ).squeeze(-1)
    pooled_ce = pooled_lse - pooled_logits[np.arange(n), labels]
    member_logp = z - member_lse[:, :, None]
    pooled_logp = pooled_logits - pooled_lse[:, None]
    pooled_prob = np.exp(pooled_logp)
    reverse_kl = np.mean(np.sum(
        pooled_prob[None, :, :] *
        (pooled_logp[None, :, :] - member_logp), axis=-1
    ), axis=0)
    ce_gap = member_ce.mean(axis=0) - pooled_ce
    identity_error = float(np.max(np.abs(ce_gap - reverse_kl)))
    if identity_error > 1e-8:
        raise AssertionError(
            f"Stable CE/KL identity has error {identity_error} nats"
        )
    return {
        "n_test": n,
        "members": m,
        "pooled_accuracy_percent": 100.0 * pooled_acc,
        "mean_member_accuracy_percent": 100.0 * member_acc,
        "pooling_gain_pp": gain_pp,
        "pair_disagreement_percent": 100.0 * float(disagreement),
        "rescue_member_node_pairs": rescue,
        "harm_member_node_pairs": harm,
        "net_correct_member_node_pairs": rescue - harm,
        "member_ce_nats": float(member_ce.mean()),
        "pooled_ce_nats": float(pooled_ce.mean()),
        "ce_gap_nats": float(ce_gap.mean()),
        "reverse_kl_nats": float(reverse_kl.mean()),
        "ce_kl_identity_max_abs_nats": identity_error,
        "all_members_wrong_pooled_correct_nodes":
            int(((k_correct == 0) & pooled_correct).sum()),
        "some_member_correct_pooled_wrong_nodes":
            int(((k_correct > 0) & ~pooled_correct).sum()),
    }


def self_test() -> None:
    # Node 1: two wrong members rescued by pooling. Node 2: one correct
    # member harmed by pooling. Node 3 checks large-logit stability.
    logits = np.array([
        [[4, 0, 0], [5, 0, 0], [2, 0, 0], [1000, 999, 0]],
        [[4, 0, 0], [0, 2, 0], [0, 3, 0], [1002, 999, 0]],
        [[4, 0, 0], [0, 2, 0], [0, 3, 0], [1001, 999, 0]],
    ], dtype=np.float64)
    y = np.zeros(4, dtype=np.int64)
    got = calculate(logits, y)
    assert got["n_test"] == 4 and got["members"] == 3
    assert got["rescue_member_node_pairs"] == 2
    assert got["harm_member_node_pairs"] == 1
    assert math.isclose(got["pooled_accuracy_percent"], 75.0)
    assert math.isclose(got["mean_member_accuracy_percent"], 200.0 / 3.0)
    assert math.isclose(got["pooling_gain_pp"], 25.0 / 3.0)
    assert math.isclose(got["pair_disagreement_percent"], 100.0 / 3.0)
    assert got["ce_kl_identity_max_abs_nats"] < 1e-10
    print("CPU synthetic analysis passed")


def verify_complete() -> dict:
    result = subprocess.run(
        [sys.executable, str(VERIFY), "--complete"],
        cwd=REPO, capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    if report.get("verified_rows") != 10 or report.get("complete") is not True:
        raise RuntimeError("GAT verifier did not confirm ten complete rows")
    return report


def official_labels_and_masks() -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as graph:
        labels = np.asarray(graph["node_labels"], dtype=np.int64)
        masks = np.asarray(graph["test_masks"], dtype=bool)
    if masks.shape == (len(labels), 10):
        masks = masks.T
    if masks.shape != (10, len(labels)):
        raise ValueError("Unexpected official Roman Empire test masks")
    return labels, masks


def read_rows() -> dict[tuple[int, str], dict[str, str]]:
    with INPUT.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise ValueError("GAT pair needs exactly ten rows")
    by_key = {(int(row["split"]), row["variant"]): row for row in rows}
    expected = {(split, variant) for split in SPLITS for variant in VARIANTS}
    if len(by_key) != 10 or set(by_key) != expected:
        raise ValueError("GAT rows lack a unique complete mask-variant matrix")
    return by_key


def read_prediction(
    row: dict[str, str], labels: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    split = int(row["split"])
    variant = row["variant"]
    expected = ROOT / "predictions" / (
        f"roman-empire_GAT_{variant}_split{split}.npz"
    )
    actual = REPO / row["prediction_file"]
    if actual != expected or not actual.is_file():
        raise ValueError("Unexpected saved prediction path")
    with np.load(actual, allow_pickle=False) as pred:
        node_index = np.asarray(pred["node_index"])
        y = np.asarray(pred["y_true"])
        logits = np.asarray(pred["member_logits"])
        member = np.asarray(pred["member_pred"])
        pooled = np.asarray(pred["ensemble_pred"])
    official_index = np.flatnonzero(masks[split])
    if (not np.array_equal(node_index, official_index)
            or not np.array_equal(y, labels[official_index])):
        raise ValueError(f"Official node or label mismatch in mask {split}")
    if logits.shape[:2] != (4, len(official_index)):
        raise ValueError(f"Unexpected logit shape in mask {split}")
    if not np.array_equal(member, logits.argmax(axis=-1)):
        raise ValueError(f"Saved member decisions differ in mask {split}")
    if not np.array_equal(pooled, logits.mean(axis=0).argmax(axis=-1)):
        raise ValueError(f"Saved pooled decisions differ in mask {split}")
    return node_index, y, logits, actual


def check_recorded(
    row: dict[str, str], metrics: dict[str, float | int]
) -> None:
    for column, key, scale in (
        ("test_acc", "pooled_accuracy_percent", 0.01),
        ("mean_member_acc", "mean_member_accuracy_percent", 0.01),
        ("pair_disagreement", "pair_disagreement_percent", 0.01),
        ("test_loss", "pooled_ce_nats", 1.0),
    ):
        recorded = float(row[column])
        measured = float(metrics[key]) * scale
        if not math.isfinite(recorded) or abs(recorded - measured) > 1e-5:
            raise ValueError(
                f"Saved logits disagree with {column} for "
                f"{row['variant']} mask {row['split']}"
            )


def atomic_text(path: Path, content: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def render_expected() -> tuple[str, str, int]:
    verified = verify_complete()
    labels, masks = official_labels_and_masks()
    rows = read_rows()
    output_rows = []
    input_hashes = {
        "experiments_iclr/gat_failure_pair_results/protocol.json": sha256(ROOT / "protocol.json"),
        "experiments_iclr/gat_failure_pair_results/projector_controls.csv": sha256(INPUT),
        "data/roman_empire.npz": sha256(DATA),
        "experiments_iclr/verify_gat_failure_pair.py": sha256(VERIFY),
        "experiments_iclr/gat_decision_pair_analysis.py": sha256(Path(__file__).resolve()),
    }
    for split in SPLITS:
        compared = {}
        pooled_decisions = {}
        common_index = common_labels = None
        for variant in VARIANTS:
            row = rows[(split, variant)]
            index, y, logits, path = read_prediction(row, labels, masks)
            if common_index is None:
                common_index, common_labels = index, y
            elif not np.array_equal(index, common_index) or not np.array_equal(
                    y, common_labels):
                raise ValueError(f"Paired test nodes differ in mask {split}")
            metrics = calculate(logits, y)
            check_recorded(row, metrics)
            compared[variant] = metrics
            pooled_decisions[variant] = logits.mean(axis=0).argmax(axis=-1)
            input_hashes[str(path.relative_to(REPO))] = sha256(path)
        result = {
            "dataset": "roman-empire",
            "model": "GAT",
            "split": split,
            "depth": int(rows[(split, "gnnm")]["num_layers"]),
            "n_test": int(compared["gnnm"]["n_test"]),
            "members": int(compared["gnnm"]["members"]),
        }
        for variant in VARIANTS:
            for key, value in compared[variant].items():
                if key not in ("n_test", "members"):
                    result[f"{variant}_{key}"] = value
        for key in METRICS:
            result[f"gnnm_minus_untied_{key}"] = (
                float(compared["gnnm"][key]) -
                float(compared["untied_backbone"][key])
            )
        gnnm_correct = pooled_decisions["gnnm"] == common_labels
        untied_correct = pooled_decisions["untied_backbone"] == common_labels
        result["both_pooled_correct_nodes"] = int(
            (gnnm_correct & untied_correct).sum()
        )
        result["gnnm_only_pooled_correct_nodes"] = int(
            (gnnm_correct & ~untied_correct).sum()
        )
        result["untied_only_pooled_correct_nodes"] = int(
            (~gnnm_correct & untied_correct).sum()
        )
        result["both_pooled_wrong_nodes"] = int(
            (~gnnm_correct & ~untied_correct).sum()
        )
        result["pooled_decision_disagreement_percent"] = 100.0 * float(
            np.mean(pooled_decisions["gnnm"] !=
                    pooled_decisions["untied_backbone"])
        )
        if not math.isclose(
            result["gnnm_minus_untied_pooled_accuracy_percent"],
            100.0 * (
                result["gnnm_only_pooled_correct_nodes"] -
                result["untied_only_pooled_correct_nodes"]
            ) / result["n_test"],
            rel_tol=0, abs_tol=1e-10
        ):
            raise AssertionError("Paired accuracy difference identity failed")
        output_rows.append(result)
    if any(sha256(REPO / relative) != digest
           for relative, digest in input_hashes.items()):
        raise RuntimeError("An input changed during GAT analysis")
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)
    audit = {
        "schema": 1,
        "verification": verified,
        "scope": (
            "Post hoc, descriptive five-mask comparison on one graph. "
            "The failure case and mask depths were selected after archived "
            "results. Nodes and official masks are paired, not independent "
            "replications. No claim about new graph generalization follows."
        ),
        "metric_units": {
            "accuracy_and_disagreement": "percent",
            "pooling_gain_and_accuracy_differences": "percentage points",
            "ce_and_kl": "nats per test node",
            "rescue_and_harm": "member-node pairs per official mask",
        },
        "input_sha256": input_hashes,
    }
    return (
        buffer.getvalue(),
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        len(output_rows),
    )


def run() -> None:
    csv_text, audit_text, count = render_expected()
    atomic_text(OUTPUT, csv_text)
    atomic_text(AUDIT, audit_text)
    print(f"Verified and analyzed {count} paired GAT masks")


def verify_only() -> None:
    csv_text, audit_text, count = render_expected()
    for path, expected in ((OUTPUT, csv_text), (AUDIT, audit_text)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing GAT analysis output: {path}")
        if path.read_bytes() != expected.encode("utf-8"):
            raise RuntimeError(
                f"GAT analysis output differs from verified inputs: {path}"
            )
    print(f"Verified GAT decision analysis output for {count} paired masks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.verify_only:
        verify_only()
    else:
        run()


if __name__ == "__main__":
    main()
