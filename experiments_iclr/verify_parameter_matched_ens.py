"""Read-only CPU gate for the five-mask parameter-matched SAGE ensemble.

The selector is recomputed from the predeclared constructors and widths. The
complete gate then checks the five selected rows, their checkpoint and test
prediction files, the paired score table, and the JSON comparison. If an
inference profile exists, its checkpoint sizes, configuration, and units are
checked too. This script never trains, writes files, or uses a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

from select_parameter_matched_ens import (  # noqa: E402
    DEPTH, INPUT_DIM, MEMBERS, OUTPUT_DIM, TARGET_WIDTH, WIDTHS,
    count_variant, make_model,
)

ROOT = Path(__file__).resolve().parents[1]
MATCHED = ROOT / "experiments_iclr" / "parameter_matched_ens_results"
STANDARD = ROOT / "experiments_iclr" / "results"
SPLITS = tuple(range(5))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def near(actual: object, expected: object, context: str,
         atol: float = 1e-5) -> None:
    a, b = float(actual), float(expected)
    check(math.isfinite(a) and math.isfinite(b) and
          math.isclose(a, b, rel_tol=1e-6, abs_tol=atol),
          f"{context}: {a} differs from {b}")


def required_file(path: Path) -> Path:
    check(path.is_file(), f"Missing required file: {path.relative_to(ROOT)}")
    check(path.stat().st_size > 0, f"Empty required file: {path.relative_to(ROOT)}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with required_file(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames is not None, f"Missing CSV header: {path.name}")
        rows = list(reader)
    check(all(None not in row and all(value is not None for value in row.values())
              for row in rows), f"Malformed CSV row: {path.name}")
    return rows


def selected_path(row: dict[str, str], field: str, folder: str,
                  split: int, root: Path, suffix: str) -> Path:
    raw = row.get(field, "")
    relative = Path(raw)
    expected = root / folder / f"roman-empire_SAGE_ens_pooled_split{split}{suffix}"
    check(raw and not relative.is_absolute() and ".." not in relative.parts,
          f"Unsafe {field} for matched split {split}: {raw!r}")
    path = ROOT / relative
    check(path == expected and path.resolve().is_relative_to(root.resolve()),
          f"Wrong {field} for matched split {split}: {raw!r}")
    return required_file(path)


def fixed_selection() -> tuple[dict, int, int]:
    check((WIDTHS, INPUT_DIM, OUTPUT_DIM, DEPTH, MEMBERS, TARGET_WIDTH) ==
          ((192, 224, 256, 288, 320), 300, 18, 5, 4, 512),
          "Predeclared selector architecture or candidate widths changed")
    manifest = json.loads(required_file(MATCHED / "selection.json").read_text())
    check((manifest.get("dataset"), manifest.get("model"),
           manifest.get("target_variant"), manifest.get("candidate_variant")) ==
          ("roman-empire", "SAGE", "gnnm", "ens_pooled"),
          "Selection manifest has an unexpected model or dataset")
    for field, expected in (("input_dim", INPUT_DIM),
                            ("output_dim", OUTPUT_DIM),
                            ("num_layers", DEPTH), ("members", MEMBERS),
                            ("target_width", TARGET_WIDTH)):
        check(int(manifest[field]) == expected, f"Selection {field} changed")
    check(manifest.get("candidate_widths") == list(WIDTHS),
          "Selection candidate widths differ from the fixed list")
    torch.set_num_threads(1)
    target = count_variant("gnnm", TARGET_WIDTH)
    counts = {width: count_variant("ens_pooled", width) for width in WIDTHS}
    check(target == 6_737_728, f"GNNM constructor count changed: {target}")
    check(int(manifest["target_trainable_parameters"]) == target,
          "Selection target count is stale")
    recorded = {int(key): int(value) for key, value in
                manifest["candidate_trainable_parameters"].items()}
    check(recorded == counts, "Selection candidate counts are stale")
    selected = min(WIDTHS, key=lambda width: (abs(counts[width] - target), width))
    check(int(manifest["selected_width"]) == selected,
          "Selected width differs from count-only rule")
    check(int(manifest["selected_trainable_parameters"]) == counts[selected],
          "Selected parameter count is stale")
    check(int(manifest["selected_minus_target_parameters"]) ==
          counts[selected] - target, "Selection parameter difference is stale")
    near(manifest["selected_to_target_parameter_ratio"],
         counts[selected] / target, "Selection parameter ratio", 1e-10)
    check("No validation or test score participates" in
          manifest.get("selection_rule", ""),
          "Selection manifest does not state the score-blind rule")
    return manifest, selected, counts[selected]


def official_data() -> tuple[np.ndarray, np.ndarray]:
    data_path = required_file(ROOT / "data" / "roman_empire.npz")
    source = json.loads(required_file(ROOT / "experiments_iclr" /
                                      "data_manifest.json").read_text())
    check(file_sha256(data_path) == source["files"]["roman_empire.npz"]["sha256"],
          "Roman Empire data differs from the pinned source")
    with np.load(data_path, allow_pickle=False) as data:
        node_count, input_dim = data["node_features"].shape
        labels = np.asarray(data["node_labels"], dtype=np.int64).reshape(-1)
        masks = np.asarray(data["test_masks"], dtype=bool)
    if masks.ndim == 2 and masks.shape[0] < masks.shape[1]:
        masks = masks.T
    check(node_count == len(labels) == masks.shape[0] and
          input_dim == INPUT_DIM and int(labels.max()) + 1 == OUTPUT_DIM and
          masks.ndim == 2 and masks.shape[1] >= 5,
          "Roman Empire dimensions or official mask matrix differ")
    return labels, masks


def rows_by_split(path: Path, variant: str, width: int,
                  exact_file: bool = False) -> dict[int, dict[str, str]]:
    all_rows = read_csv(path)
    rows = [row for row in all_rows if
            (row.get("dataset"), row.get("model"), row.get("variant")) ==
            ("roman-empire", "SAGE", variant) and
            int(row["hidden_dim"]) == width]
    splits = [int(row["split"]) for row in rows]
    check(len(rows) == 5 and sorted(splits) == list(SPLITS),
          f"Need exactly one {variant} width-{width} row per official mask")
    if exact_file:
        check(len(all_rows) == 5, "Matched result CSV contains extra rows")
    return {int(row["split"]): row for row in rows}


def row_protocol(row: dict[str, str], split: int, width: int,
                 members: int, params: int | None = None) -> None:
    for field, expected in (("seed", split), ("num_layers", 5),
                            ("hidden_dim", width), ("m", members),
                            ("num_steps", 5000)):
        check(int(row[field]) == expected,
              f"Split {split} has wrong {field}: {row[field]}")
    near(row["lr"], 3e-5, f"Split {split} learning rate", 1e-12)
    step = int(row["best_step"])
    check(1 <= step <= 5000 and (step == 1 or step % 10 == 0),
          f"Split {split} has an invalid selected step")
    check(float(row["train_seconds"]) > 0,
          f"Split {split} has invalid full training time")
    for field in ("val_metric", "test_metric", "test_acc"):
        value = float(row[field])
        check(math.isfinite(value) and 0 <= value <= 1,
              f"Split {split} has invalid {field}")
    near(row["test_metric"], row["test_acc"],
         f"Split {split} Roman Empire test metric", 1e-7)
    if params is not None:
        check(int(row["num_params"]) == params,
              f"Split {split} parameter count differs from selected constructor")


def checkpoint_spec(width: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    args = SimpleNamespace(model="SAGE", num_layers=5, hidden_dim=width, m=4)
    model = make_model(args, "ens_pooled", INPUT_DIM, OUTPUT_DIM,
                       torch.device("cpu"))
    spec = {name: (tuple(value.shape), value.dtype)
            for name, value in model.state_dict().items()}
    del model
    return spec


def verify_checkpoint(path: Path, spec: dict, split: int) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    check(isinstance(state, dict) and set(state) == set(spec),
          f"Split {split} selected checkpoint has unexpected tensor names")
    for name, (shape, dtype) in spec.items():
        tensor = state[name]
        check(isinstance(tensor, torch.Tensor) and
              tuple(tensor.shape) == shape and tensor.dtype == dtype and
              bool(torch.isfinite(tensor).all()),
              f"Split {split} selected checkpoint tensor {name} is invalid")
    del state


def prediction_metrics(row: dict[str, str], path: Path, split: int,
                       labels: np.ndarray, masks: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as saved:
        required = {"node_index", "y_true", "member_logits", "member_pred",
                    "ensemble_pred", "ensemble_prob", "confidence"}
        check(required.issubset(saved.files),
              f"Split {split} prediction lacks required arrays")
        index = np.asarray(saved["node_index"])
        truth = np.asarray(saved["y_true"])
        logits = np.asarray(saved["member_logits"])
        member_pred = np.asarray(saved["member_pred"])
        ensemble_pred = np.asarray(saved["ensemble_pred"])
        saved_prob = np.asarray(saved["ensemble_prob"])
        saved_conf = np.asarray(saved["confidence"])
    expected_index = np.flatnonzero(masks[:, split])
    n = len(expected_index)
    check(np.array_equal(index, expected_index) and
          np.array_equal(truth, labels[expected_index]),
          f"Split {split} selected prediction is not aligned to its official test mask")
    check(logits.shape == (4, n, OUTPUT_DIM) and
          member_pred.shape == (4, n) and
          ensemble_pred.shape == (n,) and
          saved_prob.shape == (n, OUTPUT_DIM) and saved_conf.shape == (n,),
          f"Split {split} prediction array shapes are wrong")
    check(np.isfinite(logits).all() and np.isfinite(saved_prob).all(),
          f"Split {split} prediction has nonfinite logits or probabilities")
    pooled = logits.mean(axis=0).astype(np.float64)
    shifted = pooled - pooled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(axis=1, keepdims=True)
    check(np.allclose(saved_prob, prob, rtol=1e-5, atol=1e-6) and
          np.array_equal(member_pred, logits.argmax(axis=2)) and
          np.array_equal(ensemble_pred, pooled.argmax(axis=1)) and
          np.allclose(saved_conf, prob.max(axis=1), rtol=1e-5, atol=1e-6),
          f"Split {split} selected predictions differ from pooled logits")
    check(int(row["n_test"]) == n, f"Split {split} test count differs from mask")
    accuracy = float(np.mean(ensemble_pred == truth))
    near(row["test_acc"], accuracy, f"Split {split} test accuracy", 1e-6)
    loss = float(np.mean(np.log(np.exp(shifted).sum(axis=1)) -
                         shifted[np.arange(n), truth]))
    near(row["test_loss"], loss, f"Split {split} pooled cross entropy", 1e-4)
    correct = member_pred == truth[None, :]
    near(row["mean_member_acc"], correct.mean(),
         f"Split {split} mean member accuracy", 1e-6)
    near(row["at_least_one_member_correct_rate"], correct.any(axis=0).mean(),
         f"Split {split} at-least-one-member rate", 1e-6)
    pairs = list(combinations(range(4), 2))
    disagreement = np.mean([(member_pred[a] != member_pred[b]).mean()
                            for a, b in pairs])
    near(row["pair_disagreement"], disagreement,
         f"Split {split} member disagreement", 1e-6)
    jaccard = []
    for a, b in pairs:
        ea, eb = ~correct[a], ~correct[b]
        union = np.logical_or(ea, eb).sum()
        jaccard.append(np.logical_and(ea, eb).sum() / union if union else 1.0)
    near(row["error_jaccard"], np.mean(jaccard),
         f"Split {split} error Jaccard", 1e-6)
    near(row["mean_entropy"],
         np.mean(-np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)),
         f"Split {split} entropy", 1e-5)
    near(row["nll"], -np.mean(np.log(np.maximum(prob[np.arange(n), truth], 1e-12))),
         f"Split {split} NLL", 1e-5)
    one_hot = np.eye(OUTPUT_DIM)[truth]
    near(row["brier"], np.mean(np.sum((prob - one_hot) ** 2, axis=1)),
         f"Split {split} Brier score", 1e-5)
    confidence = prob.max(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1],
                            np.linspace(0, 1, 11)[1:]):
        chosen = (confidence > lower) & (confidence <= upper)
        if chosen.any():
            ece += chosen.mean() * abs((ensemble_pred[chosen] == truth[chosen]).mean()
                                       - confidence[chosen].mean())
    near(row["ece10"], ece, f"Split {split} ECE10", 1e-5)


def summaries(selection: dict, width: int, matched: dict,
              gnnm: dict, full: dict) -> None:
    paired = read_csv(MATCHED / "paired_scores.csv")
    check(len(paired) == 5 and
          sorted(int(row["split"]) for row in paired) == list(SPLITS),
          "Paired score table needs exactly five official masks")
    by_split = {int(row["split"]): row for row in paired}
    for split in SPLITS:
        a, b, c, row = gnnm[split], full[split], matched[split], by_split[split]
        expected = {
            "gnnm_test_acc_percent": 100 * float(a["test_acc"]),
            "ens_512_test_acc_percent": 100 * float(b["test_acc"]),
            "ens_matched_test_acc_percent": 100 * float(c["test_acc"]),
            "ens_512_minus_gnnm_pp": 100 * (float(b["test_acc"]) - float(a["test_acc"])),
            "ens_matched_minus_gnnm_pp": 100 * (float(c["test_acc"]) - float(a["test_acc"])),
            "gnnm_validation_acc_percent": 100 * float(a["val_metric"]),
            "ens_512_validation_acc_percent": 100 * float(b["val_metric"]),
            "ens_matched_validation_acc_percent": 100 * float(c["val_metric"]),
            "gnnm_train_seconds": float(a["train_seconds"]),
            "ens_512_train_seconds": float(b["train_seconds"]),
            "ens_matched_train_seconds": float(c["train_seconds"]),
        }
        for field, value in expected.items():
            near(row[field], value, f"Paired split {split} {field}")
        for field, value in (("gnnm_selected_step", a["best_step"]),
                             ("ens_512_selected_step", b["best_step"]),
                             ("ens_matched_selected_step", c["best_step"])):
            check(int(row[field]) == int(value),
                  f"Paired split {split} {field} is stale")
    comparison = json.loads(required_file(MATCHED / "comparison.json").read_text())
    check((comparison.get("dataset"), comparison.get("model"),
           comparison.get("official_masks"), comparison.get("all_variants_depth")) ==
          ("roman-empire", "SAGE", list(SPLITS), 5),
          "Comparison metadata or mask list is wrong")
    near(comparison["all_variants_learning_rate"], 3e-5,
         "Comparison learning rate", 1e-12)
    check(comparison["parameter_match_rule"] == selection["selection_rule"] and
          int(comparison["ens_matched_width"]) == width and
          int(comparison["gnnm_parameters"]) == int(gnnm[0]["num_params"]) and
          int(comparison["ens_512_parameters"]) == int(full[0]["num_params"]) and
          int(comparison["ens_matched_parameters"]) == int(matched[0]["num_params"]),
          "Comparison rule, width, or parameter counts are stale")
    expected_means = {
        "gnnm_mean_test_acc_percent": "gnnm_test_acc_percent",
        "ens_512_mean_test_acc_percent": "ens_512_test_acc_percent",
        "ens_matched_mean_test_acc_percent": "ens_matched_test_acc_percent",
        "ens_512_minus_gnnm_mean_paired_pp": "ens_512_minus_gnnm_pp",
        "ens_matched_minus_gnnm_mean_paired_pp": "ens_matched_minus_gnnm_pp",
        "gnnm_mean_full_training_seconds": "gnnm_train_seconds",
        "ens_512_mean_full_training_seconds": "ens_512_train_seconds",
        "ens_matched_mean_full_training_seconds": "ens_matched_train_seconds",
    }
    for output_field, paired_field in expected_means.items():
        near(comparison[output_field],
             np.mean([float(by_split[s][paired_field]) for s in SPLITS]),
             f"Comparison {output_field}")
    effects = [float(by_split[s]["ens_matched_minus_gnnm_pp"]) for s in SPLITS]
    for field, expected in (("ens_matched_test_mask_wins", sum(x > 0 for x in effects)),
                            ("ens_matched_test_mask_ties", sum(x == 0 for x in effects)),
                            ("ens_matched_test_mask_losses", sum(x < 0 for x in effects))):
        check(int(comparison[field]) == expected,
              f"Comparison {field} is stale")


def inference_profile_if_present(width: int, matched: dict,
                                 gnnm: dict, full: dict,
                                 required: bool) -> None:
    path = MATCHED / "inference_profile.json"
    if not path.exists():
        check(not required, "Inference profile is required but missing")
        return
    profile = json.loads(required_file(path).read_text())
    check((profile.get("dataset"), profile.get("model")) ==
          ("roman-empire", "SAGE") and
          "milliseconds per full-graph pooled prediction" in profile.get("units", "") and
          "checkpoint bytes" in profile.get("units", "") and
          isinstance(profile.get("device"), str) and profile["device"],
          "Inference profile metadata or units are invalid")
    configurations = (("gnnm", 512, gnnm), ("ens_pooled", 512, full),
                      ("ens_pooled", width, matched))
    results = profile.get("results")
    check(isinstance(results, list) and len(results) == 3,
          "Inference profile needs exactly three configurations")
    for index, (variant, model_width, rows) in enumerate(configurations):
        result = results[index]
        check((result.get("variant"), int(result["hidden_dim"]),
               int(result["split_for_inference"])) ==
              (variant, model_width, 0),
              f"Inference profile configuration {index} is stale")
        check(int(result["trainable_parameters"]) == int(rows[0]["num_params"]),
              f"Inference profile parameter count {index} is stale")
        expected_bytes = []
        for split in SPLITS:
            raw = Path(rows[split]["checkpoint"])
            check(not raw.is_absolute() and ".." not in raw.parts,
                  f"Unsafe inference checkpoint path for {variant} split {split}")
            checkpoint = required_file(ROOT / raw)
            check(checkpoint.resolve().is_relative_to(ROOT),
                  f"External inference checkpoint for {variant} split {split}")
            expected_bytes.append(checkpoint.stat().st_size)
        check(result["checkpoint_bytes_by_split"] == expected_bytes,
              f"Inference profile checkpoint bytes {index} are stale")
        near(result["mean_checkpoint_bytes"], np.mean(expected_bytes),
             f"Inference profile mean bytes {index}", 1e-3)
        minimum = float(result["min_inference_ms"])
        maximum = float(result["max_inference_ms"])
        check(math.isfinite(minimum) and 0 < minimum <= maximum,
              f"Inference profile latency range {index} is invalid")
        for field in ("mean_inference_ms", "median_inference_ms"):
            value = float(result[field])
            check(math.isfinite(value) and minimum <= value <= maximum,
                  f"Inference profile {field} {index} is invalid")
        sd = float(result["sample_sd_inference_ms"])
        check(math.isfinite(sd) and sd >= 0,
              f"Inference profile latency deviation {index} is invalid")


def complete(selection: dict, width: int, params: int,
             require_inference_profile: bool) -> None:
    labels, masks = official_data()
    matched = rows_by_split(MATCHED / "projector_controls.csv", "ens_pooled",
                            width, exact_file=True)
    gnnm = rows_by_split(STANDARD / "projector_controls.csv", "gnnm", 512)
    full = rows_by_split(STANDARD / "projector_controls.csv", "ens_pooled", 512)
    for split in SPLITS:
        row_protocol(matched[split], split, width, 4, params)
        row_protocol(gnnm[split], split, 512, 4, 6_737_728)
        row_protocol(full[split], split, 512, 4)
    spec = checkpoint_spec(width)
    for split in SPLITS:
        row = matched[split]
        checkpoint = selected_path(row, "checkpoint", "checkpoints", split,
                                   MATCHED, ".pt")
        prediction = selected_path(row, "prediction_file", "predictions", split,
                                   MATCHED, ".npz")
        verify_checkpoint(checkpoint, spec, split)
        prediction_metrics(row, prediction, split, labels, masks)
    summaries(selection, width, matched, gnnm, full)
    inference_profile_if_present(width, matched, gnnm, full,
                                 require_inference_profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-only", action="store_true",
                        help="Check the score-blind count selection without requiring completed runs")
    parser.add_argument("--require-complete", action="store_true",
                        help="Require all five rows, selected artifacts, and fresh summaries (default)")
    parser.add_argument("--require-inference-profile", action="store_true",
                        help="Also require the optional same-device inference profile")
    args = parser.parse_args()
    check(not (args.selection_only and
               (args.require_complete or args.require_inference_profile)),
          "--selection-only cannot be combined with complete gates")
    selection, width, params = fixed_selection()
    if args.selection_only:
        print(f"PASS count-only selection: width {width}, {params:,} parameters")
        return
    complete(selection, width, params, args.require_inference_profile)
    print(f"PASS five-mask matched ENS: width {width}, selected artifacts and summaries verified")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, TypeError, RuntimeError) as error:
        raise SystemExit(f"FAIL parameter-matched ENS audit: {error}") from None
