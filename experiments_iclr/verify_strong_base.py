"""Read-only CPU gate for the timing-selected five-mask Roman Empire BASE.

Recomputes the score-blind timing choice, checks profile/selection consistency,
and verifies selected checkpoint and prediction artifacts and paired summaries.
No model is trained and no GPU or output file is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments_iclr"))
from models import Model, TABMModel  # noqa: E402
from profile_strong_base import FIELDS, WIDTHS  # noqa: E402

SPLITS = tuple(range(5))
PROFILE_KEYS = (("TABM_k4", 512), ("BASE", 512),
                ("BASE", 896), ("BASE", 1024), ("BASE", 1152))
PROTOCOL = ("Roman Empire official split 0, depth 5, lr 3e-5, AdamW weight "
            "decay 0, 50 warmup then 200 CUDA-synchronized training steps")


def require(ok: bool, context: str) -> None:
    if not ok:
        raise RuntimeError(context)


def near(actual: object, expected: object, context: str,
         atol: float = 1e-5) -> None:
    a, b = float(actual), float(expected)
    require(math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=1e-6, abs_tol=atol),
            f"{context}: {a} differs from {b}")


def file(path: Path, root: Path) -> Path:
    require(path.is_file(), f"Missing file: {path.relative_to(root)}")
    require(path.stat().st_size > 0, f"Empty file: {path.relative_to(root)}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path, root: Path) -> tuple[list[str], list[dict[str, str]]]:
    with file(path, root).open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        rows = list(reader)
    require(header is not None and all(None not in row and all(
            value is not None for value in row.values()) for row in rows),
            f"Malformed CSV: {path.relative_to(root)}")
    return header, rows


def param_count(variant: str, width: int) -> int:
    if variant == "TABM_k4":
        model = TABMModel("SAGE", 5, 300, width, 18, 1, 8,
                          "LayerNorm", 0.2, 4, "cpu")
    else:
        model = Model("SAGE", 5, 300, width, 18, 1, 8,
                      "LayerNorm", 0.2)
    count = sum(parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
    del model
    return count


def selection_profile(root: Path) -> tuple[dict, int, dict]:
    require(WIDTHS == (896, 1024, 1152),
            "Timing candidates differ from predeclared widths")
    strong = root / "experiments_iclr" / "strong_base_results"
    selection = json.loads(file(strong / "strong_base_selection.json", root).read_text())
    header, rows = csv_rows(strong / "strong_base_profile.csv", root)
    require(header == FIELDS and len(rows) == len(PROFILE_KEYS),
            "Timing profile columns or row count differ from protocol")
    keys = [(row["variant"], int(row["hidden_dim"])) for row in rows]
    require(keys == list(PROFILE_KEYS),
            "Timing profile does not contain the five fixed configurations in order")
    measurements = selection.get("measurements")
    require(isinstance(measurements, list) and len(measurements) == 5,
            "Selection manifest lacks five timing measurements")
    manifest_keys = [(row["variant"], int(row["hidden_dim"]))
                     for row in measurements]
    require(manifest_keys == keys, "Selection and profile configurations differ")
    require(selection.get("profile_protocol") == PROTOCOL and
            "No validation or test metric participates" in
            selection.get("selection_rule", ""),
            "Selection protocol or score-blind rule differs from source")
    device = selection.get("device")
    require(isinstance(device, str) and "A100" in device,
            "Timing profile does not identify the claimed A100 device")
    require(isinstance(selection.get("torch_version"), str) and
            selection["torch_version"], "Timing profile lacks PyTorch version")
    torch.set_num_threads(1)
    counts = {key: param_count(*key) for key in PROFILE_KEYS}
    require(counts[("TABM_k4", 512)] == 6_737_728,
            "GNNM timing constructor parameter count changed")
    numeric_fields = ("num_layers", "lr", "num_params", "mean_step_ms",
                      "std_step_ms", "peak_allocated_mib")
    for profile_row, manifest_row, key in zip(rows, measurements, PROFILE_KEYS):
        require((profile_row["dataset"], profile_row["model"],
                 profile_row["device"]) ==
                ("roman-empire", "SAGE", device),
                f"Wrong dataset, model, or device in timing row {key}")
        require(all(str(manifest_row[field]) == profile_row[field]
                    for field in ("dataset", "model", "variant", "device")),
                f"Manifest identifiers differ from timing CSV at {key}")
        require(int(profile_row["num_layers"]) == 5,
                f"Timing depth differs at {key}")
        near(profile_row["lr"], 3e-5, f"Timing learning rate {key}", 1e-12)
        require(int(profile_row["num_params"]) == counts[key],
                f"Timing parameter count differs from constructor at {key}")
        for field in numeric_fields:
            near(manifest_row[field], profile_row[field],
                 f"Manifest timing row {key} {field}", 1e-6)
        require(float(profile_row["mean_step_ms"]) > 0 and
                float(profile_row["std_step_ms"]) >= 0 and
                float(profile_row["peak_allocated_mib"]) > 0,
                f"Invalid synchronized timing or allocated MiB at {key}")
    by_key = {key: row for key, row in zip(keys, rows)}
    target = float(by_key[("TABM_k4", 512)]["mean_step_ms"])
    winner = min(WIDTHS, key=lambda width: (
        abs(math.log(float(by_key[("BASE", width)]["mean_step_ms"]) / target)),
        width))
    chosen = by_key[("BASE", winner)]
    require(int(selection["selected_width"]) == winner,
            "Selected BASE width differs from score-blind timing rule")
    near(selection["target_step_ms"], target, "GNNM target step time", 1e-6)
    near(selection["selected_step_ms"], chosen["mean_step_ms"],
         "Selected BASE step time", 1e-6)
    near(selection["selected_to_target_step_ratio"],
         float(chosen["mean_step_ms"]) / target, "BASE/GNNM step ratio", 1e-8)
    require(int(selection["selected_num_params"]) == counts[("BASE", winner)] and
            int(selection["gnnm_num_params"]) == counts[("TABM_k4", 512)],
            "Selected or target parameter count differs from timed constructor")
    return selection, winner, by_key


def official_data(root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = file(root / "data" / "roman_empire.npz", root)
    manifest = json.loads(file(root / "experiments_iclr" /
                               "data_manifest.json", root).read_text())
    require(sha256(path) == manifest["files"]["roman_empire.npz"]["sha256"],
            "Roman Empire data differs from its pinned manifest")
    with np.load(path, allow_pickle=False) as source:
        shape = source["node_features"].shape
        labels = np.asarray(source["node_labels"], dtype=np.int64).reshape(-1)
        masks = np.asarray(source["test_masks"], dtype=bool)
    if masks.ndim == 2 and masks.shape[0] < masks.shape[1]:
        masks = masks.T
    require(len(shape) == 2 and shape[1] == 300 and len(labels) == shape[0]
            and labels.min() >= 0 and labels.max() == 17 and
            masks.ndim == 2 and masks.shape[0] == len(labels)
            and masks.shape[1] >= 5,
            "Roman Empire dimensions or official masks differ")
    return labels, masks


def results(root: Path, strong: Path, width: int,
            selection: dict) -> tuple[dict, dict]:
    _, all_base = csv_rows(strong / "projector_controls.csv", root)
    require(len(all_base) == 5, "Selected BASE CSV has extra or missing rows")
    base = {}
    for row in all_base:
        split = int(row["split"])
        require((row["dataset"], row["model"], row["variant"]) ==
                ("roman-empire", "SAGE", "base") and
                split in SPLITS and split not in base,
                "Selected BASE CSV has a wrong or duplicate split")
        base[split] = row
    require(set(base) == set(SPLITS), "Selected BASE lacks an official split")
    _, standard_rows = csv_rows(root / "experiments_iclr" / "results" /
                                "projector_controls.csv", root)
    gnnm_rows = [row for row in standard_rows if
                 (row["dataset"], row["model"], row["variant"]) ==
                 ("roman-empire", "SAGE", "gnnm")]
    require(len(gnnm_rows) == 5 and
            sorted(int(row["split"]) for row in gnnm_rows) == list(SPLITS),
            "Selected GNNM needs five official masks")
    gnnm = {int(row["split"]): row for row in gnnm_rows}
    for split in SPLITS:
        for variant, row, expected_width, members, params in (
                ("BASE", base[split], width, 1, int(selection["selected_num_params"])),
                ("GNNM", gnnm[split], 512, 4, int(selection["gnnm_num_params"]))):
            for field, expected in (("seed", split), ("num_layers", 5),
                                    ("hidden_dim", expected_width),
                                    ("m", members), ("num_steps", 5000),
                                    ("num_params", params)):
                require(int(row[field]) == expected,
                        f"{variant} split {split} has wrong {field}")
            near(row["lr"], 3e-5, f"{variant} split {split} learning rate", 1e-12)
            step = int(row["best_step"])
            require(1 <= step <= 5000 and (step == 1 or step % 10 == 0),
                    f"{variant} split {split} has invalid checkpoint step")
            require(float(row["train_seconds"]) > 0,
                    f"{variant} split {split} has invalid full training time")
            for field in ("val_metric", "test_metric", "test_acc"):
                value = float(row[field])
                require(math.isfinite(value) and 0 <= value <= 1,
                        f"{variant} split {split} has invalid {field}")
            near(row["test_metric"], row["test_acc"],
                 f"{variant} split {split} test metric", 1e-7)
    return base, gnnm


def selected_artifact(row: dict, root: Path, strong: Path, split: int,
                      field: str, folder: str, suffix: str) -> Path:
    raw = row.get(field, "")
    relative = Path(raw)
    expected = strong / folder / f"roman-empire_SAGE_base_split{split}{suffix}"
    require(raw and not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe selected BASE {field} in split {split}")
    path = root / relative
    require(path == expected and path.resolve().is_relative_to(strong.resolve()),
            f"Wrong selected BASE {field} path in split {split}")
    return file(path, root)


def checkpoint_schema(width: int) -> dict:
    model = Model("SAGE", 5, 300, width, 18, 1, 8, "LayerNorm", 0.2)
    schema = {name: (tuple(tensor.shape), tensor.dtype)
              for name, tensor in model.state_dict().items()}
    del model
    return schema


def checkpoint(path: Path, schema: dict, split: int) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and set(state) == set(schema),
            f"Selected BASE checkpoint keys differ in split {split}")
    for name, (shape, dtype) in schema.items():
        tensor = state[name]
        require(isinstance(tensor, torch.Tensor) and
                tuple(tensor.shape) == shape and tensor.dtype == dtype and
                bool(torch.isfinite(tensor).all()),
                f"Invalid selected BASE checkpoint tensor {name} in split {split}")
    del state


def prediction(row: dict, path: Path, split: int,
               labels: np.ndarray, masks: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        keys = {"node_index", "y_true", "member_logits", "member_pred",
                "ensemble_pred", "ensemble_prob", "confidence"}
        require(keys.issubset(archive.files),
                f"Selected BASE prediction lacks arrays in split {split}")
        index = np.asarray(archive["node_index"])
        truth = np.asarray(archive["y_true"])
        logits = np.asarray(archive["member_logits"])
        member_pred = np.asarray(archive["member_pred"])
        ensemble_pred = np.asarray(archive["ensemble_pred"])
        saved_prob = np.asarray(archive["ensemble_prob"])
        saved_conf = np.asarray(archive["confidence"])
    expected_index = np.flatnonzero(masks[:, split])
    count = len(expected_index)
    require(np.array_equal(index, expected_index) and
            np.array_equal(truth, labels[expected_index]),
            f"Selected BASE prediction is not aligned to official mask {split}")
    require(logits.shape == (1, count, 18) and
            member_pred.shape == (1, count) and
            ensemble_pred.shape == (count,) and
            saved_prob.shape == (count, 18) and saved_conf.shape == (count,),
            f"Selected BASE prediction shapes differ in split {split}")
    require(np.isfinite(logits).all() and np.isfinite(saved_prob).all(),
            f"Nonfinite selected BASE prediction in split {split}")
    pooled = logits.mean(axis=0).astype(np.float64)
    shifted = pooled - pooled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(axis=1, keepdims=True)
    require(np.allclose(saved_prob, prob, rtol=1e-5, atol=1e-6) and
            np.array_equal(member_pred, logits.argmax(axis=2)) and
            np.array_equal(ensemble_pred, pooled.argmax(axis=1)) and
            np.allclose(saved_conf, prob.max(axis=1), rtol=1e-5, atol=1e-6),
            f"Selected BASE arrays differ from pooled logits in split {split}")
    require(int(row["n_test"]) == count,
            f"Selected BASE test count differs from mask {split}")
    accuracy = float(np.mean(ensemble_pred == truth))
    loss = float(np.mean(np.log(exp.sum(axis=1)) -
                         shifted[np.arange(count), truth]))
    near(row["test_acc"], accuracy, f"BASE split {split} accuracy", 1e-6)
    near(row["test_loss"], loss, f"BASE split {split} pooled loss", 1e-4)
    near(row["mean_member_acc"], accuracy,
         f"BASE split {split} member accuracy", 1e-6)
    near(row["at_least_one_member_correct_rate"], accuracy,
         f"BASE split {split} any-member accuracy", 1e-6)
    near(row["pair_disagreement"], 0, f"BASE split {split} disagreement", 1e-7)
    near(row["error_jaccard"], 1, f"BASE split {split} error Jaccard", 1e-7)
    near(row["mean_entropy"],
         np.mean(-np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)),
         f"BASE split {split} entropy")
    near(row["nll"],
         -np.mean(np.log(np.maximum(prob[np.arange(count), truth], 1e-12))),
         f"BASE split {split} NLL")
    one_hot = np.eye(18)[truth]
    near(row["brier"], np.mean(np.sum((prob - one_hot) ** 2, axis=1)),
         f"BASE split {split} Brier score")
    confidence = prob.max(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1],
                            np.linspace(0, 1, 11)[1:]):
        chosen = (confidence > lower) & (confidence <= upper)
        if chosen.any():
            ece += chosen.mean() * abs((ensemble_pred[chosen] == truth[chosen]).mean()
                                       - confidence[chosen].mean())
    near(row["ece10"], ece, f"BASE split {split} ECE10")


def summaries(root: Path, strong: Path, selection: dict,
              base: dict, gnnm: dict) -> None:
    _, paired = csv_rows(strong / "strong_base_paired_scores.csv", root)
    require(len(paired) == 5 and
            sorted(int(row["split"]) for row in paired) == list(SPLITS),
            "BASE paired score table needs five official masks")
    paired_by_split = {int(row["split"]): row for row in paired}
    for split in SPLITS:
        a, b, row = gnnm[split], base[split], paired_by_split[split]
        expected = {
            "gnnm_val_acc_percent": 100 * float(a["val_metric"]),
            "gnnm_test_acc_percent": 100 * float(a["test_acc"]),
            "gnnm_train_seconds": float(a["train_seconds"]),
            "base_val_acc_percent": 100 * float(b["val_metric"]),
            "base_test_acc_percent": 100 * float(b["test_acc"]),
            "base_train_seconds": float(b["train_seconds"]),
            "base_minus_gnnm_test_pp": 100 * (float(b["test_acc"]) -
                                                float(a["test_acc"])),
        }
        for field, value in expected.items():
            near(row[field], value, f"BASE paired split {split} {field}")
        require(int(row["gnnm_best_step"]) == int(a["best_step"]) and
                int(row["base_best_step"]) == int(b["best_step"]),
                f"BASE paired selected step differs in split {split}")
    comparison = json.loads(file(strong / "strong_base_comparison.json", root).read_text())
    require((comparison.get("dataset"), comparison.get("model"),
             comparison.get("official_splits"), comparison.get("fixed_depth"),
             comparison.get("gnnm_width"),
             comparison.get("base_width_selected_from_timing_only")) ==
            ("roman-empire", "SAGE", list(SPLITS), 5, 512,
             int(selection["selected_width"])),
            "BASE comparison metadata or chosen width is stale")
    near(comparison["learning_rate"], 3e-5,
         "BASE comparison learning rate", 1e-12)
    require(int(comparison["gnnm_num_params"]) ==
            int(selection["gnnm_num_params"]) and
            int(comparison["base_num_params"]) ==
            int(selection["selected_num_params"]),
            "BASE comparison parameter counts are stale")
    for field, source in (("gnnm_synchronized_step_ms", "target_step_ms"),
                          ("base_synchronized_step_ms", "selected_step_ms"),
                          ("base_to_gnnm_step_ratio",
                           "selected_to_target_step_ratio")):
        near(comparison[field], selection[source], f"BASE comparison {field}")
    expected_means = {
        "gnnm_mean_test_acc_percent": "gnnm_test_acc_percent",
        "base_mean_test_acc_percent": "base_test_acc_percent",
        "base_minus_gnnm_mean_paired_test_pp": "base_minus_gnnm_test_pp",
        "gnnm_mean_full_training_seconds": "gnnm_train_seconds",
        "base_mean_full_training_seconds": "base_train_seconds",
    }
    for output, column in expected_means.items():
        near(comparison[output],
             np.mean([float(paired_by_split[s][column]) for s in SPLITS]),
             f"BASE comparison {output}")
    effects = [float(paired_by_split[s]["base_minus_gnnm_test_pp"])
               for s in SPLITS]
    for field, value in (("base_test_split_wins", sum(x > 0 for x in effects)),
                         ("base_test_split_ties", sum(x == 0 for x in effects)),
                         ("base_test_split_losses", sum(x < 0 for x in effects))):
        require(int(comparison[field]) == value,
                f"BASE comparison {field} is stale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO,
                        help="Repository root; used for a synthetic in-repo fixture")
    parser.add_argument("--require-complete", action="store_true",
                        help="Require the complete five-mask study (default)")
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == REPO or root.is_relative_to(REPO),
            "Audit root must stay inside the repository")
    strong = root / "experiments_iclr" / "strong_base_results"
    selection, width, _ = selection_profile(root)
    labels, masks = official_data(root)
    base, gnnm = results(root, strong, width, selection)
    for split in SPLITS:
        require(int(base[split]["n_test"]) == int(masks[:, split].sum()) and
                int(gnnm[split]["n_test"]) == int(masks[:, split].sum()),
                f"Official test-mask size differs in split {split}")
    schema = checkpoint_schema(width)
    for split in SPLITS:
        row = base[split]
        ckpt = selected_artifact(row, root, strong, split,
                                 "checkpoint", "checkpoints", ".pt")
        pred = selected_artifact(row, root, strong, split,
                                 "prediction_file", "predictions", ".npz")
        checkpoint(ckpt, schema, split)
        prediction(row, pred, split, labels, masks)
    summaries(root, strong, selection, base, gnnm)
    print(f"PASS timing-selected BASE: width {width}, five artifacts and summaries verified")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, TypeError, RuntimeError) as error:
        raise SystemExit(f"FAIL timing-selected BASE audit: {error}") from None
