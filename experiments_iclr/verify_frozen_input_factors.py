"""Read-only, fail-closed audit of the frozen-input SAGE control.

Prefix mode checks ordered artifacts during the guarded run. Complete mode
also independently replays all five selected checkpoints on CPU.
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

import numpy as np
import torch
import torch_geometric

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = HERE / "frozen_input_factor_results"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from datasets import load_dataset  # noqa: E402
from projector_controls import FIELDS, all_logits, make_model  # noqa: E402

PROTOCOL_HASH = "6bbb3889c4f50e386b75450d6fdb1591ee14442c772ca5cacda1b8f3b968eadf"
STATIC_HASHES = {
    "experiments_iclr/projector_controls.py": "3b24562b132c439733f6e0599a84e62aa5d1d5b0ce04d0239574d3b07eea1fcf",
    "ablation/models_ablation.py": "2c03998028ff80dc3e0ea49149d8281da073797b52c663f8229bc8b61ace51a8",
    "models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
    "datasets.py": "33e4d1c7fb8033e45065db8f4dde67f4f076bf524488e301ceddf5586c9c7893",
    "run_common.py": "e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc",
    "data/roman_empire.npz": "a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428",
    "experiments_iclr/results/projector_controls.csv": "fa5682e1e2bd2c1d2f4d5ccdb6efb8e67850992df1acaf6c8a8e859b4573d47a",
}
BASE_CHECKPOINTS = (
    "072b0e53196c519dcc32efaa019020e9fceb27ad3dd7c2025d30705fa5371c12",
    "e55a995978848538fca4384797955d4eead7d18bd0fe2aa6b3dfa41c6a4611f7",
    "fe3c00526816aa0686bb890946b8defd492fbca9bd66880dc72535ecbd99f7c9",
    "7f85dbd05a969a7939ac2043f43a0d28cb780b3b376466c066f17c403a8188b9",
    "51890f54550bde90c06e38c1fcdaa3dedd890aabf118afc0560f51f2a8764775",
)
BASE_PREDICTIONS = (
    "bbbd675b8ea1d4c30968e7318c307311e33a458a0339313b0a6f66261e035754",
    "b537043f1edbd5e0e9c63a1c4a03ec10b27174f60e9c0f32c1779da88d5c02d5",
    "3c56df27e30f4fb9999ca9fb712a20853b514ec7a3905ac1595c8c470e1c5d87",
    "7e8257b6f7641a47cc617ecfa7081cc2e4975e64b6da151b7c9331a169911298",
    "0f4254ce05f717813fa26bd82b0cffb9861b3129c1ff87fbe85b9ca4102ff1cb",
)
CODE_FILES = (
    "experiments_iclr/frozen_input_factors.py",
    "experiments_iclr/verify_frozen_input_factors.py",
    "experiments_iclr/run_frozen_input_factors_once.sh",
)
PRED_FIELDS = {
    "node_index", "y_true", "ensemble_pred", "member_pred", "member_logits",
    "ensemble_prob", "confidence", "degree", "local_homophily",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_plain_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing, empty, or symlinked file: {path}")


def read_json(path: Path) -> dict:
    require_plain_file(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def check_static_inputs() -> list[dict[str, str]]:
    protocol = ROOT / "PROTOCOL.md"
    require_plain_file(protocol)
    if sha256(protocol) != PROTOCOL_HASH:
        raise ValueError("Prespecified protocol changed")
    for relative, expected in STATIC_HASHES.items():
        path = REPO / relative
        require_plain_file(path)
        if sha256(path) != expected:
            raise ValueError(f"Locked source/data/baseline changed: {relative}")
    csv_path = REPO / "experiments_iclr/results/projector_controls.csv"
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Existing control CSV header changed")
        baseline = [row for row in reader if row["dataset"] == "roman-empire"
                    and row["model"] == "SAGE" and row["variant"] == "gnnm"]
    if len(baseline) != 5 or [int(row["split"]) for row in baseline] != list(range(5)):
        raise ValueError("Existing GNNM baseline is not exactly masks 0-4")
    for split, row in enumerate(baseline):
        if (int(row["seed"]) != split or int(row["num_layers"]) != 5
                or int(row["hidden_dim"]) != 512 or int(row["m"]) != 4
                or int(row["num_steps"]) != 5000
                or not math.isclose(float(row["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"Existing GNNM settings differ for mask {split}")
        checkpoint = REPO / row["checkpoint"]
        prediction = REPO / row["prediction_file"]
        expected_stem = f"roman-empire_SAGE_gnnm_split{split}"
        if checkpoint != REPO / "experiments_iclr/results/checkpoints" / f"{expected_stem}.pt":
            raise ValueError("Existing GNNM checkpoint path changed")
        if prediction != REPO / "experiments_iclr/results/predictions" / f"{expected_stem}.npz":
            raise ValueError("Existing GNNM prediction path changed")
        require_plain_file(checkpoint)
        require_plain_file(prediction)
        if sha256(checkpoint) != BASE_CHECKPOINTS[split] or sha256(prediction) != BASE_PREDICTIONS[split]:
            raise ValueError(f"Existing GNNM artifact changed for mask {split}")
    return baseline


def expected_prepared_lock() -> dict:
    return {
        "schema": 1,
        "protocol_sha256": PROTOCOL_HASH,
        "code_sha256": {relative: sha256(REPO / relative) for relative in CODE_FILES},
        "cpu_initial_match_sha256": sha256(ROOT / "cpu_initial_match.json"),
        "cpu_smoke_report_sha256": sha256(ROOT / "cpu_smoke/smoke_report.json"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
    }


def check_prepared(*, require_cpu: bool = True) -> list[dict[str, str]]:
    baseline = check_static_inputs()
    lock = read_json(ROOT / "prepared_lock.json")
    if lock != expected_prepared_lock():
        raise ValueError("Prepared code/environment lock changed")
    if require_cpu:
        report = read_json(ROOT / "cpu_initial_match.json")
        matches = report.get("matches")
        if (report.get("schema") != 1 or report.get("source") != "real Roman Empire graph"
                or not isinstance(matches, list) or len(matches) != 5
                or [entry.get("split") for entry in matches] != list(range(5))):
            raise ValueError("All-five CPU initial-match report is absent or malformed")
        for split, entry in enumerate(matches):
            if (entry.get("seed") != split or entry.get("device") != "cpu"
                    or entry.get("changed_requires_grad") != [
                        "input_be_block.B", "input_be_block.R", "input_be_block.S"]
                    or int(entry.get("trainable_parameters", -1)) != 6732432):
                raise ValueError(f"CPU initial-match report changed for mask {split}")
            for field in ("state_sha256", "member_logits_sha256", "rng_after_constructor_sha256"):
                if not isinstance(entry.get(field), str) or len(entry[field]) != 64:
                    raise ValueError(f"Malformed CPU match {field}")
            factors = entry.get("input_factor_sha256")
            if (not isinstance(factors, dict) or set(factors) != {"R", "S", "B"}
                    or any(not isinstance(value, str) or len(value) != 64
                           for value in factors.values())):
                raise ValueError(f"Malformed CPU factor digests for mask {split}")
        smoke = read_json(ROOT / "cpu_smoke/smoke_report.json")
        if smoke.get("roundtrip") != "exact" or smoke.get("num_steps") != 2:
            raise ValueError("CPU train/artifact roundtrip missing")
        for field, path in (("checkpoint_sha256", ROOT / "cpu_smoke/checkpoints/cpu-smoke_SAGE_freeze_input_factors_split0.pt"),
                            ("prediction_sha256", ROOT / "cpu_smoke/predictions/cpu-smoke_SAGE_freeze_input_factors_split0.npz")):
            require_plain_file(path)
            if sha256(path) != smoke.get(field):
                raise ValueError(f"CPU smoke artifact changed: {field}")
        smoke_state = torch.load(ROOT / "cpu_smoke/checkpoints/cpu-smoke_SAGE_freeze_input_factors_split0.pt",
                                 map_location="cpu", weights_only=True)
        for factor in ("R", "S", "B"):
            value = smoke_state[f"input_be_block.{factor}"].contiguous().numpy().tobytes()
            if hashlib.sha256(value).hexdigest() != smoke["initial"]["input_factor_sha256"][factor]:
                raise ValueError(f"CPU smoke changed frozen input {factor}")
    return baseline


def expected_run_manifest() -> dict:
    return {"schema": 1, "prepared_lock_sha256": sha256(ROOT / "prepared_lock.json"),
            "protocol_sha256": PROTOCOL_HASH, "dataset": "roman-empire",
            "model": "SAGE", "variant": "freeze_input_factors",
            "splits": list(range(5)), "seed_by_split": list(range(5)),
            "num_layers": 5, "hidden_dim": 512, "lr": 3e-5,
            "m": 4, "num_steps": 5000, "device": "cuda:0",
            "selection": "max pooled validation accuracy; first tie retained"}


def finite_float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {name}")
    return value


def close(a: float, b: float, name: str, tol: float = 1e-5) -> None:
    if not math.isclose(a, b, rel_tol=0, abs_tol=tol):
        raise ValueError(f"{name} differs: {a} versus {b}")


def prediction_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    logits = values["member_logits"]
    y = values["y_true"]
    member = values["member_pred"]
    pool = values["ensemble_pred"]
    mean_logits = logits.mean(axis=0)
    shifted = mean_logits - mean_logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(axis=-1, keepdims=True)
    if not np.allclose(prob, values["ensemble_prob"], rtol=1e-6, atol=2e-6):
        raise ValueError("Saved probabilities disagree with pooled logits")
    if not np.array_equal(member, logits.argmax(axis=-1)) or not np.array_equal(pool, mean_logits.argmax(axis=-1)):
        raise ValueError("Saved argmax disagrees with member or pooled logits")
    if not np.allclose(values["confidence"], prob.max(axis=-1), rtol=1e-6, atol=2e-6):
        raise ValueError("Saved confidence differs")
    correct = member == y[None, :]
    pairs = list(combinations(range(4), 2))
    jaccard = []
    for i, j in pairs:
        ei, ej = ~correct[i], ~correct[j]
        union = np.logical_or(ei, ej).sum()
        jaccard.append(np.logical_and(ei, ej).sum() / union if union else 1.0)
    entropy = -(prob * np.log(np.maximum(prob, 1e-12))).sum(axis=-1)
    nll = -np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)).mean()
    brier = ((prob - np.eye(prob.shape[-1])[y]) ** 2).sum(axis=-1).mean()
    confidence = prob.max(axis=-1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs((pool[selected] == y[selected]).mean() - confidence[selected].mean())
    return {
        "test_metric": float((pool == y).mean()),
        "test_acc": float((pool == y).mean()),
        "mean_member_acc": float(correct.mean()),
        "at_least_one_member_correct_rate": float(correct.any(axis=0).mean()),
        "pair_disagreement": float(np.mean([(member[i] != member[j]).mean() for i, j in pairs])),
        "error_jaccard": float(np.mean(jaccard)),
        "mean_entropy": float(entropy.mean()), "nll": float(nll),
        "brier": float(brier), "ece10": float(ece),
    }


def check_row(row: dict[str, str], split: int, data=None, masks=None) -> None:
    if None in row or any(value is None for value in row.values()):
        raise ValueError("Truncated or extended control row")
    if (row["dataset"] != "roman-empire" or row["model"] != "SAGE"
            or row["variant"] != "freeze_input_factors" or int(row["split"]) != split
            or int(row["seed"]) != split or int(row["num_layers"]) != 5
            or int(row["hidden_dim"]) != 512 or int(row["m"]) != 4
            or int(row["num_steps"]) != 5000 or int(row["num_params"]) != 6732432
            or not math.isclose(float(row["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"Mixed protocol in mask {split}")
    step = int(row["best_step"])
    if not (1 <= step <= 5000 and (step == 1 or step % 10 == 0)):
        raise ValueError(f"Invalid selected step in mask {split}")
    if finite_float(row, "train_seconds") <= 0:
        raise ValueError("Invalid runtime")
    for field in ("val_metric", "test_metric", "test_acc", "mean_member_acc",
                  "at_least_one_member_correct_rate", "pair_disagreement",
                  "error_jaccard", "ece10"):
        if not 0 <= finite_float(row, field) <= 1:
            raise ValueError(f"Out-of-range {field}")
    for field in ("test_loss", "mean_entropy", "nll", "brier"):
        if finite_float(row, field) < 0:
            raise ValueError(f"Negative {field}")
    if int(row["n_test"]) != 5666:
        raise ValueError("Unexpected test node count")
    stem = f"roman-empire_SAGE_freeze_input_factors_split{split}"
    checkpoint_rel = Path("experiments_iclr/frozen_input_factor_results/checkpoints") / f"{stem}.pt"
    prediction_rel = Path("experiments_iclr/frozen_input_factor_results/predictions") / f"{stem}.npz"
    if Path(row["checkpoint"]) != checkpoint_rel or Path(row["prediction_file"]) != prediction_rel:
        raise ValueError(f"Unexpected artifact paths for mask {split}")
    checkpoint = REPO / checkpoint_rel
    prediction = REPO / prediction_rel
    require_plain_file(checkpoint)
    require_plain_file(prediction)
    initial = read_json(ROOT / f"initial_split{split}.json")
    if (initial.get("split") != split or initial.get("seed") != split
            or initial.get("device") != "cuda:0"
            or initial.get("changed_requires_grad") != [
                "input_be_block.B", "input_be_block.R", "input_be_block.S"]
            or initial.get("trainable_parameters") != 6732432):
        raise ValueError(f"Initial match record malformed for mask {split}")
    for field in ("state_sha256", "member_logits_sha256", "rng_after_constructor_sha256"):
        if not isinstance(initial.get(field), str) or len(initial[field]) != 64:
            raise ValueError(f"Initial match digest malformed: {field}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint is not a state dictionary")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid checkpoint tensor {name}")
    if set(initial.get("input_factor_sha256", {})) != {"R", "S", "B"}:
        raise ValueError("Initial factor digest fields malformed")
    for factor in ("R", "S", "B"):
        value = state[f"input_be_block.{factor}"]
        if hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() != initial["input_factor_sha256"][factor]:
            raise ValueError(f"Frozen input {factor} changed in mask {split}")
    if not bool(torch.all((state["input_be_block.R"] == 1) | (state["input_be_block.R"] == -1))):
        raise ValueError("Input R has values other than initial signs")
    if not bool(torch.all(state["input_be_block.S"] == 1)) or not bool(torch.all(state["input_be_block.B"] == 0)):
        raise ValueError("Input S/B are not initial values")
    with np.load(prediction, allow_pickle=False) as archive:
        if set(archive.files) != PRED_FIELDS:
            raise ValueError("Prediction archive fields changed")
        values = {name: archive[name] for name in PRED_FIELDS}
    n = 5666
    if (values["node_index"].shape != (n,) or values["y_true"].shape != (n,)
            or values["ensemble_pred"].shape != (n,)
            or values["member_pred"].shape != (4, n)
            or values["member_logits"].shape != (4, n, 18)
            or values["ensemble_prob"].shape != (n, 18)):
        raise ValueError("Prediction archive shape changed")
    for name in ("confidence", "degree", "local_homophily"):
        if values[name].shape != (n,):
            raise ValueError(f"Unexpected {name} shape")
    if any(not np.isfinite(values[name]).all() for name in
           ("member_logits", "ensemble_prob", "confidence", "degree", "local_homophily")):
        raise ValueError("Nonfinite prediction archive")
    if data is not None:
        mask = masks[2][:, split]
        expected_indices = mask.nonzero(as_tuple=False).flatten().numpy()
        if not np.array_equal(values["node_index"], expected_indices):
            raise ValueError(f"Wrong official test mask in split {split}")
        if not np.array_equal(values["y_true"], data.y[mask].numpy()):
            raise ValueError(f"Wrong official labels in split {split}")
        src, dst = data.edge_index
        keep = src != dst
        src, dst = src[keep], dst[keep]
        degree = torch.bincount(dst, minlength=data.y.numel()).float()
        same = data.y[src].eq(data.y[dst]).float()
        count = torch.zeros_like(degree)
        count.index_add_(0, dst, same)
        homophily = count / degree.clamp_min(1)
        if not np.array_equal(values["degree"], degree[mask].numpy()):
            raise ValueError("Saved degree differs from graph")
        if not np.allclose(values["local_homophily"], homophily[mask].numpy(), atol=1e-7):
            raise ValueError("Saved homophily differs from graph")
    measured = prediction_metrics(values)
    for field, value in measured.items():
        close(finite_float(row, field), value, field, tol=2e-5)
    close(finite_float(row, "test_loss"), measured["nll"], "test_loss", tol=2e-5)
    if data is not None:
        from types import SimpleNamespace
        args = SimpleNamespace(model="SAGE", num_layers=5, hidden_dim=512, m=4)
        model = make_model(args, "gnnm", 300, 18, torch.device("cpu"))
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            replay_logits = all_logits(model, "gnnm", data, 4)
        if not np.allclose(replay_logits[:, mask].numpy(), values["member_logits"],
                           rtol=2e-4, atol=2e-4):
            raise ValueError(f"Checkpoint replay differs in mask {split}")
        val = masks[1][:, split]
        val_acc = (replay_logits.mean(0)[val].argmax(-1) == data.y[val]).float().mean().item()
        close(finite_float(row, "val_metric"), val_acc, "replayed val accuracy", tol=1e-3)


def check_prefix(count: int, *, replay: bool) -> list[dict[str, str]]:
    if not 0 <= count <= 5:
        raise ValueError("Prefix count must be 0-5")
    check_prepared(require_cpu=True)
    manifest = ROOT / "run_manifest.json"
    if count or manifest.exists():
        if read_json(manifest) != expected_run_manifest():
            raise ValueError("Run manifest changed")
    result = ROOT / "projector_controls.csv"
    if count == 0 and not result.exists():
        rows = []
    else:
        require_plain_file(result)
        with result.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("Control CSV header changed")
            rows = list(reader)
    if len(rows) != count:
        raise ValueError(f"Expected {count} complete rows, found {len(rows)}")
    for directory, suffix in ((ROOT / "checkpoints", ".pt"), (ROOT / "predictions", ".npz")):
        found = set(directory.glob(f"*{suffix}")) if directory.exists() else set()
        expected = {directory / f"roman-empire_SAGE_freeze_input_factors_split{s}{suffix}"
                    for s in range(count)}
        if found != expected:
            raise ValueError(f"Unexpected or missing {suffix} artifact set")
    initial_files = set(ROOT.glob("initial_split*.json"))
    if initial_files != {ROOT / f"initial_split{s}.json" for s in range(count)}:
        raise ValueError("Unexpected or missing initial-match records")
    data = masks = None
    if replay:
        data, train, val, test, _, output_dim, binary = load_dataset(
            "roman-empire", add_self_loops=True, device="cpu", data_dir=str(REPO / "data"))
        if binary or output_dim != 18 or data.x.shape != (22662, 300):
            raise ValueError("Roman Empire dimensions changed")
        masks = (train, val, test)
    for split, row in enumerate(rows):
        check_row(row, split, data, masks)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--prefix-count", type=int)
    group.add_argument("--complete", action="store_true")
    options = parser.parse_args()
    if options.preflight:
        baseline = check_prepared(require_cpu=True)
        check_prefix(0, replay=False)
        print("FROZEN_INPUT_PREFLIGHT_OK baseline_masks=5 cpu_initial_matches=5")
    elif options.complete:
        baseline = check_prepared(require_cpu=True)
        rows = check_prefix(5, replay=True)
        effects = [float(row["test_acc"]) - float(base["test_acc"])
                   for row, base in zip(rows, baseline)]
        print("FROZEN_INPUT_COMPLETE_OK")
        print("CONTROL_MINUS_GNNM", json.dumps(effects))
        print("DESCRIPTIVE_MEAN", sum(effects) / len(effects))
    else:
        check_prefix(options.prefix_count, replay=False)
        print(f"FROZEN_INPUT_PREFIX_OK rows={options.prefix_count}")


if __name__ == "__main__":
    main()
