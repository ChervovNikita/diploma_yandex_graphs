"""Five-mask Roman Empire SAGE control with BatchEnsemble factors in every linear map.

This is a post hoc, fixed-protocol control. It reuses the training, pooled
validation selection, and test export in projector_controls.py, but writes to
its own result root. It can start after the original queue succeeds and the
optional GAT launcher exits. No GPU work is started by importing this module.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import projector_controls as controls  # noqa: E402
from ablation.models_ablation import TABMAblationModel  # noqa: E402
from models import SAGEModule  # noqa: E402


VARIANT = "all_layer_be"
OFFICIAL_MASKS = tuple(range(5))
RESULT_REL = Path("experiments_iclr/all_layer_be_sage_results")
RESULT_ROOT = REPO / RESULT_REL
RESULT_CSV = RESULT_ROOT / "projector_controls.csv"
HASHES_PATH = RESULT_ROOT / "artifact_hashes.json"
MANIFEST_PATH = RESULT_ROOT / "protocol.json"
IDENTITY_PATH = RESULT_ROOT / "initialization_audit.json"
GAT_ROOT = REPO / "experiments_iclr/gat_failure_pair_results"
SAGE_CSV = REPO / "experiments_iclr/results/projector_controls.csv"
SPLIT0_AUDIT = REPO / "experiments_iclr/results/gnnm_split0_verification_audit.json"
BASE_PARAMS = 6_737_728
ADDED_PARAMS = 133_120
TOTAL_PARAMS = BASE_PARAMS + ADDED_PARAMS
SOURCES = (
    "experiments_iclr/all_layer_be_sage.py",
    "experiments_iclr/projector_controls.py",
    "experiments_iclr/verify_control_artifacts.py",
    "experiments_iclr/verify_gat_failure_pair.py",
    "ablation/models_ablation.py",
    "models.py",
    "datasets.py",
    "run_common.py",
    "data/roman_empire.npz",
)
PROTOCOL = {
    "purpose": "Post hoc all-linear-layer BatchEnsemble SAGE control",
    "dataset": "roman-empire",
    "backbone": "SAGE",
    "official_masks": list(OFFICIAL_MASKS),
    "seed_rule": "seed equals official mask index",
    "variants": [VARIANT],
    "num_layers": 5,
    "hidden_dim": 512,
    "dropout": 0.2,
    "members": 4,
    "learning_rate": 3e-5,
    "weight_decay": 0,
    "max_steps": 5000,
    "validation": "step 1 and every 10 steps; pooled logit accuracy",
    "early_stop": "300 steps without pooled validation accuracy improvement",
    "train_objective": "mean of four member cross entropies",
    "selection": "one joint checkpoint at maximum pooled validation accuracy; earliest tie",
    "factor_placement": "input/output projectors plus SAGE neighbor/root and both FFN linear maps in each residual block",
    "factor_initialization": "hidden R=1, S=1, B=0; original shared weights retained",
    "result_root": str(RESULT_REL),
    "base_trainable_parameters": BASE_PARAMS,
    "added_trainable_parameters": ADDED_PARAMS,
    "total_trainable_parameters": TOTAL_PARAMS,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selected_sage_controls() -> None:
    """Require the complete selected-control artifact audit before freezing it."""
    verifier = REPO / "experiments_iclr/verify_control_artifacts.py"
    completed = subprocess.run(
        [sys.executable, str(verifier), "--require-complete",
         "--check-checkpoints"],
        cwd=REPO,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
             "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=1200,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"Selected SAGE control artifact verification failed: {detail}")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class MemberFactorLinear(nn.Module):
    """Member-specific rank-one factors around an existing shared affine map."""

    def __init__(self, shared: nn.Module, members: int):
        super().__init__()
        weight = getattr(shared, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError("Expected a linear map with a two-dimensional weight")
        outputs, inputs = weight.shape
        self.shared = shared
        self.R = nn.Parameter(weight.new_ones((members, inputs)))
        self.S = nn.Parameter(weight.new_ones((members, outputs)))
        self.B = nn.Parameter(weight.new_zeros((members, outputs)))
        self.members = members
        self.active_member: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        member = self.active_member
        if member is None or not 0 <= member < self.members:
            raise RuntimeError("Select a BatchEnsemble member before forwarding")
        return self.shared(x * self.R[member]) * self.S[member] + self.B[member]

    @property
    def added_parameters(self) -> int:
        return self.R.numel() + self.S.numel() + self.B.numel()


class AllLayerBESAGEModel(nn.Module):
    """Original projector-only GNNM with identity-initialized hidden factors."""

    def __init__(self, args: SimpleNamespace, input_dim: int,
                 output_dim: int, device: torch.device):
        super().__init__()
        if args.model != "SAGE" or args.m != 4:
            raise ValueError("This control is defined for four-member SAGE only")
        self.base = TABMAblationModel(
            model_name="SAGE", num_layers=args.num_layers,
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            output_dim=output_dim, hidden_dim_multiplier=1, num_heads=8,
            normalization="LayerNorm", dropout=0.2, tabm_inits=args.m,
            init_scheme="default", device=device,
        )
        factors: list[MemberFactorLinear] = []
        if len(self.base.residual_modules) != args.num_layers:
            raise RuntimeError("Unexpected SAGE residual-block count")
        for residual in self.base.residual_modules:
            sage = residual.module
            if not isinstance(sage, SAGEModule):
                raise RuntimeError("Unexpected residual module in SAGE model")
            for owner, name in (
                (sage.conv, "lin_l"), (sage.conv, "lin_r"),
                (sage.feed_forward_module, "linear_1"),
                (sage.feed_forward_module, "linear_2"),
            ):
                wrapped = MemberFactorLinear(getattr(owner, name), args.m)
                setattr(owner, name, wrapped)
                factors.append(wrapped)
        # The modules are already registered inside self.base; this plain list
        # only routes the current member through the unmodified backbone call.
        self._factor_layers = factors
        if len(factors) != 4 * args.num_layers:
            raise RuntimeError("A hidden linear map was not factorized")

    @property
    def added_parameters(self) -> int:
        return sum(layer.added_parameters for layer in self._factor_layers)

    def forward(self, graph, x: torch.Tensor, member: int) -> torch.Tensor:
        if not 0 <= member < 4:
            raise ValueError(member)
        for layer in self._factor_layers:
            layer.active_member = member
        return self.base(graph, x, tabm_seed=member)


def fixed_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset="roman-empire", model="SAGE", num_layers=5,
        hidden_dim=512, lr=3e-5, m=4, num_steps=5000,
        common_backbone_init=False,
    )


def identity_check(split: int, args: SimpleNamespace | None = None,
                   input_dim: int = 300, output_dim: int = 18) -> dict:
    """Require exact initial logits from GNNM and the factorized control."""
    args = fixed_args() if args is None else args
    graph = SimpleNamespace(edge_index=torch.tensor(
        [list(range(11)) + list(range(1, 12)),
         list(range(1, 12)) + list(range(11))], dtype=torch.long))
    x = torch.linspace(-1, 1, 12 * input_dim).reshape(12, input_dim)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(split)
        reference = TABMAblationModel(
            model_name="SAGE", num_layers=args.num_layers,
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            output_dim=output_dim, hidden_dim_multiplier=1, num_heads=8,
            normalization="LayerNorm", dropout=0.2, tabm_inits=args.m,
            init_scheme="default", device="cpu",
        ).eval()
        reference_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(split)
        candidate = AllLayerBESAGEModel(
            args, input_dim, output_dim, torch.device("cpu")).eval()
        if not torch.equal(reference_rng, torch.random.get_rng_state()):
            raise RuntimeError(f"Model construction changed RNG state on mask {split}")
        base_count = sum(p.numel() for p in reference.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
        if base_count + candidate.added_parameters != total_count:
            raise RuntimeError("Factor parameter accounting is inconsistent")
        for member in range(args.m):
            with torch.no_grad():
                original = reference(graph, x, tabm_seed=member)
                factored = candidate(graph, x, member)
            if not torch.equal(original, factored):
                delta = float((original - factored).abs().max())
                raise RuntimeError(
                    f"Initial logits differ on mask {split}, member {member}: {delta}")
    return {
        "mask": split, "members_checked": args.m,
        "initial_logit_max_abs_difference": 0.0,
        "post_construction_rng_matches": True,
        "base_trainable_parameters": base_count,
        "added_trainable_parameters": candidate.added_parameters,
        "total_trainable_parameters": total_count,
    }


def require_queue_finished_and_gat_idle() -> dict:
    """Require queue completion and no active GAT study; record its outcome."""
    from verify_gat_failure_pair import queue_terminal_marker

    queue_marker = queue_terminal_marker()
    processes = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True,
    )
    active_gat = []
    for line in processes.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        command = parts[1]
        launcher = re.search(r"run_gat_failure_pair\.sh(?:\s|$)", command)
        training = ("projector_controls.py" in command and
                    re.search(r"--model\s+GAT(?:\s|$)", command))
        if launcher or training:
            active_gat.append(parts[0])
    if active_gat:
        raise RuntimeError(f"GAT launcher or GPU training is still active: {active_gat}")
    gat_log = GAT_ROOT / "run.log"
    gat_csv = GAT_ROOT / "projector_controls.csv"
    gat_marker = (gat_log.is_file() and
                  gat_log.read_text().rstrip().endswith("GAT_PAIR_COMPLETE"))
    verification_exit_code = None
    if gat_marker:
        result = subprocess.run(
            [sys.executable, str(REPO / "experiments_iclr/verify_gat_failure_pair.py"),
             "--complete"], capture_output=True, text=True,
        )
        verification_exit_code = result.returncode
    gat_rows = None
    if gat_csv.is_file():
        try:
            with gat_csv.open(newline="") as stream:
                gat_rows = sum(1 for _ in csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error):
            pass
    return {
        "queue_success_marker": queue_marker,
        "queue_log_sha256": sha256(REPO / "experiments_iclr/logs/run_after_sage.log"),
        "gat_status": ("complete" if gat_marker and verification_exit_code == 0
                       else "incomplete_or_failed_after_launcher_exit"),
        "gat_final_marker_present": bool(gat_marker),
        "gat_verification_exit_code": verification_exit_code,
        "gat_rows_observed": gat_rows,
        "gat_protocol_sha256": (sha256(GAT_ROOT / "protocol.json")
                                if (GAT_ROOT / "protocol.json").is_file() else None),
        "gat_results_sha256": sha256(gat_csv) if gat_csv.is_file() else None,
        "gat_run_log_sha256": sha256(gat_log) if gat_log.is_file() else None,
    }


@contextmanager
def hold_gat_run_lock():
    """Exclude a later GAT launcher for the entire all-layer GPU run."""
    GAT_ROOT.mkdir(parents=True, exist_ok=True)
    if not GAT_ROOT.resolve().is_relative_to(REPO.resolve()):
        raise RuntimeError("GAT lock path escaped the repository")
    with (GAT_ROOT / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("The GAT study holds its run lock") from exc
        yield


def completed_sage_comparators() -> dict:
    """Freeze the completed paired SAGE matrix and split-0 adoption decision."""
    if not SAGE_CSV.is_file() or not SPLIT0_AUDIT.is_file():
        raise FileNotFoundError("Completed SAGE controls or split-0 adoption audit missing")
    required = {(variant, split) for variant in (
        "gnnm", "ens_pooled", "untied_backbone", "base")
        for split in OFFICIAL_MASKS}
    seen: set[tuple[str, int]] = set()
    with SAGE_CSV.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != controls.FIELDS:
            raise ValueError("Unexpected completed SAGE comparison columns")
        for row in reader:
            if row["model"] != "SAGE" or row["dataset"] != "roman-empire":
                continue
            key = (row["variant"], int(row["split"]))
            if key in required:
                if key in seen:
                    raise ValueError(f"Duplicate SAGE comparator {key}")
                seen.add(key)
    missing = required - seen
    if missing:
        raise ValueError(f"Incomplete SAGE comparator matrix: {sorted(missing)}")
    audit = json.loads(SPLIT0_AUDIT.read_text())
    if audit.get("adopted") not in ("pilot", "fresh"):
        raise ValueError("Split-0 adoption audit has no valid decision")
    return {
        "sage_controls_csv": str(SAGE_CSV.relative_to(REPO)),
        "sage_controls_csv_sha256": sha256(SAGE_CSV),
        "split0_adoption_audit": str(SPLIT0_AUDIT.relative_to(REPO)),
        "split0_adoption_audit_sha256": sha256(SPLIT0_AUDIT),
        "split0_adopted": audit["adopted"],
        "paired_variants": ["gnnm", "ens_pooled", "untied_backbone", "base"],
        "official_masks": list(OFFICIAL_MASKS),
    }


def expected_manifest(upstream: dict) -> dict:
    return {
        "schema": 1,
        "protocol": PROTOCOL,
        "upstream": upstream,
        "sage_comparators": completed_sage_comparators(),
        "source_sha256": {relative: sha256(REPO / relative) for relative in SOURCES},
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_geometric_version": __import__("torch_geometric").__version__,
    }


def ensure_manifest(expected: dict) -> None:
    if MANIFEST_PATH.is_file():
        if json.loads(MANIFEST_PATH.read_text()) != expected:
            raise RuntimeError("Frozen all-layer protocol or source differs on resume")
    else:
        if RESULT_CSV.exists() or HASHES_PATH.exists():
            raise RuntimeError("Results exist without a frozen protocol")
        write_json_atomic(MANIFEST_PATH, expected)


def verify_frozen_manifest(expected: dict) -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Frozen all-layer protocol is missing")
    if json.loads(MANIFEST_PATH.read_text()) != expected:
        raise RuntimeError("Frozen all-layer protocol or source differs")


def expected_artifact(split: int, kind: str) -> tuple[str, Path]:
    directory, suffix = (("checkpoints", ".pt") if kind == "checkpoint"
                         else ("predictions", ".npz"))
    relative = RESULT_REL / directory / f"roman-empire_SAGE_{VARIANT}_split{split}{suffix}"
    return str(relative), REPO / relative


def validate_artifact(row: dict[str, str], split: int, kind: str,
                      hashes: dict[str, dict[str, str]] | None) -> Path:
    recorded, path = expected_artifact(split, kind)
    if row[kind] != recorded or not path.resolve().is_relative_to(RESULT_ROOT.resolve()):
        raise ValueError(f"Unexpected {kind} path in mask {split}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {kind}: {path}")
    if hashes is not None:
        if hashes.get(str(split), {}).get(kind) != sha256(path):
            raise ValueError(f"Changed {kind} on resumed mask {split}")
    return path


def validate_row(row: dict[str, str], split: int,
                 hashes: dict[str, dict[str, str]] | None = None) -> None:
    if None in row or any(value is None for value in row.values()):
        raise ValueError("Truncated or extended results row")
    if (row["dataset"], row["model"], row["variant"]) != (
            "roman-empire", "SAGE", VARIANT):
        raise ValueError("Unexpected result identity")
    if (int(row["split"]) != split or int(row["seed"]) != split
            or int(row["num_layers"]) != 5 or int(row["hidden_dim"]) != 512
            or int(row["m"]) != 4 or int(row["num_steps"]) != 5000
            or int(row["num_params"]) != TOTAL_PARAMS
            or not math.isclose(float(row["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"Mixed protocol in mask {split}")
    step = int(row["best_step"])
    if not 1 <= step <= 5000 or (step != 1 and step % 10):
        raise ValueError(f"Invalid selected step in mask {split}")
    for field in ("train_seconds", "val_metric", "test_metric", "test_acc",
                  "test_loss", "mean_member_acc", "pair_disagreement"):
        if not math.isfinite(float(row[field])):
            raise ValueError(f"Nonfinite {field} in mask {split}")
    if float(row["train_seconds"]) <= 0:
        raise ValueError(f"Invalid training duration in mask {split}")
    checkpoint = validate_artifact(row, split, "checkpoint", hashes)
    prediction = validate_artifact(row, split, "prediction_file", hashes)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
            for value in state.values()):
        raise ValueError(f"Invalid checkpoint in mask {split}")
    if sum(value.numel() for value in state.values()) != TOTAL_PARAMS:
        raise ValueError(f"Unexpected checkpoint parameter count in mask {split}")
    with np.load(prediction, allow_pickle=False) as archive:
        if set(archive.files) != {
                "node_index", "y_true", "ensemble_pred", "member_pred",
                "member_logits", "ensemble_prob", "confidence", "degree",
                "local_homophily"}:
            raise ValueError(f"Unexpected prediction schema in mask {split}")
        n = int(row["n_test"])
        logits = archive["member_logits"]
        labels = archive["y_true"]
        if n != 5666 or logits.shape != (4, n, 18) or labels.shape != (n,):
            raise ValueError(f"Unexpected prediction shape in mask {split}")
        if (archive["member_pred"].shape != (4, n)
                or archive["ensemble_pred"].shape != (n,)
                or archive["ensemble_prob"].shape != (n, 18)):
            raise ValueError(f"Unexpected prediction array shape in mask {split}")
        if any(archive[field].shape != (n,) for field in (
                "node_index", "confidence", "degree", "local_homophily")):
            raise ValueError(f"Unexpected diagnostic array shape in mask {split}")
        if (not np.isfinite(logits).all()
                or not np.isfinite(archive["ensemble_prob"]).all()):
            raise ValueError(f"Nonfinite predictions in mask {split}")
        pooled = logits.mean(axis=0).argmax(axis=-1)
        member_predictions = logits.argmax(axis=-1)
        if not np.array_equal(pooled, archive["ensemble_pred"]) or not np.array_equal(
                member_predictions, archive["member_pred"]):
            raise ValueError(f"Saved predictions disagree with logits in mask {split}")
        accuracy = float(np.mean(pooled == labels))
        member_accuracy = float(np.mean(member_predictions == labels[None, :]))
        if abs(accuracy - float(row["test_metric"])) > 1e-6 or abs(
                member_accuracy - float(row["mean_member_acc"])) > 1e-6:
            raise ValueError(f"Recorded accuracy disagrees with predictions in mask {split}")


def validate_resume(require_complete: bool = False) -> tuple[list[dict[str, str]], dict]:
    hashes = json.loads(HASHES_PATH.read_text()) if HASHES_PATH.is_file() else {}
    if not isinstance(hashes, dict):
        raise ValueError("Invalid artifact hash manifest")
    if RESULT_CSV.is_file():
        with RESULT_CSV.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != controls.FIELDS:
                raise ValueError("Unexpected result CSV columns")
            rows = list(reader)
    else:
        rows = []
    if len(rows) > len(OFFICIAL_MASKS):
        raise ValueError("More than five result rows")
    for split, row in enumerate(rows):
        validate_row(row, split, hashes)
    if require_complete and len(rows) != len(OFFICIAL_MASKS):
        raise ValueError(f"Expected five masks, found {len(rows)}")
    return rows, hashes


def install_model_factory() -> None:
    original = controls.make_model

    def make_model(args, variant, input_dim, output_dim, device):
        if variant == VARIANT:
            return AllLayerBESAGEModel(args, input_dim, output_dim, device).to(device)
        return original(args, variant, input_dim, output_dim, device)

    controls.make_model = make_model


def append_result(row: dict, split: int, hashes: dict) -> None:
    validate_row(row, split)
    hashes[str(split)] = {
        kind: sha256(expected_artifact(split, kind)[1])
        for kind in ("checkpoint", "prediction_file")
    }
    # Hashes are committed first. An interruption before the CSV append leaves
    # an orphan hash entry; the incomplete mask is simply rerun on resume.
    write_json_atomic(HASHES_PATH, hashes)
    with RESULT_CSV.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=controls.FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args_cli = parser.parse_args()
    if args_cli.require_complete and not args_cli.verify_only:
        parser.error("--require-complete is for --verify-only")
    os.chdir(REPO)
    if args_cli.verify_only:
        if not RESULT_ROOT.resolve().is_relative_to(REPO.resolve()):
            raise RuntimeError("Result root escaped the repository")
        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError("Frozen all-layer protocol is missing")
        verify_selected_sage_controls()
        manifest = expected_manifest(require_queue_finished_and_gat_idle())
        verify_frozen_manifest(manifest)
        rows, _ = validate_resume(args_cli.require_complete)
        print(json.dumps({"verified_masks": len(rows),
                          "complete": len(rows) == 5}, sort_keys=True))
        return

    # Check the queue before creating either lock file. Holding the same lock
    # used by the GAT launcher then excludes any later GAT GPU start.
    from verify_gat_failure_pair import queue_terminal_marker
    queue_terminal_marker()
    with hold_gat_run_lock():
        upstream = require_queue_finished_and_gat_idle()
        verify_selected_sage_controls()
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        if not RESULT_ROOT.resolve().is_relative_to(REPO.resolve()):
            raise RuntimeError("Result root escaped the repository")
        with (RESULT_ROOT / ".run.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = expected_manifest(upstream)
            if MANIFEST_PATH.is_file():
                verify_frozen_manifest(manifest)
            elif RESULT_CSV.exists() or HASHES_PATH.exists():
                raise RuntimeError("Results exist without a frozen protocol")
            rows, hashes = validate_resume()
            checks = [identity_check(split) for split in OFFICIAL_MASKS]
            if any(check["base_trainable_parameters"] != BASE_PARAMS or
                   check["added_trainable_parameters"] != ADDED_PARAMS or
                   check["total_trainable_parameters"] != TOTAL_PARAMS
                   for check in checks):
                raise RuntimeError("Predeclared parameter count differs from the model")
            identity_audit = {"schema": 1, "checks": checks,
                              "source_sha256": manifest["source_sha256"]}
            if IDENTITY_PATH.is_file() and json.loads(
                    IDENTITY_PATH.read_text()) != identity_audit:
                raise RuntimeError("Initialization audit changed on resume")
            if not torch.cuda.is_available():
                raise RuntimeError("The fixed five-mask training protocol requires CUDA")
            # A failed CPU or CUDA preflight cannot leave a frozen empty run.
            ensure_manifest(manifest)
            if not IDENTITY_PATH.is_file():
                write_json_atomic(IDENTITY_PATH, identity_audit)
            device = torch.device("cuda:0")
            data, train_masks, valid_masks, test_masks, _, output_dim, is_binary = \
                controls.load_dataset("roman-empire", add_self_loops=True,
                                      device=device, data_dir="data")
            if data.x.shape[1] != 300 or output_dim != 18 or is_binary:
                raise RuntimeError("Roman Empire feature or class dimensions changed")
            masks = (train_masks, valid_masks, test_masks)
            train_args = fixed_args()
            install_model_factory()
            print("CONFIG", json.dumps(manifest, sort_keys=True), flush=True)
            for split in OFFICIAL_MASKS[len(rows):]:
                if expected_manifest(require_queue_finished_and_gat_idle()) != manifest:
                    raise RuntimeError("Source, data, or upstream results changed during run")
                row = controls.train_one(train_args, VARIANT, split, data, masks,
                                         output_dim, is_binary, device, RESULT_REL)
                append_result(row, split, hashes)
                print("RESULT", json.dumps(row, sort_keys=True), flush=True)
            validate_resume(require_complete=True)
            print("ALL_LAYER_BE_SAGE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
