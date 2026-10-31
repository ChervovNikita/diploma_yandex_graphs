"""Initially matched frozen-input-factor control for Roman Empire SAGE.

The existing projector_controls.train_one supplies the exact optimizer,
validation, checkpoint, and prediction path. This wrapper replaces only the
model factory for the new variant. Nothing here starts a GPU job implicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import projector_controls as controls  # noqa: E402
from datasets import load_dataset  # noqa: E402
from run_common import set_seed  # noqa: E402

ROOT = HERE / "frozen_input_factor_results"
VARIANT = "freeze_input_factors"
FACTORS = ("R", "S", "B")
ORIGINAL_MAKE_MODEL = controls.make_model


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    return digest_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def rng_snapshot(device: torch.device) -> tuple:
    cuda = tuple(x.clone() for x in torch.cuda.get_rng_state_all()) if device.type == "cuda" else ()
    numpy = np.random.get_state()
    return (random.getstate(),
            (numpy[0], numpy[1].copy(), numpy[2], numpy[3], numpy[4]),
            torch.get_rng_state().clone(), cuda)


def same_rng(a: tuple, b: tuple) -> bool:
    return (a[0] == b[0]
            and a[1][0] == b[1][0]
            and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:]
            and torch.equal(a[2], b[2])
            and len(a[3]) == len(b[3])
            and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))


def restore_rng(state: tuple) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(list(state[3]))


def rng_digest(state: tuple) -> str:
    h = hashlib.sha256()
    h.update(repr(state[0]).encode())
    h.update(state[1][0].encode())
    h.update(state[1][1].tobytes())
    h.update(repr(state[1][2:]).encode())
    h.update(state[2].numpy().tobytes())
    for value in state[3]:
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def make_pair(args: SimpleNamespace, data: Data, output_dim: int,
              split: int, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Prove equal starts, then return a model with only input R/S/B frozen."""
    set_seed(split)
    reference = ORIGINAL_MAKE_MODEL(args, "gnnm", data.x.shape[1], output_dim, device)
    reference_rng = rng_snapshot(device)
    set_seed(split)
    candidate = ORIGINAL_MAKE_MODEL(args, "gnnm", data.x.shape[1], output_dim, device)
    candidate_rng = rng_snapshot(device)
    if not same_rng(reference_rng, candidate_rng):
        raise RuntimeError(f"mask {split}: constructor RNG states differ")
    left = reference.state_dict()
    right = candidate.state_dict()
    if left.keys() != right.keys() or not all(torch.equal(left[k], right[k]) for k in left):
        raise RuntimeError(f"mask {split}: initial model states differ")
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        reference_logits = controls.all_logits(reference, "gnnm", data, args.m)
        candidate_logits = controls.all_logits(candidate, "gnnm", data, args.m)
    if not torch.equal(reference_logits, candidate_logits):
        raise RuntimeError(f"mask {split}: initial member logits differ")
    if not same_rng(candidate_rng, rng_snapshot(device)):
        raise RuntimeError(f"mask {split}: eval consumed random state")
    initial_factors = {
        name: tensor_digest(getattr(candidate.input_be_block, name)) for name in FACTORS
    }
    for name in FACTORS:
        getattr(candidate.input_be_block, name).requires_grad_(False)
    ref_flags = {name: p.requires_grad for name, p in reference.named_parameters()}
    candidate_flags = {name: p.requires_grad for name, p in candidate.named_parameters()}
    expected_changed = {f"input_be_block.{name}" for name in FACTORS}
    changed = {name for name in ref_flags if ref_flags[name] != candidate_flags[name]}
    if changed != expected_changed or any(candidate_flags[name] for name in changed):
        raise RuntimeError(f"mask {split}: incorrect frozen parameter set: {changed}")
    h = hashlib.sha256()
    for name, value in right.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    record = {
        "split": split,
        "seed": split,
        "device": str(device),
        "state_sha256": h.hexdigest(),
        "member_logits_sha256": tensor_digest(candidate_logits),
        "rng_after_constructor_sha256": rng_digest(candidate_rng),
        "input_factor_sha256": initial_factors,
        "changed_requires_grad": sorted(changed),
        "trainable_parameters": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
    }
    del reference, reference_logits, candidate_logits
    restore_rng(candidate_rng)
    return candidate, record


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def args_for(*, dataset: str = "roman-empire", depth: int = 5,
             width: int = 512, steps: int = 5000) -> SimpleNamespace:
    return SimpleNamespace(dataset=dataset, model="SAGE", num_layers=depth,
                           hidden_dim=width, lr=3e-5, m=4, num_steps=steps,
                           common_backbone_init=False)


def preflight() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES='' ")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight found initialized CUDA")
    from verify_frozen_input_factors import check_static_inputs
    check_static_inputs()
    device = torch.device("cpu")
    data, train, val, test, _, output_dim, binary = load_dataset(
        "roman-empire", add_self_loops=True, device=device, data_dir=str(REPO / "data")
    )
    if binary or data.x.shape[0] != 22662 or output_dim != 18:
        raise RuntimeError("Roman Empire data dimensions changed")
    records = []
    started = time.monotonic()
    for split in range(5):
        data.train_mask = train[:, split]
        data.val_mask = val[:, split]
        data.test_mask = test[:, split]
        model, record = make_pair(args_for(), data, output_dim, split, device)
        records.append(record)
        del model
        print(f"CPU_INITIAL_MATCH split={split} OK", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight initialized CUDA")
    result = {"schema": 1, "source": "real Roman Empire graph",
              "elapsed_seconds": time.monotonic() - started,
              "matches": records}
    atomic_json(ROOT / "cpu_initial_match.json", result)
    print("CPU_INITIAL_MATCH_ALL_FIVE_OK", flush=True)


def smoke() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke requires masked CUDA")
    from verify_frozen_input_factors import check_static_inputs
    check_static_inputs()
    set_seed(9217)
    n, features, classes = 30, 7, 3
    x = torch.randn(n, features)
    y = torch.arange(n) % classes
    src = torch.arange(n)
    edge_index = torch.stack((torch.cat((src, src)),
                              torch.cat((src.roll(1), src.roll(-1)))))
    data = Data(x=x, y=y, edge_index=edge_index)
    masks = []
    for lo, hi in ((0, 18), (18, 24), (24, 30)):
        mask = torch.zeros(n, 1, dtype=torch.bool)
        mask[lo:hi, 0] = True
        masks.append(mask)
    args = args_for(dataset="cpu-smoke", depth=1, width=16, steps=2)
    root = ROOT / "cpu_smoke"
    root.mkdir(exist_ok=False)
    original_factory = controls.make_model
    capture = {}

    def factory(factory_args, variant, input_dim, classes_arg, factory_device):
        if variant != VARIANT:
            raise RuntimeError(f"unexpected variant {variant}")
        model, capture["initial"] = make_pair(factory_args, data, classes_arg, 0,
                                                factory_device)
        return model

    controls.make_model = factory
    try:
        row = controls.train_one(args, VARIANT, 0, data, tuple(masks), classes,
                                 False, torch.device("cpu"), root)
    finally:
        controls.make_model = original_factory
    ckpt = Path(row["checkpoint"])
    prediction = Path(row["prediction_file"])
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    restored = original_factory(args, "gnnm", features, classes, torch.device("cpu"))
    restored.load_state_dict(state)
    restored.eval()
    data.test_mask = masks[2][:, 0]
    with torch.no_grad():
        replay = controls.all_logits(restored, "gnnm", data, 4)[:, data.test_mask]
    with np.load(prediction, allow_pickle=False) as saved:
        if not np.array_equal(replay.numpy(), saved["member_logits"]):
            raise RuntimeError("CPU smoke checkpoint/prediction roundtrip mismatch")
        if not np.array_equal(saved["ensemble_pred"],
                              saved["member_logits"].mean(axis=0).argmax(axis=-1)):
            raise RuntimeError("CPU smoke pooled predictions mismatch")
    for name in FACTORS:
        if tensor_digest(state[f"input_be_block.{name}"]) != capture["initial"]["input_factor_sha256"][name]:
            raise RuntimeError(f"CPU smoke changed frozen {name}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke initialized CUDA")
    atomic_json(root / "smoke_report.json", {
        "schema": 1, "split": 0, "num_steps": 2,
        "best_step": row["best_step"], "trainable_parameters": row["num_params"],
        "initial": capture["initial"], "checkpoint_sha256": digest_bytes(ckpt.read_bytes()),
        "prediction_sha256": digest_bytes(prediction.read_bytes()),
        "roundtrip": "exact",
    })
    print("CPU_SMOKE_TRAIN_ARTIFACT_ROUNDTRIP_OK", flush=True)


def run_split(split: int) -> None:
    from verify_frozen_input_factors import check_prepared, check_prefix
    check_prepared(require_cpu=True)
    check_prefix(split, replay=False)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    data, train, val, test, _, output_dim, binary = load_dataset(
        "roman-empire", add_self_loops=True, device=device, data_dir=str(REPO / "data")
    )
    if binary or data.x.shape[0] != 22662 or output_dim != 18:
        raise RuntimeError("Roman Empire data dimensions changed")
    args = args_for()
    original_factory = controls.make_model

    def factory(factory_args, variant, input_dim, classes_arg, factory_device):
        if variant != VARIANT or input_dim != data.x.shape[1] or classes_arg != output_dim:
            raise RuntimeError("unexpected model factory call")
        model, record = make_pair(factory_args, data, classes_arg, split, factory_device)
        atomic_json(ROOT / f"initial_split{split}.json", record)
        return model

    controls.make_model = factory
    try:
        row = controls.train_one(args, VARIANT, split, data, (train, val, test),
                                 output_dim, binary, device,
                                 Path("experiments_iclr/frozen_input_factor_results"))
    finally:
        controls.make_model = original_factory
    result_file = ROOT / "projector_controls.csv"
    with result_file.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=controls.FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    print("RESULT", json.dumps(row, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-preflight", action="store_true")
    group.add_argument("--cpu-smoke", action="store_true")
    group.add_argument("--split", type=int, choices=range(5))
    options = parser.parse_args()
    os.chdir(REPO)
    if options.cpu_preflight:
        preflight()
    elif options.cpu_smoke:
        smoke()
    else:
        run_split(options.split)


if __name__ == "__main__":
    main()
