# Copyright 2019 PaddlePaddle Authors. Licensed under Apache-2.0; see LICENSE.
"""Disabled full-schedule WikiCS reference runner; imports stdlib before release.

Uses only exact TRAIN-only and VALID-only safe arrays supplied by root. No author
dataset loader, TEST interface, source grid, resume, retry or remote operation.
Derived from the pinned Apache-2.0 PGL driver. This Torch/VALID-only adaptation
does not claim equivalent legacy Paddle execution; see ATTRIBUTION.md and LICENSE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from unimp_v2 import ATTENTION_MASK_POLICY, author_topology, build_model


AUTHORIZED_HOST = "anogena-2-0"
AUTHORIZED_GPU_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
AUTHORIZED_SSH_DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
AUTHORIZED_RESEARCH_ROOT = Path(
    "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/"
    "experiments_iclr/postsubmission_20260930"
)


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def verify_route():
    """Verify the literal authorized allocation before metadata/numerical access."""
    if socket.gethostname() != AUTHORIZED_HOST:
        raise RuntimeError("Wrong allocation hostname; excluded before metadata access")
    observed = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    uuids = [line.strip() for line in observed.stdout.splitlines() if line.strip()]
    if uuids != [AUTHORIZED_GPU_UUID]:
        raise RuntimeError("The sole physical GPU must match the literal authorized UUID")
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility is not None and visibility.strip() != AUTHORIZED_GPU_UUID:
        raise RuntimeError("CUDA visibility must use the authorized literal UUID, not an index/name")
    return {"hostname": AUTHORIZED_HOST, "sole_nvidia_smi_uuid": uuids[0]}


def confined_path(value, *, existing=True):
    """Constrain explicit study paths; ordinary runtime caches remain untouched."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Missing explicit absolute study path")
    path = Path(value)
    if not path.is_absolute() or not within(path, AUTHORIZED_RESEARCH_ROOT):
        raise ValueError("Study path must be inside the authorized research root")
    root = AUTHORIZED_RESEARCH_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=existing)
    if not within(resolved, root):
        raise ValueError("Resolved study path escapes the authorized research root")
    if existing and not resolved.is_file():
        raise ValueError("Existing input/metadata must be a regular file")
    return resolved


def fresh_output(value):
    output = confined_path(value, existing=False)
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a fresh directory, never an existing artifact")
    source = Path(__file__).resolve().parent
    predecessor = source.parent / "unimp_v2_torch_reference_source_20261010_v1"
    for packet in (source, predecessor):
        if within(output, packet) or within(packet, output):
            raise ValueError("Output cannot overwrite or extend sealed source packets")
    return output


def digest(path):
    return hashlib.sha256(confined_path(path).read_bytes()).hexdigest()


def check_release(freeze_path, release_path, output_path):
    route = verify_route()
    freeze_file = confined_path(freeze_path)
    release_file = confined_path(release_path)
    output = fresh_output(output_path)
    if freeze_file.suffix != ".json" or release_file.suffix != ".json":
        raise ValueError("Freeze/release must be explicit JSON records")
    freeze = json.loads(freeze_file.read_text())
    release = json.loads(release_file.read_text())
    required_true = ["execution_authorized", "positive_complete_retrieval_decision_adopted",
                     "Torch_operator_numerically_qualified", "full_input_resource_replay_qualified",
                     "float_mask_adaptation_adopted"]
    if any(release.get(k) is not True for k in required_true):
        raise RuntimeError("Root release and complete numerical/resource qualification required")
    for key in ["positive_complete_retrieval_decision_binding", "numerical_qualification_binding",
                "full_input_resource_replay_binding"]:
        binding = release.get(key)
        if not isinstance(binding, dict) or not binding.get("path") or not binding.get("sha256"):
            raise RuntimeError("Missing root adoption receipt: " + key)
        receipt_path = confined_path(binding["path"])
        if digest(receipt_path) != binding["sha256"]:
            raise RuntimeError("Root adoption receipt byte identity mismatch: " + key)
    if release.get("freeze_sha256") != digest(freeze_file):
        raise RuntimeError("Freeze byte identity mismatch")
    if release.get("model_source_sha256") != digest(Path(__file__).with_name("unimp_v2.py")):
        raise RuntimeError("Model source identity mismatch")
    if release.get("runner_source_sha256") != digest(__file__):
        raise RuntimeError("Runner source identity mismatch")
    if freeze.get("attention_mask_policy") != ATTENTION_MASK_POLICY:
        raise RuntimeError("Unadopted mask policy")
    if freeze.get("runtime_host") != AUTHORIZED_HOST or freeze.get("gpu_uuid") != AUTHORIZED_GPU_UUID:
        raise RuntimeError("Literal root-bound runtime host mismatch")
    if freeze.get("ssh_destination") != AUTHORIZED_SSH_DESTINATION or freeze.get("ssh_port") != 2222:
        raise RuntimeError("Freeze must retain the explicit authorized anogena-2 route")
    if fresh_output(freeze.get("output_path")) != output:
        raise RuntimeError("CLI output differs from the frozen fresh destination")
    if freeze.get("seeds") != [7301, 7403, 7507] or freeze.get("epochs") != 1500:
        raise RuntimeError("The fixed complete reference roster/horizon changed")
    for role in ["train", "valid"]:
        binding = freeze[role]
        role_path = confined_path(binding["path"])
        if role_path.suffix != ".npz":
            raise ValueError("Role inputs must be root-bound safe NPZ files")
        if digest(role_path) != binding["sha256"]:
            raise RuntimeError(role + " safe-array byte identity mismatch")
        binding["path"] = str(role_path)
    if freeze["train"]["path"] == freeze["valid"]["path"]:
        raise ValueError("TRAIN and VALID must be separate role-restricted files")
    return freeze, output, route, freeze_file


def cpu_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freeze, output, route, freeze_file = check_release(args.freeze, args.release, args.output)
    # Only an adopted release reaches numerical imports or payload loading.
    import numpy as np
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One root-bound visible GPU required")
    if freeze.get("torch_version") != torch.__version__:
        raise RuntimeError("Root-qualified Torch provider version changed")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with np.load(freeze["train"]["path"], allow_pickle=False) as saved:
        if set(saved.files) != {"x", "edge_index", "ids", "y"}:
            raise ValueError("TRAIN safe arrays must contain only x/edge_index/ids/y")
        train = {key: saved[key].copy() for key in saved.files}
    with np.load(freeze["valid"]["path"], allow_pickle=False) as saved:
        if set(saved.files) != {"ids", "y"}:
            raise ValueError("VALID safe arrays must contain only ids/y")
        valid = {key: saved[key].copy() for key in saved.files}
    if train["x"].shape != (11701, 300) or train["x"].dtype != np.float32 or not np.isfinite(train["x"]).all():
        raise ValueError("Frozen original WikiCS feature shape/dtype changed")
    for role, rows in [(train, 580), (valid, 5274)]:
        if role["ids"].shape != (rows,) or role["y"].shape != (rows,):
            raise ValueError("Frozen complete-role shape changed")
        if role["ids"].dtype != np.int64 or role["y"].dtype != np.int64:
            raise ValueError("IDs and labels must be int64")
        if len(np.unique(role["ids"])) != rows or role["ids"].min() < 0 or role["ids"].max() >= 11701:
            raise ValueError("Invalid role identities")
        if role["y"].min() < 0 or role["y"].max() >= 10:
            raise ValueError("TRAIN-schema class mismatch")
    if np.intersect1d(train["ids"], valid["ids"]).size:
        raise ValueError("TRAIN/VALID overlap")
    if train["edge_index"].dtype != np.int64 or train["edge_index"].ndim != 2 or train["edge_index"].shape[0] != 2:
        raise ValueError("Invalid factual edge shape/dtype")
    device = torch.device("cuda:0")
    topology = author_topology(torch, torch.from_numpy(train["x"]).to(device),
                               torch.from_numpy(train["edge_index"]).to(device))
    train_ids = torch.from_numpy(train["ids"]).to(device)
    train_y = torch.from_numpy(train["y"]).to(device)
    valid_ids = torch.from_numpy(valid["ids"]).to(device)
    valid_y = torch.from_numpy(valid["y"]).to(device)
    # VALID truth has no route into the model or optimizer loss.
    results = []
    for seed in freeze["seeds"]:
        block = output / ("seed" + str(seed))
        block.mkdir()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # NumPy controls label episodes only; model edge/feature masks use Torch.
        masks = np.random.RandomState(seed + 1900001)
        model = build_model(torch, 300, 10, attention_mask_policy=ATTENTION_MASK_POLICY).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005,
                                     betas=(0.9, 0.999), eps=1e-8)
        best, selected_epoch, selected_logits = -1, None, None
        torch.cuda.reset_peak_memory_stats()
        block_start = time.monotonic()
        order = np.arange(580, dtype=np.int64)
        with (block / "trace.jsonl").open("w") as trace:
            for epoch in range(1, 1501):
                masks.shuffle(order)
                visible = torch.from_numpy(order[:377].copy()).to(device)
                query = torch.from_numpy(order[377:].copy()).to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(topology, train_ids[visible], train_y[visible])
                loss = functional.cross_entropy(logits[train_ids[query]], train_y[query])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite complementary TRAIN CE")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError("Nonfinite source-model gradient")
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation_logits = model(topology, train_ids, train_y)[valid_ids]
                    if not bool(torch.isfinite(validation_logits).all()):
                        raise FloatingPointError("Nonfinite complete VALID prediction")
                    correct = int((validation_logits.argmax(-1) == valid_y).sum())
                    nll = float(functional.cross_entropy(validation_logits.double(), valid_y))
                trace.write(json.dumps({"epoch": epoch, "TRAIN_CE": float(loss.detach()),
                                        "VALID_correct": correct, "VALID_NLL": nll}) + "\n")
                if correct > best:
                    best, selected_epoch = correct, epoch
                    selected_logits = validation_logits.detach().cpu().clone()
                    numpy_state = masks.get_state()
                    torch.save({"model": cpu_tree(torch, model.state_dict()),
                                "optimizer": cpu_tree(torch, optimizer.state_dict()),
                                "cpu_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all(),
                                "mask_rng": [numpy_state[0], numpy_state[1].tolist(),
                                             *numpy_state[2:]],
                                "mask_order": order.tolist(),
                                "selected_epoch": epoch, "selected_correct": correct,
                                "selected_VALID_logits": selected_logits,
                                "freeze_sha256": digest(freeze_file)}, block / "selected.pt")
        state = torch.load(block / "selected.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        model.eval()
        with torch.no_grad():
            replay = model(topology, train_ids, train_y)[valid_ids].detach().cpu()
        if not torch.allclose(replay, state["selected_VALID_logits"], atol=2e-6, rtol=0):
            raise RuntimeError("Selected replay exceeds source engineering tolerance")
        if int((replay.argmax(-1) == torch.from_numpy(valid["y"])).sum()) != best:
            raise RuntimeError("Selected replay changed served decisions")
        np.savez(block / "selected_VALID.npz", ids=valid["ids"], y=valid["y"], logits=replay.numpy())
        result = {"seed": seed, "epochs": 1500, "selected_epoch": selected_epoch,
                  "VALID_accuracy_pct": 100 * best / 5274,
                  "VALID_NLL": float(functional.cross_entropy(replay.double(), torch.from_numpy(valid["y"]))),
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "elapsed_seconds": time.monotonic() - block_start,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "TRAIN_forward_calls": 1500, "VALID_forward_calls": 1500,
                  "selected_replay_forward_calls": 1, "TEST_access": False}
        (block / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        del model, optimizer, logits, validation_logits, replay, state, selected_logits
    (output / "COMPLETE.json").write_text(json.dumps({"complete": True, "seeds": results,
        "new_single_fits": 3, "TEST_access": False, "elapsed_seconds": time.monotonic() - start,
        "freeze_sha256": digest(freeze_file), "route": route,
        "model_source_sha256": digest(Path(__file__).with_name("unimp_v2.py")),
        "runner_source_sha256": digest(__file__)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
