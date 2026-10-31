"""Validation-only BUDDY family training and locked final scoring."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from cache_builder import HERE, file_sha, load_manifest, verify_family_lock, write_json
from models import IndependentBUDDY, make_model, member_logits, parameter_count
from guards import (implementation_hashes, read_json, finite_number, checkpoint_metadata,
                    selection_record_sha, validate_epoch_record, validate_history, validate_run_metadata)
from checkpoint_io import validate_selected_checkpoint


def qualification(path):
    sources = implementation_hashes()
    result = read_json(path)
    if result.get("status") != "synthetic_cpu_pass" or result.get("test_count") != 7 or result["implementation_hashes"] != sources or result.get("torch_version") != torch.__version__:
        raise RuntimeError("A passing CPU qualification for these exact sources is required.")


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def batch_rows(length, size, generator=None):
    indices = torch.randperm(length, generator=generator) if generator else torch.arange(length)
    # Keep every row; merge a singleton tail because native BatchNorm cannot
    # train on one example. The same deterministic rule applies to every arm.
    stop = length - 1 if length % size == 1 and length > size else length
    for start in range(0, stop, size):
        end = min(start + size, stop)
        if end == stop and stop < length:
            end = length
        yield indices[start:end]


def hydrated(common, data, rows, device):
    links = data["links"][rows]
    degrees = common["degrees"][links]
    inputs = (data["sf"][rows].to(device), common["x"][links].to(device), degrees[:, 0].to(device), degrees[:, 1].to(device))
    labels = (rows < data["n_positive"]).float().to(device)
    return inputs, labels


@torch.no_grad()
def predictions(model, common, data, device, size):
    model.eval()
    scores = []
    for rows in batch_rows(len(data["links"]), size):
        inputs, _ = hydrated(common, data, rows, device)
        values = member_logits(model, inputs).mean(dim=1).cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Nonfinite prediction; this cell cannot be complete.")
        scores.append(values)
    return torch.cat(scores)


def hits50(scores, n_positive):
    if not torch.isfinite(scores).all() or n_positive <= 0 or n_positive >= len(scores):
        raise RuntimeError("Hits@50 requires finite scores and a nonempty positive/negative pool.")
    negative = scores[n_positive:]
    if len(negative) != 100000:
        raise RuntimeError("Official collab evaluation requires its unchanged 100000-negative pool.")
    threshold = torch.topk(negative, 50).values[-1]
    return float(torch.sum(scores[:n_positive] > threshold)) / n_positive


def optimization(model, config):
    if isinstance(model, IndependentBUDDY):
        return [torch.optim.Adam(member.parameters(), lr=config["lr"], weight_decay=0) for member in model.models]
    return [torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0)]


def finite_step(loss, model, optimizer):
    if not torch.isfinite(loss).all():
        raise RuntimeError("Nonfinite BCE loss.")
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("Nonfinite gradient; optimizer step rejected.")
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("Nonfinite parameter/BatchNorm state after optimizer step.")
    for state in optimizer.state.values():
        for name, value in state.items():
            valid = torch.isfinite(value).all() if isinstance(value, torch.Tensor) else type(value) in (int, float) and math.isfinite(value)
            if not valid:
                raise RuntimeError(f"Nonfinite Adam state: {name}")


def train_epoch(model, optimizers, common, data, device, config, seed, epoch):
    model.train()
    generator = torch.Generator().manual_seed(seed + 100000 * epoch)
    torch.manual_seed(seed + 1000000 + epoch)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed + 1000000 + epoch)
    total = 0.0
    for rows in batch_rows(len(data["links"]), config["batch_size"], generator):
        inputs, labels = hydrated(common, data, rows, device)
        if isinstance(model, IndependentBUDDY):
            # Separate optimizers and native own-member loss for each untied fit.
            # Synchronized epoch selection is based on their pooled scores.
            loss_value = 0.0
            for member, optimizer in zip(model.models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                logits = member(*inputs).reshape(-1)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite native member logits.")
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                finite_step(loss, member, optimizer)
                loss_value += float(loss.detach()) / len(optimizers)
        else:
            optimizer = optimizers[0]
            optimizer.zero_grad(set_to_none=True)
            logits = member_logits(model, inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite predictor logits.")
            loss = F.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits))
            finite_step(loss, model, optimizer)
            loss_value = float(loss.detach())
        total += loss_value * len(rows)
        finite_number(total, "accumulated training BCE")
    return total / len(data["links"])


def load_training(cache):
    manifest = load_manifest(cache)
    if manifest["config_sha256"] != file_sha(HERE / "CONFIG.json") or manifest["source_pin_sha256"] != file_sha(HERE / "SOURCE_PINS.json") or manifest["builder_sha256"] != file_sha(HERE / "cache_builder.py"):
        raise RuntimeError("Cache does not belong to this fixed recipe/source identity.")
    common = torch.load(Path(cache) / "common.pt", map_location="cpu")
    # Hashes/cards are retained on disk for final hydration, not duplicated
    # inside four predictors or transferred to their dense training device.
    common.pop("hashes"); common.pop("cards")
    train = torch.load(Path(cache) / "train.pt", map_location="cpu")
    valid = torch.load(Path(cache) / "valid.pt", map_location="cpu")
    return manifest, common, train, valid


def fit(args, resource_only=False):
    qualification(args.qualification)
    config = read_json(HERE / "CONFIG.json")
    if args.seed not in config["seeds"]:
        raise RuntimeError("Seed is outside the frozen family.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _, common, train, valid = load_training(args.cache)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = make_model(config, args.arm, args.seed).to(device)
    actual_parameters = sum(p.numel() for p in model.parameters())
    optimizers = optimization(model, config)
    identity = {"arm": args.arm, "seed": args.seed, "cache_manifest_sha256": file_sha(Path(args.cache) / "manifest.json"), "config_sha256": file_sha(HERE / "CONFIG.json"), "implementation_hashes": implementation_hashes(), "parameters": actual_parameters, "device": str(device), "torch_version": str(torch.__version__), "optimizer_fits": len(optimizers), "pooling": "mean_raw_logits"}
    write_json(output / "identity.json", identity)
    best = -1.0
    epochs = 1 if resource_only else config["epochs"]
    total_started = time.monotonic()
    for epoch in range(1, epochs + 1):
        synchronize(device); start = time.monotonic()
        loss = train_epoch(model, optimizers, common, train, device, config, args.seed, epoch)
        synchronize(device); train_seconds = time.monotonic() - start
        start = time.monotonic()
        scores = predictions(model, common, valid, device, config["eval_batch_size"])
        synchronize(device); valid_seconds = time.monotonic() - start
        record = {"epoch": epoch, "train_bce": loss, "train_seconds": train_seconds, "validation_forward_seconds": valid_seconds}
        if not resource_only:
            value = hits50(scores, valid["n_positive"])
            labels = (torch.arange(len(scores)) < valid["n_positive"]).float()
            validation_loss = F.binary_cross_entropy_with_logits(scores, labels)
            if not torch.isfinite(validation_loss).all():
                raise RuntimeError("Nonfinite validation BCE.")
            record.update({"validation_hits50": value, "validation_bce_sampled_pool": float(validation_loss)})
            validate_epoch_record(record, epoch)
            if value > best:  # strict improvement => earliest epoch on ties
                best = value; best_epoch = epoch
                checkpoint = output / "selected.pt"
                temporary = output / "selected.tmp.pt"
                torch.save({"metadata": checkpoint_metadata(identity, record), "model_state": model.state_dict()}, temporary)
                temporary.replace(checkpoint)
                record["selected_checkpoint_sha256"] = file_sha(checkpoint)
        validate_epoch_record(record, epoch, selection=not resource_only)
        with (output / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    synchronize(device)
    summary = {**identity, "status": "resource_only_complete" if resource_only else "training_complete", "epochs_completed": epochs, "total_seconds": time.monotonic() - total_started, "peak_cuda_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, "peak_cuda_reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None, "test_loaded_or_scored": False}
    if not resource_only:
        selected = validate_history(output / "epochs.jsonl", config["epochs"])
        summary.update({"selected_epoch": best_epoch, "selected_validation_hits50": best,
                        "selected_checkpoint_sha256": file_sha(output / "selected.pt"),
                        "identity_sha256": file_sha(output / "identity.json"),
                        "epochs_sha256": file_sha(output / "epochs.jsonl"),
                        "selected_record_sha256": selection_record_sha(selected)})
        validate_selected_checkpoint(output / "selected.pt", checkpoint_metadata(identity, selected))
    finite_number(summary["total_seconds"], "total_seconds")
    write_json(output / "completion.json", summary)


def lock(args):
    sources = implementation_hashes()
    config = read_json(HERE / "CONFIG.json")
    load_manifest(args.cache)
    cache_hash = file_sha(Path(args.cache) / "manifest.json")
    runs = []
    for arm in config["arms"]:
        for seed in config["seeds"]:
            folder = Path(args.runs).resolve() / f"{arm}_seed{seed}"
            run, metadata = validate_run_metadata(folder, arm, seed, cache_hash, config, sources)
            validate_selected_checkpoint(folder / "selected.pt", metadata)
            runs.append(run)
    write_json(args.output, {"schema": "buddy-family-lock-v2", "status": "family_locked", "cache_manifest_sha256": cache_hash, "config_sha256": file_sha(HERE / "CONFIG.json"), "implementation_hashes": sources, "runs": runs})


def final_test(args):
    qualification(args.qualification)
    cache = Path(args.cache).resolve()
    load_manifest(cache)
    family = verify_family_lock(args.family_lock, file_sha(cache / "manifest.json"), file_sha(HERE / "CONFIG.json"))
    if family["implementation_hashes"] != implementation_hashes():
        raise RuntimeError("Locked source identity changed.")
    test_manifest = read_json(cache / "test_manifest.json")
    if test_manifest["cache_manifest_sha256"] != file_sha(cache / "manifest.json") or test_manifest["family_lock_sha256"] != file_sha(args.family_lock) or file_sha(cache / "test.pt") != test_manifest["test_sha256"]:
        raise RuntimeError("Final test cache/lock mismatch.")
    common = torch.load(cache / "common.pt", map_location="cpu")
    common.pop("hashes"); common.pop("cards")
    data = torch.load(cache / "test.pt", map_location="cpu")
    config = json.loads((HERE / "CONFIG.json").read_text())
    for run in family["runs"]:
        folder = Path(run["run_directory"])
        if file_sha(folder / "completion.json") != run["completion_sha256"] or file_sha(folder / "selected.pt") != run["selected_checkpoint_sha256"]:
            raise RuntimeError("Locked training artifacts changed.")
        if (folder / "final_test.json").exists():
            existing = read_json(folder / "final_test.json")
            expected = {"arm": run["arm"], "seed": run["seed"], "family_lock_sha256": file_sha(args.family_lock), "test_manifest_sha256": file_sha(cache / "test_manifest.json"), "checkpoint_sha256": run["selected_checkpoint_sha256"]}
            if any(existing.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Existing final-test result belongs to different artifacts.")
            finite_number(existing.get("hits50"), "hits50", upper=1.0)
            finite_number(existing.get("prediction_seconds"), "prediction_seconds")
            print(json.dumps({**existing, "existing_result_reused": True}), flush=True)
            continue
        device = torch.device(args.device)
        model = make_model(config, run["arm"], run["seed"]).to(device)
        model.load_state_dict(torch.load(folder / "selected.pt", map_location=device, weights_only=True)["model_state"], strict=True)
        synchronize(device); start = time.monotonic()
        scores = predictions(model, common, data, device, config["eval_batch_size"])
        synchronize(device)
        result = {"arm": run["arm"], "seed": run["seed"], "hits50": hits50(scores, data["n_positive"]), "prediction_seconds": time.monotonic() - start, "family_lock_sha256": file_sha(args.family_lock), "test_manifest_sha256": file_sha(cache / "test_manifest.json"), "checkpoint_sha256": run["selected_checkpoint_sha256"]}
        write_json(folder / "final_test.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "resource", "lock", "test"])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output")
    parser.add_argument("--runs")
    parser.add_argument("--arm", choices=["native1024", "single256", "factorized4", "independent4", "matched_single"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qualification")
    parser.add_argument("--family-lock")
    args = parser.parse_args()
    if args.stage in {"fit", "resource"}:
        if args.arm is None or args.seed is None or not args.output or not args.qualification:
            parser.error("fit/resource require --arm --seed --output --qualification")
        fit(args, resource_only=args.stage == "resource")
    elif args.stage == "lock":
        if not args.runs or not args.output:
            parser.error("lock requires --runs --output")
        lock(args)
    else:
        if not args.family_lock or not args.qualification:
            parser.error("test requires --family-lock --qualification")
        final_test(args)
