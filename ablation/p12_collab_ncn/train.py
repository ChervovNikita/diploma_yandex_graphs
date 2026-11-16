"""Resumable source-faithful NCN training and selected-checkpoint scoring."""
import importlib.metadata
import hashlib
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch_geometric.utils import negative_sampling

from ablation.common.resumable import atomic_torch_save, atomic_write_json, capture_rng, restore_rng
from ablation.common.worker import Preempted
from .data import load_bundle, masked_adjacency
from .models import NCN
from .protocol import digest_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def runtime_identity(config, bundle):
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torch-geometric", "torch-sparse", "torch-scatter", "ogb", "numpy")}
    return {"config_sha256": digest_json(config), "data_sha256": bundle.identity,
            "versions": versions, "torch_cuda": torch.version.cuda}


def metrics(predictions):
    evaluator = Evaluator(name="ogbl-collab")
    evaluator.K = 50
    result = {}
    for part in ("train", "valid", "test"):
        if f"{part}_pos" not in predictions:
            continue
        negative = predictions["valid_neg" if part == "train" else f"{part}_neg"]
        result[f"{part}_hits@50"] = float(evaluator.eval({
            "y_pred_pos": predictions[f"{part}_pos"], "y_pred_neg": negative})["hits@50"])
    return result


@torch.no_grad()
def predict(model, bundle, batch_size, store=None, test_only=False, return_members=False):
    """Preserve source edge order and average raw member prediction scores."""
    model.eval()
    adjacency = bundle.full_adj if test_only else bundle.adj
    if adjacency is None:
        raise ValueError("TEST adjacency is unavailable")
    parts = ("test",) if test_only else ("train", "valid")
    all_predictions, member_predictions = {}, {}
    for member in range(model.members):
        model.select_member(member)
        hidden = model.encoder(bundle.x, adjacency)
        for part in parts:
            for name, edges in bundle.split[part].items():
                field = f"{part}_{'pos' if name == 'edge' else 'neg'}"
                chunks = []
                for start in range(0, len(edges), batch_size):
                    prediction = model.predictor(hidden, adjacency, edges[start:start + batch_size].t())
                    chunks.append(prediction.reshape(-1).cpu())
                    if store is not None:
                        store.heartbeat()
                member_prediction = torch.cat(chunks)
                if not torch.isfinite(member_prediction).all():
                    raise FloatingPointError("NCN member predictions are non-finite")
                if return_members:
                    member_predictions.setdefault(field, []).append(member_prediction)
                if field in all_predictions:
                    all_predictions[field].add_(member_prediction / model.members)
                else:
                    all_predictions[field] = member_prediction / model.members
    if return_members:
        return all_predictions, {key: torch.stack(value) for key, value in member_predictions.items()}
    return all_predictions


def optimizer_step(model, bundle, positive_indices, negative_edges, optimizer):
    model.train()
    optimizer.zero_grad()
    adjacency = masked_adjacency(bundle, positive_indices)
    positive = bundle.split["train"]["edge"][positive_indices].t()
    negative = negative_edges[:, positive_indices]
    loss_value = 0.0
    for member in range(model.members):
        model.select_member(member)
        hidden = model.encoder(bundle.x, adjacency)
        positive_scores = model.predictor.multidomainforward(
            hidden, adjacency, positive, cndropprobs=[])
        negative_scores = model.predictor.multidomainforward(
            hidden, adjacency, negative, cndropprobs=[])
        loss = (-F.logsigmoid(positive_scores).mean()
                - F.logsigmoid(-negative_scores).mean()) / model.members
        if not torch.isfinite(loss):
            raise FloatingPointError("NCN training loss is non-finite")
        loss.backward()
        loss_value += float(loss.detach())
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise FloatingPointError("NCN training gradient is non-finite")
    optimizer.step()
    return loss_value


def train_fit(config, store, stop, bundle, device, save_every_sec=120, on_update=None):
    seed_everything(config["seed"])
    model = NCN(bundle.x.shape[1], config).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": config["gnnlr"]},
        {"params": model.predictor.parameters(), "lr": config["prelr"]},
    ])
    identity = runtime_identity(config, bundle)
    progress = {
        "epoch": 1, "next_batch": 0, "updates": 0, "epochs_completed": 0,
        "best_epoch": None, "best_update": None, "best_valid": -math.inf,
        "best_model": None, "best_predictions": None, "best_member_predictions": None, "evaluated_update": 0, "train_seconds": 0.0,
        "evaluation_seconds": 0.0, "resumes": 0, "curve": [], "stale_checks": 0, "continuations": [], "loss_sum": 0.0, "loss_count": 0,
    }
    permutation = negatives = None
    saved = store.load_state()
    if saved is not None:
        if saved["identity"] != identity or saved["scheduler"] is not None:
            raise ValueError("Checkpoint source/config/data/runtime identity does not match this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        progress = saved["progress"]
        progress["resumes"] += 1
        permutation = saved["permutation"]
        negatives = saved["negative_edges"]
        if permutation is not None:
            permutation = permutation.to(device)
            negatives = negatives.to(device)
        restore_rng(saved["rng"], device)
        print(f"[resume] {store.dir} epoch={progress['epoch']} updates={progress['updates']}", flush=True)

    def save():
        store.save_state({
            "identity": identity, "model": cpu_state(model),
            "optimizer": optimizer.state_dict(), "scheduler": None,
            "progress": progress, "rng": capture_rng(device),
            "permutation": None if permutation is None else permutation.cpu(),
            "negative_edges": None if negatives is None else negatives.cpu(),
        })
        store.heartbeat()

    batch_size = config["batch_size"]
    train_size = len(bundle.split["train"]["edge"])
    batches = train_size // batch_size
    if batches == 0:
        raise ValueError("The native drop-last training iterator requires at least one complete batch")
    last_save = time.monotonic()
    patience_reached = bool(config.get("patience") and progress["stale_checks"] >= config["patience"])
    while progress["epoch"] <= config["epochs"]:
        if config.get("patience") and progress["stale_checks"] >= config["patience"]:
            patience_reached = True
            break
        if stop.requested:
            save()
            raise Preempted()
        if (config["smoke_updates"] and progress["updates"] >= config["smoke_updates"]
                and progress["evaluated_update"] == progress["updates"]):
            break
        if permutation is None:
            # Match the upstream RNG ordering: negatives first, then permutation.
            negatives = negative_sampling(bundle.edge_index, bundle.num_nodes)
            permutation = torch.randperm(train_size, device=device)
            if negatives.shape[1] < train_size:
                raise RuntimeError("Source negative sampler returned fewer edges than the training split")
        for batch_index in range(progress["next_batch"], batches):
            if config["smoke_updates"] and progress["updates"] >= config["smoke_updates"]:
                break
            start = batch_index * batch_size
            indices = permutation[start:start + batch_size]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.monotonic()
            loss = optimizer_step(model, bundle, indices, negatives, optimizer)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            progress["train_seconds"] += time.monotonic() - started
            progress["updates"] += 1
            progress["next_batch"] = batch_index + 1
            progress["loss_sum"] += loss
            progress["loss_count"] += 1
            if on_update is not None:
                on_update(progress["updates"])
            store.heartbeat()
            if stop.requested or time.monotonic() - last_save >= save_every_sec:
                save()
                last_save = time.monotonic()
            if stop.requested:
                raise Preempted()
            if config["smoke_updates"] and progress["updates"] >= config["smoke_updates"]:
                break

        started = time.monotonic()
        predictions, member_predictions = predict(model, bundle, config["testbs"], store, return_members=True)
        scores = metrics(predictions)
        progress["evaluation_seconds"] += time.monotonic() - started
        progress["evaluated_update"] = progress["updates"]
        valid = scores["valid_hits@50"]
        if not math.isfinite(valid):
            raise RuntimeError("Non-finite validation Hits@50")
        if valid > progress["best_valid"]:
            progress["best_valid"] = valid
            progress["best_epoch"] = progress["epoch"]
            progress["best_update"] = progress["updates"]
            progress["best_model"] = cpu_state(model)
            progress["best_predictions"] = predictions
            progress["best_member_predictions"] = member_predictions
            progress["stale_checks"] = 0
        elif progress["epoch"] > config.get("min_epochs", 0):
            progress["stale_checks"] += 1
        if progress["epoch"] == 100 and not config["smoke_updates"]:
            atomic_torch_save({"model": progress["best_model"], "config": config,
                              "identity": identity, "predictions": progress["best_predictions"],
                              "member_predictions": progress["best_member_predictions"],
                              "selected_epoch": progress["best_epoch"]},
                             store.dir / "reference100_model.pt")
        progress.setdefault("curve", []).append({
            "epoch": progress["epoch"], "updates": progress["updates"],
            "valid_hits@50": valid, "best_valid": progress["best_valid"],
            "train_loss": progress["loss_sum"] / max(1, progress["loss_count"]),
            "train_seconds": progress["train_seconds"],
            "evaluation_seconds": progress["evaluation_seconds"],
        })
        atomic_write_json(progress["curve"], store.dir / "curve.json")
        patience_reached = (config.get("patience") is not None
                            and progress["stale_checks"] >= config["patience"])
        if (not config["smoke_updates"] and progress["epoch"] in config.get("cap_schedule", [])[:-1]
                and not patience_reached):
            progress["continuations"].append({
                "completed_epoch": progress["epoch"], "next_cap": config["epochs"],
                "reason": "patience_plateau_not_reached", "optimizer_and_rng_preserved": True})
        print(f"[collab] {store.dir} epoch={progress['epoch']} updates={progress['updates']} "
              f"valid_hits@50={valid:.6f} best={progress['best_valid']:.6f}", flush=True)
        if progress["next_batch"] == batches:
            progress["epochs_completed"] += 1
            progress["epoch"] += 1
            progress["next_batch"] = 0
            progress["loss_sum"] = 0.0
            progress["loss_count"] = 0
            permutation = negatives = None
        save()
        last_save = time.monotonic()
        if patience_reached or (config["smoke_updates"] and progress["updates"] >= config["smoke_updates"]):
            break

    if progress["best_model"] is None:
        raise RuntimeError("Training ended without a selected checkpoint")
    model.load_state_dict(progress["best_model"])
    predictions = dict(progress["best_predictions"])
    member_predictions = dict(progress["best_member_predictions"])
    test_edge_order_sha256 = None
    if not config["smoke_updates"] and not config.get("defer_test", False):
        # The fixed config is frozen before the full run. TEST is scored once
        # from the validation-selected checkpoint at ordinary run completion.
        test_bundle = load_bundle(config["data_dir"], device, release_test=True)
        if test_bundle.identity != bundle.identity:
            raise ValueError("Dataset changed before selected-checkpoint TEST scoring")
        test_edge_order_sha256 = test_bundle.test_edge_order_sha256
        if not test_edge_order_sha256:
            raise ValueError("TEST edge-order identity is unavailable")
        test_predictions, test_members = predict(
            model, test_bundle, config["testbs"], store, test_only=True, return_members=True)
        predictions.update(test_predictions)
        member_predictions.update(test_members)
    artifact = {
        "identity": identity, "config": config, "selected_epoch": progress["best_epoch"],
        "selected_update": progress["best_update"], "predictions": predictions,
        "member_predictions": member_predictions,
        "test_edge_order_sha256": test_edge_order_sha256,
        "test_role": ("closed_smoke" if config["smoke_updates"] else
                      "deferred" if config.get("defer_test") else "selected_checkpoint_once"),
    }
    atomic_torch_save(dict(artifact, model=progress["best_model"]), store.dir / "selected_model.pt")
    atomic_torch_save(artifact, store.dir / "selected_predictions.pt")
    result = dict(config, **metrics(predictions), identity=identity,
                  best_epoch=progress["best_epoch"], best_update=progress["best_update"],
                  updates=progress["updates"], epochs_completed=progress["epochs_completed"],
                  train_seconds=progress["train_seconds"], evaluation_seconds=progress["evaluation_seconds"],
                  resumes=progress["resumes"], num_params=sum(p.numel() for p in model.parameters()),
                  test_role=artifact["test_role"], selected_predictions="selected_predictions.pt",
                  selected_model="selected_model.pt", test_edge_order_sha256=test_edge_order_sha256)
    result["selected_checkpoint_sha256"] = hashlib.sha256(
        (store.dir / "selected_model.pt").read_bytes()).hexdigest()
    reference = store.dir / "reference100_model.pt"
    result["reference100_available"] = reference.exists()
    result["reference100_checkpoint_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest() if reference.exists() else None
    result["continuations"] = progress["continuations"]
    result["stale_validation_checks"] = progress["stale_checks"]
    result["cap_hit"] = not config["smoke_updates"] and progress["epochs_completed"] >= config["epochs"]
    result["stop_reason"] = ("smoke" if config["smoke_updates"] else
                             "patience" if patience_reached else "epoch_cap")
    result["needs_convergence_extension"] = not config["smoke_updates"] and not patience_reached
    result["val_metric"] = result["valid_hits@50"]
    if "test_hits@50" in result:
        result["test_metric"] = result["test_hits@50"]
    return result
