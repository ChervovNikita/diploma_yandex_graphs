"""Train-only graph-unit gradient collection for the proposed sharing policy.

The caller supplies an already constructed tied probe and ONLY training
units. This module neither downloads data nor loads checkpoints/held-out
labels. A unit's complete graph is the fold unit for the first PPI study.
Models must expose forward_member(graph, features, member), probe_blocks(),
and members; SelectiveProjectorEnsemble supplies this interface.
"""

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
COLLECTOR_SCHEMA = "gnnm-recurrence-collector-v2"
FOLD_WEIGHTING = "equal_mean_per_training_unit"
LOSS_RULE = "unscaled own member loss; mean of unit-specific mean finite observed-entry losses"
ADAM_DIAGNOSTIC_CONVENTION = {
    "optimizer": "virtual_Adam_first_step", "moment_initialization": "zero",
    "learning_rate": 0.001, "epsilon": 1e-8, "weight_decay": 0.0,
    "shared_gradient": "mean own-member gradients",
    "private_gradient": "own-member gradient / members",
    "prediction": "negative linearized mean own-member scoring loss difference; symmetric A/B average",
    "scope": "secondary local zero-moment diagnostic; not actual checkpoint AdamW or final benefit",
}


@dataclass(frozen=True)
class TrainingUnit:
    unit_id: str
    graph: Any
    features: torch.Tensor
    targets: torch.Tensor
    split: str = "train"


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    metadata = {"dtype": str(value.dtype), "shape": list(value.shape)}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return metadata | {"sha256": digest.hexdigest()}


def state_hash(model):
    manifest = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def validate_training_units(units, fit_ids, partitions):
    by_id = {}
    for unit in units:
        if unit.split != "train":
            raise ValueError("Collector accepts training units only")
        if unit.unit_id in by_id:
            raise ValueError("Duplicate training-unit ID")
        by_id[unit.unit_id] = unit
    fit = set(fit_ids)
    if not fit or len(fit) != len(fit_ids) or not fit <= by_id.keys():
        raise ValueError("Invalid probe-fit training-unit IDs")
    score_set = None
    names = set()
    for partition in partitions:
        if partition["partition_id"] in names:
            raise ValueError("Duplicate partition ID")
        names.add(partition["partition_id"])
        a, b = set(partition["A"]), set(partition["B"])
        if not a or not b or len(a) != len(partition["A"]) or len(b) != len(partition["B"]):
            raise ValueError("Empty or duplicate scoring units")
        if a & b or fit & (a | b) or not (a | b) <= by_id.keys():
            raise ValueError("Probe-fit/A/B units overlap or refer to non-training data")
        if score_set is not None and score_set != a | b:
            raise ValueError("All partitions must cover the same score-only Q units")
        score_set = a | b
    if not partitions or fit | score_set != by_id.keys():
        raise ValueError("Require a complete disjoint P/Q partition of supplied training units")
    return by_id


def unit_fingerprint(unit):
    # Only supplied training-unit labels enter this fingerprint.
    result = {"unit_id": unit.unit_id, "split": unit.split,
              "features": tensor_hash(unit.features), "targets": tensor_hash(unit.targets)}
    for field in ("edge_index", "edge_attr", "pos"):
        value = getattr(unit.graph, field, None)
        if value is not None:
            result[field] = tensor_hash(value)
    return result


def loss_sum_and_count(logits, targets, objective):
    if objective == "bce":
        if logits.shape != targets.shape:
            raise ValueError("BCE logits/targets shape mismatch")
        finite = torch.isfinite(targets)
        count = int(finite.sum())
        if not count:
            raise ValueError("Training unit has no finite BCE targets")
        selected = targets[finite]
        if not bool(((selected == 0) | (selected == 1)).all()):
            raise ValueError("BCE targets must be binary or missing")
        return F.binary_cross_entropy_with_logits(logits[finite], selected, reduction="sum"), count
    if objective == "ce":
        targets = targets.reshape(-1)
        if logits.ndim != 2 or logits.shape[0] != len(targets) or targets.dtype != torch.long:
            raise ValueError("CE logits/targets shape or dtype mismatch")
        if not len(targets):
            raise ValueError("Empty CE targets")
        return F.cross_entropy(logits, targets, reduction="sum"), len(targets)
    raise ValueError("Unknown training objective")


def fold_gradients(model, by_id, ids, blocks, objective):
    """Equal-unit mean of own-member unit losses; no division by members."""
    parameters = tuple(parameter for block in blocks.values() for parameter in block)
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Empty/overlapping block parameters")
    counts = []
    for unit_id in ids:
        targets = by_id[unit_id].targets
        counts.append(int(torch.isfinite(targets).sum()) if objective == "bce" else targets.numel())
    total = sum(counts)
    if not ids or any(count <= 0 for count in counts):
        raise ValueError("Empty scoring fold or unit without observed labels")
    rows = {key: [] for key in blocks}
    losses = []
    for member in range(model.members):
        sums = {key: torch.zeros(sum(p.numel() for p in block), dtype=torch.float64)
                for key, block in blocks.items()}
        member_loss = 0.0
        for unit_id, expected_count in zip(ids, counts):
            unit = by_id[unit_id]
            logits = model.forward_member(unit.graph, unit.features, member)
            loss, count = loss_sum_and_count(logits, unit.targets, objective)
            if count != expected_count:
                raise ValueError("Training label count changed during probe")
            gradients = torch.autograd.grad(loss / (count * len(ids)), parameters, allow_unused=False)
            offset = 0
            for key, block in blocks.items():
                vector = torch.cat([g.detach().reshape(-1).double().cpu()
                                    for g in gradients[offset:offset + len(block)]])
                if not bool(torch.isfinite(vector).all()):
                    raise ValueError("Nonfinite training gradient")
                sums[key] += vector
                offset += len(block)
            member_loss += float(loss.detach()) / (count * len(ids))
        for key in blocks:
            rows[key].append(sums[key])
        losses.append(member_loss)
    return {key: torch.stack(value) for key, value in rows.items()}, {
        "unit_ids": list(ids), "unit_count": len(ids), "observed_entries": total,
        "observed_entries_by_unit": counts, "unit_weights": [1.0 / len(ids)] * len(ids),
        "member_mean_loss": losses, "fold_weighting": FOLD_WEIGHTING, "loss_rule": LOSS_RULE,
    }


def virtual_native_adam_first_step(a, b):
    """Native mean-loss step scale; a cheap diagnostic from existing vectors."""
    if a.shape != b.shape or a.ndim != 2 or len(a) != 4:
        raise ValueError("Invalid virtual Adam member gradients")
    lr, epsilon = 0.001, 1e-8

    def directional(fit, score):
        shared_gradient = fit.mean(0)
        private_gradient = fit / len(fit)
        shared_delta = -lr * shared_gradient / (shared_gradient.abs() + epsilon)
        private_delta = -lr * private_gradient / (private_gradient.abs() + epsilon)
        return -float((score * private_delta).sum(1).mean()
                      - (score.mean(0) * shared_delta).sum())

    ab, ba = directional(a, b), directional(b, a)
    return {"convention": dict(ADAM_DIAGNOSTIC_CONVENTION),
            "gain_A_to_B": ab, "gain_B_to_A": ba, "symmetric_gain": (ab + ba) / 2}


def block_statistics(a, b):
    ca, cb = a - a.mean(0), b - b.mean(0)
    c = float((ca * cb).sum(1).mean())
    n = float((ca - cb).square().sum(1).mean() / 4)
    cross = ca @ cb.T
    permutation_c = [float(sum(cross[i, p[i]] for i in range(len(a))) / len(a))
                     for p in itertools.permutations(range(len(a)))]
    return {
        "C": c, "N": n,
        "raw_gram": {"AA": (a @ a.T).tolist(), "BB": (b @ b.T).tolist(),
                     "AB": (a @ b.T).tolist()},
        "centered_gram": {"AA": (ca @ ca.T).tolist(), "BB": (cb @ cb.T).tolist(),
                          "AB": cross.tolist()},
        "member_gradient_sha256": {"A": [tensor_hash(row)["sha256"] for row in a],
                                   "B": [tensor_hash(row)["sha256"] for row in b]},
        "member_permutation_C": permutation_c,
        "permutation_mean_C": sum(permutation_c) / len(permutation_c),
        "virtual_native_adam_first_step": virtual_native_adam_first_step(a, b),
    }


def collect_record(model, units, probe_fit_ids, partitions, checkpoint_epoch,
                   optimizer_seed, dataset_identity, objective="bce"):
    """Collect one checkpoint; never changes parameters, grads, modes or RNG.

    dataset_identity should contain the new study's pretraining data/source
    manifest hash. It is metadata, not authority to load labels elsewhere.
    """
    if checkpoint_epoch not in (0, 10, 25):
        raise ValueError("Checkpoint is outside the predeclared 0/10/25 probe window")
    if model.members != 4:
        raise ValueError("The first frozen sharing policy requires four members")
    by_id = validate_training_units(units, probe_fit_ids, partitions)
    blocks = model.probe_blocks()
    parameter_names = {id(p): name for name, p in model.named_parameters()}
    before_hash = state_hash(model)
    before_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    record = {
        "schema": COLLECTOR_SCHEMA, "dataset_identity": dataset_identity,
        "optimizer_seed": int(optimizer_seed), "checkpoint_epoch": int(checkpoint_epoch),
        "model_state_sha256": before_hash, "members": model.members, "objective": objective,
        "fold_weighting": FOLD_WEIGHTING,
        "unit_kind": "whole_training_graph", "probe_fit_unit_ids": list(probe_fit_ids),
        "training_unit_fingerprints": [unit_fingerprint(by_id[k]) for k in sorted(by_id)],
        "block_parameters": {key: {"count": sum(p.numel() for p in block),
                                   "names": [parameter_names[id(p)] for p in block]}
                             for key, block in blocks.items()},
        "partitions": [],
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
    }
    try:
        model.eval()
        with torch.enable_grad():
            for partition in partitions:
                a, meta_a = fold_gradients(model, by_id, partition["A"], blocks, objective)
                b, meta_b = fold_gradients(model, by_id, partition["B"], blocks, objective)
                record["partitions"].append({
                    "partition_id": partition["partition_id"], "A": meta_a, "B": meta_b,
                    "blocks": {key: block_statistics(a[key], b[key]) for key in blocks},
                })
    finally:
        for module, mode in modes:
            module.training = mode
    if before_hash != state_hash(model) or not torch.equal(before_rng, torch.get_rng_state()):
        raise RuntimeError("Probe changed state or CPU RNG")
    if cuda_rng is not None:
        after = torch.cuda.get_rng_state_all()
        if len(after) != len(cuda_rng) or any(not torch.equal(a, b) for a, b in zip(after, cuda_rng)):
            raise RuntimeError("Probe changed CUDA RNG")
    record["state_and_rng_unchanged"] = True
    return record


def save_record(record, output):
    output = Path(output).resolve()
    if HERE not in output.parents:
        raise ValueError("Collector output must remain inside its sharing_policy folder")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
