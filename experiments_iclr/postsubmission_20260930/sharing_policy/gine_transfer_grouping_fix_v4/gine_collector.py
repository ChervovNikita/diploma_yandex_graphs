"""GINE batching port; mathematical gradient statistic/selector is unchanged.

Each molecule graph receives equal weight, even when its node count or
finite target count differs. Exact duplicate molecule groups determine
fold allocation, not the numerical loss weight. No validation path exists.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from common import sha
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from recurrence_collector import (
    COLLECTOR_SCHEMA, FOLD_WEIGHTING, LOSS_RULE, block_statistics, state_hash,
    tensor_hash, unit_fingerprint, validate_training_units,
)


def equal_graph_bce(logits, targets):
    if logits.ndim != 3 or targets.shape != logits.shape[1:]:
        raise ValueError("Require logits[M,B,T] and targets[B,T]")
    finite = torch.isfinite(targets)
    observed = finite.sum(-1)
    if not bool((observed > 0).all()):
        raise ValueError("Supply observed graphs only; hidden graphs cannot enter loss")
    if not bool(((targets[finite] == 0) | (targets[finite] == 1)).all()):
        raise ValueError("Observed labels must be binary")
    safe = torch.where(finite, targets, torch.zeros_like(targets))
    element = F.binary_cross_entropy_with_logits(logits, safe[None].expand_as(logits), reduction="none")
    return ((element * finite[None]).sum(-1) / observed[None]).mean()


def fold_gradients_batched(model, by_id, ids, blocks, batch_size=128):
    """Average own-member graph loss gradients, without division by members."""
    params = tuple(p for block in blocks.values() for p in block)
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError("Empty or overlapping probed blocks")
    counts = [int(torch.isfinite(by_id[key].targets).sum()) for key in ids]
    if not ids or min(counts) <= 0 or batch_size <= 0:
        raise ValueError("Empty/hidden graph in fold")
    rows, losses = {key: [] for key in blocks}, []
    for member in range(model.members):
        sums = {key: torch.zeros(sum(p.numel() for p in block), dtype=torch.float64)
                for key, block in blocks.items()}
        total_loss = 0.0
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            graphs = []
            for key in selected:
                unit = by_id[key]
                graph = unit.graph.clone()
                graph.x, graph.y = unit.features, unit.targets
                graphs.append(graph)
            batch = Batch.from_data_list(graphs).to(params[0].device)
            logits = model.forward_member(batch, batch.x, member)
            # Each chunk mean is weighted by its graph count, including last chunk.
            loss = equal_graph_bce(logits[None], batch.y.to(logits.dtype)) * len(selected) / len(ids)
            grads = torch.autograd.grad(loss, params, allow_unused=False)
            offset = 0
            for key, block in blocks.items():
                vector = torch.cat([g.detach().reshape(-1).double().cpu()
                                    for g in grads[offset:offset + len(block)]])
                if not bool(torch.isfinite(vector).all()):
                    raise ValueError("Nonfinite GINE fold gradient")
                sums[key] += vector
                offset += len(block)
            total_loss += float(loss.detach())
        for key in blocks:
            rows[key].append(sums[key])
        losses.append(total_loss)
    return {key: torch.stack(value) for key, value in rows.items()}, {
        "unit_ids": list(ids), "unit_count": len(ids), "observed_entries": sum(counts),
        "observed_entries_by_unit": counts, "unit_weights": [1 / len(ids)] * len(ids),
        "member_mean_loss": losses, "fold_weighting": FOLD_WEIGHTING, "loss_rule": LOSS_RULE,
    }


def block_contract(model):
    blocks = model.probe_blocks()
    names = {id(p): name for name, p in model.named_parameters()}
    result = {key: {"count": sum(p.numel() for p in block),
                    "names": [names[id(p)] for p in block]} for key, block in blocks.items()}
    if set(result) != {"0", "1", "2"} or len({v["count"] for v in result.values()}) != 1:
        raise ValueError("Require three equal-cost full GINE blocks")
    for key, value in result.items():
        if not any("bond_encoder" in name for name in value["names"]):
            raise ValueError("Private block omitted its learned bond embeddings")
    return result


def collect_record(model, units, fit_ids, partitions, epoch, seed, dataset_identity, batch_size=128):
    if epoch not in (0, 10, 25) or model.members != 4:
        raise ValueError("Preserve four members and checkpoints0/10/25")
    by_id = validate_training_units(units, fit_ids, partitions)
    blocks, contract = model.probe_blocks(), block_contract(model)
    before = state_hash(model)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    grads = {name: None if p.grad is None else p.grad.detach().clone()
             for name, p in model.named_parameters()}
    record = {"schema": COLLECTOR_SCHEMA, "dataset_identity": dataset_identity,
              "optimizer_seed": seed, "checkpoint_epoch": epoch,
              "model_state_sha256": before, "members": 4, "objective": "bce",
              "fold_weighting": FOLD_WEIGHTING, "unit_kind": "whole_observed_training_graph",
              "probe_fit_unit_ids": list(fit_ids),
              "training_unit_fingerprints": [unit_fingerprint(by_id[k]) for k in sorted(by_id)],
              "block_parameters": contract, "partitions": [],
              "collector_source_sha256": sha(Path(__file__)), "torch_version": torch.__version__,
              "batch_size": batch_size,
              "numerical_note": "Batched graph-gradient sums are algebraically equal to per-graph sums; float32 order may differ"}
    try:
        model.eval()
        with torch.enable_grad():
            for part in partitions:
                a, ma = fold_gradients_batched(model, by_id, part["A"], blocks, batch_size)
                b, mb = fold_gradients_batched(model, by_id, part["B"], blocks, batch_size)
                record["partitions"].append({"partition_id": part["partition_id"], "A": ma, "B": mb,
                                            "blocks": {k: block_statistics(a[k], b[k]) for k in blocks}})
    finally:
        for module, mode in modes:
            module.training = mode
    if before != state_hash(model) or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError("Collection changed model or CPU RNG")
    if cuda_rng is not None and any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
        raise RuntimeError("Collection changed CUDA RNG")
    for name, p in model.named_parameters():
        expected = grads[name]
        if (expected is None) != (p.grad is None) or (expected is not None and not torch.equal(expected, p.grad)):
            raise RuntimeError("Collection changed an optimizer gradient buffer")
    record["state_and_rng_unchanged"] = True
    return record
