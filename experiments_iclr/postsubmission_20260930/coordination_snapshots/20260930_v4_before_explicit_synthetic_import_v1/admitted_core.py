"""Actual K8 AdamW implementation for qualified real or explicit synthetic units.

The v1/v2 source-only functions are unchanged. This sibling repeats their
scientific update and scoring order, adding complete snapshot/trace capture.
"""
import hashlib
import math
import torch
from torch_geometric.data import Batch

from admission import Admission, canonical, LOSSES, PLACEMENTS
from lookahead_contract import RECIPE, RULE
from lookahead_core import (checked_units, initial_models, objective_values,
                            population_summary, state_hash, tensor_hash, update_schedule)
from gpu_lookahead_core_v2 import gpu_environment, score_gpu, GPU_RECIPE


def tree_manifest(value):
    if torch.is_tensor(value):
        return {"tensor": tensor_hash(value)}
    if isinstance(value, dict):
        return {"dict": [[str(k), tree_manifest(v)] for k, v in sorted(value.items(), key=lambda x: str(x[0]))]}
    if isinstance(value, (list, tuple)):
        return {"sequence": [tree_manifest(v) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Unsupported optimizer snapshot type")


def tree_hash(value):
    return hashlib.sha256(canonical(tree_manifest(value))).hexdigest()


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def snapshot(model, optimizer, logits, trace):
    return {"state_dict": to_cpu(model.state_dict()), "optimizer": to_cpu(optimizer.state_dict()),
            "gradients": {name: None if p.grad is None else to_cpu(p.grad) for name, p in model.named_parameters()},
            "member_logits": to_cpu(logits), "trace": trace,
            "rng_cpu": torch.get_rng_state().clone(), "rng_cuda": torch.cuda.get_rng_state(0).clone(),
            "model_state_sha256": state_hash(model), "optimizer_sha256": tree_hash(optimizer.state_dict()),
            "member_logits_sha256": tensor_hash(logits)}


def run_cell(common, update, screen, source_fit_groups, seed, *, admission=None,
             fraction=None, synthetic=False, include_null=False, capture=None):
    if not synthetic:
        if not isinstance(admission, Admission) or (seed, fraction) not in {
                (d["seed"], d["label_fraction"]) for d in admission.cells}:
            raise ValueError("Real inputs require the completed six-source admission object")
        if include_null:
            raise ValueError("Real design is exactly eight arms, never extra null trajectories")
    if seed not in (0, 1, 2):
        raise ValueError("Only three declared source seeds")
    environment = gpu_environment()
    if environment["name"] != "NVIDIA A100-SXM4-80GB":
        raise ValueError("Require the authorized A10080GB runtime")
    if any(p.device != torch.device("cuda:0") or p.dtype != torch.float32 for p in common.parameters()):
        raise ValueError("Require native float32 common source on cuda:0")
    groups = checked_units(update, screen, source_fit_groups, synthetic)
    before = state_hash(common)
    def inputs():
        return {u.unit_id: {k: tensor_hash(getattr(u.graph, k)) for k in ("x", "edge_index", "edge_attr", "y")}
                for u in [*update, *screen]}
    input_hashes = inputs()
    schedule = update_schedule(update, seed)
    seen = {u.unit_id: u for step in schedule for u in step}
    coverage = {"update_fold": population_summary(update), "screen_fold": population_summary(screen),
                "seen_update_unique": population_summary(list(seen.values())),
                "update_graph_presentations": sum(len(step) for step in schedule),
                "update_unique_graph_fraction": len(seen) / len(update)}
    rows, artifacts = {}, {}
    with torch.random.fork_rng(devices=[0]):
        states, block_count = initial_models(common, include_null)
        probe = Batch.from_data_list([u.graph for u in schedule[0]]).to("cuda:0")
        initial_logits = None
        for name, (model, optimizer) in states.items():
            model.eval()
            with torch.no_grad():
                logits = model(probe, probe.x)
            if initial_logits is None:
                initial_logits = logits.detach().clone()
            elif not torch.equal(initial_logits, logits):
                raise ValueError("Private layout changes the initial source function")
            parameters = list(model.parameters())
            opt_parameters = [p for group in optimizer.param_groups for p in group["params"]]
            if len(opt_parameters) != len(parameters) or {id(p) for p in opt_parameters} != {id(p) for p in parameters}:
                raise ValueError("Optimizer must cover every actual trainable parameter exactly once")
        for name, (model, optimizer) in states.items():
            loss = name.split("_")[0]
            initial, screen_logits = score_gpu(model, screen)
            if capture is not None:
                capture(name, "initial", snapshot(model, optimizer, screen_logits, []))
            trace = []
            for step, units in enumerate(schedule):
                model.train()
                torch.manual_seed(27431 + 1000 * seed + step)
                batch = Batch.from_data_list([u.graph for u in units]).to("cuda:0")
                optimizer.zero_grad(set_to_none=True)
                pre_state = state_hash(model)
                logits = model(batch, batch.x)
                values, _ = objective_values(logits, batch.y.float())
                if not bool(torch.isfinite(values[loss])):
                    raise ValueError("Nonfinite update loss")
                values[loss].backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError("Missing/nonfinite actual GPU gradient")
                gradient_hash = tree_hash({n: p.grad for n, p in model.named_parameters()})
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError("Nonfinite actual AdamW update")
                row = {"step": step + 1, "unit_ids": [u.unit_id for u in units],
                       "dropout_seed": 27431 + 1000 * seed + step,
                       "own_loss": float(values["own"].detach()), "pooled_loss": float(values["pooled"].detach()),
                       "state_before_sha256": pre_state, "training_member_logits_sha256": tensor_hash(logits),
                       "gradient_sha256": gradient_hash, "state_after_sha256": state_hash(model),
                       "optimizer_sha256": tree_hash(optimizer.state_dict())}
                trace.append(row)
                if capture is not None:
                    capture(name, "step%02d" % (step + 1), snapshot(model, optimizer, logits, list(trace)))
            final, logits = score_gpu(model, screen)
            steps = sorted({float(s["step"]) for s in optimizer.state.values()})
            if steps != [float(RECIPE["steps"])]:
                raise ValueError("Optimizer did not perform exactly K updates")
            if capture is not None:
                capture(name, "final", snapshot(model, optimizer, logits, list(trace)))
            rows[name] = {"training_loss": loss, "initial_screen": initial, "final_screen": final,
                          "steps": trace, "initial_optimizer_empty": True, "optimizer_step_counts": steps,
                          "final_state_sha256": state_hash(model), "final_optimizer_sha256": tree_hash(optimizer.state_dict()),
                          "parameter_count": sum(p.numel() for p in model.parameters()),
                          "private_block_parameter_count": block_count}
            artifacts[name] = (model, optimizer, logits)
        for name, row in rows.items():
            tied = rows[row["training_loss"] + "_tied"]["final_screen"]
            row["paired_pooled_bce_gain_vs_actual_tied_after_K"] = tied["pooled_bce"] - row["final_screen"]["pooled_bce"]
            row["paired_graph_gains_vs_actual_tied_after_K"] = [x - y for x, y in zip(
                tied["pooled_bce_by_graph"], row["final_screen"]["pooled_bce_by_graph"])]
    torch.cuda.synchronize(0)
    if state_hash(common) != before or inputs() != input_hashes:
        raise ValueError("Source or training input tensors changed")
    if gpu_environment() != environment:
        raise ValueError("GPU runtime flags changed during lookahead")
    return {"schema": "gnnm-molhiv-GPU-actual-optimizer-lookahead-cell-v3",
            "scope": "synthetic GPU fixture" if synthetic else "qualified training-only GPU mechanism screen",
            "seed": seed, "label_fraction": fraction, "source_state_sha256": before,
            "groups": groups, "coverage": coverage, "recipe": GPU_RECIPE, "rule": RULE,
            "environment": environment, "trajectories": rows, "input_hashes": input_hashes,
            "source_and_inputs_preserved": True, "screen_scored_completely": True,
            "predictive_accuracy_claimed": False, "novelty_claimed": False,
            "CPU_molecular_sibling_required": False, "validation_or_test_loaded": False}, artifacts
