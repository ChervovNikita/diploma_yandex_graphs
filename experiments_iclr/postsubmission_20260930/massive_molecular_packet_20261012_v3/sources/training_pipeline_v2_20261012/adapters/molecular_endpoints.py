"""Selected molecular logits, post-pool scalar CAL and masked binary scoring.

Only endpoint jobs open their corresponding target authority. Export opens
graph-only packs and own-selected checkpoint parents, never CAL/evaluate truth.
"""
from functools import wraps
import gc
import hashlib
import json
import math
from pathlib import Path

try:
    from ..engine import restore_checkpoint
    from ..run_study import load_factory, resolve_selected_config
    from ..spec import fingerprint
    from .molecular import bound_file, validate_prediction_pack, validate_independent_roster
except ImportError:
    from engine import restore_checkpoint
    from run_study import load_factory, resolve_selected_config
    from spec import fingerprint
    from adapters.molecular import bound_file, validate_prediction_pack, validate_independent_roster


def receipt(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Missing, symlinked or unconfined endpoint artifact")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return {"path": str(path.resolve().relative_to(root)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def load_bound(torch, root, row):
    return torch.load(bound_file(root, row), map_location="cpu", weights_only=True)


def retain_costs(function):
    """Retain completed lower bounds even if the root runner catches failure."""
    @wraps(function)
    def wrapped(**kwargs):
        counts = {"prediction_pack_attempts": 0, "prediction_packs_completed": 0,
                  "model_forwards": 0, "route_predictions": 0,
                  "scalar_optimizer_update_attempts": 0, "scalar_optimizer_updates": 0,
                  "scalar_backward_calls": 0, "scientific_backbone_fits": 0}
        try:
            result = function(operation_counts=counts, **kwargs)
            result["operation_counts"] = dict(counts)
            return result
        except BaseException as error:
            failure = {"status": "failed", "failure_type": type(error).__name__,
                       "failure": str(error), "operation_counts": counts,
                       "partial_inference_counts_are_completed_lower_bounds": True}
            destination = Path(kwargs["output_dir"])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "failure_costs.json").write_text(json.dumps(failure, indent=2, allow_nan=False)+"\n")
            error.endpoint_costs = failure
            raise
    return wrapped


def pooled_logits(torch, logits, fusion):
    if (fusion != "mean_logits" or not torch.is_tensor(logits) or logits.ndim != 3
            or min(logits.shape) < 1 or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())):
        raise ValueError("Molecular serving requires finite [members,graphs,tasks] raw logits and mean_logits")
    return logits.mean(0)


def _bank(torch, bank, fusion):
    if not isinstance(bank, dict) or set(bank) != {"ids", "logits"}:
        raise ValueError("An exact exported ID/raw-logit bank is required")
    ids = bank["ids"]
    if (not torch.is_tensor(ids) or ids.dtype != torch.long or ids.ndim != 1
            or len(torch.unique(ids)) != len(ids) or bool((ids < 0).any())):
        raise ValueError("Invalid exported graph identities")
    scores = pooled_logits(torch, bank["logits"].double(), fusion)
    if scores.shape[0] != len(ids):
        raise ValueError("Exported logits and graph IDs differ")
    return scores


def _targets(torch, targets, ids, outputs, policy):
    if not isinstance(targets, dict) or set(targets) != {"ids", "y"}:
        raise ValueError("Targets require a separate exact {ids,y} authority")
    if (not torch.is_tensor(targets["ids"]) or targets["ids"].dtype != torch.long
            or not torch.equal(ids, targets["ids"])):
        raise ValueError("Target identities/order differ from the exported bank")
    y = targets["y"]
    if not torch.is_tensor(y) or not y.is_floating_point() or y.requires_grad:
        raise ValueError("Immutable floating binary target tensors required")
    if y.ndim == 1 and outputs == 1:
        y = y[:, None]
    if y.shape != (len(ids), outputs) or bool(torch.isinf(y).any()):
        raise ValueError("Target shape changed; missing binary labels must be NaN")
    mask = torch.isfinite(y)
    if policy not in {"complete_binary", "observed_binary_mask"}:
        raise ValueError("Declare the binary target policy")
    if policy == "complete_binary" and not bool(mask.all()):
        raise ValueError("Complete binary endpoint contains missing labels")
    if bool(((y[mask] != 0) & (y[mask] != 1)).any()):
        raise ValueError("Observed endpoint targets must be binary")
    return y.double(), mask


def _endpoint_inputs(study, config):
    inputs = config["endpoint_inputs"]
    if set(inputs) != {"cal_graphs", "evaluate_graphs", "cal_targets", "evaluate_targets"}:
        raise ValueError("Declare graph-only and separate target receipts for both endpoints")
    declared = study["data"]["evidence"]["endpoint_inputs"]
    if inputs != declared or config["target_policy"] != study["data"]["target_policy"]:
        raise ValueError("Endpoint authority or target policy differs from the study")
    return inputs


def _complete_parent(study, registry, run, identifier, kind):
    if identifier not in run["parents"]:
        raise ValueError("Endpoint parent is outside the declared dependency roster")
    row = registry.run(study["study_id"], identifier)
    if row["state"] != "complete" or row["contract"]["kind"] != kind:
        raise ValueError("Endpoint requires a completed " + kind + " parent")
    return row


@retain_costs
def export_bank(*, project_root, study, run, output_root, output_dir, registry, device, operation_counts):
    import torch
    config = run["config"]
    inputs = _endpoint_inputs(study, config)
    fits = config["fit_parents"]
    if len(fits) not in (1, 4) or len(set(fits)) != len(fits):
        raise ValueError("Export needs one own-selected fit or exactly four distinct I4 bodies")
    if config["fusion"] != "mean_logits":
        raise ValueError("Molecular serving is mean raw logits then sigmoid")
    packs = {role: validate_prediction_pack(torch, load_bound(torch, project_root, inputs[role+"_graphs"]))
             for role in ("cal", "evaluate")}
    if bool(torch.isin(packs["cal"]["ids"], packs["evaluate"]["ids"]).any()):
        raise ValueError("CAL/evaluation graph identities overlap")
    outputs, ancestry, contracts = {role: [] for role in packs}, [], []
    for identifier in fits:
        row = _complete_parent(study, registry, run, identifier, "model_fit")
        actual = resolve_selected_config(study, row["contract"], registry)
        if actual["config"]["target_policy"] != config["target_policy"]:
            raise ValueError("Fit and endpoint target policy differ")
        if len(fits) == 4:
            if "independent4_body" not in actual["config"]["arm"]:
                raise ValueError("Genuine I4 requires typed separately paid body runs")
            validate_independent_roster(study, actual)
        elif "independent4_body" in actual["config"]["arm"]:
            equivalents = {"independent4_body": "single", "independent4_body_contrastive": "single_contrastive",
                           "factorized_independent4_body": "factorized_single",
                           "factorized_independent4_body_contrastive": "factorized_single_contrastive"}
            if actual["body"] != 0 or config.get("single_reference_arm") != equivalents[actual["config"]["arm"]]:
                raise ValueError("A singleton I4 export must be the prospectively fixed body0 reference")
            validate_independent_roster(study, actual)
        factory = load_factory(project_root, actual["adapter"])
        adapter = factory(torch=torch, project_root=project_root, study=study, run=actual, device=device)
        selected = row["result"]["selected"]
        selected_path = Path(selected["path"])
        if selected_path.is_absolute() or ".." in selected_path.parts:
            raise ValueError("Selected checkpoint must stay inside its own fit output")
        checkpoint = Path(output_root) / study["study_id"] / identifier / selected_path
        restore_checkpoint(torch, checkpoint, adapter, expected_fit_id=identifier)
        for role, pack in packs.items():
            if any(bool(torch.isin(pack["ids"], adapter_pack["ids"]).any()) for adapter_pack in (adapter.train, adapter.select)):
                raise ValueError("Heldout endpoint graphs overlap TRAIN/SELECT")
            operation_counts["prediction_pack_attempts"] += 1
            logits = adapter.predict_pack(pack)
            pooled_logits(torch, logits, config["fusion"])
            if logits.shape[1:] != (len(pack["ids"]), actual["config"]["model"]["outputs"]):
                raise ValueError("Molecular prediction shape differs from graph/task roster")
            if len(fits) == 4 and logits.shape[0] != 1:
                raise ValueError("Each independently selected body must supply exactly one route")
            operation_counts["prediction_packs_completed"] += 1
            operation_counts["model_forwards"] += logits.shape[0] * math.ceil(len(pack["ids"])/actual["config"]["training"]["eval_batch_size"])
            operation_counts["route_predictions"] += logits.shape[0] * len(pack["ids"])
            outputs[role].append(logits.detach().cpu())
        ancestry.append({"run_id": identifier, "body": actual["body"], "checkpoint": receipt(checkpoint, project_root),
                         "selected_epoch": selected["step"], "config_hash": fingerprint(actual["config"]),
                         **{key: actual["config"][key] for key in ("initialization_seed", "dropout_seed", "shuffle_seed")}})
        contracts.append(actual)
        del adapter, logits
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if len(fits) == 4:
        group = contracts[0]["config"]["independence_group"]
        expected = {r["run_id"] for r in study["runs"] if r["kind"] == "model_fit" and r["config"].get("independence_group") == group}
        if set(fits) != expected:
            raise ValueError("Export does not bind the complete declared I4 group")
        excluded = {"initialization_seed", "dropout_seed", "shuffle_seed"}
        recipes = [{k:v for k,v in row["config"].items() if k not in excluded} for row in contracts]
        if any(value != recipes[0] for value in recipes[1:]):
            raise ValueError("Resolved independent body recipes differ")
    files = {}
    for role in packs:
        path = Path(output_dir) / (role+"_logits.pt")
        torch.save({"ids": packs[role]["ids"], "logits": torch.cat(outputs[role], 0)}, path)
        files[role] = receipt(path, project_root)
    return {"status": "complete", "files": files, "ancestry": ancestry,
            "fusion": "mean_logits", "probability_law": "sigmoid(mean_raw_member_logits)",
            "target_policy": config["target_policy"], "graph_inputs": {r: inputs[r+"_graphs"] for r in packs},
            "independent_complete_learners": len(fits) == 4,
            "singleton_reference_is_fixed_body0": "single_reference_arm" in config, "CAL_targets_read": False,
            "evaluation_targets_read": False, "scientific_backbone_fits": 0}


@retain_costs
def scalar_calibration(*, project_root, study, run, output_root, output_dir, registry, device, operation_counts):
    import torch
    config = run["config"]
    inputs = _endpoint_inputs(study, config)
    identifier = config["export_parent"]
    exported = _complete_parent(study, registry, run, identifier, "ensemble_assembly")["result"]
    if exported["fusion"] != config["fusion"] or exported["target_policy"] != config["target_policy"]:
        raise ValueError("CAL and export serving/target laws differ")
    if config["placement"] != "after_serving_pool" or config["steps"] != 500 or config["learning_rate"] != .01:
        raise ValueError("Scalar CAL fixes post-pool placement, 500 Adam steps and learning rate .01")
    bank = load_bound(torch, project_root, exported["files"]["cal"])
    scores = _bank(torch, bank, config["fusion"]).detach()
    targets = load_bound(torch, project_root, inputs["cal_targets"])
    y, mask = _targets(torch, targets, bank["ids"], scores.shape[1], config["target_policy"])
    observed = int(mask.sum())
    if not observed:
        raise ValueError("Scalar CAL requires at least one observed binary target")
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores[mask] / parameter.exp(), y[mask])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite scalar molecular CAL loss")
        loss.backward()
        operation_counts["scalar_backward_calls"] += 1
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad)):
            raise FloatingPointError("Nonfinite scalar molecular CAL gradient")
        operation_counts["scalar_optimizer_update_attempts"] += 1
        optimizer.step()
        operation_counts["scalar_optimizer_updates"] += 1
        with torch.no_grad():
            parameter.clamp_(-4, 4)
        if not bool(torch.isfinite(parameter)):
            raise FloatingPointError("Nonfinite scalar molecular CAL parameter")
    return {"status": "complete", "temperature": float(parameter.detach().exp()),
            "placement": "after_serving_pool", "fusion": config["fusion"],
            "target_policy": config["target_policy"], "CAL_graphs": len(y),
            "CAL_observed_targets": observed, "CAL_missing_targets": y.numel()-observed,
            "scalar_optimizer_updates": 500, "export_parent": identifier,
            "CAL_target_receipt": inputs["cal_targets"], "export_CAL_receipt": exported["files"]["cal"],
            "evaluation_targets_read": False, "scientific_backbone_fits": 0}


def binary_metrics(torch, scores, y, mask):
    """Observed-target proper losses; macro AUC only over OGB-eligible tasks."""
    from sklearn.metrics import roc_auc_score
    observed = int(mask.sum())
    probability = scores.sigmoid()
    tasks, aucs = [], []
    for task in range(y.shape[1]):
        values, z = y[mask[:, task], task], scores[mask[:, task], task]
        positives, negatives = int((values == 1).sum()), int((values == 0).sum())
        eligible = positives > 0 and negatives > 0
        auc = float(roc_auc_score(values.numpy(), z.numpy())) if eligible else None
        if eligible:
            aucs.append(auc)
        tasks.append({"task": task, "observed": len(values), "missing": len(y)-len(values),
                      "positive": positives, "negative": negatives, "rocauc_eligible": eligible, "rocauc": auc})
    return {"rocauc": sum(aucs)/len(aucs) if aucs else None,
            "rocauc_eligible_tasks": len(aucs), "rocauc_ineligible_tasks": y.shape[1]-len(aucs),
            "rocauc_task_accounting": tasks,
            "nll_observed_targets": float(torch.nn.functional.binary_cross_entropy_with_logits(scores[mask], y[mask])) if observed else None,
            "brier_observed_targets": float((probability[mask]-y[mask]).square().mean()) if observed else None,
            "graphs": len(y), "tasks": y.shape[1], "observed_targets": observed,
            "missing_targets": y.numel()-observed, "proper_loss_denominator": observed,
            "rocauc_aggregation": "macro_mean_over_tasks_with_both_observed_binary_classes"}


@retain_costs
def evaluate_bank(*, project_root, study, run, output_root, output_dir, registry, device, operation_counts):
    import torch
    config = run["config"]
    inputs = _endpoint_inputs(study, config)
    export_id, cal_id = config["export_parent"], config["calibration_parent"]
    exported = _complete_parent(study, registry, run, export_id, "ensemble_assembly")["result"]
    calibrated = _complete_parent(study, registry, run, cal_id, "calibration")["result"]
    if (calibrated["export_parent"] != export_id or calibrated["export_CAL_receipt"] != exported["files"]["cal"]
            or calibrated["fusion"] != config["fusion"] or exported["fusion"] != config["fusion"]
            or calibrated["target_policy"] != config["target_policy"] or exported["target_policy"] != config["target_policy"]
            or calibrated["placement"] != "after_serving_pool"):
        raise ValueError("Evaluation and calibration bank custody differ")
    temperature = calibrated["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Scalar molecular temperature must be finite and positive")
    bank = load_bound(torch, project_root, exported["files"]["evaluate"])
    scores = _bank(torch, bank, config["fusion"])
    targets = load_bound(torch, project_root, inputs["evaluate_targets"])
    y, mask = _targets(torch, targets, bank["ids"], scores.shape[1], config["target_policy"])
    results = {name: binary_metrics(torch, scores/value, y, mask)
               for name, value in (("raw", 1.), ("calibrated", temperature))}
    return {"status": "complete", "metrics": results, "temperature": temperature,
            "fusion": config["fusion"], "probability_law": "sigmoid(mean_raw_member_logits / temperature)",
            "target_policy": config["target_policy"], "ancestry": exported["ancestry"],
            "evaluation_target_receipt": inputs["evaluate_targets"], "export_parent": export_id,
            "calibration_parent": cal_id, "evaluation_scope": study["data"]["prior_exposure"],
            "statistical_inference": "whole_paired_seed_cohort_required; graph_rows_are_not_independent_replicates",
            "scientific_backbone_fits": 0}
