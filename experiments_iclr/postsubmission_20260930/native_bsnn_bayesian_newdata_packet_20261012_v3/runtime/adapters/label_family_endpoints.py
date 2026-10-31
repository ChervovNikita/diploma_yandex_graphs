"""Selected banks, fixed scalar calibration and closed-cohort node scoring.

Export reads label-free CAL/evaluation IDs. Calibration and evaluation targets
are opened only in their separate declared endpoint jobs after their barriers.
"""
import gc
import hashlib
import json
from pathlib import Path

try:
    from ..engine import restore_checkpoint
    from ..run_study import load_factory, resolve_selected_config
    from ..spec import confined, fingerprint
except ImportError:
    from engine import restore_checkpoint
    from run_study import load_factory, resolve_selected_config
    from spec import confined, fingerprint


def receipt(path, root):
    raw = path.read_bytes()
    return {"path": str(path.resolve().relative_to(Path(root).resolve())), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def export_bank(*, project_root, study, run, output_root, output_dir, registry, device):
    import torch
    config = run["config"]
    fits = config["fit_parents"]
    if len(fits) not in (1, 4) or not set(fits).issubset(run["parents"]):
        raise ValueError("Export must identify one selected bank or four separately trained members")
    role_ids = {role: torch.load(confined(project_root, config["endpoint_inputs"][role + "_ids"]),
                                 map_location="cpu", weights_only=True)
                for role in ("cal", "evaluate")}
    if any(not torch.is_tensor(ids) or ids.ndim != 1 or ids.dtype != torch.long for ids in role_ids.values()):
        raise ValueError("Export ID files must contain ordered int64 tensors only")
    if torch.isin(role_ids["cal"], role_ids["evaluate"]).any():
        raise ValueError("CAL and evaluation overlap")
    outputs = {role: [] for role in role_ids}
    ancestry = []
    contracts = []
    for identifier in fits:
        row = registry.run(study["study_id"], identifier)
        if row["state"] != "complete" or row["contract"]["kind"] != "model_fit":
            raise ValueError("Prediction export requires completed selected model fits")
        actual = resolve_selected_config(study, row["contract"], registry)
        factory = load_factory(project_root, actual["adapter"])
        adapter = factory(torch=torch, project_root=project_root, study=study, run=actual, device=device)
        checkpoint_role = config.get("checkpoint_role", "selected")
        if checkpoint_role not in ("selected", "final"):
            raise ValueError("Declared selected or final checkpoint required")
        checkpoint_name = "final_training.pt" if checkpoint_role == "final" else row["result"]["selected"]["path"]
        actual_step = row["result"]["completed_steps"] if checkpoint_role == "final" else row["result"]["selected"]["step"]
        checkpoint = output_root / study["study_id"] / identifier / checkpoint_name
        restore_checkpoint(torch, checkpoint, adapter, expected_fit_id=identifier)
        adapter.model.eval()
        with torch.no_grad():
            logits = adapter.logits_fn(adapter.model, False, actual_step)
            if config.get("readout") == "native_at_corrected_selected":
                logits = adapter.source_family._last_native_logits
            if logits.ndim != 3 or not torch.isfinite(logits).all():
                raise ValueError("Export expects finite [routes,nodes,classes] logits")
            if len(fits) == 4 and logits.shape[0] != 1:
                raise ValueError("Genuine I4 requires four single-view complete learners")
            for role, ids in role_ids.items():
                outputs[role].append(logits[:, ids.to(logits.device)].detach().cpu())
        ancestry.append({"run_id": identifier, "checkpoint": receipt(checkpoint, project_root),
                         "selected_step": actual_step, "checkpoint_role": checkpoint_role,
                         "actual_model_seed": adapter.provenance.get("actual_model_seed"),
                         "config_hash": fingerprint(actual["config"])})
        contracts.append(actual)
        del adapter, logits
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if len(fits) == 4:
        seeds = [a["actual_model_seed"] for a in ancestry]
        if None in seeds or len(set(seeds)) != 4:
            raise ValueError("Independent ensemble seed ancestry is incomplete or duplicated")
        # Width/objective/fusion differences cannot be hidden in an I4 label.
        excluded = {"initialization_seed", "resolved_selection", "reservation_mib", "estimated_seconds"}
        views = [{k:v for k,v in row["config"].items() if k not in excluded} for row in contracts]
        if any(v != views[0] for v in views[1:]):
            raise ValueError("Equipped I4 member configurations differ")
    files = {}
    for role in outputs:
        path = output_dir / (role + "_logits.pt")
        torch.save({"ids": role_ids[role], "logits": torch.cat(outputs[role], dim=0)}, path)
        files[role] = receipt(path, project_root)
    return {"status": "complete", "ancestry": ancestry, "files": files,
            "fusion": config["fusion"], "independent_complete_learners": len(fits) == 4,
            "CAL_targets_read": False, "evaluation_targets_read": False,
            "scientific_backbone_fits": 0}


def pooled_log_scores(torch, logits, fusion):
    if logits.ndim != 3 or not torch.isfinite(logits).all():
        raise ValueError("Finite bank logits required")
    if fusion == "mean_logits":
        return logits.mean(0)
    if fusion == "mean_probabilities":
        import math
        return torch.logsumexp(logits.log_softmax(-1), dim=0) - math.log(logits.shape[0])
    raise ValueError("Unknown serving law")


def load_bound(torch, project_root, row):
    path = confined(project_root, row["path"])
    observed = receipt(path, project_root)
    if observed != row:
        raise ValueError("Prediction bank changed after export")
    return torch.load(path, map_location="cpu", weights_only=True)


def scalar_calibration(*, project_root, study, run, output_root, output_dir, registry, device):
    import torch
    from torch.nn import functional as F
    config = run["config"]
    export_id = config["export_parent"]
    row = registry.run(study["study_id"], export_id)
    if row["state"] != "complete" or export_id not in run["parents"]:
        raise ValueError("Calibration requires its completed label-free bank export")
    bank = load_bound(torch, project_root, row["result"]["files"]["cal"])
    targets = torch.load(confined(project_root, config["endpoint_inputs"]["cal_targets"]),
                         map_location="cpu", weights_only=True)
    if set(targets) != {"ids", "y"} or not torch.equal(bank["ids"], targets["ids"]):
        raise ValueError("CAL target ordering differs from exported rows")
    if config["placement"] != "after_serving_pool" or config["steps"] != 500 or config["learning_rate"] != .01:
        raise ValueError("Calibration packet fixes one scalar, 500 Adam steps and placement prospectively")
    scores = pooled_log_scores(torch, bank["logits"].double(), config["fusion"]).detach()
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=config["learning_rate"])
    for step in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(scores / parameter.exp(), targets["y"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite scalar CAL loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(-4, 4)
    return {"status": "complete", "temperature": float(parameter.detach().exp()),
            "placement": config["placement"], "fusion": config["fusion"],
            "CAL_examples": len(targets["y"]), "scalar_optimizer_updates": config["steps"],
            "export_parent": export_id, "evaluation_targets_read": False,
            "scientific_backbone_fits": 0}


def evaluate_bank(*, project_root, study, run, output_root, output_dir, registry, device):
    import torch
    config = run["config"]
    export_id, cal_id = config["export_parent"], config["calibration_parent"]
    if not {export_id, cal_id}.issubset(run["parents"]):
        raise ValueError("Evaluation must bind its export and scalar CAL dependencies")
    exported = registry.run(study["study_id"], export_id)["result"]
    calibration = registry.run(study["study_id"], cal_id)["result"]
    if calibration["export_parent"] != export_id or calibration["fusion"] != config["fusion"]:
        raise ValueError("Calibration belongs to another bank or serving law")
    bank = load_bound(torch, project_root, exported["files"]["evaluate"])
    targets = torch.load(confined(project_root, config["endpoint_inputs"]["evaluate_targets"]),
                         map_location="cpu", weights_only=True)
    if set(targets) != {"ids", "y"} or not torch.equal(bank["ids"], targets["ids"]):
        raise ValueError("Evaluation target ordering differs")
    logits, labels = bank["logits"].double(), targets["y"]
    scores = pooled_log_scores(torch, logits, config["fusion"])
    results = {}
    for name, temperature in (("raw", 1.), ("calibrated", calibration["temperature"])):
        lp = (scores / temperature).log_softmax(-1)
        probability = lp.exp()
        correct = probability.argmax(-1).eq(labels)
        one_hot = torch.nn.functional.one_hot(labels, probability.shape[1])
        results[name] = {"accuracy": float(correct.double().mean()),
                         "correct": int(correct.sum()), "examples": len(labels),
                         "nll": float(-lp.gather(1, labels[:,None]).mean()),
                         "brier_sum_classes": float((probability-one_hot).square().sum(-1).mean())}
    members = {"accuracies": [float(z.argmax(-1).eq(labels).double().mean()) for z in logits],
               "oracle_accuracy_descriptive_only": float(logits.argmax(-1).eq(labels[None]).any(0).double().mean()),
               "all_members_wrong_fraction": float((~logits.argmax(-1).eq(labels[None]).any(0)).double().mean())}
    # Node rows are not treated as independent statistical replicates. Paired
    # seed/split inference belongs to the whole-cohort analysis endpoint.
    return {"status": "complete", "metrics": results, "member_diagnostics": members,
            "evaluation_scope": study["data"]["prior_exposure"], "fusion": config["fusion"],
            "ancestry": exported["ancestry"], "statistical_inference": "whole_seed_split_cohort_required",
            "scientific_backbone_fits": 0}
