"""Own-selected PolyFormer export with independent construction/dropout custody.

Fixed post-probability-pool scalar CAL/evaluation reuse the node endpoint laws.
Export remains separate because independent PolyFormer streams are distinct.
"""
import gc
import math
from pathlib import Path

try:
    from ..engine import restore_checkpoint
    from ..run_study import load_factory, resolve_selected_config
    from ..spec import confined, fingerprint
    from .node_endpoints import receipt, scalar_calibration, evaluate_bank
except ImportError:
    from engine import restore_checkpoint
    from run_study import load_factory, resolve_selected_config
    from spec import confined, fingerprint
    from adapters.node_endpoints import receipt, scalar_calibration, evaluate_bank


def independent_contracts(study, contracts):
    """Four paid M1 trajectories share recipe, never RNG streams or selectors."""
    if len(contracts) != 4 or {r["body"] for r in contracts} != {0, 1, 2, 3}:
        raise ValueError("PolyFormer I4 requires four separately paid body IDs0..3")
    group = contracts[0]["config"].get("independence_group")
    if not group or {r["run_id"] for r in study["runs"] if r["kind"] == "model_fit" and r["config"].get("independence_group") == group} != {r["run_id"] for r in contracts}:
        raise ValueError("Export must bind the complete prospectively declared independent group")
    if len({r["config"]["initialization_seed"] for r in contracts}) != 4:
        raise ValueError("Independent constructors must have distinct seed bindings")
    streams = [set(r["config"]["dropout_seeds"]) for r in contracts]
    if any(streams[i] & streams[j] for i in range(4) for j in range(i)):
        raise ValueError("Independent bodies must have disjoint private dropout stream seeds")
    excluded = {"initialization_seed", "dropout_seeds", "factor_seed", "resolved_selection"}
    recipes = [{k:v for k,v in r["config"].items() if k not in excluded} for r in contracts]
    if any(recipe != recipes[0] for recipe in recipes[1:]) or any(
            any(r[key] != contracts[0][key] for key in ("method", "adapter", "split", "seed", "budget", "selection")) for r in contracts[1:]):
        raise ValueError("Independent PolyFormer architecture/source/roles/objective/budget/selector differ")
    if contracts[0]["config"]["factor_initialization"] == "first_rademacher" and len({r["config"].get("factor_seed") for r in contracts}) != 4:
        raise ValueError("Independent Rademacher constructors require distinct factor seeds")


def export_bank(*, project_root, study, run, output_root, output_dir, registry, device):
    import torch
    config = run["config"]
    fits = config["fit_parents"]
    if config["fusion"] != "mean_probabilities" or len(fits) not in (1, 4) or len(set(fits)) != len(fits) or not set(fits).issubset(run["parents"]):
        raise ValueError("Selected PolyFormer export requires one bank or four distinct paid M1 parents")
    rows = [registry.run(study["study_id"], identifier) for identifier in fits]
    if any(row["state"] != "complete" or row["contract"]["kind"] != "model_fit" for row in rows):
        raise ValueError("Own-selected fit parents must be complete")
    contracts = [resolve_selected_config(study, row["contract"], registry) for row in rows]
    if len(fits) == 4:
        independent_contracts(study, contracts)
    elif contracts[0]["config"].get("independence_group"):
        if contracts[0]["body"] != 0 or config.get("singleton_reference") != "fixed_body0":
            raise ValueError("A singleton independent-group reference must be prospectively fixed body0")
    ids = {role: torch.load(confined(project_root, config["endpoint_inputs"][role+"_ids"]),
                           map_location="cpu", weights_only=True) for role in ("cal", "evaluate")}
    if any(not torch.is_tensor(value) or value.ndim != 1 or value.dtype != torch.long
           or not len(value) or len(torch.unique(value)) != len(value) or bool((value < 0).any()) for value in ids.values()):
        raise ValueError("Unique ordered label-free endpoint ID tensors required")
    if bool(torch.isin(ids["cal"], ids["evaluate"]).any()):
        raise ValueError("CAL/evaluation IDs overlap")
    outputs, ancestry = {role: [] for role in ids}, []
    for identifier, row, actual in zip(fits, rows, contracts):
        factory = load_factory(project_root, actual["adapter"])
        adapter = factory(torch=torch, project_root=project_root, study=study, run=actual, device=device)
        selected = row["result"]["selected"]
        relative = Path(selected["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Selected checkpoint path escapes its own paid fit")
        checkpoint = Path(output_root)/study["study_id"]/identifier/relative
        restore_checkpoint(torch, checkpoint, adapter, expected_fit_id=identifier)
        with torch.no_grad():
            adapter.model.eval()
            logits = adapter.logits_fn(adapter.model, False, selected["step"])
            if logits.ndim != 3 or not bool(torch.isfinite(logits).all()):
                raise ValueError("Finite native [routes,nodes,classes] bank required")
            if len(fits) == 4 and logits.shape[0] != 1:
                raise ValueError("Every independently selected fit must serve exactly one route")
            for role, positions in ids.items():
                if int(positions.max()) >= logits.shape[1] or any(bool(torch.isin(positions, training.cpu()).any()) for training in (adapter.train_ids, adapter.select_ids)):
                    raise ValueError("Endpoint IDs overlap TRAIN/SELECT or leave the native graph")
                outputs[role].append(logits[:, positions.to(logits.device)].cpu())
        ancestry.append({"run_id": identifier, "body": actual["body"], "selected_step": selected["step"],
            "checkpoint": receipt(checkpoint, project_root), "actual_model_seed": adapter.provenance["actual_model_seed"],
            "private_dropout_seeds": adapter.provenance["private_dropout_seeds"],
            "factor_seed": adapter.provenance["factor_seed"], "config_hash": fingerprint(actual["config"])})
        del adapter, logits
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    files = {}
    for role in ids:
        path = Path(output_dir)/(role+"_logits.pt")
        torch.save({"ids": ids[role], "logits": torch.cat(outputs[role], 0)}, path)
        files[role] = receipt(path, project_root)
    return {"status": "complete", "files": files, "ancestry": ancestry, "fusion": "mean_probabilities",
            "independent_complete_learners": len(fits) == 4,
            "singleton_reference_is_fixed_body0": config.get("singleton_reference") == "fixed_body0",
            "CAL_targets_read": False, "evaluation_targets_read": False, "scientific_backbone_fits": 0}
