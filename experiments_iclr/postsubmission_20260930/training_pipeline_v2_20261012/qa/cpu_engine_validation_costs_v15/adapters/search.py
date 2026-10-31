"""Freeze one complete search cell using TRAIN/SELECT records only."""
import copy
import math

try:
    from ..spec import fingerprint
except ImportError:
    from spec import fingerprint


def select_configuration(*, project_root, study, run, output_root, output_dir, registry, device):
    config = run["config"]
    if config["ranking_metric"] not in {"accuracy", "pooled_nll", "objective", "rocauc"}:
        raise ValueError("Unsupported predeclared search metric")
    direction = config["ranking_direction"]
    if direction not in {"max", "min"}:
        raise ValueError("Explicit ranking direction required")
    ranking, failed = [], []
    for candidate in config["configurations"]:
        values, losses = [], []
        for identifier in candidate["run_ids"]:
            row = registry.run(study["study_id"], identifier)
            if row["state"] == "failed":
                failed.append({"config_id": candidate["config_id"], "run_id": identifier,
                               "failure": row["result"]})
                continue
            if row["state"] != "complete" or row["contract"]["config"].get("stage") != "search":
                raise ValueError("Every declared search seed must be terminal before ranking")
            result = row["result"]
            if result["completed_steps"] != row["contract"]["budget"]["maximum"]:
                raise ValueError("A partial trajectory cannot enter full-budget search ranking")
            metrics = result["selected"]["validation"]
            value = float(metrics[config["ranking_metric"]])
            proper_loss = float(metrics.get("pooled_nll", metrics.get("ce", metrics["objective"])))
            if not math.isfinite(value) or not math.isfinite(proper_loss):
                raise ValueError("A nonfinite selected endpoint requires an explicit failure receipt")
            values.append(value)
            losses.append(proper_loss)
        if len(values) == len(candidate["run_ids"]):
            ranking.append({"config_id": candidate["config_id"], "parameters": candidate["parameters"],
                            "run_ids": candidate["run_ids"], "mean_select_metric": sum(values)/len(values),
                            "mean_select_loss": sum(losses)/len(losses), "seed_select_metrics": values})
    if not ranking:
        raise ValueError("No configuration completed all declared search seeds")
    ranking.sort(key=lambda r: ((-1 if direction == "max" else 1)*r["mean_select_metric"],
                                r["mean_select_loss"], r["config_id"]))
    selected = copy.deepcopy({k: ranking[0][k] for k in ("config_id", "parameters", "run_ids")})
    return {"status": "complete", "ranking_source": "whole_declared_TRAIN_SELECT_search_cell",
            "selected_configuration": selected, "choice_hash": fingerprint(selected),
            "ranking": ranking, "failed_search_trajectories": failed,
            "ranking_metric": config["ranking_metric"], "ranking_direction": direction,
            "CAL_read": False, "evaluate_read": False, "scientific_backbone_fits": 0}
