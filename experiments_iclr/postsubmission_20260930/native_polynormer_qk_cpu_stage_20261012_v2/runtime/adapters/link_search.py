"""Whole declared link search selection from own-selected TRAIN/SELECT records."""
import copy
import math
try:
    from ..spec import fingerprint
except ImportError:
    from spec import fingerprint


def select_configuration(*, project_root, study, run, output_root, output_dir, registry, device):
    config = run["config"]
    if config["ranking_metric"] != "hits@50" or config["ranking_direction"] != "max":
        raise ValueError("This link packet freezes native SELECT Hits@50 ranking")
    ranking, failed = [], []
    for candidate in config["configurations"]:
        values, losses = [], []
        for identifier in candidate["run_ids"]:
            row = registry.run(study["study_id"], identifier)
            if row["state"] == "failed":
                failed.append({"config_id": candidate["config_id"], "run_id": identifier, "failure": row["result"]})
                continue
            if row["state"] != "complete" or row["contract"]["config"].get("stage") != "search":
                raise ValueError("Whole declared link search must close before configuration ranking")
            result = row["result"]
            if result["completed_steps"] != row["contract"]["budget"]["maximum"]:
                raise ValueError("Partial native trajectories cannot rank as full-budget configurations")
            metrics = result["selected"]["validation"]
            score, loss = float(metrics["hits@50"]), float(metrics["ce"])
            if not math.isfinite(score) or not math.isfinite(loss):
                raise ValueError("Nonfinite selected link endpoint needs an explicit failure")
            values.append(score); losses.append(loss)
        if len(values) == len(candidate["run_ids"]):
            ranking.append({"config_id": candidate["config_id"], "parameters": candidate["parameters"],
                "run_ids": candidate["run_ids"], "mean_select_metric": sum(values)/len(values),
                "mean_select_loss": sum(losses)/len(losses), "seed_select_metrics": values})
    if not ranking:
        raise ValueError("No link configuration completed every declared search seed")
    ranking.sort(key=lambda row: (-row["mean_select_metric"], row["mean_select_loss"], row["config_id"]))
    selected = copy.deepcopy({key: ranking[0][key] for key in ("config_id", "parameters", "run_ids")})
    return {"status": "complete", "ranking_source": "whole_declared_TRAIN_SELECT_link_search",
        "selected_configuration": selected, "choice_hash": fingerprint(selected), "ranking": ranking,
        "failed_search_trajectories": failed, "ranking_metric": "hits@50", "ranking_direction": "max",
        "CAL_read": False, "evaluate_read": False, "scientific_backbone_fits": 0}
