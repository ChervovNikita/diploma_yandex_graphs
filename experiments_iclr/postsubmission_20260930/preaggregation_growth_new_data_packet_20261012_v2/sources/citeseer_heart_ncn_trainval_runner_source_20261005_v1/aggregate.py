#!/usr/bin/env python3
"""Four independently selected native constituents; aggregate VALID only."""
import argparse
import json
from pathlib import Path
import time
from run import authorize, phase_file, sha, write_json, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    job = authorize(args.job, output)
    if job.get("mode") != "aggregate" or job.get("complete_frozen_cohort") is not True:
        raise ValueError("Aggregation release requires root's complete frozen cohort")
    family = job.get("ensemble_family")
    if family not in ("ordinary_independent4", "independent_frame4") or len(job.get("component_freezes", [])) != 4:
        raise ValueError("Exactly four declared independent components required")
    import torch
    started = time.monotonic()
    output.mkdir()
    try:
        components, positions, predictions = [], set(), []
        for index, record in enumerate(job["component_freezes"]):
            freeze_path = phase_file(record["path"])
            if freeze_path.name != "FREEZE.json" or sha(freeze_path) != record["sha256"]:
                raise ValueError("Exact selected component freeze differs")
            freeze = json.loads(freeze_path.read_text())
            root = freeze_path.parent
            config_path = root / "CONFIG.json"
            logits_path = root / "selected_VALID_logits.pt"
            checkpoint_path = root / "selected_checkpoint.pt"
            if sha(config_path) != freeze["config_sha256"] or sha(logits_path) != freeze["VALID_logits_sha256"] or sha(checkpoint_path) != freeze["checkpoint_sha256"]:
                raise ValueError("Component weights/config/logits differ from freeze")
            config = json.loads(config_path.read_text())
            component_job = config["job"]
            expected_arm = "independent_member" if family == "ordinary_independent4" else "independent_frame_member"
            reused = index == 0 and family == "ordinary_independent4" and job.get("reuse_native_single_as_member0") is True and freeze["arm"] == "native_single"
            if not reused and (freeze["arm"] != expected_arm or component_job["ensemble_member_index"] != index):
                raise ValueError("Component operation/index does not match ordinary ensemble")
            if family == "independent_frame4" and component_job["axis_index"] != index:
                raise ValueError("Identical-operation ensemble requires independent axes0,1,2,3")
            if freeze["member_count"] != 1 or freeze["paired_seed_block"] != job["paired_seed_block"] or freeze["cohort_plan_sha256"] != job["cohort_plan_sha256"] or freeze["TEST_access"] is not False:
                raise ValueError("Component cohort/native-single role differs")
            if freeze["seed"] in positions:
                raise ValueError("Independent constituent seeds must be distinct")
            positions.add(freeze["seed"])
            saved = torch.load(logits_path, map_location="cpu", weights_only=True)
            if saved["checkpoint_sha256"] != freeze["checkpoint_sha256"] or saved["input_identities"] != freeze["input_identities"] or saved["selected_epoch"] != freeze["selected_epoch"]:
                raise ValueError("Prediction/checkpoint/data binding differs")
            if saved["pos"].shape != (227,) or saved["neg"].shape != (227, 500):
                raise ValueError("Incomplete same-pool VALID arrays")
            if predictions and saved["input_identities"] != predictions[0]["input_identities"]:
                raise ValueError("Ensemble components used different exact benchmark bytes")
            predictions.append(saved)
            components.append({"path": record["path"], "freeze_sha256": record["sha256"], "seed": freeze["seed"],
                               "selected_epoch": freeze["selected_epoch"], "checkpoint_sha256": freeze["checkpoint_sha256"],
                               "inclusive_component_seconds": freeze["inclusive_seconds"]})
        pos = torch.stack([item["pos"] for item in predictions]).mean(0)
        neg = torch.stack([item["neg"] for item in predictions]).mean(0)
        if not bool(torch.isfinite(pos).all() and torch.isfinite(neg).all()):
            raise FloatingPointError("Nonfinite fixed ensemble VALID logits")
        torch.save({"pos": pos, "neg": neg, "components": components, "input_identities": predictions[0]["input_identities"]}, output / "selected_VALID_logits.pt")
        result = {"schema": "citeseer_independent_ensemble_VALID_freeze_v1", "ensemble_family": family,
                  "member_count": 4, "components": components, "VALID_metric": metric(pos, neg),
                  "selection": "each native constituent selected independently by VALID MRR; no ensemble retuning/selection",
                  "serving": "mean raw logits", "TEST_access": False, "fit_count": 0, "optimizer_updates": 0,
                  "aggregate_seconds": time.monotonic() - started, "job_sha256": sha(args.job),
                  "VALID_logits_sha256": sha(output / "selected_VALID_logits.pt")}
        write_json(output / "ENSEMBLE_FREEZE.json", result)
        print(json.dumps({"status": "independent_ensemble_fixed_TEST_closed", "family": family}))
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "FAILURE.json", {"error": type(error).__name__ + ": " + str(error), "TEST_access": False, "retry": False})
        raise


if __name__ == "__main__":
    main()
