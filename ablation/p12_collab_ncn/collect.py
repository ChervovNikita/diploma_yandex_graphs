"""Score selected independent-member arrays with the official OGB evaluator."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from ablation.common.resumable import atomic_torch_save, atomic_write_json
from .protocol import REPOSITORY, digest_json, job_key
from .train import metrics


def read_fit(root, cohort, method):
    directory = root / job_key(cohort, method)
    if not (directory / "done.json").exists():
        return None
    result = json.loads((directory / "done.json").read_text())
    artifact = torch.load(directory / "selected_predictions.pt", map_location="cpu", weights_only=False)
    if artifact["config"]["cohort"] != cohort or artifact["config"]["method"] != method:
        raise ValueError(f"Selected artifact occupies a conflicting job directory: {directory}")
    if artifact["config"] != json.loads((directory / "config.json").read_text()):
        raise ValueError(f"Selected artifact/config mismatch in {directory}")
    if artifact["identity"]["config_sha256"] != digest_json(artifact["config"]):
        raise ValueError(f"Selected artifact config digest mismatch in {directory}")
    if any(result[key] != value for key, value in artifact["config"].items()):
        raise ValueError(f"Completed result/config mismatch in {directory}")
    if artifact["identity"] != result["identity"]:
        raise ValueError(f"Selected artifact/result identity mismatch in {directory}")
    test_order = artifact.get("test_edge_order_sha256")
    if "test_pos" in artifact["predictions"] and not test_order:
        raise ValueError(f"Selected TEST edge-order identity is missing in {directory}")
    if test_order != result.get("test_edge_order_sha256"):
        raise ValueError(f"Selected TEST edge-order identity differs from result in {directory}")
    return result, artifact


def ensemble_artifact(fits):
    if len(fits) != 4:
        raise ValueError("Independent4 requires exactly four selected native members")
    reference = fits[0][1]
    test_order = reference.get("test_edge_order_sha256")
    if "test_pos" in reference["predictions"] and not test_order:
        raise ValueError("Independent-member TEST edge-order identity is missing")
    identities = [fit[1]["identity"]["config_sha256"] for fit in fits]
    if len(set(identities)) != 4:
        raise ValueError("Duplicate independent-member identities")
    common_config = {key: value for key, value in reference["config"].items()
                     if key not in ("method", "seed")}
    for member, (result, artifact) in enumerate(fits):
        if artifact["config"]["method"] != f"native{member}":
            raise ValueError("Independent4 requires native0 through native3 in order")
        if artifact["identity"]["config_sha256"] != digest_json(artifact["config"]):
            raise ValueError("Independent-member configuration digest mismatch")
        expected_seed = artifact["config"]["cohort"] * 1000 + member
        if artifact["config"]["seed"] != expected_seed:
            raise ValueError("Conflicting independent-member seed identity")
        member_config = {key: value for key, value in artifact["config"].items()
                         if key not in ("method", "seed")}
        if member_config != common_config:
            raise ValueError("Independent-member configurations conflict")
        for key in ("data_sha256", "versions", "torch_cuda"):
            if artifact["identity"][key] != reference["identity"][key]:
                raise ValueError(f"Independent-member identity mismatch for {key}")
        if artifact.get("test_edge_order_sha256") != test_order:
            raise ValueError("Independent-member TEST edge ordering differs")
        if artifact["predictions"].keys() != reference["predictions"].keys():
            raise ValueError("Independent members have different evaluation roles")
    predictions = {}
    for name, values in reference["predictions"].items():
        members = [fit[1]["predictions"][name] for fit in fits]
        if any(member.shape != values.shape for member in members):
            raise ValueError(f"Independent-member prediction shape mismatch for {name}")
        predictions[name] = torch.stack(members).mean(dim=0)
    return {
        "cohort": reference["config"]["cohort"],
        "members": [fit[1]["identity"] for fit in fits],
        "selected_epochs": [fit[1]["selected_epoch"] for fit in fits],
        "selected_updates": [fit[1]["selected_update"] for fit in fits],
        "result_role": reference["config"]["result_role"],
        "test_role": reference["test_role"],
        "test_edge_order_sha256": test_order,
        "aggregation": "arithmetic_mean_of_selected_raw_prediction_scores",
        "predictions": predictions,
    }


def row_for(cohort, setting, result):
    return {
        "cohort": cohort, "setting": setting, "result_role": result["result_role"],
        "valid_hits@50": result["valid_hits@50"],
        "test_hits@50": result.get("test_hits@50", ""),
        "train_seconds": result["train_seconds"], "num_params": result["num_params"],
        "test_role": result["test_role"],
    }


def collect(root, out, cohorts, include_smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    rows, incomplete, missing_groups = [], [], []
    cohort_groups = []
    study_identity = None
    for cohort in cohorts:
        native = [read_fit(root, cohort, f"native{member}") for member in range(4)]
        gnnm = read_fit(root, cohort, "gnnm")
        if not include_smoke:
            native = [fit if fit is None or fit[0]["result_role"] == "full" else None
                      for fit in native]
            if gnnm is not None and gnnm[0]["result_role"] != "full":
                gnnm = None
        available = [fit for fit in [*native, gnnm] if fit is not None]
        for _, artifact in available:
            excluded = ("cohort", "method", "seed", "members", "route_boundaries")
            current_identity = {
                "protocol": {key: value for key, value in artifact["config"].items() if key not in excluded},
                "runtime_and_data": {key: artifact["identity"][key]
                                     for key in ("data_sha256", "versions", "torch_cuda")},
                "test_edge_order_sha256": artifact.get("test_edge_order_sha256"),
            }
            if study_identity is None:
                study_identity = current_identity
            elif current_identity != study_identity:
                raise ValueError("Requested cohort protocol/source/data/runtime or TEST order identities conflict")
        cohort_groups.append((cohort, native, gnnm))
    # Validate the complete requested study before emitting any pooled result.
    for cohort, native, gnnm in cohort_groups:
        if native[0] is not None:
            rows.append(row_for(cohort, "BASE", native[0][0]))
        if gnnm is not None:
            rows.append(row_for(cohort, "GNNM4", gnnm[0]))
        if all(fit is not None for fit in native):
            artifact = ensemble_artifact(native)
            atomic_torch_save(artifact, out / f"cohort{cohort}_independent4_predictions.pt")
            result = dict(metrics(artifact["predictions"]),
                          result_role=artifact["result_role"], test_role=artifact["test_role"],
                          train_seconds=sum(fit[0]["train_seconds"] for fit in native),
                          num_params=sum(fit[0]["num_params"] for fit in native))
            rows.append(row_for(cohort, "independent4", result))
        missing_fits = [f"native{member}" for member, fit in enumerate(native) if fit is None]
        missing_settings = []
        if native[0] is None:
            missing_settings.append("BASE")
        if missing_fits:
            missing_settings.append("independent4")
        if gnnm is None:
            missing_fits.append("gnnm")
            missing_settings.append("GNNM4")
        if missing_fits:
            incomplete.append(cohort)
            missing_groups.append({"cohort": cohort, "missing_fits": missing_fits,
                                   "missing_settings": missing_settings})
    fields = ("cohort", "setting", "result_role", "valid_hits@50", "test_hits@50",
              "train_seconds", "num_params", "test_role")
    with (out / "cohorts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"metric": "OGB Hits@50", "aggregation": "mean and sample standard deviation",
               "incomplete_cohorts": incomplete, "include_smoke": include_smoke,
               "requested_cohorts": list(cohorts), "missing_groups": missing_groups,
               "complete": bool(cohorts) and not incomplete,
               "study_identity": study_identity,
               "settings": {}, "paired_differences": {}}
    for role in sorted({row["result_role"] for row in rows}):
        for setting in ("BASE", "independent4", "GNNM4"):
            selected = [row for row in rows if row["setting"] == setting and row["result_role"] == role]
            key = f"{role}/{setting}"
            summary["settings"][key] = {}
            for metric in ("valid_hits@50", "test_hits@50"):
                values = [row[metric] for row in selected if row[metric] != ""]
                if values:
                    summary["settings"][key][metric] = {
                        "n": len(values), "mean": statistics.mean(values),
                        "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
        for comparison in ("BASE", "independent4"):
            for metric in ("valid_hits@50", "test_hits@50"):
                gnnm = {row["cohort"]: row[metric] for row in rows
                        if row["setting"] == "GNNM4" and row["result_role"] == role and row[metric] != ""}
                other = {row["cohort"]: row[metric] for row in rows
                         if row["setting"] == comparison and row["result_role"] == role and row[metric] != ""}
                common = sorted(gnnm.keys() & other.keys())
                deltas = [gnnm[cohort] - other[cohort] for cohort in common]
                if deltas:
                    summary["paired_differences"][f"{role}/GNNM4-minus-{comparison}/{metric}"] = {
                        "cohorts": common, "deltas": deltas, "mean": statistics.mean(deltas),
                        "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else None}
    atomic_write_json(summary, out / "summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cohorts", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--include-smoke", action="store_true",
                        help="Include explicitly tagged engineering smoke results")
    args = parser.parse_args(argv)
    if len(set(args.cohorts)) != len(args.cohorts) or any(c < 0 or c > 4 for c in args.cohorts):
        parser.error("--cohorts must contain distinct values from 0 through 4")
    if any(not path.resolve().is_relative_to(REPOSITORY) for path in (args.root, args.out)):
        parser.error("All input/output directories must be inside the authorized repository")
    summary = collect(args.root, args.out, args.cohorts, args.include_smoke)
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
