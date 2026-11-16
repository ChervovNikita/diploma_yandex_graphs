"""Post-training comparisons, fixed-budget search, and transfer evaluation."""
from collections import defaultdict
import csv
import itertools
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

import numpy as np

from .tasks import output, result, read_json, write_json

def paired_summary(deltas):
    values = np.asarray(deltas, dtype=float)
    n = len(values)
    if not np.isfinite(values).all():
        raise ValueError('Paired comparison contains nonfinite differences')
    if n == 0:
        return dict(n=0)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    interval = None
    if n > 1:
        from scipy.stats import t
        radius = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        interval = [mean - radius, mean + radius]
    p = None
    if n <= 16:
        observed = abs(mean)
        count = sum(abs(float(np.mean(values * signs))) >= observed - 1e-14
                    for signs in itertools.product((-1, 1), repeat=n))
        p = count / (2 ** n)
    return dict(n=n, deltas=list(map(float, values)), mean=mean, sample_sd=sd,
                descriptive_paired_t_interval=interval, exact_sign_flip_p=p,
                caveat="Overlapping graph splits and fixed OGB splits limit population inference")

def holm(comparisons):
    ordered = sorted((row["exact_sign_flip_p"], key) for key, row in comparisons.items()
                     if row.get("exact_sign_flip_p") is not None)
    previous = 0.0
    for index, (value, key) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * value))
        comparisons[key]["holm_adjusted_p"] = previous

def comparison(scores):
    failures = [row for row in scores if row.get("scientific_status") == "numerical_failure"]
    requested_scores = len(scores)
    scores = [row for row in scores if row.get("scientific_status") != "numerical_failure"]
    grouped = defaultdict(dict)
    for row in scores:
        key = (row["dataset"], row["model"], row.get("num_layers") if row["stage"] == "roman_ablation" else "selected")
        split = row["split"]
        setting_rows = grouped[key].setdefault(row["setting"], {})
        if split in setting_rows:
            raise ValueError(f'Duplicate comparison cell {key}/{row["setting"]}/{split}')
        if any(not math.isfinite(row[role + "_metric"]) for role in ("val", "test")):
            raise ValueError(f'Nonfinite score in comparison cell {key}')
        setting_rows[split] = row
    contrasts, summaries = {}, {}
    for group, settings in grouped.items():
        for setting, rows in settings.items():
            for role in ("val", "test"):
                values = [row[role + "_metric"] for row in rows.values()]
                summaries["/".join(map(str, (*group, setting, role)))] = dict(
                    n=len(values), mean=statistics.mean(values),
                    sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                    unresolved_convergence=[row.get("unresolved_convergence") for row in rows.values()
                                            if row.get("unresolved_convergence")])
        for setting, rows in settings.items():
            if setting in ("BASE", "ENS"):
                continue
            references = ["BASE", "ENS"]
            gnnm_name = "GNNM" if "GNNM" in settings else "gnnm"
            if setting != gnnm_name:
                references.append(gnnm_name)
            for control in references:
                if control not in settings:
                    continue
                common = sorted(rows.keys() & settings[control].keys())
                for role in ("val", "test"):
                    first, second = (control, setting) if control in ("GNNM", "gnnm") else (setting, control)
                    deltas = [settings[first][split][role + "_metric"] - settings[second][split][role + "_metric"]
                              for split in common]
                    key = "/".join(map(str, (*group, first + "-minus-" + second, role)))
                    contrasts[key] = dict(paired_summary(deltas), paired_splits=common)
    holm(contrasts)
    return dict(settings=summaries, paired_comparisons=contrasts,
                inference="Descriptive fixed-benchmark paired variation. Sign-flip probabilities assume independent symmetric pair differences and are not calibrated for overlapping graph splits.",
                multiplicity="Holm over all emitted contrasts and roles in this report",
                requested_score_cells=requested_scores, numerical_failures=failures,
                complete=not failures and not any(row.get("unresolved_convergence") for row in scores),
                units="original metric scale", all_results_retained=True)

def score_transfer(meta, root, out, device):
    import torch
    from ogb.graphproppred import Evaluator as GraphEvaluator
    from ablation.p09_molecular.training import score as molecular_metric
    from ablation.p11_collab_buddy.run import evaluate_scores as buddy_metric
    from ablation.p12_collab_ncn.train import metrics as ncn_metrics

    artifacts, reference_artifacts, fit_results, directories = {}, {}, {}, {}
    common = None
    excluded = {"method", "variant", "seed", "member", "members", "reports_base",
                "normalization", "route_boundaries"}
    for task_id in meta["fits"]:
        trained = result(root, task_id)
        method = trained["method"]
        if method in artifacts:
            raise ValueError("Duplicate transfer method")
        directory = Path(trained["fit_dir"])
        if meta["module"] == "p12_collab_ncn":
            from ablation.p12_collab_ncn.score import score_fit
            score_dir = out / method
            scored = score_fit(directory, score_dir, device)
        else:
            argv = list(trained["argv"]) + ["--score-only"]
            lock_fds = tuple(int(value) for value in os.environ.get("GNNM_CAMPAIGN_LOCK_FDS", "").split(",") if value)
            subprocess.run(argv, check=True, pass_fds=lock_fds)
            base = Path(argv[argv.index("--root") + 1])
            score_dir = base / "scoring" / directory.relative_to(base)
            scored = read_json(score_dir / "done.json")
        artifact = torch.load(score_dir / "predictions.pt", map_location="cpu", weights_only=False)
        config = artifact["config"]
        expected_members = 1 if method.startswith("native") else 4
        expected_seed = 1000 * meta["cohort"] + (int(method[-1]) if method.startswith("native") else 0)
        if config["cohort"] != meta["cohort"] or config["seed"] != expected_seed:
            raise ValueError("Transfer cohort/seed identity mismatch")
        if config.get("members", 1) != expected_members:
            raise ValueError("Transfer member count mismatch")
        binding = {key: value for key, value in config.items() if key not in excluded}
        if "data_binding" in artifact:
            binding["data_binding"] = artifact["data_binding"]
        if "identity" in artifact:
            binding["runtime_identity"] = {key: value for key, value in artifact["identity"].items()
                                           if key != "config_sha256"}
        if common is not None and binding != common:
            raise ValueError("Transfer source/protocol/data/runtime binding differs")
        common = binding
        artifacts[method] = artifact
        fit_results[method] = scored
        directories[method] = str(score_dir)
        reference = score_dir / "reference100_predictions.pt"
        if reference.exists():
            reference_artifacts[method] = torch.load(reference, map_location="cpu", weights_only=False)

    required = {f"native{index}" for index in range(4)}
    if not required <= artifacts.keys():
        raise ValueError("Four distinct native transfer fits are required")

    def summarize_artifacts(values):
        rows = {}
        reference = values["native0"]
        native = [values[f"native{index}"] for index in range(4)]
        if meta["module"] == "p09_molecular":
            evaluator = GraphEvaluator(name=reference["config"]["dataset"])
            for method, artifact in values.items():
                row = {}
                for role in ("valid", "test"):
                    current, expected = artifact["splits"][role], reference["splits"][role]
                    if (not torch.equal(current["indices"], expected["indices"]) or
                            not torch.allclose(current["y_true"], expected["y_true"], equal_nan=True)):
                        raise ValueError("Molecular prediction identity mismatch")
                    if current["member_logits"].shape != (
                            artifact["config"]["members"], len(current["indices"]), 1):
                        raise ValueError("Molecular prediction cardinality mismatch")
                    key = ("val" if role == "valid" else role) + "_metric"
                    row[key] = molecular_metric(evaluator, current["y_true"], current["member_logits"].mean(0))
                rows[method] = row
            ensemble = {}
            for role in ("valid", "test"):
                logits = torch.stack([item["splits"][role]["member_logits"].mean(0) for item in native]).mean(0)
                ensemble[("val" if role == "valid" else role) + "_metric"] = molecular_metric(
                    evaluator, reference["splits"][role]["y_true"], logits)
        elif meta["module"] == "p11_collab_buddy":
            for method, artifact in values.items():
                row = {}
                for role in ("valid", "test"):
                    current, expected = artifact[role], reference[role]
                    if (current["edge_order_sha256"] != expected["edge_order_sha256"]
                            or current["num_positive"] != expected["num_positive"]
                            or current["scores"].shape != expected["scores"].shape):
                        raise ValueError("BUDDY prediction edge ordering/cardinality differs")
                    row[("val" if role == "valid" else role) + "_metric"] = buddy_metric(
                        current["scores"], current["num_positive"])
                rows[method] = row
            ensemble = {("val" if role == "valid" else role) + "_metric": buddy_metric(
                torch.stack([item[role]["scores"] for item in native]).mean(0),
                reference[role]["num_positive"]) for role in ("valid", "test")}
        else:
            for method, artifact in values.items():
                if (artifact["identity"]["data_sha256"] != reference["identity"]["data_sha256"]
                        or artifact["test_edge_order_sha256"] != reference["test_edge_order_sha256"]
                        or not artifact["test_edge_order_sha256"]):
                    raise ValueError("NCN VALID/TEST edge ordering differs")
                if set(artifact["predictions"]) != set(reference["predictions"]):
                    raise ValueError("NCN prediction fields differ")
                if any(value.shape != reference["predictions"][key].shape
                       for key, value in artifact["predictions"].items()):
                    raise ValueError("NCN prediction cardinality differs")
                metric = ncn_metrics(artifact["predictions"])
                rows[method] = dict(val_metric=metric["valid_hits@50"], test_metric=metric["test_hits@50"])
            predictions = {name: torch.stack([item["predictions"][name] for item in native]).mean(0)
                           for name in reference["predictions"]}
            metric = ncn_metrics(predictions)
            ensemble = dict(val_metric=metric["valid_hits@50"], test_metric=metric["test_hits@50"])
        rows["BASE"] = rows["native0"]
        rows["ENS"] = ensemble
        return {key: value for key, value in rows.items() if not key.startswith("native")}

    rows = summarize_artifacts(artifacts)
    reference_rows = {}
    if reference_artifacts:
        if set(reference_artifacts) != set(artifacts):
            raise ValueError("Incomplete reference100 paired prediction bank")
        for method, artifact in reference_artifacts.items():
            if artifact["config"] != artifacts[method]["config"]:
                raise ValueError("Reference100 fit config differs")
        reference_rows = summarize_artifacts(reference_artifacts)
    unresolved = [method for method, fit in fit_results.items()
                  if fit.get("stop_reason") == "epoch_cap"]
    return dict(pair=meta["pair"], cohort=meta["cohort"], settings=rows,
                directories=directories, unresolved_convergence=unresolved,
                reference100_settings=reference_rows,
                reference100_results={method: {key: value for key, value in fit.items()
                                      if key.startswith("reference100_")} for method, fit in fit_results.items()})

def transfer_report(scores):
    grouped = defaultdict(list)
    for score in scores:
        grouped[score["pair"]].append(score)
    horizons = {}
    for horizon, bank in (("extended", "settings"), ("reference100", "reference100_settings")):
        summaries, contrasts, missing = {}, {}, []
        for pair, rows in grouped.items():
            cohorts = [row["cohort"] for row in rows]
            if len(cohorts) != len(set(cohorts)):
                raise ValueError(f"Duplicate transfer cohort in {pair}")
            available = [row for row in rows if row.get(bank)]
            missing.extend(dict(pair=pair, cohort=row["cohort"]) for row in rows if not row.get(bank))
            if not available:
                continue
            expected = set(available[0][bank])
            if any(set(row[bank]) != expected for row in available):
                raise ValueError(f"Inconsistent transfer methods in {pair}/{horizon}")
            for setting in sorted(expected):
                for role in ("val", "test"):
                    values = [row[bank][setting][role + "_metric"] for row in available]
                    if any(not math.isfinite(value) for value in values):
                        raise ValueError("Nonfinite transfer score")
                    summaries[f"{pair}/{setting}/{role}"] = dict(n=len(values),
                        mean=statistics.mean(values),
                        sample_sd=statistics.stdev(values) if len(values) > 1 else None)
                if setting not in ("BASE", "ENS"):
                    controls = ["BASE", "ENS"] + (["gnnm"] if setting != "gnnm" else [])
                    for control in controls:
                        deltas = [row[bank][setting]["test_metric"] - row[bank][control]["test_metric"]
                                  for row in available]
                        contrasts[f"{pair}/{setting}-minus-{control}"] = dict(
                            paired_summary(deltas), cohorts=[row["cohort"] for row in available])
        holm(contrasts)
        horizons[horizon] = dict(settings=summaries, paired_comparisons=contrasts,
                                 missing_cohorts=missing, complete=not missing)
    unresolved = [dict(pair=row["pair"], cohort=row["cohort"], methods=row["unresolved_convergence"])
                  for row in scores if row["unresolved_convergence"]]
    return dict(settings=horizons["extended"]["settings"],
                paired_comparisons=horizons["extended"]["paired_comparisons"],
                horizons=horizons, unresolved_convergence=unresolved,
                complete=not unresolved and all(item["complete"] for item in horizons.values()),
                cohort_results=scores,
                replication="Repeated training seeds on official fixed OGB splits")


def budget_profile(meta, root, out, device):
    from ablation.p07_cost_small_setup.selected import profile_selected
    groups = {"BASE": meta["fits"][:1], "ENS": meta["fits"][:4], "GNNM": meta["fits"][4:]}
    profiles = {}
    for setting, names in groups.items():
        if any(result(root, name).get("scientific_status") == "numerical_failure" for name in names):
            profiles[setting] = dict(scientific_status="infeasible")
            continue
        profiles[setting] = profile_selected([output(root, name) for name in names],
            out / setting, device=device, warmup=20, measured=100)
    return dict(dataset=meta["dataset"], model=meta["model"], config_name=meta["config_name"],
                depth=meta["depth"], profiles=profiles, measurement="exclusive GPU, 100 measured steps")

def budget_report(meta, root):
    cells = [result(root, name) for name in meta["cells"]]
    profiles = {(item["config_name"], item["depth"]): item
                for item in [result(root, name) for name in meta["profiles"]]}
    grouped = defaultdict(list)
    for cell in cells:
        grouped[(cell["config_name"], cell["depth"])].append(cell)
    candidates = sorted(grouped)
    random.Random(20261111).shuffle(candidates)
    trace, totals = {}, {}
    for setting in ("BASE", "ENS", "GNNM"):
        rows, elapsed, incumbent = [], 0.0, -math.inf
        for key in candidates:
            group = grouped[key]
            profile = profiles[key]["profiles"][setting]
            if profile.get("scientific_status") == "infeasible" or any(cell["scores"][setting] is None for cell in group):
                # A failed candidate has no defensible measured time-to-completion.
                # Keep the failure and mark this replay unavailable.
                return dict(role="val", complete=False, ready=False,
                            ready_BASE=False, ready_ENS=False, ready_GNNM=False,
                            reason="numerically infeasible search candidate prevents complete budget replay",
                            config_name=key[0], depth=key[1], setting=setting, no_test_read=True)
            seconds_per_step = profile["full_step_ms"] / 1000
            costs = []
            for cell in group:
                fits = [result(root, name) for name in cell["fit_ids"]]
                if setting == "BASE":
                    cost = fits[0]["stopped_step"] * seconds_per_step
                elif setting == "GNNM":
                    cost = fits[4]["stopped_step"] * seconds_per_step
                else:
                    # Ensemble profile measures a four-model update. Count each
                    # model's true duration using the separately measured native update.
                    native_rate = profiles[key]["profiles"]["BASE"]["full_step_ms"] / 1000
                    cost = sum(fit["stopped_step"] for fit in fits[:4]) * native_rate
                costs.append(cost)
            elapsed += sum(costs)
            score = statistics.mean(cell["scores"][setting] for cell in group)
            incumbent = max(incumbent, score)
            rows.append(dict(config_name=key[0], depth=key[1], cumulative_gpu_seconds=elapsed,
                             mean_valid=score, incumbent_valid=incumbent))
        trace[setting], totals[setting] = rows, elapsed
    budget = min(totals.values())
    matched, selections, ready = {}, {}, {}
    for setting, rows in trace.items():
        eligible = [row for row in rows if row["cumulative_gpu_seconds"] <= budget]
        matched[setting] = dict(completed_candidates=len(eligible),
            best_valid=max((row["mean_valid"] for row in eligible), default=None))
        ready["ready_" + setting] = bool(eligible)
        if eligible:
            chosen = sorted(eligible, key=lambda item: (-item["mean_valid"], item["depth"], item["config_name"]))[0]
            key = (chosen["config_name"], chosen["depth"])
            selections[setting] = dict(chosen, spec=grouped[key][0]["specs"][setting])
    return dict(role="val", common_gpu_seconds=budget, matched=matched, full_trace=trace,
        ready=all(ready.values()), selections=selections, **ready,
        cost_definition="measured isolated update+validation rate multiplied by actual update counts",
        caveat="Retrospective budget-matched replay of a predeclared candidate order; excludes IO/startup",
        order_seed=20261111, no_test_read=True)

def final_report(meta, root, out, plan):
    from .schema import TERMINAL
    from .store import CampaignStore
    tasks = {task["id"]: task for task in plan["tasks"]}
    states = CampaignStore(root, plan).snapshot()
    own = {name for name in tasks if output(root, name) == out}
    relevant = {name: state for name, state in states.items() if name not in own}
    allowed_skips = {}

    def declared_skip(name):
        if name in allowed_skips:
            return allowed_skips[name]
        task, state = tasks[name], states[name]
        direct = state.get("reason") == "validation_gate_closed" and bool(task.get("when"))
        inherited = state.get("reason") == "required_dependency_skipped" and any(
            states[parent]["status"] == "skipped" and declared_skip(parent)
            for parent in task.get("dependencies", []))
        allowed_skips[name] = direct or inherited
        return allowed_skips[name]

    incomplete = {name: state for name, state in relevant.items()
                  if state["status"] != "complete"
                  and not (state["status"] == "skipped" and declared_skip(name))}
    available = {name: result(root, name) for name in meta["reports"]
                 if (output(root, name) / "done.json").exists()}
    missing_reports = [name for name in meta["reports"] if name not in available
                       and not (states[name]["status"] == "skipped" and declared_skip(name))]
    scientific_issues = []

    def inspect(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if ((key in {"complete", "ready", "convergence_ready"} and item is False)
                        or (key in {"unresolved_convergence", "selected_unresolved", "numerical_failures"} and item)
                        or (key == "needs_convergence_extension" and item)
                        or (key == "scientific_status" and item in {"numerical_failure", "infeasible"})):
                    scientific_issues.append(dict(path=path + [key], reason=key))
                inspect(item, path + [key])
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, path + [str(index)])

    for name, report in available.items():
        inspect(report, [name])
    for name, task in tasks.items():
        if task.get("metadata", {}).get("action") in {"select", "expand"} and (output(root, name) / "done.json").exists():
            inspect(result(root, name), [name])
    execution_complete = not incomplete and not missing_reports
    scientific_complete = execution_complete and not scientific_issues
    return dict(campaign_id=plan["campaign_id"], reports=available, task_states=states,
                queue_terminal=all(state["status"] in TERMINAL for state in relevant.values()),
                execution_complete=execution_complete, scientific_complete=scientific_complete,
                complete=scientific_complete, incomplete_tasks=incomplete,
                missing_reports=missing_reports, scientific_issues=scientific_issues,
                declared_skips=sorted(name for name, allowed in allowed_skips.items() if allowed),
                original_paper_scores="unchanged",
                acceptance="Reviewer judgment remains external to the experiment queue")


def write_score_table(rows, out):
    fields = ["dataset", "model", "num_layers", "split", "setting", "stage",
              "val_metric", "test_metric", "scientific_status", "unresolved_convergence"]
    with (out / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def execute_analysis(action, meta, root, out, device, plan):
    if action == "comparison":
        missing = [name for name in meta["scores"] if not (output(root, name) / "done.json").exists()]
        rows = [result(root, name) for name in meta["scores"] if name not in missing]
        write_score_table(rows, out)
        report = comparison(rows)
        report["missing_score_tasks"] = missing
        report["requested_score_cells"] = len(meta["scores"])
        report["complete"] = report["complete"] and not missing
        return report
    if action == "architecture_select":
        from .architecture_search import execute_selection
        return execute_selection(meta, root)
    if action == "architecture_comparison":
        from .architecture_search import execute_comparison
        return execute_comparison(meta, root)
    if action == "scaling_profile":
        from .tasks import selected_fit_dirs
        from ablation.p07_cost_small_setup.selected import profile_selected
        directories = selected_fit_dirs(meta, root)
        if not directories:
            return dict(scientific_status="infeasible")
        windows = {}
        for count in (25, 50, 100):
            windows[str(count)] = profile_selected(directories, out / f"steps{count}",
                  device=device, warmup=20, measured=count)
        return dict(dataset=meta["dataset"], model=meta["model"], depth=meta["depth"],
                    setting=meta["setting"], windows=windows,
                    scope="Fixed-depth short-run execution measurements; all saved models remain unchanged")
    if action == "transfer_score":
        return score_transfer(meta, root, out, device)
    if action == "transfer_report":
        missing = [name for name in meta["scores"] if not (output(root, name) / "done.json").exists()]
        report = transfer_report([result(root, name) for name in meta["scores"] if name not in missing])
        report["missing_cohort_tasks"] = missing
        report["complete"] = report["complete"] and not missing
        return report
    if action == "budget_profile":
        return budget_profile(meta, root, out, device)
    if action == "budget_report":
        return budget_report(meta, root)
    if action == "final_report":
        return final_report(meta, root, out, plan)
    raise ValueError(f"Unsupported analysis action {action}")
