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
    declared, grouped = defaultdict(dict), defaultdict(dict)
    for row in scores:
        key = (row["dataset"], row["model"], row.get("num_layers") if row["stage"] == "roman_ablation" else "selected")
        split = row["split"]
        declared_rows = declared[key].setdefault(row["setting"], {})
        if split in declared_rows:
            raise ValueError(f'Duplicate comparison cell {key}/{row["setting"]}/{split}')
        declared_rows[split] = row
        if row.get("scientific_status") == "numerical_failure":
            continue
        if any(not math.isfinite(row[role + "_metric"]) for role in ("val", "test")):
            raise ValueError(f'Nonfinite score in comparison cell {key}')
        grouped[key].setdefault(row["setting"], {})[split] = row
    contrasts, summaries = {}, {}
    for group, declared_settings in declared.items():
        settings = grouped[group]
        for setting, rows in settings.items():
            for role in ("val", "test"):
                values = [row[role + "_metric"] for row in rows.values()]
                summaries["/".join(map(str, (*group, setting, role)))] = dict(
                    n=len(values), mean=statistics.mean(values),
                    sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                    unresolved_convergence=[row.get("unresolved_convergence") for row in rows.values()
                                            if row.get("unresolved_convergence")])
        for setting in declared_settings:
            if setting in ("BASE", "ENS"):
                continue
            references = ["BASE", "ENS"]
            gnnm_name = "GNNM" if "GNNM" in declared_settings else "gnnm"
            if setting != gnnm_name:
                references.append(gnnm_name)
            if setting == "xavier_activation_rms":
                references.append("xavier_same_weights")
            for control in references:
                if control not in declared_settings:
                    continue
                first, second = (control, setting) if control in ("GNNM", "gnnm") else (setting, control)
                paired_methods = (first, second)
                planned = sorted(declared_settings[first].keys() | declared_settings[second].keys())
                first_rows, second_rows = settings.get(first, {}), settings.get(second, {})
                common = sorted(first_rows.keys() & second_rows.keys())
                missing = sorted(set(planned) - set(common))
                missing_by_arm = {method: sorted(set(planned) - settings.get(method, {}).keys())
                                  for method in paired_methods}
                failed_by_arm = {
                    method: sorted(split for split, row in declared_settings[method].items()
                                   if row.get("scientific_status") == "numerical_failure")
                    for method in paired_methods
                }
                unresolved = [
                    dict(split=split, methods={
                        method: declared_settings[method].get(split, {}).get("unresolved_convergence", [])
                        for method in paired_methods})
                    for split in planned
                    if any(declared_settings[method].get(split, {}).get("unresolved_convergence")
                           for method in paired_methods)
                ]
                for role in ("val", "test"):
                    deltas = [first_rows[split][role + "_metric"] - second_rows[split][role + "_metric"]
                              for split in common]
                    key = "/".join(map(str, (*group, first + "-minus-" + second, role)))
                    contrasts[key] = dict(
                        paired_summary(deltas), paired_splits=common,
                        planned_splits=planned, available_paired_splits=common,
                        missing_paired_splits=missing, missing_splits_by_arm=missing_by_arm,
                        numerical_failure_splits=failed_by_arm,
                        unresolved_paired_splits=[item["split"] for item in unresolved],
                        unresolved_convergence=unresolved,
                        estimand="Available-case paired differences over completed score pairs with observed values only")
    holm(contrasts)
    incomplete_pairs = any(row["missing_paired_splits"] for row in contrasts.values())
    unresolved_scores = any(row.get("unresolved_convergence") for row in scores)
    return dict(settings=summaries, paired_comparisons=contrasts,
                inference="Descriptive fixed-benchmark paired variation. Sign-flip probabilities assume independent symmetric pair differences and are not calibrated for overlapping graph splits.",
                multiplicity="Holm over all emitted contrasts and roles in this report",
                requested_score_cells=requested_scores, numerical_failures=failures,
                complete=not failures and not incomplete_pairs and not unresolved_scores,
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
    sources, method_statuses = {}, {}
    expected_methods = list(meta.get("methods", []))
    if "method_scores" in meta:
        for entry in meta["method_scores"]:
            method, task_id = entry["method"], entry["task"]
            if method in method_statuses:
                raise ValueError("Duplicate transfer method")
            if method not in expected_methods:
                expected_methods.append(method)
            state_path = output(root, task_id).parent / "state.json"
            state = read_json(state_path) if state_path.exists() else {"status": "missing"}
            receipt_path = output(root, task_id) / "done.json"
            if not receipt_path.exists():
                status = dict(task=task_id, fit=entry["fit"], task_status=state["status"],
                              scientific_status="unavailable", failure_kind="infrastructure_or_scoring",
                              reason=state.get("reason"))
                fit_path = output(root, entry["fit"]) / "done.json"
                if fit_path.exists():
                    fit = read_json(fit_path)
                    if (fit.get("scientific_status") == "numerical_failure"
                            or fit.get("fit", {}).get("scientific_status") == "numerical_failure"):
                        status.update(scientific_status="numerical_failure", failure_kind="numerical")
                method_statuses[method] = status
                continue
            receipt = result(root, task_id)
            if receipt["pair"] != meta["pair"] or receipt["cohort"] != meta["cohort"]:
                raise ValueError("Transfer method receipt identity mismatch")
            method_statuses[method] = dict(receipt["method_statuses"][method], task=task_id)
            if method in receipt["directories"]:
                sources[method] = (None, Path(receipt["directories"][method]),
                                   receipt["scoring_results"][method])
    else:
        for task_id in meta["fits"]:
            trained = result(root, task_id)
            method = trained["method"]
            if method in method_statuses:
                raise ValueError("Duplicate transfer method")
            if method not in expected_methods:
                expected_methods.append(method)
            method_statuses[method] = dict(fit=task_id, task_status="complete",
                                           scientific_status="complete")
            if (trained.get("scientific_status") == "numerical_failure"
                    or trained.get("fit", {}).get("scientific_status") == "numerical_failure"):
                method_statuses[method].update(scientific_status="numerical_failure",
                                               failure_kind="numerical")
                continue
            sources[method] = (trained, None, None)
    for method, (trained, score_dir, scored) in sources.items():
        if trained is not None:
            directory = Path(trained["fit_dir"])
            if meta["module"] == "p12_collab_ncn":
                from ablation.p12_collab_ncn.score import score_fit
                score_dir = out / method
                receipt_path = score_dir / "done.json"
                previous = read_json(receipt_path) if receipt_path.exists() else {}
                if previous.get("scientific_status") == "numerical_failure":
                    scored = previous
                else:
                    try:
                        scored = score_fit(directory, score_dir, device)
                    except FloatingPointError as error:
                        scored = dict(scientific_status="numerical_failure", error=str(error),
                                      error_type=type(error).__name__, phase="scoring",
                                      config=trained.get("fit", {}))
                        write_json(scored, receipt_path)
            else:
                argv = list(trained["argv"])
                if "--device" in argv:
                    argv[argv.index("--device") + 1] = str(device)
                else:
                    argv.extend(["--device", str(device)])
                argv.append("--score-only")
                lock_fds = tuple(int(value) for value in os.environ.get("GNNM_CAMPAIGN_LOCK_FDS", "").split(",") if value)
                subprocess.run(argv, check=True, pass_fds=lock_fds)
                base = Path(argv[argv.index("--root") + 1])
                score_dir = base / "scoring" / directory.relative_to(base)
                scored = read_json(score_dir / "done.json")
        if scored.get("scientific_status") == "numerical_failure":
            method_statuses[method].update(scientific_status="numerical_failure",
                                           failure_kind="numerical", phase="scoring",
                                           error=scored.get("error"))
            fit_results[method] = scored
            continue
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
    def summarize_artifacts(values):
        if not values:
            return {}
        rows = {}
        reference = values.get("native0", next(iter(values.values())))
        native = [values[f"native{index}"] for index in range(4)] if required <= values.keys() else []
        ensemble = None
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
            if native:
                ensemble = {}
                for role in ("valid", "test"):
                    logits = torch.stack([item["splits"][role]["member_logits"].mean(0) for item in native]).mean(0)
                    ensemble[("val" if role == "valid" else role) + "_metric"] = molecular_metric(
                        evaluator, reference["splits"][role]["y_true"], logits)
        elif meta["module"] in ("p11_collab_buddy", "p14_collab_ogb"):
            from ablation.p14_collab_ogb.run import all_hits

            def link_metrics(scores, num_positive, prefix):
                # Hits@50 stays the primary metric; Hits@20 and Hits@100 are reported beside it.
                row = {prefix + "_metric": buddy_metric(scores, num_positive)}
                row.update({f"{prefix}_{key}": value for key, value in all_hits(scores, num_positive).items()})
                return row

            for method, artifact in values.items():
                row = {}
                for role in ("valid", "test"):
                    current, expected = artifact[role], reference[role]
                    if (current["edge_order_sha256"] != expected["edge_order_sha256"]
                            or current["num_positive"] != expected["num_positive"]
                            or current["scores"].shape != expected["scores"].shape):
                        raise ValueError("Collab prediction edge ordering/cardinality differs")
                    row.update(link_metrics(current["scores"], current["num_positive"],
                                            "val" if role == "valid" else role))
                rows[method] = row
            if native:
                ensemble = {}
                for role in ("valid", "test"):
                    ensemble.update(link_metrics(
                        torch.stack([item[role]["scores"] for item in native]).mean(0),
                        reference[role]["num_positive"], "val" if role == "valid" else role))
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
            if native:
                predictions = {name: torch.stack([item["predictions"][name] for item in native]).mean(0)
                               for name in reference["predictions"]}
                metric = ncn_metrics(predictions)
                ensemble = dict(val_metric=metric["valid_hits@50"], test_metric=metric["test_hits@50"])
        if "native0" in rows:
            rows["BASE"] = rows["native0"]
        if ensemble is not None:
            rows["ENS"] = ensemble
        return {key: value for key, value in rows.items() if not key.startswith("native")}

    rows = summarize_artifacts(artifacts)
    reference_rows = {}
    if reference_artifacts:
        for method, artifact in reference_artifacts.items():
            if artifact["config"] != artifacts[method]["config"]:
                raise ValueError("Reference100 fit config differs")
        reference_rows = summarize_artifacts(reference_artifacts)
    unresolved = [method for method, fit in fit_results.items()
                  if fit.get("stop_reason") == "epoch_cap"]
    expected_settings = [method for method in expected_methods if not method.startswith("native")]
    if "native0" in expected_methods:
        expected_settings.append("BASE")
    if required <= set(expected_methods):
        expected_settings.append("ENS")
    missing_methods = sorted(set(expected_methods) - artifacts.keys())
    missing_reference_methods = sorted(set(expected_methods) - reference_artifacts.keys())
    return dict(pair=meta["pair"], cohort=meta["cohort"], settings=rows,
                directories=directories, scoring_results=fit_results,
                expected_methods=expected_methods, expected_settings=expected_settings,
                missing_methods=missing_methods, missing_reference100_methods=missing_reference_methods,
                method_statuses=method_statuses, unresolved_convergence=unresolved,
                complete=not missing_methods and not unresolved,
                reference100_optional=True,
                reference100_settings=reference_rows,
                reference100_results={method: {key: value for key, value in fit.items()
                                      if key.startswith("reference100_")} for method, fit in fit_results.items()})

def transfer_report(scores):
    grouped = defaultdict(list)
    for score in scores:
        grouped[score["pair"]].append(score)
    horizons = {}
    constituents = {"BASE": {"native0"}, "ENS": {f"native{index}" for index in range(4)}}
    for horizon, bank in (("extended", "settings"), ("reference100", "reference100_settings")):
        summaries, contrasts, missing = {}, {}, []
        for pair, rows in grouped.items():
            rows = sorted(rows, key=lambda row: row["cohort"])
            cohorts = [row["cohort"] for row in rows]
            if len(cohorts) != len(set(cohorts)):
                raise ValueError(f"Duplicate transfer cohort in {pair}")
            # Legacy receipts have no declared partial-method contract.
            legacy = [row for row in rows if "expected_settings" not in row and row.get(bank)]
            if legacy and any(set(row[bank]) != set(legacy[0][bank]) for row in legacy):
                raise ValueError(f"Inconsistent transfer methods in {pair}/{horizon}")
            expected = set().union(*(set(row.get("expected_settings", row.get(bank, {}))) for row in rows))
            for row in rows:
                absent = sorted(expected - row.get(bank, {}).keys())
                if absent:
                    missing.append(dict(pair=pair, cohort=row["cohort"], missing_settings=absent))
            for setting in sorted(expected):
                available = [row for row in rows if setting in row.get(bank, {})]
                missing_cohorts = [row["cohort"] for row in rows if setting not in row.get(bank, {})]
                for role in ("val", "test"):
                    values = [row[bank][setting][role + "_metric"] for row in available]
                    if any(not math.isfinite(value) for value in values):
                        raise ValueError("Nonfinite transfer score")
                    summaries[f"{pair}/{setting}/{role}"] = dict(n=len(values),
                        mean=statistics.mean(values) if values else None,
                        sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                        cohorts=[row["cohort"] for row in available], missing_cohorts=missing_cohorts)
                if setting in ("BASE", "ENS"):
                    continue
                controls = ["BASE", "ENS"] + (["gnnm"] if setting != "gnnm" else [])
                for control in controls:
                    if control not in expected:
                        continue
                    paired = [row for row in available if control in row.get(bank, {})]
                    deltas = [row[bank][setting]["test_metric"] - row[bank][control]["test_metric"]
                              for row in paired]
                    paired_cohorts = [row["cohort"] for row in paired]
                    absent = [dict(cohort=row["cohort"], missing_settings=[
                              name for name in (setting, control) if name not in row.get(bank, {})])
                              for row in rows if row["cohort"] not in paired_cohorts]
                    methods = constituents.get(setting, {setting}) | constituents.get(control, {control})
                    unresolved = [dict(cohort=row["cohort"], methods=sorted(
                                  methods & set(row.get("unresolved_convergence", []))))
                                  for row in paired if horizon == "extended"
                                  and methods & set(row.get("unresolved_convergence", []))]
                    contrasts[f"{pair}/{setting}-minus-{control}"] = dict(
                        paired_summary(deltas), cohorts=paired_cohorts,
                        missing_cohorts=absent, unresolved_convergence=unresolved,
                        complete=not absent and not unresolved)
        holm(contrasts)
        horizons[horizon] = dict(required=horizon == "extended", settings=summaries, paired_comparisons=contrasts,
                                 missing_cohorts=missing, complete=not missing)
    unresolved = [dict(pair=row["pair"], cohort=row["cohort"], methods=row["unresolved_convergence"])
                  for row in scores if row.get("unresolved_convergence")]
    missing_methods = [dict(pair=row["pair"], cohort=row["cohort"],
                            methods=row["missing_methods"], method_statuses=row.get("method_statuses", {}))
                       for row in scores if row.get("missing_methods")]
    return dict(settings=horizons["extended"]["settings"],
                paired_comparisons=horizons["extended"]["paired_comparisons"],
                horizons=horizons, unresolved_convergence=unresolved, missing_methods=missing_methods,
                complete=not unresolved and not missing_methods
                         and all(row.get("complete", True) for row in scores)
                         and horizons["extended"]["complete"],
                reference100_policy="Optional: reported only for runs that actually reached epoch 100",
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
    indices = {"BASE": [0], "ENS": [0, 1, 2, 3], "GNNM": [4]}
    trace, totals, lower_bounds, accounting_gaps, numerical_failures = {}, {}, {}, {}, []
    for setting, members in indices.items():
        rows, gaps, elapsed, incumbent = [], [], 0.0, None
        prefix_accounted = True
        for key in candidates:
            group = grouped[key]
            # ENS charges each native fit at the measured single-model rate.
            profile_setting = "BASE" if setting == "ENS" else setting
            profile = profiles[key]["profiles"][profile_setting]
            step_ms = profile.get("full_step_ms")
            if step_ms is not None and (not math.isfinite(step_ms) or step_ms <= 0):
                raise ValueError("Budget profile requires a finite positive update rate")
            known_cost, candidate_gaps, failed_fits = 0.0, [], []
            for cell in group:
                for index in members:
                    identifier = cell["fit_ids"][index]
                    fit = result(root, identifier)
                    failed = fit.get("scientific_status") == "numerical_failure"
                    if failed:
                        failed_fits.append(identifier)
                    counter = "attempted_steps" if failed else "stopped_step"
                    updates = fit.get(counter)
                    if updates is not None and (isinstance(updates, bool)
                            or not isinstance(updates, (int, float))
                            or not math.isfinite(updates) or updates < 0 or int(updates) != updates):
                        raise ValueError("Budget accounting requires a nonnegative integer update count")
                    missing = []
                    if updates is None:
                        missing.append(counter)
                    if step_ms is None:
                        missing.append("isolated_update_profile")
                    if missing:
                        candidate_gaps.append(dict(fit_id=identifier, missing=missing,
                                                   scientific_status=fit.get("scientific_status", "complete")))
                    else:
                        known_cost += updates * step_ms / 1000
            values = [cell["scores"][setting] for cell in group]
            if not failed_fits and any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("Missing/nonfinite budget score requires a numerical-failure receipt")
            score = None if failed_fits else statistics.mean(values)
            if failed_fits:
                numerical_failures.append(dict(setting=setting, config_name=key[0],
                                               depth=key[1], failed_fit_ids=failed_fits))
            elapsed += known_cost
            prefix_accounted = prefix_accounted and not candidate_gaps
            if score is not None and prefix_accounted:
                incumbent = score if incumbent is None else max(incumbent, score)
            gaps.extend(dict(config_name=key[0], depth=key[1], **gap) for gap in candidate_gaps)
            rows.append(dict(config_name=key[0], depth=key[1],
                             candidate_gpu_seconds=None if candidate_gaps else known_cost,
                             candidate_gpu_seconds_lower_bound=known_cost,
                             cumulative_gpu_seconds=elapsed if prefix_accounted else None,
                             cumulative_gpu_seconds_lower_bound=elapsed,
                             mean_valid=score, incumbent_valid=incumbent,
                             scientific_status="numerical_failure" if failed_fits else "complete",
                             failed_fit_ids=failed_fits, accounting_gaps=candidate_gaps))
        trace[setting], lower_bounds[setting] = rows, elapsed
        totals[setting] = None if gaps else elapsed
        accounting_gaps[setting] = gaps
    exact_totals = [value for value in totals.values() if value is not None]
    exact_minimum = min(exact_totals) if exact_totals else None
    minimum_identified = (exact_minimum is not None
                          and all(value >= exact_minimum for value in lower_bounds.values()))
    budget = exact_minimum if minimum_identified else min(lower_bounds.values())
    budget_mode = "minimum_full_search_cost" if minimum_identified else "conservative_lower_bound_sensitivity"
    matched, selections, ready = {}, {}, {}
    for setting, rows in trace.items():
        accounted_prefix = [row for row in rows if row["cumulative_gpu_seconds"] is not None
                            and row["cumulative_gpu_seconds"] <= budget]
        eligible = [row for row in accounted_prefix if row["mean_valid"] is not None]
        matched[setting] = dict(completed_candidates=len(eligible),
            attempted_candidates=len(accounted_prefix),
            failed_candidates=sum(row["scientific_status"] == "numerical_failure" for row in accounted_prefix),
            best_valid=max((row["mean_valid"] for row in eligible), default=None),
            accounting_complete=not accounting_gaps[setting])
        ready["ready_" + setting] = bool(eligible)
        if eligible:
            chosen = sorted(eligible, key=lambda item: (-item["mean_valid"], item["depth"], item["config_name"]))[0]
            key = (chosen["config_name"], chosen["depth"])
            selections[setting] = dict(chosen, spec=grouped[key][0]["specs"][setting])
    accounting_complete = not any(accounting_gaps.values())
    return dict(role="val", common_gpu_seconds=budget, common_budget_mode=budget_mode,
        common_budget_identified=minimum_identified, accounting_complete=accounting_complete,
        complete=accounting_complete and all(ready.values()),
        full_search_gpu_seconds=totals, full_search_gpu_seconds_lower_bounds=lower_bounds,
        accounting_gaps=accounting_gaps, numerical_failures=numerical_failures,
        matched=matched, full_trace=trace, ready=all(ready.values()), selections=selections, **ready,
        cost_definition="measured isolated update+validation rate multiplied by recorded updates; numerical failures charge every attempted update as one full step",
        partial_budget_interpretation="Unpriced costs make accounting incomplete. Only a fully priced prefix is eligible. An unidentified minimum uses a conservative common-budget lower bound for sensitivity analysis.",
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
            if value.get("required") is False:
                return
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
    from .owner_reports import training_tables
    tables = training_tables(tasks, root, out)
    execution_complete = not incomplete and not missing_reports
    scientific_complete = execution_complete and not scientific_issues
    return dict(campaign_id=plan["campaign_id"], reports=available, task_states=states, tables=tables,
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
    if action in {"diversity_report", "historical_contrast_report"}:
        from .owner_reports import build_report
        return build_report(action, meta, root, out)
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
        rows = [result(root, name) for name in meta["scores"] if name not in missing]
        tasks = {task["id"]: task for task in plan["tasks"]}
        missing_states = {}
        for name in missing:
            state_path = output(root, name).parent / "state.json"
            state = read_json(state_path) if state_path.exists() else {"status": "missing"}
            missing_states[name] = state
            metadata = tasks[name]["metadata"]
            methods = metadata.get("methods", [])
            settings = [method for method in methods if not method.startswith("native")]
            if "native0" in methods:
                settings.append("BASE")
            if {f"native{index}" for index in range(4)} <= set(methods):
                settings.append("ENS")
            rows.append(dict(pair=metadata["pair"], cohort=metadata["cohort"],
                             settings={}, reference100_settings={}, expected_settings=settings,
                             expected_methods=methods, missing_methods=methods,
                             method_statuses={method: dict(task=name, task_status=state["status"],
                                 scientific_status="unavailable", failure_kind="cohort_scoring_unavailable")
                                 for method in methods}, complete=False, unresolved_convergence=[]))
        report = transfer_report(rows)
        report["missing_cohort_tasks"] = missing
        report["missing_cohort_task_states"] = missing_states
        report["complete"] = report["complete"] and not missing
        return report
    if action == "budget_profile":
        return budget_profile(meta, root, out, device)
    if action == "budget_report":
        return budget_report(meta, root)
    if action == "final_report":
        return final_report(meta, root, out, plan)
    raise ValueError(f"Unsupported analysis action {action}")
