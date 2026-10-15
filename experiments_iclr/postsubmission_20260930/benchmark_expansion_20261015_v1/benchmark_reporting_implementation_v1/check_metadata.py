"""Concrete JSON-only reporter fixtures and existing molecular extraction checks."""
import copy
import hashlib
import json
from pathlib import Path
import reporter as r

HERE = Path(__file__).resolve().parent
E = HERE.parent


def receipt(name):
    return {"path": "fixture/" + name, "bytes": 1, "sha256": hashlib.sha256(name.encode()).hexdigest()}


def fixture():
    endpoint = {"name": "accuracy", "units": "fraction", "direction": "max"}
    bindings = {field: hashlib.sha256(field.encode()).hexdigest() for field in r.HASH_FIELDS}
    rows, expected, cohorts = [], [], []
    for seed in range(10):
        cohort = {"cohort_id": "paired/" + str(seed), "seed": seed, "split_index": seed, "literal_split_seed": seed + 1000}
        cohorts.append(cohort)
        for method, metric in (("native_M1", .5), ("gnnm_boundary_native", .625)):
            row = dict(dataset="fixture-graph", preset_id="fixed-preset", track="matched_extended", cohort_id=cohort["cohort_id"],
                cohort_seed=seed, split_index=seed, literal_split_seed=seed + 1000, scientific_method_id=method,
                runtime_procedure_id="native_single" if method == "native_M1" else method, readout="actual",
                fit_id=method + "/" + str(seed), actual_seed=seed, endpoint=endpoint, bindings=bindings,
                scope="expanded_exploratory", status="complete", selected_served_metric=metric, native_proper_loss=.4,
                selected_epoch=5, trained_epochs=149, stop_reason="patience", own_selection=True,
                selected_checkpoint=receipt(method + str(seed)), selected_export=receipt("export" + method + str(seed)),
                descriptor_hash=r.digest([method, seed]), costs={"physical_fit_count": 1, "latest_invocation_wall_seconds": 2.0})
            rows.append(row)
            expected.append({key: copy.deepcopy(row[key]) for key in r.IDENTITY_FIELDS + ("endpoint", "bindings")})
    contrast = {"id": "G0-vs-M1", "lhs": {"method_id": "gnnm_boundary_native", "readout": "actual"},
                "rhs": {"method_id": "native_M1", "readout": "actual"}}
    reverse = {"id": "M1-vs-G0", "lhs": contrast["rhs"], "rhs": contrast["lhs"]}
    preset = {"id": "fixed-preset", "matched_cohorts": cohorts, "contrasts": [contrast, reverse]}
    registry = {"schema": "expanded-ten-cohort-report-registry-v1", "presets": [preset], "expected_predictors": expected}
    plan = {"schema": "expanded-prescribed-contrast-plan-v1", "presets": {"fixed-preset": preset["contrasts"]},
            "cohort_keys": {"fixed-preset": [{k: c[k] for k in ("seed", "split_index", "literal_split_seed")} for c in cohorts]}}
    return registry, rows, plan


def expect_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
        return
    raise AssertionError("Expected rejection: " + fragment)


def run():
    directory = HERE / "metadata_fixtures"
    directory.mkdir(exist_ok=True)
    registry, rows, plan = fixture()
    results = []
    out = r.report(registry, rows, [{"attempt_id": "failed-attempt", "worker_wall_seconds": 3.0, "status": "failed"}], plan)
    c = out["contrasts"][0]
    assert c["n_score_pairs"] == c["n_resolved_pairs"] == 10 and c["wins_ties_losses_missing"] == [10, 0, 0, 0]
    assert c["mean"] == .125 and c["descriptive_marginal_95pct_paired_percentile_interval"] == [.125, .125]
    assert out["cost_ledger"][0]["status"] == "failed" and out["cost_summary"]["known_attempt_worker_wall_seconds_sum"] == 3.0
    results.append("Complete ten paired cohorts: fixed bootstrap, exact signs, no extra M1/I4 fit or row-cost summation.")
    missing = rows[:-1]
    incomplete = r.report(registry, missing, contrast_plan=plan)["contrasts"][0]
    assert incomplete["n_planned"] == 10 and incomplete["n_score_pairs"] == 9 and incomplete["n_missing_pairs"] == 1
    assert len(incomplete["rows"]) == 10 and incomplete["descriptive_marginal_95pct_paired_percentile_interval"] is None
    assert incomplete["mean"] == .125 and incomplete["available_pair_summary_incomplete"] is True
    results.append("A missing cohort remains in all ten statuses; available nine-pair mean is incomplete; no survivor interval.")
    capped = copy.deepcopy(rows); capped[-1]["stop_reason"] = "cap_without_patience_plateau"
    cap = r.report(registry, capped, contrast_plan=plan)["contrasts"][0]
    assert cap["n_score_pairs"] == 10 and cap["n_unresolved_scored_pairs"] == 1 and cap["n_resolved_pairs"] == 9
    assert cap["wins_ties_losses_missing"] == [10, 0, 0, 0] and cap["descriptive_marginal_95pct_paired_percentile_interval"] is None
    results.append("A capped selected score retains its directional sign and costs but blocks the complete-preset interval.")
    wrong = copy.deepcopy(rows); wrong[-1]["actual_seed"] = 2
    expect_error(lambda: r.report(registry, wrong, contrast_plan=plan), "actual_seed")
    wrong = copy.deepcopy(rows); wrong[-1]["literal_split_seed"] += 1
    expect_error(lambda: r.report(registry, wrong, contrast_plan=plan), "literal_split_seed")
    wrong = copy.deepcopy(rows); wrong[-1]["endpoint"]["units"] = "percent"
    expect_error(lambda: r.report(registry, wrong, contrast_plan=plan), "endpoint")
    expect_error(lambda: r.report(registry, rows + rows[:1], contrast_plan=plan), "Duplicate observation")
    wrong_registry = copy.deepcopy(registry); wrong_registry["presets"][0]["contrasts"].pop()
    expect_error(lambda: r.report(wrong_registry, rows, contrast_plan=plan), "contrasts omitted")
    results.append("Wrong seed/split/units, duplicate records and omitted prescribed contrasts are rejected.")
    tie_loss = copy.deepcopy(rows); tie_loss[1]["selected_served_metric"] = .5; tie_loss[3]["selected_served_metric"] = .25
    tally = r.report(registry, tie_loss, contrast_plan=plan)["contrasts"][0]
    assert tally["wins_ties_losses_missing"] == [8, 1, 1, 0]
    results.append("Exact saved equality is a tie; unfavorable contrasts remain signed losses.")

    # Authentic native I4 RESULT schema with four independently identified body
    # metadata rows. No logits or models are constructed.
    identity = copy.deepcopy(registry["expected_predictors"][0]); identity.update(scientific_method_id="native_I4", runtime_procedure_id="aggregate_I4", fit_id="I4/0", actual_seed=None)
    components, members = [], []
    for body, offset in enumerate(r.OFFSETS):
        part = copy.deepcopy(rows[0]); part.update(fit_id="native_M1/0" if body == 0 else "body/" + str(body), actual_seed=offset)
        part["descriptor_hash"] = r.digest([part["fit_id"], offset]); part["selected_checkpoint"] = receipt("body-checkpoint" + str(body))
        components.append(part)
        members.append({"body": body, "seed": offset, "job_id": part["fit_id"], "job_hash": part["descriptor_hash"],
                        "selected_epoch": part["selected_epoch"], "selected_checkpoint": part["selected_checkpoint"]})
    raw_i4 = {"schema": "independent-own-selected-VALID-bank-result-v1", "status": "complete", "cohort_id": identity["cohort_id"],
        "TEST_accessed": False, "extra_training_fits": 0, "aggregation": "mean_raw_logits_after_four_own_selectors",
        "metrics": {"accuracy": .7}, "loss": .3, "members": members, "M1_exact_member0_job_hash": members[0]["job_hash"], "bank_receipt": receipt("I4bank")}
    extracted = r.extract_i4(raw_i4, identity, components)
    assert extracted["M1_exact_member0_fit_id"] == components[0]["fit_id"] and extracted["costs"]["physical_fit_count"] == 0
    assert "readout" not in extracted["bindings"]  # Missing aggregate source authority is never filled from expected values.
    wrong_i4 = copy.deepcopy(raw_i4); wrong_i4["M1_exact_member0_job_hash"] = "0" * 64
    expect_error(lambda: r.extract_i4(wrong_i4, identity, components), "M1/body0 hash")
    wrong_i4 = copy.deepcopy(raw_i4); wrong_i4["members"][1]["seed"] = 0
    expect_error(lambda: r.extract_i4(wrong_i4, identity, components), "seed offset")
    results.append("I4 actual aggregate schema: exact four bodies, offsets, M1/body0 checkpoint/hash, no extra fit; wrong identities rejected.")

    # Preserve source-reference rows and avoid mistaking their fixed horizon
    # for failed matched convergence.
    source = copy.deepcopy(rows[0]); source.update(track="source_reference", fit_id="source/0", stop_reason="fixed_source_horizon")
    source_reg = copy.deepcopy(registry); source_reg["expected_predictors"].append({key: copy.deepcopy(source[key]) for key in r.IDENTITY_FIELDS + ("endpoint", "bindings")})
    source_report = r.report(source_reg, rows + [source], contrast_plan=plan)
    assert source_report["source_reference_rows"] == [source] and not r.unresolved(source)
    results.append("Source-reference fixed-horizon rows remain separate from matched contrasts/convergence.")

    compact_path = E / "pilot_20261015_molbace_v1/complete_analysis_v1/REPORT.json"
    compact = json.loads(compact_path.read_text()); compact_hash = r.pin(compact_path)
    pilot = r.extract_compact_molecular(compact)
    assert len(pilot["rows"]) == 18 and pilot["n_cohorts"] == 3 and not pilot["eligible_for_ten_cohort_bootstrap"]
    assert pilot["costs"]["observed_optimizer_completed"] == 139840 and pilot["raw_fit_rows"] == compact["fits"]
    assert r.pin(compact_path) == compact_hash
    pilot["metadata_inputs"] = [compact_hash]
    (HERE / "COMPLETED_MOLECULAR_EXTRACTION.json").write_text(json.dumps(pilot, indent=2, sort_keys=True) + "\n")
    results.append("Completed MolBACE24 metadata: all24fitrows/18aggregate rows, exact scores/IDs/epochs/costs, unchanged input hash; three-seed pilot remains outside ten-cohort bootstrap.")

    lifecycle = E / "qualification/runtime_v4_molecular_selected_mean_lifecycle_v1"
    native_job = json.loads((lifecycle / "JOB_CPU.json").read_text())
    native_result = json.loads((lifecycle / "fetched_attempt01/RESULT.json").read_text())["invocations"][-1]["result"]
    native_identity = copy.deepcopy(rows[0]); native_identity.update(fit_id=native_job["id"], cohort_id=native_job["cohort_id"],
        dataset=native_job["recipe"]["dataset"], preset_id=native_job["recipe"]["id"], actual_seed=native_job["seed"],
        split_index=native_job["split"], runtime_procedure_id=native_job["procedure"], scientific_method_id=native_job["procedure"],
        endpoint={"name": "rocauc", "units": "fraction", "direction": "max"})
    actual = r.extract_runtime(native_job, native_result, native_identity)
    assert actual["selected_served_metric"] == native_result["selected_VALID"]["metrics"]["rocauc"]
    assert actual["bindings"] == {} and actual["scope"] == "noncampaign_pilot_or_engineering"
    results.append("Authentic current v4 runtime RESULT/descriptor extraction matches its own selected endpoint; engineering scope and missing external binding proof remain unresolved.")

    for name, value in (("REGISTRY.json", registry), ("CONTRAST_PLAN.json", plan), ("OBSERVATIONS_COMPLETE.json", rows),
                        ("OBSERVATIONS_MISSING.json", missing), ("OBSERVATIONS_CAPPED.json", capped)):
        (directory / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    checks = {"schema": "exploratory-report-metadata-checks-v1", "status": "passed", "checks": results,
        "numpy_version": out["bootstrap"]["numpy_version"], "native_or_tensor_work": 0, "remote_operations": 0,
        "completed_molecular_input": compact_hash, "bootstrap_seed": 130102026, "bootstrap_resamples": 10000,
        "scores_modified": False, "reviewer_calls": 0}
    (HERE / "CHECKS.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "checks": len(results), "numpy": out["bootstrap"]["numpy_version"]}))


if __name__ == "__main__":
    run()
