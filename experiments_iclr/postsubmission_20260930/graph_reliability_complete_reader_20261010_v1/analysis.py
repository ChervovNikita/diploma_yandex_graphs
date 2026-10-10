"""Post-closure fixed45-bank reliability reader; no fits, forwards or TEST."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT_DIR = "common_wrapper_graph_reliability_root_20261010_v2"
SOURCE_DIR = "common_wrapper_graph_reliability_source_20261010_v2"
MATH_READER = "private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py"
MATH_SHA = "82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d"
DECISION_SHA = "87f52640edc7ad62aa2faa8539406da073f0e9d0aafd15992a9a99083f902e86"
FREEZE_SHA = "0cc67c8e426ffa2bc11d2ba279dcd630b7114d35b92abc8029afccf1c3c94679"
SUPPORT_SHA = "e0f39c505e7b493c853f676d704691b02ae9e157a61f923065ad8b7503020d17"
BACKBONES, SEEDS = ("SAGE", "GCN", "GAT"), (7301, 7403, 7507)
ARMS = ("ordinary_M1", "ordinary_genuine_I4", "factorized_allmap_M1", "factorized_allmap_genuine_I4", "shared4_unchanged")
O1, O4, F1, F4, S = ARMS
RULES = ("temperature_global", "temperature_member", "reliability_self", "linear_stacking", "reliability_graph")
GRAPH, SELF, NATIVE, UNIFORM = "reliability_graph", "reliability_self", "native_archived", "uniform_FP64"
PREDICTORS = (NATIVE, UNIFORM) + RULES


def sha(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
        return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def math_reader(root):
    path = root / MATH_READER
    require(sha(path) == MATH_SHA, "Immutable paired/count reader required")
    spec = importlib.util.spec_from_file_location("reliability_paired_count_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Stdlib at module scope; never calls its main.
    return module


def scoped(path, root):
    path = Path(path).resolve()
    require(path.is_relative_to(root), "In-scope artifact identity required")
    return path


def admitted(root, helper):
    here, source = root / ROOT_DIR, root / SOURCE_DIR
    require(sha(here / "DECISION.md") == DECISION_SHA and sha(here / "FREEZE.json") == FREEZE_SHA and
            sha(source / "FROZEN_SUPPORT.json") == SUPPORT_SHA, "Frozen policy/support identity mismatch")
    freeze, support = read(here / "FREEZE.json"), read(source / "FROZEN_SUPPORT.json")
    for bound in freeze["bound_files"]:
        require(sha(scoped(root / bound["path"], root)) == bound["sha256"], "Execution/source binding changed: " + bound["path"])
    require(support["backbones"] == list(BACKBONES) and support["seeds"] == list(SEEDS) and support["arms"] == list(ARMS) and
            support["banks"] == 45 and support["TEST_access"] is False and support["fusion"]["nonduplicate_fit_calls"] == 855 and
            support["fusion"]["updates"] == 500 and support["fusion"]["folds"] == 5 and support["fusion"]["fold_seed"] == 11709 and
            support["fusion"]["rule_order"] == list(RULES), "Fixed complete support required")
    # The frozen owner checks COMPLETE_STUDY45/855. The root completion summary
    # repeats those counts without predictions or quality metrics.
    owner = helper.closed_owner(here / "OWNER_END.json")
    summary = read(here / "COMPLETE.json")
    require(summary["TEST_access"] is False and summary["new_base_fits"] == 0 and summary["study"]["complete_roster"] is True and
            summary["study"]["banks"] == 45 and summary["study"]["fit_calls"] == 855 and summary["export"]["complete"] is True and
            summary["export"]["banks"] == 45 and summary["export"]["selected_model_forward_calls"] == 89 and
            summary["export"]["prior_prefix_forward_calls"] == 10 and summary["export"]["represented_selected_model_forward_calls"] == 99,
            "Full45/855 and45-bank89+10 export completion required before result descriptors")
    study_dir, export_dir = here / "actual_study_v1", here / "actual_export_v1"
    study_path, export_path = study_dir / "COMPLETE_STUDY.json", export_dir / "COMPLETE_EXPORT.json"
    require(sha(study_path) == owner["complete_sha256"], "Owner must bind exact complete study")
    # Complete descriptors provide archive/OOF hashes; no outcome array or
    # numerical metric is interpreted until every identity below is admitted.
    study, export, folds = read(study_path), read(export_path), read(study_dir / "FOLDS.json")
    require(study["schema"] == "common-wrapper-reliability-development-v1" and study["complete_roster"] is True and
            len(study["banks"]) == 45 and study["fit_calls"] == 855 and study["maximum_small_head_updates"] == 427500 and
            study["support_sha256"] == SUPPORT_SHA and study["export_sha256"] == sha(export_path) and study["TEST_access"] is False and
            study["new_base_fits_or_forwards"] == 0 and study["no_final_refit_or_held_fold_checkpoint_selection"] is True,
            "Full fixed study descriptor required")
    require(export["complete"] is True and len(export["banks"]) == 45 and export["support_sha256"] == SUPPORT_SHA and
            export["TEST_access"] is False and export["labels_exported"] is False and export["new_base_fits"] == 0 and
            export["selected_model_forward_calls"] == 89 and export["native_fullgraph_member_trajectories"] == 116 and
            export["prior_prefix_forward_calls"] == export["prior_prefix_member_trajectories"] == 10 and
            export["represented_selected_model_forward_calls"] == 99 and export["represented_native_fullgraph_member_trajectories"] == 126 and
            export["reused_prefix_banks"] == 4 and export["newly_exported_banks"] == 41, "Complete amended export accounting required")
    require(study["ordered_VALID_hashes"] == support["ordered_VALID_hashes"] and folds["seed"] == 11709 and folds["count"] == 5 and
            folds["counts"] == [1055, 1055, 1055, 1055, 1054] and folds["ids_sha256"] == support["ordered_VALID_hashes"]["ids"] and
            folds["fold_assignment_sha256"] == study["fold_assignment_sha256"], "Exact fixed label-free fold identity required")
    expected = {bank["key"]: bank for family in support["families"] for bank in family["banks"]}
    index, exports = {r["key"]: r for r in study["banks"]}, {r["key"]: r for r in export["banks"]}
    roster = {f"{b}_{a}_seed{s}" for b in BACKBONES for a in ARMS for s in SEEDS}
    require(len(expected) == len(index) == len(exports) == 45 and set(expected) == set(index) == set(exports) == roster, "Exact45-bank roster required")
    for family in support["families"]:
        old_dir = scoped(family["family_root"], root)
        require(sha(old_dir / "COMPLETE_FAMILY.json") == family["complete_sha256"] and
                sha(old_dir.parent / "OWNER_END.json") == family["owner_end_sha256"] and
                sha(scoped(family["config"], root)) == family["config_sha256"], "Original native closed identities changed")
        helper.closed_owner(old_dir.parent / "OWNER_END.json")
    fits, failed = 0, []
    for key in sorted(roster):
        bank, record, exported = expected[key], index[key], exports[key]
        require((record["backbone"], record["arm"], record["seed"], record["members"]) ==
                (key.split("_", 1)[0], bank["arm"], bank["seed"], bank["members"]) and record["changed_base_member_predictions"] is False,
                "Native bank identity/unchanged member predictions required")
        require(sha(scoped(bank["archive"], root)) == bank["archive_sha256"] == record["archive_sha256"] == exported["archive_sha256"] and
                exported["archive"] == bank["archive"] and sha(scoped(exported["cache"], root)) == exported["cache_sha256"] == record["neighbor_cache_sha256"] and
                exported["discrepancies"]["within_fixed_tolerance"] is True, "Original archive/cache identity changed")
        oof = study_dir / (key + "_OOF_VALID.npz")
        require(scoped(record["VALID_OOF_path"], root) == oof and sha(oof) == record["VALID_OOF_sha256"] and set(record["rules"]) == set(RULES), "OOF/rule identity required")
        for rule in RULES:
            rule_record = record["rules"][rule]
            duplicate = bank["members"] == 1 and rule in ("temperature_member", SELF, GRAPH)
            endpoints = rule_record["fits"]
            require(len(endpoints) == (0 if duplicate else 5), "All declared nonduplicate endpoints required")
            fits += len(endpoints)
            for endpoint in endpoints:
                fold = endpoint["fold"]
                require(fold in range(5), "Fixed fold endpoint required")
                if endpoint["status"] == "failed_retained":
                    failed.append({"key": key, "rule": rule, **endpoint})
                else:
                    require(endpoint["status"] == "finite_fixed_endpoint" and endpoint["updates"] == 500 and
                            endpoint["fit_nodes"] == 5274 - folds["counts"][fold] and endpoint["held_nodes"] == folds["counts"][fold] and
                            endpoint["selected_endpoint"] == "fixed_final_update_no_heldout_selector", "Fixed final endpoint/supervision required")
            if endpoints:
                require([e["fold"] for e in endpoints] == list(range(5)), "Every fixed fold retained")
    failure_index = lambda values: {(v["key"], v["rule"], v["fold"]): v for v in values}
    require(fits == 855 and len(failed) == len(study["failures"]) == len(failure_index(failed)) and
            failure_index(failed) == failure_index(study["failures"]) and study["all_fixed_endpoints_finite"] is (not failed), "Complete retained failure/fit accounting required")
    return dict(here=here, study_dir=study_dir, owner=owner, summary=summary, support=support, study=study, export=export,
                folds=folds, expected=expected, index=index, study_sha256=sha(study_path), export_sha256=sha(export_path))


def payloads(context, np):
    arrays, oof, ids, labels, fold_values = {}, {}, None, None, None
    keys = {"ids", "y", "raw_logits", "probability_mean", "member_errors", "pooled_errors"}
    for key, bank in context["expected"].items():
        with np.load(bank["archive"], allow_pickle=False) as archive:
            require(set(archive.files) == keys, "Original native VALID schema required")
            a = {k: archive[k].copy() for k in keys}
        path = context["study_dir"] / (key + "_OOF_VALID.npz")
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"ids", "folds", "rule_names", "log_probability"}, "Exact OOF schema required")
            o = {k: archive[k].copy() for k in archive.files}
        members = bank["members"]
        require(a["ids"].dtype == a["y"].dtype == o["ids"].dtype == o["folds"].dtype == np.int64 and
                a["ids"].shape == a["y"].shape == o["ids"].shape == o["folds"].shape == (5274,) and
                a["raw_logits"].shape == (members, 5274, 10) and a["raw_logits"].dtype == np.float32 and
                a["probability_mean"].shape == (5274, 10) and a["probability_mean"].dtype == np.float32 and
                a["member_errors"].shape == (members, 5274) and a["pooled_errors"].shape == (5274,) and
                a["member_errors"].dtype == a["pooled_errors"].dtype == np.bool_, "Exact native identity/layout required")
        require(o["rule_names"].tolist() == list(RULES) and o["log_probability"].shape == (5, 5274, 10) and
                o["log_probability"].dtype == np.float64 and np.isfinite(a["raw_logits"]).all() and np.isfinite(a["probability_mean"]).all(), "Fixed rule/native payload layout required")
        if ids is None:
            ids, labels, fold_values = a["ids"], a["y"], o["folds"]
            require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10)), "Exact VALID identity required")
            require({k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in (("ids", ids), ("labels", labels))} == context["support"]["ordered_VALID_hashes"] and
                    hashlib.sha256(fold_values.tobytes()).hexdigest() == context["study"]["fold_assignment_sha256"] and
                    np.bincount(fold_values, minlength=5).tolist() == context["folds"]["counts"], "Bound ordered roles/folds required")
        require(np.array_equal(ids, a["ids"]) and np.array_equal(labels, a["y"]) and np.array_equal(ids, o["ids"]) and
                np.array_equal(fold_values, o["folds"]) and np.array_equal(a["probability_mean"].argmax(1) != labels, a["pooled_errors"]), "All45 bank identities/native decisions must agree")
        for ordinal, rule in enumerate(RULES):
            values, record = o["log_probability"][ordinal], context["index"][key]["rules"][rule]
            if record["fits"]:
                for endpoint in record["fits"]:
                    held = values[fold_values == endpoint["fold"]]
                    require(np.isfinite(held).all() if endpoint["status"] == "finite_fixed_endpoint" else np.isnan(held).all(), "Saved endpoint/failure predictions disagree")
            elif rule == "temperature_member":
                require(np.array_equal(values, o["log_probability"][0], equal_nan=True), "Prespecified M1 temperature duplicate must remain identical")
            else:
                require(np.isfinite(values).all(), "Prespecified structural M1 endpoint must be finite")
        arrays[key], oof[key] = a, o
    return arrays, oof, ids, labels  # Every identity precedes scoring.


def score(helper, np, a, native, log_q, strict, native_override=False):
    y = a["y"]
    correct = native["P"] if native_override else log_q.argmax(1) == y
    probability = a["probability_mean"].astype(np.float64) if native_override else np.exp(log_q)
    nll = -log_q[np.arange(len(y)), y]
    truth = np.eye(10, dtype=np.float64)[y]
    brier = np.square(probability - truth).sum(1)
    covered, member = native["V"], native["C"]
    native_loss = covered & ~native["P"]

    def counts(mask):
        c = {"nodes": int(mask.sum()), "pooled_correct": int(correct[mask].sum()), "coverage": int(covered[mask].sum()),
             "lost_correct_alternatives": int((covered & ~correct)[mask].sum()), "aggregation_only_correct": int((~covered & correct)[mask].sum()),
             "member_correct": member[:, mask].sum(1).tolist(), "native_pooling_loss_nodes": int(native_loss[mask].sum()),
             "repaired_native_pooling_losses": int((native_loss & correct)[mask].sum()), "remaining_native_pooling_losses": int((native_loss & ~correct)[mask].sum()),
             "native_common_strict_rival_nodes": int(strict[mask].sum()), "corrected_native_common_strict_rivals": int((strict & correct)[mask].sum())}
        require(c["pooled_correct"] == c["coverage"] - c["lost_correct_alternatives"] + c["aggregation_only_correct"], "Exact coverage identity failed")
        return c

    classes = []
    for k in range(10):
        mask = y == k
        classes.append({**native["classes"][k], "counts": counts(mask), "pooled_accuracy_pct": float(correct[mask].mean() * 100),
                        "pooled_nll": float(nll[mask].mean()), "pooled_brier": float(brier[mask].mean())})
    return {"quality": {**native["quality"], "pooled_accuracy_pct": float(correct.mean() * 100), "pooled_nll": float(nll.mean()), "pooled_brier": float(brier.mean())},
            "counts": counts(np.ones(len(y), dtype=np.bool_)), "classes": classes, "P": correct, "V": covered, "C": member,
            "state": 2 * covered.astype(np.int64) + correct.astype(np.int64)}


def public_record(record):
    # Trained head states remain in their original server-side artifact.
    result = {k: v for k, v in record.items() if k != "rules"}
    result["rules"] = {rule: {**{k: v for k, v in value.items() if k != "fits"},
                             "fits": [{k: v for k, v in fit.items() if k != "state"} for fit in value["fits"]]} for rule, value in record["rules"].items()}
    return result


def bank_scores(helper, np, context, arrays, oof, labels):
    scores, detail = {}, []
    for key, a in arrays.items():
        record = context["index"][key]
        native = helper.score(a)
        raw = a["raw_logits"].astype(np.float64)
        log_p = raw - helper.logsumexp(raw, -1)[..., None]
        uniform = helper.logsumexp(log_p, 0) - math.log(len(raw))
        member_probability = np.exp(log_p)
        true_probability = np.take_along_axis(member_probability, labels[None, :, None], 2)
        strict = (member_probability > true_probability).all(0).any(1)
        native_readout = score(helper, np, a, native, uniform, strict, True)
        variants = {}
        for rule in PREDICTORS:
            log_q = uniform if rule in (NATIVE, UNIFORM) else oof[key]["log_probability"][RULES.index(rule)]
            if not np.isfinite(log_q).all():
                variants[rule] = {"available": False, "status": "failed_endpoint_retained_no_partial_metric", "original_rule_record": public_record(record)["rules"][rule]}
                scores[record["backbone"], record["arm"], record["seed"], rule] = None
                continue
            value = score(helper, np, a, native, log_q, strict, rule == NATIVE)
            expected_metric = record[NATIVE if rule == NATIVE else UNIFORM] if rule in (NATIVE, UNIFORM) else record["rules"][rule]["metrics"]
            require(value["counts"]["pooled_correct"] == expected_metric["correct"], "Stored native/OOF correct count mismatch")
            scores[record["backbone"], record["arm"], record["seed"], rule] = value
            variants[rule] = {"available": True, **{k: value[k] for k in ("quality", "counts", "classes")},
                              "versus_original_native": helper.comparison(value, native_readout, labels)}
        detail.append({"key": key, "backbone": record["backbone"], "arm": record["arm"], "seed": record["seed"],
                       "original_record": public_record(record), "predictions": variants, "changed_native_members": False})
    return scores, detail


def contrast(helper, scores, backbone, candidate, reference, labels):
    a = [scores[backbone, candidate[0], s, candidate[1]] for s in SEEDS]
    b = [scores[backbone, reference[0], s, reference[1]] for s in SEEDS]
    missing = [s for s, u, v in zip(SEEDS, a, b) if u is None or v is None]
    if missing:
        return {"available": False, "candidate": list(candidate), "reference": list(reference), "failed_seeds_retained": missing,
                "scope": "No surviving-seed estimate or replacement endpoint."}
    return {"available": True, "candidate": list(candidate), "reference": list(reference),
            "accuracy_pp": helper.paired(100 * (u["counts"]["pooled_correct"] - v["counts"]["pooled_correct"]) / 5274 for u, v in zip(a, b)),
            "nll": helper.paired(u["quality"]["pooled_nll"] - v["quality"]["pooled_nll"] for u, v in zip(a, b)),
            "brier": helper.paired(u["quality"]["pooled_brier"] - v["quality"]["pooled_brier"] for u, v in zip(a, b)),
            "per_seed_error_diagnostics": [{"seed": s, **helper.comparison(u, v, labels)} for s, u, v in zip(SEEDS, a, b)],
            "classes": [{"class": k, "accuracy_pp": helper.paired(u["classes"][k]["pooled_accuracy_pct"] - v["classes"][k]["pooled_accuracy_pct"] for u, v in zip(a, b)),
                         "nll": helper.paired(u["classes"][k]["pooled_nll"] - v["classes"][k]["pooled_nll"] for u, v in zip(a, b)),
                         "brier": helper.paired(u["classes"][k]["pooled_brier"] - v["classes"][k]["pooled_brier"] for u, v in zip(a, b))} for k in range(10)]}


def family_report(helper, scores, backbone, labels):
    shared_controls = [(S, NATIVE)] + [(S, r) for r in RULES if r != GRAPH]
    processed_i4 = [(O4, GRAPH), (F4, GRAPH)]
    primary = {arm + "/" + rule: contrast(helper, scores, backbone, (S, GRAPH), (arm, rule), labels) for arm, rule in shared_controls + processed_i4}
    accuracy, protection = {}, {}
    for name, value in primary.items():
        threshold = .2 if name == S + "/" + NATIVE else .1
        accuracy[name] = {"available_all_three_seeds": value["available"], "mean_required_accuracy_pp": threshold,
                          "positive_mean_and_threshold": value["available"] and value["accuracy_pp"]["mean"] >= threshold,
                          "all_seed_nonnegative": value["available"] and value["accuracy_pp"]["nonnegative_seed_count"] == 3,
                          "at_least_two_positive_seeds": value["available"] and value["accuracy_pp"]["positive_seed_count"] >= 2}
        if name != S + "/" + NATIVE:
            protection[name] = {"available_all_three_seeds": value["available"],
                                "mean_nll_deterioration_at_most_0_02": value["available"] and value["nll"]["mean"] <= .02,
                                "each_seed_nll_deterioration_at_most_0_05": value["available"] and value["nll"]["max"] <= .05}
    return {"primary_shared_graph_contrasts": primary, "accuracy_transfer_criteria": accuracy,
            "accuracy_transfer_within_backbone": all(v[k] for v in accuracy.values() for k in ("available_all_three_seeds", "positive_mean_and_threshold", "all_seed_nonnegative", "at_least_two_positive_seeds")),
            "proper_risk_protection_criteria": protection, "proper_risk_protection_within_backbone": all(all(v.values()) for v in protection.values()),
            "graph_minus_identical_P_equals_I_every_bank": {arm: contrast(helper, scores, backbone, (arm, GRAPH), (arm, SELF), labels) for arm in ARMS},
            "each_rule_minus_original_native_every_bank": {arm + "/" + rule: contrast(helper, scores, backbone, (arm, rule), (arm, NATIVE), labels) for arm in ARMS for rule in RULES + (UNIFORM,)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root, report_path = args.research_root.resolve(), args.report.resolve()
    require(not report_path.exists(), "Fresh report path required")
    manifest = read(Path(__file__).with_name("SOURCE.json"))
    require(manifest["files"]["analysis.py"] == sha(Path(__file__)), "Sealed reader source mismatch")
    helper = math_reader(root)
    context = admitted(root, helper)  # All closure/support/source/fold/archive/OOF hashes first.
    import numpy as np
    helper.np = np
    arrays, oof, ids, labels = payloads(context, np)
    scores, detail = bank_scores(helper, np, context, arrays, oof, labels)  # All45 identities before scores.
    families = {b: family_report(helper, scores, b, labels) for b in BACKBONES}
    accuracy = all(families[b]["accuracy_transfer_within_backbone"] for b in BACKBONES)
    protection = all(families[b]["proper_risk_protection_within_backbone"] for b in BACKBONES)
    report = {"complete": True, "banks": 45, "fit_calls": 855, "backbones": list(BACKBONES), "arms": list(ARMS), "seeds": list(SEEDS), "rules": list(PREDICTORS),
              "TEST_access": False, "analysis_source_sha256": sha(Path(__file__)), "reader_manifest_sha256": sha(Path(__file__).with_name("SOURCE.json")),
              "immutable_paired_count_helper_sha256": MATH_SHA, "decision_sha256": DECISION_SHA, "frozen_support_sha256": SUPPORT_SHA,
              "owner_end_sha256": sha(context["here"] / "OWNER_END.json"), "complete_study_sha256": context["study_sha256"], "complete_export_sha256": context["export_sha256"],
              "folds": context["folds"], "ordered_VALID_hashes": context["support"]["ordered_VALID_hashes"], "banks_detail": detail, "families": families,
              "original_native_family_identities": [{k: v for k, v in family.items() if k != "banks"} for family in context["support"]["families"]],
              "all_fixed_endpoints_finite": context["study"]["all_fixed_endpoints_finite"], "failed_endpoints_retained": context["study"]["failures"],
              "frozen_decision": {"accuracy_rule_all_three_backbones": accuracy, "proper_risk_protection_all_three_backbones": protection,
                                  "transferable_development_clue_without_failed_endpoint_promotion": accuracy and context["study"]["all_fixed_endpoints_finite"]},
              "costs": {"owner": context["owner"], "root_complete_summary": context["summary"], "head_fitting_seconds": context["study"]["seconds"],
                        "declared_head_updates": 427500, "fit_calls_including_failures": 855, "export_seconds": context["export"]["seconds"],
                        "new_forwards": 89, "prior_forwards": 10, "new_member_trajectories": 116, "prior_member_trajectories": 10,
                        "complete_export_records_and_costs": context["export"],
                        "prior_prefix_failure_and_costs": context["support"]["prefix_reuse"], "all_fit_and_rule_costs_preserved_in_bank_records": True},
              "native_authority": "Original float32 archived pooled decisions/probabilities are unchanged; stable raw-logit float64 NLL is preserved. uniform_FP64 is a separate descriptive readout.",
              "strict_rival_definition": "Exists a wrong class whose original member probability strictly exceeds truth in every member. No epsilon or outcome-conditioned margin. Corrections counted for every finite rule, with floating/tie caveats; a scalar convex pool cannot reverse a common strict rival in exact arithmetic.",
              "coverage_identity": "pooled correct = any-member coverage - lost correct alternatives + aggregation-only correct",
              "interpretation_limits": ["Aggregator OOF uses encountered development labels after full-VALID base checkpoint selection and diagnosis; this is not full-pipeline cross-fitting or unused confirmation.",
                  "Paired df2 intervals describe three optimization seeds on the selected development graph; connected nodes are not independent graph replications.",
                  "Every temperature, self and stacking rule receives its declared equal fitting opportunity. A failed endpoint has no partial metric or surviving-seed promotion.",
                  "Accuracy and NLL protection are separate; better NLL/Brier alone supports calibration utility rather than the requested accuracy contribution.",
                  "No source tolerance is an accuracy gate, no native member changed, no final refit or new model/TEST forward occurs.",
                  "Full costs/failures remain visible; observed concurrent host costs establish no isolated speed or inference-route saving claim.",
                  "A pass requires a frozen whole pipeline, capable matched references and unused graph confirmation; a failure closes this exact fixed rule without a rescue grid."]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"report": str(report_path), "complete": True, "frozen_decision": report["frozen_decision"]}))


if __name__ == "__main__":
    main()
