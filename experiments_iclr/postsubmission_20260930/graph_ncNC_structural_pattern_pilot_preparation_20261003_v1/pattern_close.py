"""Stdlib-only closure of the fixed complete pair and its diagnostics."""
from pathlib import Path
from pilot_common import require, read_bound_json, file_sha, atomic_json, utc


def bound_payload(directory, pin):
    directory=Path(directory).resolve()
    path=(directory/pin["path"]).resolve()
    require(path.is_relative_to(directory), "Receipt path escaped its output")
    require(path.stat().st_size==pin["bytes"] and file_sha(path)==pin["sha256"], "Output custody differs: "+pin["path"])
    return path


def run(context,output,attempts):
    records={}; selections={}; fit_pins=context["release"]["fit_receipts"]
    attempts.phase("complete_pair_receipt_source_state_stream_and_support_custody")
    for arm in ("J","F"):
        pin=fit_pins[arm]
        r=read_bound_json(pin["path"],pin["sha256"])
        require(r["schema"]=="ncnc-pattern-complete-fit-v1" and r["identity"]==context["identity"] and r["arm"]==arm and r["seed"]==0, "Fit identity differs")
        require(r["epochs"]==100 and r["optimizer_steps"]==1700 and r["selection_candidates"]==100 and r["full_VALID_evaluations"]==102 and r["selected_roundtrip_and_full_served_replay"] is True, "Fit coverage incomplete")
        require(r["resource_state_donor"] is False and r["test_file_opened"] is False, "Fit provenance differs")
        directory=Path(pin["path"]).resolve().parent
        bound_payload(directory,r["selected_checkpoint"])
        bound_payload(directory,r["journal"])
        path=bound_payload(directory,r["private_selection"])
        selection=read_bound_json(path,r["private_selection"]["sha256"])
        require(selection["identity"]==context["identity"] and selection["arm"]==arm and selection["seed"]==0 and selection["checkpoint"]==r["selected_checkpoint"], "Selected JSON identity/custody differs")
        require(len(r["epoch_streams"])==100 and [e["epoch"] for e in r["epoch_streams"]]==list(range(1,101)), "Epoch stream coverage incomplete")
        require(all(len(e["support_digests"])==17 and e["stream"]["full_batches"]==17 for e in r["epoch_streams"]), "Native full-batch support coverage incomplete")
        records[arm]=r;selections[arm]=selection["selection"]
    a,b=records["J"],records["F"]
    require(a["initial_state_sha256"]==b["initial_state_sha256"] and a["initial_rng_sha256"]==b["initial_rng_sha256"] and a["teacher"]==b["teacher"], "Pair initial state/RNG/source teacher differs")
    require(a["epoch_streams"]==b["epoch_streams"], "Pair native negative/permutation/RNG/actual candidate supports differ")
    attempts.phase("complete_selected_representation_diagnostic_custody")
    pin=context["release"]["diagnostics_receipt"]
    diagnostic=read_bound_json(pin["path"],pin["sha256"])
    require(diagnostic["schema"]=="ncnc-pattern-diagnostics-complete-v1" and diagnostic["identity"]==context["identity"] and diagnostic["arms"]==["J","F"] and diagnostic["status"]=="COMPLETE", "Diagnostics incomplete/different")
    require(diagnostic["full_selected_VALID_evaluations"]==2 and diagnostic["full_TRAIN_mask_epochs"]==2 and diagnostic["matched_mask_supports"] is True and diagnostic["test_file_opened"] is False, "Diagnostic coverage differs")
    path=bound_payload(Path(pin["path"]).resolve().parent,diagnostic["private_diagnostics"])
    detail=read_bound_json(path,diagnostic["private_diagnostics"]["sha256"])
    require(detail["identity"]==context["identity"] and set(detail["arms"])=={"J","F"}, "Diagnostic identity differs")
    for arm in ("J","F"):
        d=detail["arms"][arm]
        require(d["VALID"]["served_hits50"]==selections[arm]["hits50"] and d["selected_epoch"]==selections[arm]["order"], "Diagnostics used another selection")
    costs={"fit_"+arm:records[arm]["inclusive_accounting"] for arm in ("J","F")}
    costs["diagnostics"]=diagnostic["inclusive_accounting"]
    for stage in ("numerical","full_graph"):
        qpin=context["release"]["qualification"][stage]
        q=read_bound_json(qpin["path"],qpin["sha256"])
        costs[stage]=q["inclusive_accounting"]
    failures=context["release"].get("prior_failure_receipts",[])
    require(len({str(Path(p["path"]).resolve()) for p in failures})==len(failures),"Prior failure receipt repeated")
    completed_directories={Path(p["path"]).resolve().parent for p in [*fit_pins.values(),context["release"]["diagnostics_receipt"],*context["release"]["qualification"].values()]}
    for index,pin in enumerate(failures,start=1):
        require(Path(pin["path"]).resolve().parent not in completed_directories,"Failure cost already included in completed output's own attempt ledger")
        failed=read_bound_json(pin["path"],pin["sha256"])
        require(failed["schema"]=="ncnc-pattern-failed-stage-v1" and failed["identity"]==context["identity"] and failed["status"]=="FAILED", "Prior failure identity differs")
        costs["prior_failed_stage_"+str(index)]=failed["inclusive_accounting"]
    result={"schema":"ncnc-pattern-pair-results-v1","identity":context["identity"],"selections":selections,
            "selected_served_VALID_Hits50_J_minus_F":selections["J"]["hits50"]-selections["F"]["hits50"],
            "representation_diagnostics":detail["arms"],"all_100_native_streams_RNG_and_actual_supports_match":True,
            "scope":"single_seed_validation_selected_development_pilot_not_confirmatory",
            "source_pattern_scope":"TRAIN_observation_incidence_not_latent_link_truth",
            "stronger_claim_requires_covariance_aware_competent_single":True,"test_file_opened":False}
    atomic_json(output/"PAIR_RESULTS.json",result)
    path=output/"PAIR_RESULTS.json"
    return {"schema":"ncnc-pattern-pair-closure-v1","identity":context["identity"],"status":"CLOSED",
            "fit_receipts":fit_pins,"diagnostics_receipt":pin,
            "pair_results":{"path":path.name,"bytes":path.stat().st_size,"sha256":file_sha(path)},
            "unique_scientific_fits":2,"scientific_optimizer_steps":3400,"scientific_selector_candidates":200,
            "complete_scientific_VALID_evaluations_including_replay_and_diagnostics":206,
            "preclosure_bound_stage_accounting":costs,"prior_failure_receipts":failures,
            "preclosure_bound_closed_attempt_wall_seconds":sum(r["closed_attempt_wall_seconds"] for r in costs.values()),
            "preclosure_bound_observed_interrupted_wall_seconds":sum(r["observed_interrupted_wall_seconds"] for r in costs.values()),
            "preclosure_bound_cost_exact":all(r["total_cost_exact"] for r in costs.values()),
            "cost_scope":"bound_stage_receipts_and_their_own_ledgers_plus_root_listed_prior_failed_stage_receipts; external_unlisted_attempts_not_certified",
            "matched_pair_streams_RNG_and_supports":True,"TEST_supported":False,"UTC":utc()}
