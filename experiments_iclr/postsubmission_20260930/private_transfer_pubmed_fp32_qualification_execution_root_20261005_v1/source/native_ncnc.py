#!/usr/bin/env python3
"""Disabled allocation-only native NCNC member fits and discarded TRAIN gate.

Calls unchanged native factory/train/validation bodies. One member per fresh
process/output; individual full native selection, no jointly trained bank.
No TEST input and no old engineering/fitting checkpoint donor.
"""
import json
import time
from pathlib import Path
import custody
from native_state import capture, clone
from native_family_custody import record

QUALIFY = "native_ncnc_discarded_TRAIN_qualification"
FIT = "TRAIN_VALID_native_ncnc_member_fit"


def setup(job, include_valid):
    torch, device, versions = custody.runtime(job)
    import numpy as np
    import sys
    sys.path.insert(0, str(custody.ROOT/"vendor"))
    bodies = custody.load_module("allocation_pubmed_native_bodies", custody.ROOT/"native_bodies.py")
    from evalutors import evaluate_mrr, eval_mrr
    for name, relative in {"gnn_model": "vendor/gnn_model.py", "scoring": "vendor/scoring.py",
        "evalutors": "vendor/evalutors.py", "baseline_models.NCN.model": "vendor/baseline_models/NCN/model.py",
        "baseline_models.NCN.util": "vendor/baseline_models/NCN/util.py"}.items():
        if Path(sys.modules[name].__file__).resolve() != custody.ROOT/relative:
            raise ValueError("Pinned native dependency shadowed: "+name)
    x, train, valid, pool, identities = custody.load_inputs(torch, job, include_valid=include_valid)
    return torch, np, bodies, evaluate_mrr, eval_mrr, x, train, valid, pool, identities, versions


def serve(bodies, unit, valid, pool, torch):
    model, predictor, _, x, data, _ = unit
    scope = dict(vars(bodies))
    scope.update(model=model, predictor=predictor, model_name="NCNC", x=x, data=data,
        device=torch.device("cuda:0"), valid=valid, valid_neg=pool,
        split_edge={"valid": {"edge": valid, "edge_neg": pool}})
    pos, neg = clone(bodies.bound_validation("candidate", "NCNC", scope)(), torch)
    if pos.shape != (2216,) or neg.shape != (2216, 500) or not bool(torch.isfinite(pos).all() and torch.isfinite(neg).all()):
        raise ValueError("Complete native Pubmed VALID geometry/finite logits differ")
    return pos, neg


def validate_fit_plan(job):
    path = custody.phase_file(job["cohort_plan_relative"])
    if custody.sha(path) != job["cohort_plan_sha256"]:
        raise ValueError("Separate native cohort source changed")
    plan = json.loads(path.read_text())
    custody.validate_cohort_admission(plan)
    if plan.get("root_adopted_after_TRAIN_cost") is not True or plan.get("TEST_closed") is not True:
        raise ValueError("Native fit cohort remains disabled")
    matches = [row for row in plan["native_fits"] if row["fit_id"] == job.get("fit_id")]
    if len(matches) != 1 or any(job.get(key) != matches[0][key] for key in ("base_seed", "member", "seed")):
        raise ValueError("Native independent member identity differs")
    if job["seed"] != job["base_seed"]+5*job["member"] or job["base_seed"] not in (0, 1, 2) or job["member"] not in (0, 1, 2, 3):
        raise ValueError("Native member seed policy differs")
    if plan.get("native_selection") != {"max_epochs": 9999, "eval_every_epochs": 5,
        "validation_miss_limit": 11, "selection": "first_maximum_complete_VALID_MRR_rounded4"}:
        raise ValueError("Source-native NCNC selection differs")
    if job.get("fits_authorized") is not True or job.get("VALID_values_access") is not True:
        raise ValueError("Separate native member fit admission absent")


def main():
    started = time.monotonic()
    args = custody.parser(__doc__).parse_args()
    purpose = json.loads(args.job.read_text()).get("purpose")
    if purpose not in (QUALIFY, FIT): raise ValueError("Explicit native member purpose required")
    job, output = custody.authorize(__file__, args, purpose)
    is_fit = purpose == FIT
    if is_fit:
        validate_fit_plan(job)
    elif job.get("fits_authorized") is not False or job.get("VALID_values_access") is not False or job.get("discarded_full_epochs") != 2 or job.get("seed") != 0:
        raise ValueError("Native gate is exactly two discarded seed0 TRAIN epochs; no VALID")
    output.mkdir()
    custody.write_json(output/"START.json", {"purpose": purpose, "source_manifest_sha256": job["source_manifest_sha256"],
        "job_sha256": custody.sha(args.job), "fits_authorized": is_fit, "TEST_access": False})
    try:
        torch, np, bodies, evaluate_mrr, eval_mrr, x, train, valid, pool, identities, versions = setup(job, is_fit)
        unit = bodies.candidate_factory("NCNC", job["seed"], x, train)
        model, predictor, optimizer, _, data, _ = unit
        counts = {"started": 0, "completed": 0}
        def before(opt, args, kwargs): counts["started"] += 1
        def after(opt, args, kwargs): counts["completed"] += 1
        optimizer.register_step_pre_hook(before); optimizer.register_step_post_hook(after)
        best, selected_score, selected_epoch, misses = 0., None, None, 0
        limit = 9999 if is_fit else 2
        identity = {"source_manifest_sha256": job["source_manifest_sha256"], "input_files": identities,
            "seed": job["seed"], "base_seed": job.get("base_seed"), "member": job.get("member"),
            "fit_id": job.get("fit_id"), "model": "source_native_NCNC", "runtime": versions,
            "native_bodies_sha256": custody.sha(custody.ROOT/"native_bodies.py"),
            "native_evaluator_sha256": custody.sha(custody.ROOT/"vendor/evalutors.py")}
        custody.write_json(output/"CONFIG.json", identity)
        with (output/"HISTORY.jsonl").open("x") as history:
            for epoch in range(1, limit+1):
                loss = bodies.candidate_ncnc_train(model, predictor, data, {"train": {"edge": train}}, optimizer, 1024, True, [], None)
                if counts != {"started": epoch*(37676//1024), "completed": epoch*(37676//1024)}:
                    raise ValueError("Native shuffled dropped-tail Adam accounting differs")
                row = {"epoch": epoch, "TRAIN_loss": float(loss), "Adam_updates": counts["completed"]}
                if is_fit and epoch % 5 == 0:
                    pre = capture(unit, torch, np)
                    pos, neg = serve(bodies, unit, valid, pool, torch)
                    result = evaluate_mrr(None, pos, neg)
                    per_query = clone(eval_mrr(pos, neg), torch)
                    post = capture(unit, torch, np)
                    score = result["MRR"]
                    if score > best: best, misses = score, 0
                    else: misses += 1
                    if selected_score is None or score > selected_score:
                        selected_score, selected_epoch = score, epoch
                        chosen = {**identity, "selected_epoch": epoch, "selected_VALID_MRR": score}
                        torch.save({**chosen, "positive_scores": pos, "negative_scores": neg,
                            "native_per_query_metrics": per_query, "native_metrics_rounded4": result}, output/"selected_VALID_scores.pt")
                        torch.save({**chosen, "state_before_VALID": pre, "state_after_VALID": post,
                            "selected_VALID_scores_sha256": custody.sha(output/"selected_VALID_scores.pt")}, output/"selected_state.pt")
                    row.update(VALID=result, misses=misses, selected_epoch=selected_epoch)
                history.write(json.dumps(row, allow_nan=False)+"\n"); history.flush()
                if time.monotonic()-started > job["soft_seconds"]:
                    raise TimeoutError("Native owned bound exceeded; preserve incomplete attempt, no retry")
                if is_fit and misses > 10: break
        torch.cuda.synchronize()
        final = {**identity, "scope": "selected_native_member" if is_fit else "discarded_two_epoch_native_TRAIN_gate",
            "status": "COMPLETE", "epochs": epoch, "Adam_updates": counts["completed"],
            "selected_epoch": selected_epoch, "selected_VALID_MRR": selected_score,
            "TEST_access": False, "states_discarded": not is_fit,
            "inclusive_seconds": time.monotonic()-started,
            "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(),
            "selected_replay_qualified": False, "evaluator_integration_qualified": False}
        if is_fit:
            if selected_epoch is None: raise ValueError("No complete native VALID selection")
            final.update(selected_state_sha256=custody.sha(output/"selected_state.pt"),
                selected_VALID_scores_sha256=custody.sha(output/"selected_VALID_scores.pt"))
        custody.write_json(output/("FIT_FREEZE.json" if is_fit else "TRAIN_GATE.json"), final)
        if is_fit:
            metadata_keys = ("source_manifest_sha256", "input_files", "runtime", "base_seed", "member", "seed", "fit_id", "model")
            custody.write_json(output/"FIT_ARTIFACT_CUSTODY.json", {"schema": "owned_native_member_artifact_custody_v1",
                "status": "COMPLETE", **{key: identity[key] for key in metadata_keys},
                "output_directory": str(output), "job_sha256": custody.sha(args.job),
                "files": [record(output/name) for name in ("START.json", "CONFIG.json", "HISTORY.jsonl", "FIT_FREEZE.json", "selected_state.pt", "selected_VALID_scores.pt")],
                "artifacts": {role: record(output/name) for role, name in {"freeze": "FIT_FREEZE.json", "state": "selected_state.pt", "scores": "selected_VALID_scores.pt"}.items()}})
    except (Exception, KeyboardInterrupt) as error:
        custody.write_json(output/"FAILURE.json", {"error": type(error).__name__+": "+str(error),
            "TEST_access": False, "retry": False, "partial_outputs_preserved": True})
        raise


if __name__ == "__main__": main()
