#!/usr/bin/env python3
"""Disabled zero-update replay/pool of four individually selected native fits.

Member0 is the declared single. No ensemble checkpoint search or retraining.
Uses the saved replay tolerance; exact full state/RNG and metrics are required.
"""
import json
import time
from pathlib import Path
import custody
from native_state import capture, restore, equal
from native_ncnc import setup, serve


def main():
    started = time.monotonic()
    args = custody.parser(__doc__).parse_args()
    job, output = custody.authorize(__file__, args, "native_ncnc_selected_pool_replay")
    if job.get("fits_authorized") is not False or job.get("VALID_values_access") is not True or job.get("base_seed") not in (0, 1, 2):
        raise ValueError("Explicit zero-update VALID replay of one frozen independent-four group required")
    members = job.get("members", [])
    if len(members) != 4 or [r["member"] for r in members] != [0, 1, 2, 3]:
        raise ValueError("All four native independently selected members required")
    custody.require_evidence(job, "owned_complete_native_fit_family")
    output.mkdir()
    try:
        torch, np, bodies, evaluate_mrr, eval_mrr, x, train, valid, pool, identities, versions = setup(job, True)
        positives, negatives, reports = [], [], []
        for row in members:
            seed = job["base_seed"]+5*row["member"]
            paths = {}
            for role in ("freeze", "state", "scores"):
                p = custody.phase_file(row[role]["path"])
                if custody.sha(p) != row[role]["sha256"]: raise ValueError("Owned selected member hash differs")
                paths[role] = p
            frozen = json.loads(paths["freeze"].read_text())
            if frozen.get("status") != "COMPLETE" or frozen.get("seed") != seed or frozen.get("base_seed") != job["base_seed"] or frozen.get("member") != row["member"]:
                raise ValueError("Native independent fit family differs")
            if frozen.get("fit_id") != "pubmed_native_s"+str(job["base_seed"])+"_m"+str(row["member"]) or frozen.get("model") != "source_native_NCNC" or frozen.get("runtime") != versions:
                raise ValueError("Native member model/runtime/fit identity differs")
            if frozen.get("source_manifest_sha256") != job["source_manifest_sha256"] or frozen.get("input_files") != identities:
                raise ValueError("Native fit source/inputs differ")
            if frozen.get("selected_state_sha256") != custody.sha(paths["state"]) or frozen.get("selected_VALID_scores_sha256") != custody.sha(paths["scores"]):
                raise ValueError("Native freeze does not bind supplied selected files")
            state = torch.load(paths["state"], map_location="cpu", weights_only=True)
            scores = torch.load(paths["scores"], map_location="cpu", weights_only=True)
            for key in ("source_manifest_sha256", "input_files", "seed", "base_seed", "member", "fit_id", "model", "native_bodies_sha256", "native_evaluator_sha256", "selected_epoch", "selected_VALID_MRR"):
                if state[key] != scores[key] or state[key] != frozen[key]: raise ValueError("Selected identity differs: "+key)
            if state["selected_VALID_scores_sha256"] != custody.sha(paths["scores"]): raise ValueError("State/score binding differs")
            unit = bodies.candidate_factory("NCNC", seed, x, train)
            restore(unit, state["state_after_VALID"], torch, np)
            equal(state["state_after_VALID"], capture(unit, torch, np), torch, "exact_poststate_restore")
            restore(unit, state["state_before_VALID"], torch, np)
            equal(state["state_before_VALID"], capture(unit, torch, np), torch, "exact_prestate_restore")
            pos, neg = serve(bodies, unit, valid, pool, torch)
            # The earlier native replay source fixes 128*float32 epsilon for both.
            tolerance = 128*torch.finfo(torch.float32).eps
            for actual, expected in ((pos, scores["positive_scores"]), (neg, scores["negative_scores"])):
                if actual.shape != expected.shape or actual.dtype != expected.dtype or not bool(torch.isfinite(expected).all()):
                    raise ValueError("Saved native logit geometry differs")
                if not bool(((actual.double()-expected.double()).abs() <= tolerance+tolerance*expected.double().abs()).all()):
                    raise ValueError("Unchanged native replay score tolerance failed")
            equal(scores["native_per_query_metrics"], eval_mrr(pos, neg), torch, "exact_per_query_metrics")
            if scores["native_metrics_rounded4"] != evaluate_mrr(None, pos, neg): raise ValueError("Exact rounded metric replay failed")
            if scores["native_metrics_rounded4"]["MRR"] != state["selected_VALID_MRR"]:
                raise ValueError("Replayed selected native metric identity differs")
            equal(state["state_after_VALID"], capture(unit, torch, np), torch, "exact_postserve_state_RNG")
            positives.append(pos); negatives.append(neg)
            reports.append({"member": row["member"], "seed": seed, "selected_epoch": state["selected_epoch"], "replay": "PASS"})
            del unit, state, scores
            if time.monotonic()-started > job["soft_seconds"]:
                raise TimeoutError("Owned replay bound exceeded; no retry or partial-family promotion")
        mean_pos, mean_neg = torch.stack(positives).mean(0), torch.stack(negatives).mean(0)
        torch.save({"member_pos": torch.stack(positives), "member_neg": torch.stack(negatives),
            "mean_pos": mean_pos, "mean_neg": mean_neg, "input_files": identities,
            "members": members, "source_manifest_sha256": job["source_manifest_sha256"]}, output/"pooled_VALID_scores.pt")
        custody.write_json(output/"REPLAY.json", {"status": "PASS", "optimizer_updates": 0,
            "base_seed": job["base_seed"], "member_replays": reports,
            "single_member0": evaluate_mrr(None, positives[0], negatives[0]),
            "independent_four_mean_raw_logits": evaluate_mrr(None, mean_pos, mean_neg),
            "TEST_access": False, "ensemble_selection": "none; individual native selectors",
            "pooled_scores_sha256": custody.sha(output/"pooled_VALID_scores.pt")})
    except (Exception, KeyboardInterrupt) as error:
        custody.write_json(output/"FAILURE.json", {"error": type(error).__name__+": "+str(error), "retry": False})
        raise


if __name__ == "__main__": main()
