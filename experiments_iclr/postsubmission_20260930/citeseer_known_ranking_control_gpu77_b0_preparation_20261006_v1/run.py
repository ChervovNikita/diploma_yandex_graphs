#!/usr/bin/env python3
"""Known ranking-objective ordinary controls; source preparation only.

No TEST loader; no default seed, schedule, cohort, retry, GPU or selection arm.
Full outer TRAIN coverage includes the tail. Numeric training schedules require
root freeze after measured TRAIN-only feasibility and complete-cycle costs.
"""
import gc
import gzip
import json
import os
import random
import time
from pathlib import Path

import custody

TRAIN_PURPOSE = "TRAIN_VALID_prospective_private_transfer_fit"
COST_PURPOSE = "TRAIN_only_discarded_complete_cycle_cost"
RULES = ("ordinary_joint",)
ARMS = ("shared_f4", "capable_single", "ordinary_native4")


def metric(torch, positive, negative):
    rank = 1+.5*((negative >= positive[:, None]).sum(1)+(negative > positive[:, None]).sum(1))
    return {"MRR": round((1/rank.float()).mean().item(), 4), "Hits10": round((rank <= 10).float().mean().item(), 4)}


def admissible(job):
    if job.get("arm") not in ARMS or job.get("rule") not in RULES or job.get("geometry") not in ("endpoint", "matched_random"):
        raise ValueError("Explicit reviewed architecture/learning-rule/geometry required")
    if job["arm"] == "ordinary_native4" and job["rule"] != "ordinary_joint":
        raise ValueError("Native ordinary4 is a quality control, not a privately adapted arm")
    for key in ("outer_size", "inner_size"):
        if type(job.get(key)) is not int or job[key] < 1: raise ValueError("Root-fixed episode sizes required")
    gate = job.get("training_step_gate", {})
    if gate.get("approved") is not True: raise ValueError("New FP32 Adam/committed-state gate not admitted")
    path = custody.phase_file(gate["path"])
    if custody.sha(path) != gate["sha256"]: raise ValueError("Exact prospective training-step gate changed")
    receipt = json.loads(path.read_text())
    if receipt.get("passed") is not True or receipt.get("scope") != "discarded_FP32_training_step_qualification":
        raise ValueError("Historical SGD/float64/FD diagnostics cannot admit this Adam recipe")
    if receipt.get("source_manifest_sha256") != job["source_manifest_sha256"]:
        raise ValueError("FP32 Adam gate did not qualify these exact sealed training source bytes")
    if set(receipt.get("architectures", {})) != {"shared_f4", "capable_single", "ordinary_native4"}:
        raise ValueError("Exact three ordinary-control architecture gates missing")
    if receipt["geometry"]["endpoint"]["roles"]["outer_positive"]["queries"] != job["outer_size"]:
        raise ValueError("Training episode size differs from qualified first fixture")
    if receipt["geometry"]["endpoint"]["inner_routes"][0]["positive"]["queries"] != job["inner_size"]:
        raise ValueError("Private episode size differs from qualified fixture")
    if job["rule"] == "live_stale_transfer" and receipt.get("stale_commit_qualification") is not True:
        raise ValueError("Optional stale private serving state has not been prospectively qualified")


def freeze_schedule(job, purpose):
    if purpose == COST_PURPOSE:
        if job.get("fits_authorized") is not False or job.get("VALID_values_access") is not False or job.get("cycles") != 1:
            raise ValueError("Cost gate is one complete discarded TRAIN cycle only")
        if job.get("result_based_early_stop") is not False:
            raise ValueError("Do not shorten a paid complete-cycle measurement")
        return {"max_cycles": 1, "eval_every_cycles": None, "validation_miss_limit": None}
    if job.get("fits_authorized") is not True or job.get("VALID_values_access") is not True:
        raise ValueError("Exact prospective TRAIN/VALID fit scope required")
    path = custody.phase_file(job["cohort_plan_relative"])
    if custody.sha(path) != job["cohort_plan_sha256"]: raise ValueError("Prospective cohort plan changed")
    plan = json.loads(path.read_text())
    if plan.get("root_adopted_after_TRAIN_cost") is not True or plan.get("TEST_closed") is not True:
        raise ValueError("Root must freeze the recipe after measured TRAIN cost, before scores")
    for evidence in plan["complete_cycle_cost_evidence"]:
        if custody.sha(custody.phase_file(evidence["path"])) != evidence["sha256"]:
            raise ValueError("Complete-cycle cost evidence changed")
    if not plan.get("complete_cycle_cost_evidence"): raise ValueError("No actual complete-cycle cost evidence")
    match = [row for row in plan["cells"] if row.get("cell_id") == job.get("cell_id")]
    if len(match) != 1: raise ValueError("Fit absent/duplicated in immutable prospective cohort")
    cell = match[0]
    for key in ("arm", "rule", "geometry", "seed", "factor_seed", "paired_seed_block", "outer_size", "inner_size"):
        if job.get(key) != cell.get(key): raise ValueError("Cohort cell identity differs: "+key)
    schedule = cell["schedule"]
    if schedule != {"max_cycles": 60, "eval_every_cycles": 5, "validation_miss_limit": None, "result_based_early_stop": False}:
        raise ValueError("Require exactly sixty complete cycles and complete VALID each five cycles; result-based early stop disabled")
    if plan.get("selection") != "first_maximum_complete_VALID_MRR_rounded4":
        raise ValueError("Prospective complete validation selection differs")
    if plan.get("selection_budget_fairness_approved") is not True or not plan.get("paid_budget_description"):
        raise ValueError("Paid comparator training/selection budgets not explicitly approved")
    return schedule


def main():
    started = time.monotonic()
    args = custody.parser(__doc__).parse_args()
    proposed = json.loads(args.job.read_text())
    purpose = proposed.get("purpose")
    if purpose not in (TRAIN_PURPOSE, COST_PURPOSE): raise ValueError("No execution-purpose default")
    job, output = custody.authorize(__file__, args, purpose)
    admissible(job); schedule = freeze_schedule(job, purpose)
    is_fit = purpose == TRAIN_PURPOSE
    output.mkdir()
    custody.write_json(output/"START.json", {"job_sha256": custody.sha(args.job), "purpose": purpose,
        "arm": job["arm"], "rule": job["rule"], "geometry": job["geometry"], "seed": job["seed"],
        "schedule": schedule, "TEST_access": False, "PID": os.getpid(), "retry": False})
    try:
        torch, device, versions = custody.runtime(job)
        native, heads, geometry, cycle, adjoint, models, steps = custody.load_sources()
        import private_adam
        x, train, valid, pool, identities = custody.load_inputs(torch, job, include_valid=is_fit)
        x = x.to(device)
        endpoint_streams = geometry.route_streams(job["seed"], 4)
        random_streams = geometry.route_streams(job["seed"], 4, control=True)
        model, shared, private, partition = models.build(native, heads, arm=job["arm"], seed=job["seed"],
            factor_seed=job["factor_seed"], device=device)
        custody.write_json(output/"PARAMETER_PARTITION.json", partition)
        parameters = dict(model.named_parameters())
        streams = steps.DropoutStreams(torch, device, job["seed"])
        previous = private_adam.initial_state(torch, {name: parameters[name] for name in private}) if job["rule"] != "ordinary_joint" else None
        optimizer = torch.optim.Adam([parameters[name] for name in shared] if previous is not None else list(parameters.values()),
            lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
        config = {"job": job, "schedule": schedule, "runtime": versions, "inputs": identities,
            "native_author_commit": "c447cbff4c493b60d14b6544c3c39d3b9c5ddff0",
            "native_architecture": "NCN,width256,one_puregcn,input_projection,JK,nonlinear_CN_predictor",
            "member_count": model.member_count, "head": "unchanged_unframed_F4" if job["arm"] in ("shared_f4", "untied4") else "native_full_nonlinear_NCN",
            "private_role": "unadapted_joint_ordinary_native_heads" if job["arm"] == "ordinary_native4" else "entire_nonlinear_predictor" if job["arm"] == "capable_single" else "route_r_s_LayerNorm_beta",
            "optimizer": "native_constants_Adam_single_tensor; functional_private_exact_stable_zero_history",
            "Adam": {"lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "foreach": False, "fused": False},
            "xdrop": .4, "GNN_head_drop": .3, "dropedge": 0., "inner_dropout": "route_stream_replayed_virtual_and_recomputed",
            "outer_dropout": "off", "outer_objective": "0.5_scaled_BPR_mean_raw_logits+0.5_mean_own_scaled_BPR",
            "loss": "known_ranking_objective_CONTROL_2softplus((negative-positive)/2); fixed_scale_no_tuning",
            "historical_bitwise_initialization_verified": False,
            "serving": "committed_private_state_mean_raw_logits_without_adaptation",
            "support": "identical_union_of_outer+endpoint_inner+matched_random_inner_positive_targets; no endpoint node deletion",
            "geometry_control": "exact_full_TRAIN_degree/CN_strata; post_mask_strata_measured_not_asserted_equal",
            "coverage": "every_TRAIN_positive_once_as_outer_each_complete_cycle_including_tail",
            "native_donor_epoch_comparison": {"full_batches": len(train)//1024, "dropped_tail": len(train)%1024,
                                              "prospective_episode_batches": (len(train)+job["outer_size"]-1)//job["outer_size"]},
            "ordinary_joint": "three_committed_native_Adam_updates_inner_outer_inner; same_examples_and_mask_replay; not_identical_optimization",
            "TEST_access": False, "cost_gate_values_only": not is_fit,
            "uncertainty_limit": "one_dataset_native_fixed_split; no_novelty_or_superiority_inferred_from_control_gates"}
        custody.write_json(output/"CONFIG.json", config)
        validation_support = custody.support_tensor(torch, train, list(range(len(train))), len(x), device) if is_fit else None
        if is_fit: valid, pool = valid.to(device), pool.to(device)

        @torch.no_grad()
        def validate():
            member_positive, member_negative = [], []
            for start in range(0, len(valid), 512):
                ids = torch.arange(start, min(start+512, len(valid)), device=device)
                negative_queries = pool[ids].permute(2, 0, 1).reshape(2, -1)
                positive, negative_logits = steps.functional_forward(torch, model, dict(model.named_parameters()), x,
                    validation_support, (valid[ids].t(), negative_queries), training=False)
                if positive.shape != (len(ids), model.member_count) or negative_logits.shape != (len(ids)*500, model.member_count):
                    raise ValueError("Complete fixed500 member serving geometry differs")
                member_positive.append(positive.cpu()); member_negative.append(negative_logits.reshape(len(ids), 500, model.member_count).cpu())
            positive, negative_logits = torch.cat(member_positive), torch.cat(member_negative)
            if not bool(torch.isfinite(positive).all() and torch.isfinite(negative_logits).all()): raise FloatingPointError("Nonfinite complete VALID serving logits")
            return metric(torch, positive.mean(-1), negative_logits.mean(-1)), positive, negative_logits

        def rng_state():
            return {"global_cpu": torch.get_rng_state(), "global_cuda": torch.cuda.get_rng_state(device),
                "private_dropout": streams.state_dict(), "endpoint_sampling": [{k: r.getstate() for k, r in row.items()} for row in endpoint_streams],
                "matched_random_sampling": [{k: r.getstate() for k, r in row.items()} for row in random_streams],
                "python_global": random.getstate()}

        counters = {"episodes": 0, "functional_forward_calls": 0, "encoder_forward_count": 0,
                    "shared_updates": 0, "private_updates": 0, "ordinary_joint_updates": 0}
        selected = selected_cycle = None; best = 0.; misses = 0
        cycle_costs = []; torch.cuda.reset_peak_memory_stats()
        with custody.native_adjoint(torch, native, adjoint.make_constant_adjacency_spmm), \
             gzip.open(output/"CYCLE_DRAWS.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as draws, \
             (output/"EPISODE_HISTORY.jsonl").open("x") as episode_history, \
             (output/"CYCLE_HISTORY.jsonl").open("x") as cycle_history, \
             (output/"VALID_HISTORY.jsonl").open("x") as validation_history:
            for cycle_index in range(schedule["max_cycles"]):
                cycle_started = time.monotonic(); torch.cuda.synchronize()
                negative, order, episode_pairs = custody.make_pair(torch, geometry, cycle, train, len(x), job["seed"],
                    cycle_index, job["outer_size"], job["inner_size"], endpoint_streams, random_streams)
                geometry_seconds = time.monotonic()-cycle_started
                full_episodes = [row[0 if job["geometry"] == "endpoint" else 1] for row in episode_pairs]
                exposure = geometry.exposure_summary(train.tolist(), negative.tolist(), full_episodes, len(x), 4)
                if exposure["outer_positive_ids_unseen"] != 0 or exposure["outer_positive_instances"] != len(train):
                    raise ValueError("Incomplete complete-cycle outer TRAIN exposure")
                draws.write(json.dumps({"cycle": cycle_index+1, "negative_bank": negative.tolist(), "outer_order": order,
                    "paired_episodes": [{"endpoint": row[0], "matched_random": row[1], "kept_positive_ids": row[2],
                                         "description": row[3]} for row in episode_pairs], "exposure": exposure})+"\n"); draws.flush()
                cycle_updates = []
                for episode_index, (endpoint, random_episode, kept, description) in enumerate(episode_pairs):
                    episode = endpoint if job["geometry"] == "endpoint" else random_episode
                    support = custody.support_tensor(torch, train, kept, len(x), device)
                    inner, outer = custody.queries(torch, train, negative, episode, device)
                    torch.cuda.synchronize(); episode_started = time.monotonic()
                    if job["rule"] == "ordinary_joint":
                        update = steps.ordinary_episode_step(torch, model, optimizer, x, support, inner, outer, streams)
                        counters["ordinary_joint_updates"] += update["native_joint_Adam_updates"]
                    else:
                        previous, update = steps.episode_step(torch, model, shared, private, optimizer, previous,
                            x, support, inner, outer, streams, live=job["rule"] != "detached_transfer",
                            commit="stale_virtual" if job["rule"] == "live_stale_transfer" else "recomputed")
                        counters["shared_updates"] += update["shared_optimizer_updates"]
                        counters["private_updates"] += update["private_optimizer_updates"]
                    torch.cuda.synchronize()
                    paid = {"cycle": cycle_index+1, "episode": episode_index+1, "elapsed_seconds": time.monotonic()-episode_started,
                            "outer_positive_rows": len(episode["outer_pos_ids"]), "inner_rows_per_route": [len(row["pos_ids"]) for row in episode["inner"]],
                            "mask_positive_count": len(train)-len(kept), "support_nnz": support.nnz(), "update": update}
                    episode_history.write(json.dumps(paid)+"\n"); episode_history.flush()
                    cycle_updates.append(paid)
                    counters["episodes"] += 1; counters["functional_forward_calls"] += update["functional_forward_calls"]
                    counters["encoder_forward_count"] += update["model_encoder_forward_count"]
                    custody.write_json(output/"PROGRESS.json", {"cycle": cycle_index+1, "episode": episode_index+1,
                        "episodes_in_cycle": len(episode_pairs), "counters": counters, "inclusive_seconds": time.monotonic()-started})
                    del support, inner, outer
                    if time.monotonic()-started > job["soft_seconds"]:
                        raise TimeoutError("Root-frozen job limit exceeded; preserve partial work, no shortening/retry")
                torch.cuda.synchronize()
                cost = {"cycle": cycle_index+1, "episodes": len(episode_pairs), "geometry_seconds": geometry_seconds,
                    "complete_cycle_seconds": time.monotonic()-cycle_started, "neural_seconds": sum(r["elapsed_seconds"] for r in cycle_updates),
                    "outer_positive_instances": exposure["outer_positive_instances"], "outer_negative_instances": exposure["outer_negative_instances"],
                    "inner_positive_instances_per_route": [r["positive_instances"] for r in exposure["routes"]],
                    "inner_negative_instances_per_route": [r["negative_instances"] for r in exposure["routes"]],
                    "inner_twice_computed_positive_instances": 2*sum(r["positive_instances"] for r in exposure["routes"]),
                    "inner_twice_computed_negative_instances": 2*sum(r["negative_instances"] for r in exposure["routes"]),
                    "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "counters": dict(counters)}
                cycle_history.write(json.dumps(cost)+"\n"); cycle_history.flush(); cycle_costs.append(cost)
                if not is_fit or (cycle_index+1)%schedule["eval_every_cycles"]: continue
                validation_started = time.monotonic()
                result, positive, negative_logits = validate()
                validation_history.write(json.dumps({"cycle": cycle_index+1, "complete_VALID": result,
                    "validation_seconds": time.monotonic()-validation_started, "counters": counters})+"\n"); validation_history.flush()
                score = result["MRR"]
                if selected is None or score > selected:
                    selected, selected_cycle = score, cycle_index+1
                    checkpoint = output/"selected_checkpoint.pt"
                    torch.save({"model": model.state_dict(), "outer_or_joint_optimizer": optimizer.state_dict(), "private_Adam": previous,
                        "RNG": rng_state(), "config": config, "partition": partition, "selected_cycle": selected_cycle,
                        "selected_VALID_MRR": selected, "counters": dict(counters)}, checkpoint)
                    torch.save({"member_pos": positive, "member_neg": negative_logits, "mean_pos": positive.mean(-1),
                        "mean_neg": negative_logits.mean(-1), "selected_cycle": selected_cycle, "inputs": identities,
                        "checkpoint_sha256": custody.sha(checkpoint)}, output/"selected_VALID_logits.pt")
                if score > best: best, misses = score, 0
                else:
                    misses += 1
                    # Explicit prospective schedule change: all sixty cycles are paid.
        torch.cuda.synchronize()
        final = {"scope": "TRAIN_VALID_freeze" if is_fit else "discarded_TRAIN_complete_cycle_cost",
            "arm": job["arm"], "rule": job["rule"], "geometry": job["geometry"], "seed": job["seed"],
            "source_manifest_sha256": job["source_manifest_sha256"], "job_sha256": custody.sha(args.job),
            "config_sha256": custody.sha(output/"CONFIG.json"), "input_identities": identities, "cycle_costs": cycle_costs,
            "counters": counters, "last_cycle": cycle_index+1, "selected_VALID_MRR": selected, "selected_cycle": selected_cycle,
            "complete_VALID_scoring": is_fit, "TEST_access": False, "states_discarded": not is_fit,
            "inclusive_seconds": time.monotonic()-started, "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "native_alias_restored": True,
            "history": {name: custody.sha(output/name) for name in ("CYCLE_DRAWS.jsonl.gz", "EPISODE_HISTORY.jsonl", "CYCLE_HISTORY.jsonl", "VALID_HISTORY.jsonl")}}
        if is_fit:
            if selected is None: raise ValueError("No complete native-cadence VALID selection occurred")
            if len(cycle_costs) != 60: raise ValueError("Full sixty-cycle fit did not complete")
            final.update(checkpoint_sha256=custody.sha(output/"selected_checkpoint.pt"),
                         VALID_logits_sha256=custody.sha(output/"selected_VALID_logits.pt"), cohort_plan_sha256=job["cohort_plan_sha256"],
                         paired_seed_block=job["paired_seed_block"], stop_rule="fixed_60_complete_cycles_no_result_based_early_stop")
        custody.write_json(output/("FREEZE.json" if is_fit else "COST_RESULT.json"), final)
        print(json.dumps({"status": "complete_VALID_selected_TEST_closed" if is_fit else "complete_discarded_paid_cycle_cost", "output": str(output)}), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        custody.write_json(output/"FAILURE.json", {"error": type(error).__name__+": "+str(error),
            "inclusive_seconds": time.monotonic()-started, "TEST_access": False, "retry": False,
            "partial_outputs_preserved": True, "purpose": purpose})
        raise


if __name__ == "__main__": main()
