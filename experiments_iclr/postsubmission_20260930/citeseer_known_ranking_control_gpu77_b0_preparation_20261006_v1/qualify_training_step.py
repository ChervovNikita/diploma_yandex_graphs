#!/usr/bin/env python3
"""Discarded ordinary Adam qualification for the known ranking control.

TRAIN/features only. One first-episode ordinary history per architecture,
explicit direct-module logit VJP reference, all states discarded. No live
credit, private functional Adam, VALID/TEST values, fit or checkpoint.
"""
import gc
import json
import math
import os
import resource
import time

import custody

PARAM_RTOL, PARAM_ATOL = 2e-6, 2e-7
MOMENT_RTOL, MOMENT_ATOL = 2e-6, 1e-10
DERIV_RTOL, DERIV_ATOL, DERIV_REL_L2 = 5e-5, 1e-7, 2e-5
ARMS = ("shared_f4", "capable_single", "ordinary_native4")

def compare(torch, actual, expected, *, rtol, atol, group_l2=None):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Qualification shape/precision differs")
    a, b = actual.detach(), expected.detach()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError("Qualification tensor is nonfinite")
    difference = (a-b).abs()
    limit = atol+rtol*torch.maximum(a.abs(), b.abs())
    relative = float((a-b).double().norm()/torch.maximum(a.double().norm(), b.double().norm()).clamp_min(1e-300))
    ok = bool((difference <= limit).all()) and (group_l2 is None or relative <= group_l2)
    # Diagnostics only: 0/0 for an exact comparison is reported as zero.
    # A nonzero error at zero tolerance remains a failed criterion and has an
    # explicit null ratio/status, rather than emitting JSON NaN or Infinity.
    zero_limit = limit == 0
    safe_limit = torch.where(zero_limit, torch.ones_like(limit), limit)
    ratio = torch.where(zero_limit & (difference == 0), torch.zeros_like(difference), difference/safe_limit)
    maximum_ratio = float(ratio.max())
    maximum_absolute = float(difference.max())
    if bool((zero_limit & (difference != 0)).any()):
        maximum_ratio, ratio_status = None, "nonzero_error_with_zero_limit"
    elif not math.isfinite(maximum_ratio):
        maximum_ratio, ratio_status = None, "nonfinite_diagnostic_ratio"
    else:
        ratio_status = "finite"
    result = {"coordinates": a.numel(),
              "maximum_absolute_error": maximum_absolute if math.isfinite(maximum_absolute) else None,
              "maximum_absolute_error_status": "finite" if math.isfinite(maximum_absolute) else "nonfinite_diagnostic_difference",
              "maximum_coordinate_error_to_limit": maximum_ratio,
              "coordinate_error_to_limit_status": ratio_status,
              "relative_L2": relative if math.isfinite(relative) else None,
              "relative_L2_status": "finite" if math.isfinite(relative) else "nonfinite_diagnostic_norm",
              "passed": ok}
    if not ok: raise ValueError("Prospective immutable qualification tolerance failed: "+json.dumps(result))
    return result


def tree_equal(torch, a, b):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(tree_equal(torch, a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    return a == b


def mapping_compare(torch, actual, expected, *, moments=False, derivative=False):
    if set(actual) != set(expected): raise ValueError("Qualification omitted a parameter/moment name")
    rtol, atol = ((MOMENT_RTOL, MOMENT_ATOL) if moments else
                 (DERIV_RTOL, DERIV_ATOL) if derivative else (PARAM_RTOL, PARAM_ATOL))
    return {name: compare(torch, actual[name], expected[name], rtol=rtol, atol=atol,
                          group_l2=DERIV_REL_L2 if derivative else None) for name in actual}


def moment_compare(torch, actual, expected):
    if set(actual) != set(expected): raise ValueError("Native private moment names differ")
    rows = {}
    for name in actual:
        if actual[name]["step"] != expected[name]["step"]: raise ValueError("Native Adam step counter differs")
        rows[name] = mapping_compare(torch, {k: actual[name][k] for k in ("exp_avg", "exp_avg_sq")},
            {k: expected[name][k] for k in ("exp_avg", "exp_avg_sq")}, moments=True)
    return rows


def optimizer_moments(model, optimizer):
    return {name: {"step": int(optimizer.state[p]["step"].item()),
                   "exp_avg": optimizer.state[p]["exp_avg"],
                   "exp_avg_sq": optimizer.state[p]["exp_avg_sq"]}
            for name, p in model.named_parameters()}


def direct_forward(torch, model, x, support, queries, streams, *, training, route=None, advance=False):
    """Use the committed native modules directly, without functional_call."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    model.train(training)
    try:
        if training:
            with streams.use(route, advance=advance):
                result = model(x, support, *queries, route=route)
        else:
            result = model(x, support, *queries, route=route)
        if any(not bool(torch.isfinite(value).all()) for value in result):
            raise FloatingPointError("Nonfinite direct native-module reference logits")
        if not tree_equal(torch, dict(model.named_buffers()), buffers):
            raise ValueError("Native direct reference mutated constant buffers")
        return result
    finally:
        for module, mode in modes: module.training = mode


def reference_episode(torch, model, optimizer, x, support, inner, outer, streams):
    """Independent analytical logit VJP and three native torch.optim.Adam commits.

    d[2 softplus((n-p)/2)]/dn = sigmoid((n-p)/2). The pooled and
    member terms use their separate original means. This path calls neither
    own_loss, outer_loss nor ordinary_episode_step.
    """
    calls = [(0, (torch.cat([r[0] for r in inner], 1),
                  torch.cat([r[1] for r in inner], 1)))] if model.member_count == 1 else list(enumerate(inner))
    for pass_index in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs, cotangents = [], []
        if pass_index == 1:
            positive, negative = direct_forward(torch, model, x, support, outer, streams, training=False)
            if positive.shape != negative.shape or positive.numel() == 0:
                raise ValueError("Reference outer positional pair geometry differs")
            with torch.no_grad():
                gradient = (.5*torch.sigmoid((negative-positive)/2)/positive.numel()
                    + .5*torch.sigmoid((negative.mean(1)-positive.mean(1))/2)[:, None]/positive.numel())
                if hasattr(model, "routes"): gradient = gradient*model.member_count
            outputs += [positive, negative]; cotangents += [-gradient, gradient]
        else:
            normalization = 1 if hasattr(model, "routes") else len(calls)
            for member, queries in calls:
                positive, negative = direct_forward(torch, model, x, support, queries, streams,
                    training=True, route=member, advance=pass_index == 2)
                if positive.shape != negative.shape or positive.numel() == 0:
                    raise ValueError("Reference inner positional pair geometry differs")
                with torch.no_grad():
                    gradient = torch.sigmoid((negative-positive)/2)/(positive.numel()*normalization)
                outputs += [positive, negative]; cotangents += [-gradient, gradient]
        torch.autograd.backward(outputs, cotangents)
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("Reference parameter disconnected/nonfinite: "+name)
        optimizer.step()


def complete_cycle(torch, native, heads, geometry, cycle, models, steps, x, train, job, arm, output, stage):
    """One fresh full TRAIN cycle, including the tail; no selection or VALID."""
    started = time.monotonic()
    endpoint_streams = geometry.route_streams(job["seed"], 4)
    random_streams = geometry.route_streams(job["seed"], 4, control=True)
    negative, order, episodes = custody.make_pair(torch, geometry, cycle, train, len(x), job["seed"],
        0, job["outer_size"], job["inner_size"], endpoint_streams, random_streams)
    geometry_seconds = time.monotonic()-started
    exposure = geometry.exposure_summary(train.tolist(), negative.tolist(), [r[0] for r in episodes], len(x), 4)
    if exposure["outer_positive_ids_unseen"] != 0 or exposure["outer_positive_instances"] != 3870:
        raise ValueError("Discarded full-cycle TRAIN exposure incomplete")
    model, _, _, _ = models.build(native, heads, arm=arm, seed=job["seed"], factor_seed=job["factor_seed"], device=x.device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=.001, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    streams = steps.DropoutStreams(torch, x.device, job["seed"])
    counters = {"episodes": 0, "ordinary_joint_updates": 0, "functional_forward_calls": 0, "encoder_forward_count": 0}
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); neural_started = time.monotonic()
    with (output/(arm+"_COST_EPISODES.jsonl")).open("x") as history:
        for index, (endpoint, _, kept, _) in enumerate(episodes):
            support = custody.support_tensor(torch, train, kept, len(x), x.device)
            inner, outer = custody.queries(torch, train, negative, endpoint, x.device)
            torch.cuda.synchronize(); episode_started = time.monotonic()
            update = steps.ordinary_episode_step(torch, model, optimizer, x, support, inner, outer, streams)
            torch.cuda.synchronize()
            history.write(json.dumps({"episode": index+1, "elapsed_seconds": time.monotonic()-episode_started,
                "outer_positive_rows": len(endpoint["outer_pos_ids"]), "support_nnz": support.nnz(), "update": update})+"\n")
            history.flush()
            counters["episodes"] += 1; counters["ordinary_joint_updates"] += update["native_joint_Adam_updates"]
            counters["functional_forward_calls"] += update["functional_forward_calls"]
            counters["encoder_forward_count"] += update["model_encoder_forward_count"]
            del support, inner, outer
            stage(arm+"_cost_episode", {"episode": index+1, "total": len(episodes)})
    neural_seconds = time.monotonic()-neural_started
    if counters["episodes"] != 61 or counters["ordinary_joint_updates"] != 183:
        raise ValueError("Full discarded TRAIN cycle did not pay every episode/three-pass update")
    result = {"scope": "discarded_TRAIN_complete_cycle_cost", "arm": arm, "rule": "ordinary_joint", "geometry": "endpoint",
        "seed": job["seed"], "factor_seed": job["factor_seed"], "physical_gpu_uuid": job["physical_gpu_uuid"],
        "source_manifest_sha256": custody.sha(custody.ROOT/"SOURCE_MANIFEST.json"), "states_discarded": True,
        "VALID_TEST_values_access": False, "complete_VALID_scoring": False, "last_cycle": 1,
        "geometry_seconds": geometry_seconds, "neural_seconds": neural_seconds, "complete_cycle_seconds": time.monotonic()-started,
        "inclusive_seconds": time.monotonic()-started, "exposure": exposure, "counters": counters,
        "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved()}
    custody.write_json(output/(arm+"_COST_RESULT.json"), result)
    del model, optimizer, streams
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    started = time.monotonic(); args = custody.parser(__doc__).parse_args()
    job, output = custody.authorize(__file__, args, "fp32_discarded_training_step_qualification")
    assigned = {custody.GPU_UUIDS[0]: ["shared_f4", "capable_single"], custody.GPU_UUIDS[1]: ["ordinary_native4"]}
    if job.get("architectures") != assigned[job["physical_gpu_uuid"]] or job.get("reachable_history_steps") != 1:
        raise ValueError("Only the prospectively assigned architecture/GPU first-episode checks are admitted")
    if job.get("fits_authorized") is not False or job.get("VALID_values_access") is not False or job.get("full_TRAIN_cycles_per_architecture") != 1:
        raise ValueError("Discarded first-episode parity plus one full TRAIN-only cycle per architecture required")
    if job.get("episode_index") != 0 or job.get("outer_size") != 64 or job.get("inner_size") != 256 or job.get("seed") != 0 or job.get("factor_seed") != 283136447778829174:
        raise ValueError("Require the exact b0 first-cycle/first-episode fixture")
    tolerances = {"parameter": [PARAM_RTOL, PARAM_ATOL], "moment": [MOMENT_RTOL, MOMENT_ATOL]}
    if job.get("tolerances") != tolerances: raise ValueError("Prospective native-Adam parity tolerances differ")
    output.mkdir()
    receipt = {"scope": "discarded_FP32_training_step_qualification", "job_sha256": custody.sha(args.job),
        "source_manifest_sha256": custody.sha(custody.ROOT/"SOURCE_MANIFEST.json"), "physical_gpu_uuid": job["physical_gpu_uuid"],
        "fits": 0, "VALID_TEST_values_access": False, "states_discarded": True, "tolerances": tolerances,
        "architectures": {}, "passed": False, "PID": os.getpid(), "retry": False}
    custody.write_json(output/"START.json", receipt)

    def stage(name, values):
        row = {"stage": name, "elapsed_seconds": time.monotonic()-started, **values}
        with (output/"STAGES.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        if time.monotonic()-started > job["soft_seconds"]: raise TimeoutError("Fixed discarded qualification/cost soft bound exceeded")

    try:
        torch, device, versions = custody.runtime(job)
        native, heads, geometry, cycle, adjoint, models, steps = custody.load_sources()
        x, train, _, _, identities = custody.load_inputs(torch, job, include_valid=False); x = x.to(device)
        endpoint_streams = geometry.route_streams(job["seed"], 4); random_streams = geometry.route_streams(job["seed"], 4, control=True)
        negative, order, episodes = custody.make_pair(torch, geometry, cycle, train, len(x), job["seed"], 0, 64, 256, endpoint_streams, random_streams)
        endpoint, _, kept, description = episodes[0]
        support = custody.support_tensor(torch, train, kept, len(x), device)
        inner, outer = custody.queries(torch, train, negative, endpoint, device)
        receipt.update(runtime=versions, inputs=identities, geometry=description, complete_cycle_episodes=len(episodes), full_TRAIN_coverage=len(order))
        zpos = torch.zeros((2, 3), device=device, requires_grad=True); zneg = torch.zeros_like(zpos, requires_grad=True)
        loss = steps.own_loss(torch, zpos, zneg); dp, dn = torch.autograd.grad(loss, (zpos, zneg))
        receipt["scalar_loss_at_zero"] = compare(torch, loss, torch.tensor(2*math.log(2), device=device), rtol=PARAM_RTOL, atol=PARAM_ATOL)
        receipt["scalar_positive_gradient"] = compare(torch, dp, torch.full_like(dp, -.5/dp.numel()), rtol=PARAM_RTOL, atol=PARAM_ATOL)
        receipt["scalar_negative_gradient"] = compare(torch, dn, torch.full_like(dn, .5/dn.numel()), rtol=PARAM_RTOL, atol=PARAM_ATOL)
        for arm in job["architectures"]:
            arm_started = time.monotonic()
            model, _, _, partition = models.build(native, heads, arm=arm, seed=job["seed"], factor_seed=job["factor_seed"], device=device)
            reference, _, _, _ = models.build(native, heads, arm=arm, seed=job["seed"], factor_seed=job["factor_seed"], device=device)
            custody.write_json(output/(arm+"_PARTITION.json"), partition)
            if not tree_equal(torch, dict(model.named_parameters()), dict(reference.named_parameters())):
                raise ValueError("Fresh actual/reference initial parameters are not exactly equal")
            initial = mapping_compare(torch, dict(model.named_parameters()), dict(reference.named_parameters()))
            optimizer = torch.optim.Adam(list(model.parameters()), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
            reference_optimizer = torch.optim.Adam(list(reference.parameters()), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
            streams = steps.DropoutStreams(torch, device, job["seed"]); reference_streams = steps.DropoutStreams(torch, device, job["seed"])
            cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state(device).clone()
            modes = [m.training for m in model.modules()]; torch.cuda.reset_peak_memory_stats()
            with custody.native_adjoint(torch, native, adjoint.make_constant_adjacency_spmm):
                update = steps.ordinary_episode_step(torch, model, optimizer, x, support, inner, outer, streams)
                reference_episode(torch, reference, reference_optimizer, x, support, inner, outer, reference_streams)
                parameters = mapping_compare(torch, dict(model.named_parameters()), dict(reference.named_parameters()))
                moments = moment_compare(torch, optimizer_moments(model, optimizer), optimizer_moments(reference, reference_optimizer))
                with torch.no_grad():
                    actual = steps.functional_forward(torch, model, dict(model.named_parameters()), x, support, outer, training=False)
                    expected = direct_forward(torch, reference, x, support, outer, reference_streams, training=False)
                    serving = [compare(torch, a, b, rtol=PARAM_RTOL, atol=PARAM_ATOL) for a, b in zip(actual, expected)]
            if update["native_joint_Adam_updates"] != 3 or any(r["step"] != 3 for r in optimizer_moments(model, optimizer).values()):
                raise ValueError("Ordinary first episode did not commit exactly three native Adam updates")
            if not tree_equal(torch, streams.state_dict(), reference_streams.state_dict()): raise ValueError("Native reference dropout replay/advance differs")
            expected_calls = [1 if model.member_count == 4 or i == 0 else 0 for i in range(4)]
            if [r["committed_calls"] for r in streams.rows] != expected_calls: raise ValueError("Inner dropout streams did not advance exactly once")
            if modes != [m.training for m in model.modules()] or not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
                raise ValueError("Ordinary qualification changed global RNG/native module modes")
            receipt["architectures"][arm] = {"passed": True, "first_episode": {"initial_parameters": initial,
                "committed_parameters": parameters, "native_Adam_moments": moments, "direct_committed_serving": serving,
                "state_RNG_mode_guards": "passed", "update_accounting": update}, "first_episode_seconds": time.monotonic()-arm_started}
            del model, reference, optimizer, reference_optimizer, streams, reference_streams, actual, expected
            gc.collect(); torch.cuda.empty_cache()
            stage(arm+"_first_episode_native_Adam_parity", {"passed": True})
            with custody.native_adjoint(torch, native, adjoint.make_constant_adjacency_spmm):
                cost = complete_cycle(torch, native, heads, geometry, cycle, models, steps, x, train, job, arm, output, stage)
            receipt["architectures"][arm]["complete_cycle_cost"] = cost
            custody.write_json(output/"PARTIAL_RESULT.json", receipt)
        receipt.update(passed=True, ordinary_three_pass_qualification=True, native_alias_restored=True,
            inclusive_seconds=time.monotonic()-started, peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        custody.write_json(output/"RESULT.json", receipt)
        print(json.dumps({"status": "passed_discarded_ordinary_gate_and_full_TRAIN_cycle_cost", "fits": 0}), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(passed=False, error=type(error).__name__+": "+str(error), inclusive_seconds=time.monotonic()-started)
        custody.write_json(output/"FAILURE.json", receipt); raise


if __name__ == "__main__": main()
