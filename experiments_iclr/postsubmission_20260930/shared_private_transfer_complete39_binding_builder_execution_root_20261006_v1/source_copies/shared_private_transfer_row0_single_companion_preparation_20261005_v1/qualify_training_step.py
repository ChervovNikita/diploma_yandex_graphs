#!/usr/bin/env python3
"""UNEXECUTED finite FP32 Adam/chain-rule/committed-serving qualification.

No fit, VALID values, ranking metric, checkpoint or finite differences. Models
and two reachable optimizer-history steps are discarded after metadata gates.
Tolerances below were fixed in source before any numerical execution.
"""
import gc
import json
import math
import os
import resource
import time
from pathlib import Path

import custody

PARAM_RTOL, PARAM_ATOL = 2e-6, 2e-7
MOMENT_RTOL, MOMENT_ATOL = 2e-6, 1e-10
DERIV_RTOL, DERIV_ATOL, DERIV_REL_L2 = 5e-5, 1e-7, 2e-5
ARMS = ("row0_single",)


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


def native_private(torch, parameters, gradients, previous):
    """Independent torch.optim.Adam commit from exactly frozen native moments."""
    clones = {name: torch.nn.Parameter(value.detach().clone()) for name, value in parameters.items()}
    optimizer = torch.optim.Adam(list(clones.values()), lr=.001, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    for name, parameter in clones.items():
        old = previous[name]
        optimizer.state[parameter] = {"step": torch.tensor(float(old["step"])),
            "exp_avg": old["exp_avg"].detach().clone(), "exp_avg_sq": old["exp_avg_sq"].detach().clone()}
        parameter.grad = gradients[name].detach().clone()
    optimizer.step()
    result = {name: value.detach().clone() for name, value in clones.items()}
    moments = {name: {"step": int(optimizer.state[value]["step"].item()),
        "exp_avg": optimizer.state[value]["exp_avg"].detach().clone(),
        "exp_avg_sq": optimizer.state[value]["exp_avg_sq"].detach().clone()} for name, value in clones.items()}
    return result, moments


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


def four_stream_gradient_oracle(torch, steps, model, parameters, private, x, support, inner, streams):
    # Independent objective-first gradient: one mean of four normalized losses.
    losses=[]
    for index, queries in enumerate(inner):
        positive, negative = steps.functional_forward(torch, model, parameters, x, support, queries,
            training=True, stream=(streams,index), route=0, advance=False)
        losses.append(steps.own_loss(torch,positive,negative))
    value=torch.stack(losses).mean()
    gradients=torch.autograd.grad(value,tuple(parameters[name] for name in private))
    return dict(zip(private,gradients))


def ordinary_joint_gate(torch,native,heads,models,steps,x,support,inner,outer,job,device):
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(device).clone()
    actual,_,_,_=models.build(native,heads,arm="row0_single",seed=job["seed"],factor_seed=job["factor_seed"],device=device)
    reference,_,_,_=models.build(native,heads,arm="row0_single",seed=job["seed"],factor_seed=job["factor_seed"],device=device)
    actual_modes=[module.training for module in actual.modules()]
    reference_modes=[module.training for module in reference.modules()]
    aopt=torch.optim.Adam(actual.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    ropt=torch.optim.Adam(reference.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    ast=steps.DropoutStreams(torch,device,job["seed"]);rst=steps.DropoutStreams(torch,device,job["seed"])
    for pass_index in range(3):
        ropt.zero_grad(set_to_none=True);parameters=dict(reference.named_parameters())
        if pass_index==1:
            pos,neg=steps.functional_forward(torch,reference,parameters,x,support,outer,training=False)
            loss=steps.outer_loss(torch,pos,neg)
        else:
            losses=[]
            for index,queries in enumerate(inner):
                pos,neg=steps.functional_forward(torch,reference,parameters,x,support,queries,training=True,
                    stream=(rst,index),route=0,advance=pass_index==2)
                losses.append(steps.own_loss(torch,pos,neg))
            loss=torch.stack(losses).mean()
        loss.backward();ropt.step()
    update=steps.ordinary_episode_step(torch,actual,aopt,x,support,inner,outer,ast)
    ap=dict(actual.named_parameters());rp=dict(reference.named_parameters())
    row={"parameters":mapping_compare(torch,ap,rp),"update_accounting":update}
    def moments(opt,params):
        return {name:{"step":int(opt.state[value]["step"].item()),"exp_avg":opt.state[value]["exp_avg"],
                      "exp_avg_sq":opt.state[value]["exp_avg_sq"]} for name,value in params.items()}
    row["moments"]=moment_compare(torch,moments(aopt,ap),moments(ropt,rp))
    if not tree_equal(torch,ast.state_dict(),rst.state_dict()) or any(r["committed_calls"]!=1 for r in ast.rows):
        raise ValueError("Ordinary four-stream replay/advancement differs")
    if update["native_joint_Adam_updates"]!=3 or update["functional_forward_calls"]!=9:
        raise ValueError("Ordinary paid-work operation differs")
    with torch.no_grad():
        actual_serving=steps.functional_forward(torch,actual,ap,x,support,outer,training=False)
        expected_serving=steps.functional_forward(torch,reference,rp,x,support,outer,training=False)
        actual.eval();h=actual.encoder(x,support)
        donor_serving=(actual.predictor(h,support,outer[0]),actual.predictor(h,support,outer[1]))
        for module,mode in zip(actual.modules(),actual_modes):module.training=mode
    row["explicit_committed_serving_parity"]=[compare(torch,a,b,rtol=PARAM_RTOL,atol=PARAM_ATOL)
        for a,b in zip(actual_serving,expected_serving)]
    row["original_donor_serving_parity"]=[compare(torch,a,b,rtol=0.,atol=0.) for a,b in zip(actual_serving,donor_serving)]
    if ([module.training for module in actual.modules()]!=actual_modes or
            [module.training for module in reference.modules()]!=reference_modes):
        raise ValueError("Ordinary native training modes not restored")
    if not torch.equal(torch.get_rng_state(),cpu_rng) or not torch.equal(torch.cuda.get_rng_state(device),cuda_rng):
        raise ValueError("Ordinary four-stream work advanced global RNG")
    row["state_RNG_mode_guards"]="passed"
    return row


def main():
    started = time.monotonic()
    args = custody.parser(__doc__).parse_args()
    job, output = custody.authorize(__file__, args, "fp32_discarded_training_step_qualification")
    if job.get("fits_authorized") is not False or job.get("VALID_values_access") is not False:
        raise ValueError("This finite source qualifies steps only; TRAIN-only values")
    expected_tolerances = {"parameter": [PARAM_RTOL, PARAM_ATOL], "moment": [MOMENT_RTOL, MOMENT_ATOL],
                           "derivative": [DERIV_RTOL, DERIV_ATOL, DERIV_REL_L2]}
    if job.get("tolerances") != expected_tolerances:
        raise ValueError("Qualification tolerances differ from prospective source")
    if job.get("architectures") != list(ARMS) or job.get("reachable_history_steps") != 2:
        raise ValueError("Row0 companion and zero plus two reachable Adam histories required")
    if job.get("stale_commit_extra_discarded_step") is not False:
        raise ValueError("Companion gates recomputed commitment only")
    output.mkdir()
    receipt = {"scope": "discarded_FP32_training_step_qualification", "job_sha256": custody.sha(args.job),
        "source_manifest_sha256": custody.sha(custody.ROOT/"SOURCE_MANIFEST.json"), "fits": 0,
        "VALID_TEST_values_access": False, "finite_differences": False, "states_discarded": True,
        "tolerances": expected_tolerances, "architectures": {}, "passed": False, "PID": os.getpid()}
    custody.write_json(output/"START.json", receipt)

    def stage(name, row):
        row = {"stage": name, "elapsed_seconds": time.monotonic()-started, **row}
        with (output/"STAGES.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        if time.monotonic()-started > job["soft_seconds"]: raise TimeoutError("Root-fixed soft qualification bound exceeded")

    try:
        torch, device, versions = custody.runtime(job)
        native, heads, geometry, cycle, adjoint, models, steps = custody.load_sources()
        import private_adam
        x, train, _, _, identities = custody.load_inputs(torch, job, include_valid=False)
        x = x.to(device)
        endpoint_streams = geometry.route_streams(job["seed"], 4)
        random_streams = geometry.route_streams(job["seed"], 4, control=True)
        negative, order, episodes = custody.make_pair(torch, geometry, cycle, train, len(x), job["seed"],
            0, job["outer_size"], job["inner_size"], endpoint_streams, random_streams)
        if job.get("episode_index") != 0: raise ValueError("One prospective first episode; no outcome-selected fixture")
        endpoint, random_episode, kept, description = episodes[0]
        support = custody.support_tensor(torch, train, kept, len(x), device)
        inner, outer = custody.queries(torch, train, negative, endpoint, device)
        receipt.update(runtime=versions, inputs=identities, geometry=description, complete_cycle_episodes=len(episodes),
                       full_TRAIN_coverage=len(order), support_nnz=support.nnz())
        stage("geometry", {"full_TRAIN_coverage": len(order), "episodes": len(episodes), "first_episode_only": True})

        # Unchanged Adam scalar primitives retain their original qualification;
        # this gate uses the companion's full TRAIN-only first episode.
        with custody.native_adjoint(torch, native, adjoint.make_constant_adjacency_spmm):
            for arm in ARMS:
                arm_started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
                model, shared, private, partition = models.build(native, heads, arm=arm, seed=job["seed"],
                    factor_seed=job["factor_seed"], device=device)
                custody.write_json(output/(arm+"_PARTITION.json"), partition)
                parameters = dict(model.named_parameters())
                with torch.no_grad():
                    initial_serving = steps.functional_forward(torch, model, parameters, x, support, outer, training=False)
                reference, ref_shared, ref_private, ref_partition = models.build(native, heads, arm="shared_f4",
                    seed=job["seed"], factor_seed=job["factor_seed"], device=device)
                ref_parameters = dict(reference.named_parameters())
                if tuple(shared) != tuple(ref_shared) or tuple(private) != tuple(ref_private):
                    raise ValueError("Row0 private/outer names differ from F4 roles")
                row_maps = {name: compare(torch, parameter, ref_parameters[name][0:1] if name in private else ref_parameters[name], rtol=0., atol=0.)
                            for name, parameter in parameters.items()}
                with torch.no_grad():
                    ref_serving = steps.functional_forward(torch, reference, ref_parameters, x, support, outer, training=False, route=0)
                receipt["row0_initial_parameter_parity"] = row_maps
                receipt["row0_initial_complete_logit_parity"] = [compare(torch, a, b, rtol=0., atol=0.) for a, b in zip(initial_serving, ref_serving)]
                if (sum(parameters[n].numel() for n in shared),sum(parameters[n].numel() for n in private)) != (1409026,6402):
                    raise ValueError("Companion role counts changed")
                receipt["row0_initial_function_and_role_parity"] = True
                del reference, ref_parameters, ref_serving
                previous = private_adam.initial_state(torch, {name: parameters[name] for name in private})
                optimizer = torch.optim.Adam([parameters[name] for name in shared], lr=.001, weight_decay=0., foreach=False, fused=False)
                streams = steps.DropoutStreams(torch, device, job["seed"])
                rows = []
                for history_step in range(3):
                    commit = "recomputed"
                    pristine_private = {name: parameters[name].detach().clone() for name in private}
                    pristine_moments = private_adam.detached_state(previous)
                    pristine_streams = streams.state_dict()
                    pristine_modes = [module.training for module in model.modules()]
                    cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state(device).clone()
                    adapted, gradient, virtual_moment, _, _ = steps.virtual_state(torch, model, x, support, inner,
                        streams, private, previous, live=True)
                    mean_oracle = four_stream_gradient_oracle(torch, steps, model, parameters, private, x, support, inner, streams)
                    if not tree_equal(torch, streams.state_dict(), pristine_streams):
                        raise ValueError("Four-stream oracle mutated RNG state")
                    mean_parity = mapping_compare(torch, gradient, mean_oracle)
                    receipt["four_stream_mean_gradient_oracle"] = True
                    native_value, native_moment = native_private(torch, {name: parameters[name] for name in private}, gradient, previous)
                    row = {"mean_four_stream_gradient_parity": mean_parity, "history_step": history_step+1, "private_commit": commit, "private_virtual_native_parameters":
                           mapping_compare(torch, {name: adapted[name] for name in private}, native_value),
                           "private_virtual_native_moments": moment_compare(torch, virtual_moment, native_moment)}
                    positive, negative_logits = steps.functional_forward(torch, model, adapted, x, support, outer, training=False)
                    value = steps.outer_loss(torch, positive, negative_logits)
                    live = torch.autograd.grad(value, tuple(parameters[name] for name in shared), retain_graph=True)
                    # Independent chain rule: adapted phi is an independent leaf;
                    # derivative of native Adam is the separate scalar formula.
                    point = {**parameters, **{name: adapted[name].detach().requires_grad_(True) for name in private}}
                    direct_positive, direct_negative = steps.functional_forward(torch, model, point, x, support, outer, training=False)
                    row["live_detached_outer_logits"] = [compare(torch, positive, direct_positive, rtol=0., atol=0.),
                                                         compare(torch, negative_logits, direct_negative, rtol=0., atol=0.)]
                    direct_value = steps.outer_loss(torch, direct_positive, direct_negative)
                    direct = torch.autograd.grad(direct_value, tuple(parameters[name] for name in shared)+tuple(point[name] for name in private))
                    jacobian = private_adam.gradient_jacobian(torch, gradient, previous)
                    cotangents = tuple(direct[len(shared)+i].detach()*jacobian[name] for i, name in enumerate(private))
                    mixed = torch.autograd.grad(tuple(gradient[name] for name in private), tuple(parameters[name] for name in shared),
                                                grad_outputs=cotangents, allow_unused=True)
                    oracle = {name: direct[i].detach()+(torch.zeros_like(parameters[name]) if mixed[i] is None else mixed[i].detach())
                              for i, name in enumerate(shared)}
                    row["full_shared_chain_rule"] = mapping_compare(torch, {name: live[i] for i, name in enumerate(shared)}, oracle, derivative=True)
                    row["mixed_path_disconnected_shared_names"] = [name for name, v in zip(shared, mixed) if v is None]
                    row["mixed_path_L2"] = float(torch.sqrt(sum((v.detach().double().square().sum() for v in mixed if v is not None))))
                    row["complete_shared_coordinates"] = sum(parameters[name].numel() for name in shared)
                    shared_gradient = {name: live[i].detach().clone() for i, name in enumerate(shared)}
                    # Build a separate explicit native shared commit at the same
                    # reachable optimizer history, then recompute private loss.
                    prior_shared = {name: {"step": int(optimizer.state[parameters[name]]["step"].item()),
                        "exp_avg": optimizer.state[parameters[name]]["exp_avg"].detach().clone(),
                        "exp_avg_sq": optimizer.state[parameters[name]]["exp_avg_sq"].detach().clone()}
                        if parameters[name] in optimizer.state else {"step": 0, "exp_avg": torch.zeros_like(parameters[name]),
                                                                    "exp_avg_sq": torch.zeros_like(parameters[name])} for name in shared}
                    expected_shared, expected_shared_moment = native_private(torch, {name: parameters[name] for name in shared}, shared_gradient, prior_shared)
                    # Release virtual graph before the complete recomputation.
                    del adapted, gradient, virtual_moment, positive, negative_logits, value, point, direct_positive, direct_negative, direct_value, live, direct, mixed, oracle, cotangents, jacobian
                    gc.collect()
                    recomputed_point = {**parameters, **{name: expected_shared[name].requires_grad_(True) for name in shared}}
                    expected_streams = steps.DropoutStreams(torch, device, job["seed"])
                    expected_streams.rows = [{"cpu": pristine_streams[str(i)]["cpu"].clone(),
                        "cuda": pristine_streams[str(i)]["cuda"].clone(),
                        "committed_calls": pristine_streams[str(i)]["committed_calls"]} for i in range(4)]
                    recomputed, _, _ = steps.inner_gradients(torch, model, recomputed_point, private, x, support, inner,
                        expected_streams, create_graph=False, advance=True)
                    expected_private, expected_moments = native_private(torch, {name: parameters[name] for name in private}, recomputed, previous)
                    committed_expected = expected_private
                    committed_moments = expected_moments
                    expected_point = {**recomputed_point, **committed_expected}
                    with torch.no_grad(): expected_serving = steps.functional_forward(torch, model, expected_point, x, support, outer, training=False)
                    if not tree_equal(torch, streams.state_dict(), pristine_streams) or not tree_equal(torch, previous, pristine_moments):
                        raise ValueError("Virtual work mutated private moments/dropout streams")
                    if any(not torch.equal(parameters[name], pristine_private[name]) for name in private):
                        raise ValueError("Virtual work changed committed private parameters")
                    new_previous, update = steps.episode_step(torch, model, shared, private, optimizer, previous,
                        x, support, inner, outer, streams, live=True, commit=commit)
                    row["committed_native_private_parameters"] = mapping_compare(torch, {name: parameters[name] for name in private}, committed_expected)
                    row["committed_native_private_moments"] = moment_compare(torch, new_previous, committed_moments)
                    row["committed_native_shared_parameters"] = mapping_compare(torch, {name: parameters[name] for name in shared}, expected_shared)
                    actual_shared_moment = {name: {"step": int(optimizer.state[parameters[name]]["step"].item()),
                        "exp_avg": optimizer.state[parameters[name]]["exp_avg"], "exp_avg_sq": optimizer.state[parameters[name]]["exp_avg_sq"]} for name in shared}
                    row["committed_native_shared_moments"] = moment_compare(torch, actual_shared_moment, expected_shared_moment)
                    with torch.no_grad():
                        actual_serving = steps.functional_forward(torch, model, dict(model.named_parameters()), x, support, outer, training=False)
                        # Original unchanged donor serving forward.
                        model.eval()
                        if hasattr(model, "routes"):
                            originals = []
                            for route in model.routes:
                                h = route.encoder(x, support)
                                originals.append((route.predictor(h, support, outer[0]), route.predictor(h, support, outer[1])))
                            original_serving = (torch.cat([p[0] for p in originals], 1), torch.cat([p[1] for p in originals], 1))
                        else:
                            h = model.encoder(x, support)
                            original_serving = (model.predictor(h, support, outer[0]), model.predictor(h, support, outer[1]))
                        for module, mode in zip(model.modules(), pristine_modes): module.training = mode
                    row["explicit_committed_serving_parity"] = [compare(torch, a, b, rtol=PARAM_RTOL, atol=PARAM_ATOL) for a, b in zip(actual_serving, expected_serving)]
                    row["original_donor_serving_parity"] = [compare(torch, a, b, rtol=0., atol=0.) for a, b in zip(actual_serving, original_serving)]
                    if not tree_equal(torch, streams.state_dict(), expected_streams.state_dict()): raise ValueError("Recomputed dropout stream advances differ")
                    expected_calls = [pristine_streams[str(i)]["committed_calls"]+1 for i in range(4)]
                    if [r["committed_calls"] for r in streams.rows] != expected_calls: raise ValueError("Private streams not advanced exactly once")
                    if not tree_equal(torch, previous, pristine_moments): raise ValueError("Previous episode Adam moments mutated")
                    if [module.training for module in model.modules()] != pristine_modes: raise ValueError("Native training modes not restored")
                    if not torch.equal(torch.get_rng_state(), cpu_rng) or not torch.equal(torch.cuda.get_rng_state(device), cuda_rng):
                        raise ValueError("Functional/committed work advanced global RNG")
                    row["state_RNG_mode_guards"] = "passed"; row["update_accounting"] = update
                    rows.append(row); previous = new_previous
                    stage(arm+"_reachable_history_"+str(history_step+1), {"passed": True,
                        "shared_coordinates": row["complete_shared_coordinates"], "mixed_path_L2": row["mixed_path_L2"]})
                torch.cuda.synchronize()
                receipt["architectures"][arm] = {"rows": rows, "inclusive_seconds": time.monotonic()-arm_started,
                    "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "passed": True}
                custody.write_json(output/"PARTIAL_RESULT.json", receipt)
                del model, parameters, optimizer, previous, streams, recomputed_point, expected_point, expected_streams
                gc.collect(); torch.cuda.empty_cache()
            receipt["ordinary_joint_rows"] = ordinary_joint_gate(torch, native, heads, models, steps, x, support, inner, outer, job, device)
            receipt["four_stream_ordinary_joint_parity"] = True
        receipt["native_alias_restored"] = True
        receipt.update(passed=True, stale_commit_qualification=False, inclusive_seconds=time.monotonic()-started,
                       peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        custody.write_json(output/"RESULT.json", receipt)
        print(json.dumps({"status": "passed_discarded_FP32_gate_no_fit", "output": str(output)}), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(passed=False, error=type(error).__name__+": "+str(error),
                       inclusive_seconds=time.monotonic()-started, retry=False)
        custody.write_json(output/"FAILURE.json", receipt)
        raise


if __name__ == "__main__": main()
