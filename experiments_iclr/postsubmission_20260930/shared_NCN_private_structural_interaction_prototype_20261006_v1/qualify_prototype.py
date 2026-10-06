#!/usr/bin/env python3
"""Discarded source-bound TRAIN fixture qualification; VALID/TEST values closed.

Compare the actual functional ordinary driver with an independent direct model
forward plus native Adam reference, including parameters and optimizer moments.
No receipt passes by source inspection alone. All six controls run on the exact
assigned physical GPU; actual complete-cycle costs remain a separate gate.
"""
import copy
import json
import os
import time
import custody
from contracts import ARMS, QUALIFY_PURPOSE, validate_recipe


def adam(torch, model):
    return torch.optim.Adam(list(model.parameters()), lr=.001, betas=(.9, .999),
                            eps=1e-8, weight_decay=0., foreach=False, fused=False)


def own(torch, positive, negative):
    # Independent explicit BCE expression; no transfer_step loss/forward call.
    return -(torch.nn.functional.logsigmoid(positive).mean()+
             torch.nn.functional.logsigmoid(-negative).mean())


def direct_reference(torch, model, optimizer, x, support, inner, outer, streams):
    calls = [(0, (torch.cat([row[0] for row in inner], 1), torch.cat([row[1] for row in inner], 1)))] if model.member_count == 1 else list(enumerate(inner))
    losses, gradient_maxima = [], []
    for pass_index in range(3):
        optimizer.zero_grad(set_to_none=True)
        if pass_index == 1:
            model.eval(); positive, negative = model(x, support, *outer)
            value = .5*own(torch, positive.mean(1), negative.mean(1))+.5*own(torch, positive, negative)
            if hasattr(model, "routes"): value = value*model.member_count
        else:
            model.train(); own_losses = []
            for member, queries in calls:
                with streams.use(member, advance=pass_index == 2):
                    positive, negative = model(x, support, *queries, route=member)
                own_losses.append(own(torch, positive, negative))
            value = torch.stack(own_losses).sum() if hasattr(model, "routes") else torch.stack(own_losses).mean()
        if not bool(torch.isfinite(value)): raise FloatingPointError("Nonfinite direct reference BCE")
        value.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Direct native reference has disconnected/nonfinite gradients")
        gradient_maxima.append(max(float(p.grad.detach().abs().max()) for p in model.parameters()))
        optimizer.step(); losses.append(float(value.detach()))
    return {"task_losses": losses, "gradient_maxima": gradient_maxima}


def max_tensor_delta(torch, first, second):
    if first.shape != second.shape or first.dtype != second.dtype or not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
        raise ValueError("Tensor parity geometry/finiteness differs")
    return float((first-second).abs().max()) if first.numel() else 0.


def state_delta(torch, first_model, second_model, first_optimizer, second_optimizer):
    left, right = dict(first_model.named_parameters()), dict(second_model.named_parameters())
    if set(left) != set(right): raise ValueError("Direct/functional parameter names differ")
    parameter_max = max(max_tensor_delta(torch, left[name], right[name]) for name in left)
    moment_max, steps_equal = 0., True
    for name in left:
        a, b = first_optimizer.state[left[name]], second_optimizer.state[right[name]]
        if set(a) != {"step", "exp_avg", "exp_avg_sq"} or set(a) != set(b):
            raise ValueError("Native Adam state coverage differs")
        for key in ("exp_avg", "exp_avg_sq"):
            moment_max = max(moment_max, max_tensor_delta(torch, a[key], b[key]))
        steps_equal = steps_equal and float(a["step"]) == float(b["step"]) == 3.
    return {"parameter_max_abs": parameter_max, "moment_max_abs": moment_max, "all_steps_exactly3": steps_equal}


def synthetic_branch_gate(torch, model, topology, device):
    """Small fixed-H check: startup zeros, learned-message gradients after a moves,
    and connected zero gradients on the genuine empty-witness path."""
    neighbors = [{2, 3, 4, 5}, {2, 3, 4, 5}, {0, 1, 3}, {0, 1, 2}, {0, 1, 5}, {0, 1, 4}]
    results = []
    for branch in model.branches:
        # The enclosing arm's kind is reflected in its public partition by caller.
        kind = "count" if model._qualification_arm == "count_only" else "blind" if model._qualification_arm == "structure_blind" else "structural"
        info = topology.enumerate_queries(neighbors, [(0, 1)], kind)
        h = torch.linspace(-.5, .5, 6*256, device=device).reshape(6, 256)
        members = list(range(branch.r.shape[0]))
        work = {}
        def charge(key, value): work[key] = work.get(key, 0)+value
        branch.zero_grad(set_to_none=True)
        out = branch(h, info, members, charge)
        if not bool((out == 0).all()): raise ValueError("Residual does not include base function at initialization")
        out.sum().backward()
        psi_zero = all(p.grad is not None and bool((p.grad == 0).all()) for p in branch.psi.parameters())
        gate_zero = branch.r.grad is not None and bool((branch.r.grad == 0).all())
        readout_nonzero = branch.a.grad is not None and bool((branch.a.grad != 0).any())
        if not (psi_zero and gate_zero and readout_nonzero): raise ValueError("Startup Psi/gate/readout gradient certificate failed")
        with torch.no_grad(): branch.a.fill_(.03125)
        branch.zero_grad(set_to_none=True)
        branch(h, info, members, charge).sum().backward()
        psi_active = any(p.grad is not None and bool((p.grad != 0).any()) for p in branch.psi.parameters())
        gate_active = branch.r.grad is not None and bool((branch.r.grad != 0).any())
        if not (psi_active and gate_active): raise ValueError("Learned Psi/gate failed to receive gradients after readout moved")
        with torch.no_grad(): branch.a.zero_()
        branch.zero_grad(set_to_none=True)
        empty = topology.enumerate_queries([set(), set()], [(0, 1)], kind)
        branch(h, empty, members, charge).sum().backward()
        connected = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in branch.parameters())
        if not connected: raise ValueError("Empty-CN branch disconnects declared parameters")
        if kind != "count" and not all(bool((p.grad == 0).all()) for p in branch.parameters()):
            raise ValueError("Empty witness anchor must have exact zero derivatives")
        branch.zero_grad(set_to_none=True)
        results.append({"startup_output_exact_zero": True, "startup_Psi_gate_gradients_exact_zero": True,
                        "startup_readout_gradient_nonzero": True, "moved_readout_Psi_gate_gradients_nonzero": True,
                        "empty_CN_gradients_connected": True, "empty_witness_anchor_exact_zero": kind != "count",
                        "synthetic_branch_work": work})
    del model._qualification_arm
    return results


def main():
    started = time.monotonic()
    args = custody.parser(__doc__).parse_args()
    job, output = custody.authorize(__file__, args, QUALIFY_PURPOSE)
    validate_recipe(job, qualification=True)
    output.mkdir()
    custody.write_json(output/"START.json", {"purpose": QUALIFY_PURPOSE, "PID": os.getpid(), "TEST_access": False, "VALID_values_access": False})
    try:
        torch, device, versions = custody.runtime(job)
        native, heads, geometry, cycle, adjoint, models, steps = custody.load_sources()
        import topology
        x, train, _, _, identities = custody.load_inputs(torch, job, include_valid=False); x = x.to(device)
        es = geometry.route_streams(job["seed"], 4); rs = geometry.route_streams(job["seed"], 4, control=True)
        negative, _, episodes = custody.make_pair(torch, geometry, cycle, train, len(x), job["seed"], 0,
            job["outer_size"], job["inner_size"], es, rs)
        endpoint, _, kept, _ = episodes[0]
        support = custody.support_tensor(torch, train, kept, len(x), device)
        inner, outer = custody.queries(torch, train, negative, endpoint, device)
        results = {}; torch.cuda.reset_peak_memory_stats()
        with custody.native_adjoint(torch, native, adjoint.make_constant_adjacency_spmm):
            for arm in ARMS:
                if time.monotonic()-started > job["soft_seconds"]: raise TimeoutError("Qualification root-frozen soft bound exceeded")
                model, shared, private, partition = models.build(native, heads, arm=arm, seed=job["seed"], factor_seed=job["factor_seed"],
                    device=device, input_dim=x.shape[1], structural_seed=job["structural_seed"],
                    message_width=job["message_width"], message_hidden=job["message_hidden"], chunk_size=job["chunk_size"])
                model._qualification_arm = arm
                branch_certificate = synthetic_branch_gate(torch, model, topology, device)
                model.eval(); model.base.eval()
                with torch.no_grad():
                    base_logits = model.base(x, support, *outer)
                    new_logits = model(x, support, *outer)
                initial_delta = max(max_tensor_delta(torch, a, b) for a, b in zip(base_logits, new_logits))
                if initial_delta != 0.: raise ValueError("Zero residual failed exact live base-function check")
                initial_work = model.drain_work()
                reference = copy.deepcopy(model)
                opt, ref_opt = adam(torch, model), adam(torch, reference)
                streams, ref_streams = steps.DropoutStreams(torch, device, job["seed"]), steps.DropoutStreams(torch, device, job["seed"])
                global_cpu, global_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
                direct = direct_reference(torch, reference, ref_opt, x, support, inner, outer, ref_streams)
                actual = steps.ordinary_episode_step(torch, model, opt, x, support, inner, outer, streams)
                work, reference_work = model.drain_work(), reference.drain_work()
                differences = state_delta(torch, model, reference, opt, ref_opt)
                differences["loss_max_abs"] = max(abs(a-b) for a, b in zip(actual["task_losses"], direct["task_losses"]))
                dropout_equal = all(torch.equal(a["cpu"], b["cpu"]) and torch.equal(a["cuda"], b["cuda"]) and a["committed_calls"] == b["committed_calls"] == (1 if model.member_count == 4 or i == 0 else 0)
                                    for i, (a, b) in enumerate(zip(streams.rows, ref_streams.rows)))
                global_restored = torch.equal(global_cpu, torch.get_rng_state()) and torch.equal(global_cuda, torch.cuda.get_rng_state(device))
                model.eval(); reference.eval()
                with torch.no_grad():
                    p, n = model(x, support, *outer); rp, rn = reference(x, support, *outer)
                differences["committed_logit_max_abs"] = max(max_tensor_delta(torch, p, rp), max_tensor_delta(torch, n, rn))
                if (differences["parameter_max_abs"] > 5e-6 or differences["moment_max_abs"] > 5e-6 or differences["loss_max_abs"] > 5e-6 or
                    differences["committed_logit_max_abs"] > 2e-5 or not differences["all_steps_exactly3"] or not dropout_equal or not global_restored):
                    raise ValueError("New structural direct/native Adam parity gate failed: "+json.dumps(differences))
                if work.get("encoder_forward_count") != actual["model_encoder_forward_count"] or work.get("wrapper_forward_calls") != actual["functional_forward_calls"]:
                    raise ValueError("Measured structural forward work differs from driver accounting")
                results[arm] = {"passed": True, "differences": differences, "thresholds": {"parameters_moments_losses": 5e-6, "committed_logits": 2e-5},
                    "startup_branches": branch_certificate, "initial_base_logit_max_abs": initial_delta,
                    "base_initialization_check_encoder_calls": 4 if hasattr(model, "routes") else 1,
                    "global_RNG_restored": global_restored, "dropout_streams_parity": dropout_equal,
                    "actual_step": actual, "initial_work": initial_work, "actual_step_work": work, "direct_step_work": reference_work,
                    "poststep_check_work": model.drain_work(), "direct_poststep_check_work": reference.drain_work(),
                    "parameter_counts": {"shared_role": sum(dict(model.named_parameters())[name].numel() for name in shared),
                                         "private_role": sum(dict(model.named_parameters())[name].numel() for name in private)}}
                custody.write_json(output/(arm+".json"), results[arm])
                del model, reference, opt, ref_opt, base_logits, new_logits, p, n, rp, rn
        torch.cuda.synchronize()
        receipt = {"scope": "discarded_FP32_structural_training_step_qualification", "passed": True,
            "source_manifest_sha256": job["source_manifest_sha256"], "fixed_recipe_sha256": job["fixed_recipe_sha256"],
            "available_manifest_sha256": job["available_manifest_sha256"], "target": job["target"], "physical_gpu_uuid": job["physical_gpu_uuid"],
            "fixture": {"outer_size": job["outer_size"], "inner_size": job["inner_size"], "cycle": 1, "episode": 1},
            "architectures": results, "runtime": versions, "input_identities": identities,
            "TEST_access": False, "VALID_values_access": False, "states_discarded": True,
            "inclusive_seconds": time.monotonic()-started, "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "job_sha256": custody.sha(args.job)}
        custody.write_json(output/"QUALIFICATION.json", receipt)
        print(json.dumps({"status": "discarded_structural_first_fixture_passed", "output": str(output)}), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        custody.write_json(output/"FAILURE.json", {"passed": False, "error": type(error).__name__+": "+str(error),
            "inclusive_seconds": time.monotonic()-started, "TEST_access": False, "VALID_values_access": False, "retry": False, "partial_outputs_preserved": True})
        raise


if __name__ == "__main__": main()
