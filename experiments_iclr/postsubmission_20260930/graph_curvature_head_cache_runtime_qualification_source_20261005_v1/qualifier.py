"""Source-only bounded cache equivalence qualifier; explicit callable only.

No numerical imports, data/checkpoint loader, warm training, continuation,
remote command or score access on import/CLI. Root must admit runtime execution
and supply one custodied prescribed native warm endpoint with TRAIN inputs.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time


HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CELLS = {"Squirrel": "polyformer_mono", "Photo": "polynormer_r"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_bytes(record):
    relative = Path(record["path"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Phase-relative source descriptor required")
    path = PHASE / relative
    require(path.resolve() == path.absolute(), "Symlink source is not bound")
    data = path.read_bytes()
    require(len(data) == record["bytes"]
            and hashlib.sha256(data).hexdigest() == record["sha256"],
            "Pinned source bytes differ: " + str(path))
    return path, data


def verify_packet():
    seal = json.loads((HERE / "SEAL.json").read_text())
    manifest_path, manifest_bytes = verified_bytes(dict(seal["manifest"],
        path=HERE.name + "/" + seal["manifest"]["path"]))
    manifest = json.loads(manifest_bytes)
    for row in manifest["files"]:
        verified_bytes(dict(row, path=HERE.name + "/" + row["path"]))
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for row in bindings["files"]:
        verified_bytes(row)
    return bindings


def load_bound_module(name, record):
    path, data = verified_bytes(record)
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module, "__executed_sha256__", None) == record["sha256"]
                and getattr(module, "__executed_path__", None) == str(path),
                "Unverified cached module: " + name)
        return module
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
        module.__executed_sha256__, module.__executed_path__ = record["sha256"], str(path)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_runtime(bindings):
    """Runtime source loading only; no warm/qualification runner is invoked."""
    runner = load_bound_module("cache_equivalence_native_witness_runner_v3", bindings["runner"])
    driver, selector = runner.load_driver()
    sources = driver.load_sources()
    witness = load_bound_module("cache_equivalence_install_witness", bindings["witness"])
    require(witness.__executed_sha256__ == bindings["witness"]["sha256"],
            "Installation witness source changed")
    cache = load_bound_module("cache_equivalence_head_candidate_v2", bindings["cache"])
    constants = selector.FrozenConstants(**json.loads(
        verified_bytes(bindings["constants"])[1])["constants"])
    constants.validate()
    return runner, driver, selector, sources, witness, cache, constants


def endpoint_guard(native, metadata, graph_name, graph):
    """Same endpoint predicates as the pinned driver's native preparation."""
    backbone = CELLS[graph_name]
    spec = native.specification
    require(spec["backbone"] == backbone == graph.teacher_backbone
            and spec["seed"] == 17 and spec["config"] == 0
            and spec["family"] == "single_author" and native.members == 1,
            "Exact native single cfg0 seed17 endpoint required")
    require(metadata["fixed_last_warm"] is True and metadata["warm_stage_epoch"] == 50
            and metadata["actual_updates"] == (50 if graph_name == "Squirrel" else 250)
            and metadata["warm_stage"] == ("native" if graph_name == "Squirrel" else "global"),
            "Prescribed native warm endpoint required")
    if graph_name == "Photo":
        require(native.global_stage is True and metadata["local_model_and_adam_restored"] is True
                and metadata["native_local_rng_restored"] is False,
                "Photo local-best model/Adam, post-all200 RNG then50 global required")


def prepare_shared(native, optimizer, graph, edges, adapter, boundary, source, torch):
    """Mirror unchanged native preparation once, before the paired selectors."""
    backbone = native.specification["backbone"]
    donor = copy.deepcopy(native)
    native_frozen = adapter.named_optimizer_snapshot(native, optimizer)
    k1 = adapter.clone_boundary(donor, boundary, 1).eval()
    prototype = adapter.clone_boundary(donor, boundary, 4)
    identity = adapter.identity_logits_audit(donor, k1, prototype, graph)
    prototype.train(native.training)
    transported, transport = adapter.transport_optimizer(donor, native_frozen, prototype)
    frozen = adapter.named_optimizer_snapshot(prototype, transported)
    args = adapter.raw_arguments(graph, backbone)
    parameter = next(k1.parameters())
    S = source.symmetric_normalized_adjacency(graph.teacher_input.shape[0], edges,
                                             parameter.dtype, parameter.device)
    nodes = torch.arange(S.shape[0], dtype=torch.int64, device=parameter.device)
    return k1, prototype, transported, frozen, args, S, nodes, identity, transport


def run_selector_case(kind, fn, theta, shared, warm_rng, native, warm_snapshot,
        graph, train, constants, runner, driver, selector, adapter, source, witness, torch):
    """One unchanged selector call; witnesses observe the original installers."""
    _, prototype, _, frozen, args, S, nodes, _, _ = shared
    trial_function = selector.one_adam_trial
    recorder = witness.InstallWitnessRecorder(driver, selector, adapter)
    adapter.rng_restore(warm_rng)
    with recorder:
        outputs, receipt = driver.select_initializations(prototype, frozen, fn, theta, S, nodes,
            train.nodes, train.labels, native.specification["backbone"], args, warm_rng,
            17, constants, source, adapter)
    require(recorder.restored and selector.one_adam_trial is trial_function,
            "Selector instrumentation or original full-model trial binding was not restored")
    require(receipt["trial_maps_started"] <= 10 and receipt["trial_maps_completed"] <= 10,
            "Original ten-map bound exceeded")
    def exact_custody_state(left, right, torch, path="$"):
        # adapter.cpu_copy snapshots use dict; live state_dict uses OrderedDict.
        # Normalize only these named tensor-state containers, never their values
        # or the optimizer/RNG structure. The original exact predicate follows.
        def model_state(value):
            return (type(value) in (dict, OrderedDict) and bool(value)
                    and all(isinstance(key, str) and torch.is_tensor(item)
                            for key, item in value.items()))
        if model_state(left) and model_state(right):
            left, right = dict(left), dict(right)
        runner.exact_equal(left, right, torch, path=path)
    custody = witness.source_custody(outputs, receipt, native, warm_snapshot, selector,
        adapter, torch, exact_custody_state, runner.tensor_storage, recorder)
    custody["model_state_mapping_normalization"] = "dict/OrderedDict named tensor mappings only; original exact keys/dtype/shape/torch.equal"
    if not custody["passed"]:
        error = ValueError(kind + " actual installation/prototype custody failed")
        error.details = dict(case=kind, source_custody=custody, witness_summary=recorder.summary())
        raise error
    require(recorder.rows and recorder.rows[0]["serial"] == 1,
            "Actual common-trial installation witness is missing")
    actual_common = recorder.rows[0]["intended_slices"]
    runner.exact_equal(actual_common, actual_common[0].expand_as(actual_common), torch,
                       path=kind + ".actual_common_members")
    returned = {arm: dict(model=adapter.cpu_copy(out["model"].state_dict()),
        optimizer=adapter.named_optimizer_snapshot(out["model"], out["optimizer"]),
        rng=adapter.cpu_copy(out["rng"]),
        modes={name: module.training for name, module in out["model"].named_modules()})
        for arm, out in outputs.items()}
    installations = [dict(serial=row["serial"], backbone=row["backbone"], head=row["head"],
                          slices=row["intended_slices"]) for row in recorder.rows]
    runner.exact_equal(warm_rng, adapter.rng_snapshot(), torch, path=kind + ".rng_after_selector")
    # Return detached CPU comparisons, not usable model/optimizer continuations.
    return dict(receipt=receipt, custody=custody, actual_common=actual_common,
                returned=returned, installations=installations,
                witness_summary=recorder.summary())


def compare_cases(original, cached, selector, runner, torch):
    """Compare actual source results, never independently reconstructed heads."""
    fields = ("constants", "arms", "pairs", "construction_seeds", "common_center", "common_trial",
              "graph_bank", "spans", "D0_reference", "candidates", "selection",
              "trial_maps_started", "trial_maps_completed", "maximum_trial_maps",
              "candidate_slots_per_selected_arm", "failed_or_abstaining_arms_retained")
    checks = []
    for field in fields:
        require((field in original["receipt"]) == (field in cached["receipt"]),
                "Source receipt field presence differs: " + field)
        if field in original["receipt"]:
            runner.exact_equal(original["receipt"][field], cached["receipt"][field], torch,
                               path="paired_selector.receipt." + field)
        checks.append(field)
    runner.exact_equal(original["actual_common"], cached["actual_common"], torch,
                       path="paired_selector.actual_common")
    runner.exact_equal(original["installations"], cached["installations"], torch,
                       path="paired_selector.actual_installed_slices")
    for arm in selector.ARMS:
        runner.exact_equal(original["returned"][arm], cached["returned"][arm], torch,
                           path="paired_selector.actual_returned." + arm)
    # These are the original fixed-first reference and all original matched slots.
    gap = dict(reference=original["receipt"].get("D0_reference"),
        matched_slots=[dict(span=span, pair_index=row["pair_index"], pair=row["pair"],
            status=row["status"], radius=row.get("radius"), D=row.get("D"),
            trace=row.get("matching_trace"), evaluations=row.get("scalar_search_evaluations"))
            for span, rows in original["receipt"]["candidates"].items() for row in rows])
    return dict(passed=True, predicate="Original exact dtype/shape/torch.equal and exact scalar equality",
                compared_receipt_fields=checks, returned_arms=len(selector.ARMS),
                actual_fixed_pair_gap_and_radius_matching=gap,
                independent_fp32_head_reconstruction_performed=False)


def paired_forward_checks(original_fn, cached_fn, theta, case, runner, torch):
    points = [theta.detach().cpu()]
    for installation in case["installations"]:
        for row in installation["slices"]:
            if not any(torch.equal(row, existing) for existing in points):
                points.append(row.clone())
    require(len(points) <= 38, "Actual nine-pair head-slice bound exceeded")
    with torch.no_grad():
        for index, point in enumerate(points):
            value = point.to(device=theta.device, dtype=theta.dtype)
            left, right = original_fn(value), cached_fn(value)
            require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
                    "Nonfinite paired forward logits")
            runner.exact_equal(left, right, torch, path="paired_forward[" + str(index) + "]")
    return dict(passed=True, points=len(points), points_source="Identity and unique actual installed slices",
                source_forward_calls=2 * len(points), new_radius_grid_or_draws=False)


def paired_ad_checks(original_fn, cached_fn, theta, common, train, runner, adapter, torch):
    import torch.nn.functional as F
    rows = []
    # Use the inherited qualification's three seed17 directions/cotangents.
    for point_name, point in (("identity", theta), ("actual_common", common[0].to(theta))):
        generator = torch.Generator(device="cpu").manual_seed(17)
        left, left_pb = torch.func.vjp(original_fn, point)
        right, right_pb = torch.func.vjp(cached_fn, point)
        runner.exact_equal(left, right, torch, path="paired_ad." + point_name + ".vjp_primal")
        residual = torch.zeros_like(left)
        residual[train.nodes] = (left[train.nodes].softmax(-1).detach()
            - F.one_hot(train.labels, left.shape[1]).to(left.dtype)) / train.nodes.numel()
        left_g, right_g = left_pb(residual)[0], right_pb(residual)[0]
        runner.exact_equal(left_g, right_g, torch, path="paired_ad." + point_name + ".train_vjp")
        ordinary = []
        for fn in (original_fn, cached_fn):
            value = point.detach().clone().requires_grad_(True)
            loss = F.cross_entropy(fn(value)[train.nodes], train.labels)
            ordinary.append(torch.autograd.grad(loss, value)[0])
        runner.exact_equal(ordinary[0], ordinary[1], torch,
                           path="paired_ad." + point_name + ".ordinary_ce_gradient")
        for value in (left, right, left_g, right_g, *ordinary):
            require(bool(torch.isfinite(value).all()), "Nonfinite paired AD result")
        # This within-implementation check uses the existing integration tolerance.
        ordinary_vjp = [adapter.difference(gradient, oracle, adapter.TOLERANCES["gradient_atol"],
            adapter.TOLERANCES["gradient_rtol"]) for gradient, oracle in zip(ordinary, (left_g, right_g))]
        require(all(row["passed"] for row in ordinary_vjp), "Existing CE-gradient/VJP check failed")
        for index in range(3):
            direction = torch.randn(point.shape, generator=generator, dtype=point.dtype).to(point.device)
            direction = direction / direction.norm().clamp_min(1e-20) * (point.numel() ** 0.5)
            cotangent = torch.randn(left.shape, generator=generator, dtype=left.dtype).to(left.device)
            left_z, left_j = torch.func.jvp(original_fn, (point,), (direction,))
            right_z, right_j = torch.func.jvp(cached_fn, (point,), (direction,))
            runner.exact_equal(left_z, right_z, torch, path=f"paired_ad.{point_name}.jvp_primal[{index}]")
            runner.exact_equal(left_j, right_j, torch, path=f"paired_ad.{point_name}.jvp[{index}]")
            left_v, right_v = left_pb(cotangent)[0], right_pb(cotangent)[0]
            runner.exact_equal(left_v, right_v, torch, path=f"paired_ad.{point_name}.vjp[{index}]")
            require(all(bool(torch.isfinite(value).all()) for value in (left_j, right_j, left_v, right_v)),
                    "Nonfinite paired JVP/VJP")
        rows.append(dict(point=point_name, passed=True, paired_directions=3,
                         ordinary_gradient_vs_manual_vjp=ordinary_vjp))
        del left_pb, right_pb
    return dict(passed=True, points=rows, cross_closure_predicate="exact", seed=17,
                gradients_wrt="final_head_R_only")


def qualify_from_native_warm(native, optimizer, metadata, warm_rng, graph_name,
        graph, train, canonical_edges, admission):
    """One root-admitted representative graph17; returns JSON diagnostics only.

    The external supervisor must impose the existing per-graph 3600-second cap,
    inherited memory floor and no-retry policy. This function never acquires or
    restores a checkpoint and never calls warm training or continuation.
    """
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    require(admission.get("execute_cache_qualification") is True
            and admission.get("graph") == graph_name and admission.get("seed") == 17
            and admission.get("per_graph_child_cap_seconds") == 3600
            and admission.get("exact_actual_warm_preprocessing") is True,
            "Root admission for one graph17 under the existing 3600-second cap required")
    require(graph_name in CELLS, "Only Squirrel17/Photo17 representatives")
    bindings = verify_packet()
    runtime_policy = json.loads(verified_bytes(bindings["runtime_policy"])[1])
    require(runtime_policy["budget_policy"]["per_graph_child_cap_seconds"] == 3600,
            "Bound existing engineering cap changed")
    runner, driver, selector, sources, witness, cache, constants = load_runtime(bindings)
    import torch
    require(torch.__version__.split("+")[0] == "2.1.2", "Exactly the bound Torch 2.1.2 runtime")
    runner.verify_runtime_location()
    require(torch.cuda.device_count() == 1 and next(native.parameters()).device == torch.device("cuda:0")
            and torch.are_deterministic_algorithms_enabled()
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            and torch.get_float32_matmul_precision() == "highest", "Unchanged one-GPU native precision policy")
    endpoint_guard(native, metadata, graph_name, graph)
    adapter, boundary, source = (sources[key] for key in ("integration", "boundary", "initializer"))
    costs_rows = []
    costs = runner.Costs(torch, "cuda:0", costs_rows)
    report = dict(schema="graph-curvature-head-cache-runtime-equivalence-v1", graph=graph_name, seed=17,
        status="STARTED_ENGINEERING_ONLY", predictive_continuation=False, validation_labels_received=False,
        predictive_score_files_opened=False, original_native_qualification_not_established=True,
        qualification_promoted=False, checks={}, costs=costs_rows, supplied_warm_provenance_certified=False,
        actual_source_custody={}, installation_capture={}, selector_trials_per_case={})
    caller_rng = adapter.rng_snapshot()
    runner.exact_equal(warm_rng, caller_rng, torch, path="supplied_warm_rng_vs_current_rng")
    before = adapter.native_checkpoint(native, optimizer, metadata)
    input_fingerprints = dict(teacher_input=witness.tensor_fingerprint(graph.teacher_input),
        canonical_edges=witness.tensor_fingerprint(canonical_edges),
        teacher_edges=witness.tensor_fingerprint(graph.teacher_edge_index),
        train_nodes=witness.tensor_fingerprint(train.nodes), train_labels=witness.tensor_fingerprint(train.labels))
    report["same_supplied_input_fingerprints"] = input_fingerprints
    report["same_supplied_warm_logical_digests"] = {
        key: witness.state_fingerprint(before[key], torch)["sha256_logical_descriptor"]
        for key in ("model", "optimizer", "rng")}
    native_modes = {name: module.training for name, module in native.named_modules()}
    native_grads = adapter.cpu_copy({name: p.grad for name, p in native.named_parameters()})
    trial_function = selector.one_adam_trial
    driver_selection = driver.select_initializations
    try:
        shared = costs.call("shared_native_identity_transport_and_graph_construction", lambda:
            prepare_shared(native, optimizer, graph, canonical_edges, adapter, boundary, source, torch))
        k1, prototype, transported, frozen, args, _, _, identity, transport = shared
        proto_before = adapter.cpu_copy(prototype.state_dict())
        k1_before = adapter.cpu_copy(k1.state_dict())
        clone_modes = [{name: module.training for name, module in model.named_modules()}
                       for model in (k1, prototype)]
        clone_hooks = [{name: (len(module._forward_pre_hooks), len(module._forward_hooks),
                               len(module._backward_hooks)) for name, module in model.named_modules()}
                       for model in (k1, prototype)]
        clone_grads = [adapter.cpu_copy({name: p.grad for name, p in model.named_parameters()})
                       for model in (k1, prototype)]
        original_theta, original_fn, original_binding = costs.call("original_head_binding", lambda:
            selector.bind_head_only(k1, CELLS[graph_name], args))
        cached_theta, cached_fn, cached_binding = costs.call("cached_prehead_construction", lambda:
            cache.bind_cached_head_only(k1, CELLS[graph_name], args, selector, boundary))
        runner.exact_equal(original_theta, cached_theta, torch, path="same_k1_identity_head_slice")
        original = costs.call("original_selector_and_actual_install_custody", lambda:
            run_selector_case("original", original_fn, original_theta, shared, warm_rng, native,
                before, graph, train, constants, runner, driver, selector, adapter, source, witness, torch))
        report["actual_source_custody"]["original"] = original["custody"]
        report["installation_capture"]["original"] = original["witness_summary"]
        report["selector_trials_per_case"]["original"] = original["receipt"]["trial_maps_started"]
        cached = costs.call("cached_selector_and_actual_install_custody", lambda:
            run_selector_case("cached", cached_fn, original_theta, shared, warm_rng, native,
                before, graph, train, constants, runner, driver, selector, adapter, source, witness, torch))
        report["actual_source_custody"]["cached"] = cached["custody"]
        report["installation_capture"]["cached"] = cached["witness_summary"]
        report["selector_trials_per_case"]["cached"] = cached["receipt"]["trial_maps_started"]
        report["checks"]["paired_selector_gap_matching_decisions_and_returned_states"] = costs.call(
            "exact_actual_selector_comparison", lambda: compare_cases(original, cached, selector, runner, torch))
        report["checks"]["paired_full_logits"] = costs.call("paired_forward_points", lambda:
            paired_forward_checks(original_fn, cached_fn, original_theta, original, runner, torch))
        report["checks"]["paired_head_AD"] = costs.call("paired_JVP_VJP_and_CE_gradients", lambda:
            paired_ad_checks(original_fn, cached_fn, original_theta, original["actual_common"],
                             train, runner, adapter, torch))
        center = original["actual_common"][0].to(original_theta)
        for kind, fn in (("original", original_fn), ("cached", cached_fn)):
            interface = costs.call(kind + "_inherited_common_center_finite_differences", lambda fn=fn:
                source.qualify_gradient_interface(fn, center, train.nodes, train.labels, 17))
            report["checks"][kind + "_inherited_common_center_AD_interface"] = interface
            require(interface["passed"], "Inherited three-direction/two-epsilon interface failed: " + kind)
        runner.exact_equal(proto_before, adapter.cpu_copy(prototype.state_dict()), torch,
                           path="actual_prototype_after_checks")
        runner.exact_equal(frozen, adapter.named_optimizer_snapshot(prototype, transported), torch,
                           path="actual_transported_Adam_after_checks")
        runner.exact_equal(k1_before, adapter.cpu_copy(k1.state_dict()), torch, path="same_k1_after_checks")
        for index, model in enumerate((k1, prototype)):
            require(clone_modes[index] == {name: module.training for name, module in model.named_modules()},
                    "Original cloned module modes changed")
            require(clone_hooks[index] == {name: (len(module._forward_pre_hooks), len(module._forward_hooks),
                len(module._backward_hooks)) for name, module in model.named_modules()}, "Clone hooks changed")
            runner.exact_equal(clone_grads[index], {name: p.grad for name, p in model.named_parameters()},
                               torch, path="original_clone_gradients[" + str(index) + "]")
        report.update(status="PASSED_CACHE_EQUIVALENCE_FOR_SUPPLIED_ENDPOINT_ENGINEERING_ONLY",
            actual_source_custody={kind: result["custody"] for kind, result in (("original", original), ("cached", cached))},
            installation_capture={kind: result["witness_summary"] for kind, result in (("original", original), ("cached", cached))},
            selector_trials_per_case={kind: result["receipt"]["trial_maps_started"]
                                     for kind, result in (("original", original), ("cached", cached))},
            noncommon_coverage=any(row["pair"] is not None for row in original["receipt"]["selection"].values()),
            maximum_disposable_full_model_trials=20, additional_Adam_probes=0,
            native_identity_audit=identity, optimizer_transport=transport,
            head_bindings=dict(original=original_binding, cached=cached_binding),
            returned_models_optimizers_or_tensor_snapshots_saved=False,
            construction_cost_includes_cache_build=True, cost_phases_disjoint=True)
        reference = original["receipt"].get("D0_reference")
        matching_exercised = any(row.get("matching_trace") for rows in
            original["receipt"]["candidates"].values() for row in rows)
        report["gap_matching_coverage"] = dict(
            reference_measured=reference is not None, scalar_matching_exercised=bool(matching_exercised))
        if reference is None or not matching_exercised:
            report["status"] = "COMPARISONS_PASSED_NONCOMMON_MATCHING_COVERAGE_UNRESOLVED_ENGINEERING_ONLY"
    except Exception as error:
        report.update(status="FAILED_OR_UNRESOLVED_CACHE_EQUIVALENCE_ENGINEERING_ONLY",
                      error_type=type(error).__name__, error=str(error))
        if hasattr(error, "details"):
            report["exact_difference"] = error.details
    finally:
        final_checks = []
        witness.collect_check(final_checks, "caller_RNG_restore", lambda: adapter.rng_restore(caller_rng))
        witness.collect_check(final_checks, "original_selector_trial_bindings", lambda: require(
            selector.one_adam_trial is trial_function and driver.select_initializations is driver_selection,
            "Original selector/trial instrumentation binding changed"))
        witness.collect_check(final_checks, "native_model_Adam_RNG", lambda: runner.exact_equal(
            before, adapter.native_checkpoint(native, optimizer, metadata), torch,
            path="native_model_Adam_RNG_after_qualification"))
        witness.collect_check(final_checks, "native_gradients", lambda: runner.exact_equal(
            native_grads, {name: p.grad for name, p in native.named_parameters()}, torch,
            path="native_gradients_after_qualification"))
        witness.collect_check(final_checks, "native_modes", lambda: require(
            native_modes == {name: module.training for name, module in native.named_modules()},
            "Native module modes changed"))
        def same_inputs():
            after_inputs = dict(teacher_input=witness.tensor_fingerprint(graph.teacher_input),
                canonical_edges=witness.tensor_fingerprint(canonical_edges),
                teacher_edges=witness.tensor_fingerprint(graph.teacher_edge_index),
                train_nodes=witness.tensor_fingerprint(train.nodes), train_labels=witness.tensor_fingerprint(train.labels))
            require(input_fingerprints == after_inputs, "Supplied actual graph/TRAIN inputs changed")
        witness.collect_check(final_checks, "same_actual_graph_TRAIN_inputs", same_inputs)
        witness.collect_check(final_checks, "sealed_source_after_run", verify_packet)
        report["checks"]["final_custody"] = dict(passed=all(row["passed"] for row in final_checks),
                                                    checks=final_checks)
        if not report["checks"]["final_custody"]["passed"]:
            report["status"] = "FAILED_FINAL_CUSTODY_CACHE_EQUIVALENCE_ENGINEERING_ONLY"
    report["charged_phase_wall_seconds"] = sum(row["wall_seconds"] for row in costs_rows)
    report["total_harness_wall_seconds"] = time.perf_counter() - started_wall
    report["total_harness_process_cpu_seconds"] = time.process_time() - started_cpu
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    report["cost_scope"] = "Total includes setup, source verification, snapshots, construction and checks; phase sum is a disjoint subset; caller report IO/external supervisor extra"
    return report


if __name__ == "__main__":
    print(json.dumps(dict(status="SEALED_SOURCE_ONLY_RUNTIME_NOT_ADMITTED", numerical_execution=False,
        callable="qualifier.qualify_from_native_warm", predictive_continuation=False)))
