"""One complete-family test evaluation, paired timings and all fixed gates."""
import gc
from contextlib import ExitStack
import json
import math
import statistics
import time

from .contract import (ARMS, TASKS, HERE, PHASE, Budget, checked, completed, digest, hash_cache_report,
                       manifest, reference, within, write)
from .core import (audit_family_lock, build_model, configure, load_canonical,
                   load_execution_family, load_inputs, prescribed, prior_seconds,
                   save_tensor, split_forward)


def test_freeze(path, admitted, binding_ref, lock_ref):
    value = json.loads(within(path, HERE).read_text())
    if (value.get("schema") != "gnnm-propagation-root-test-target-freeze-v1"
            or value.get("method_binding") != binding_ref or value.get("family_lock") != lock_ref
            or value.get("root_qualified") is not True
            or set(value.get("test_exports", {})) != set(TASKS)):
        raise ValueError("Both completed post-lock test admissions must be root-frozen")
    for row in value["test_exports"].values():
        checked(row["bundle"])
        receipt = json.loads(checked(row["receipt"]).read_text())
        if receipt.get("family_lock") != lock_ref:
            raise ValueError("Test-target export does not bind this exact completed family lock")
        chain = completed(PHASE / row["supervisor_directory"])
        if chain["gpu_uuid"] != admitted["gpu_uuid"]:
            raise ValueError("Test admission used a different allocation")
    return value, reference(path)


def percentile90(values):
    # Fixed nearest-rank p90: rank ceil(.9*n), no interpolation convention.
    if len(values) != 50:
        raise ValueError("Exactly fifty timed forwards required")
    return sorted(values)[math.ceil(.9 * len(values)) - 1]


def summarize(samples):
    wall = [r["synchronized_wall_seconds"] for r in samples]
    cuda = [r["CUDA_event_seconds"] for r in samples]
    return {"median_seconds": statistics.median(wall), "p90_seconds": percentile90(wall),
            "CUDA_event_median_seconds": statistics.median(cuda),
            "CUDA_event_p90_seconds": percentile90(cuda), "raw_samples": samples,
            "p90_definition": "nearest_rank_ceil_0.9n", "primary": "synchronized_wall"}


def raw_sample(model, family, features, execution, budget):
    """Unchanged sealed inference helper; instrumentation absent here."""
    import torch
    from propagation_cost_engineering_v1.complete_models import pooled_inference
    budget.check()
    family.ledger.clear()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(0)
    begin = time.perf_counter()
    start_event.record()
    pooled = pooled_inference(model, family, features, execution)
    end_event.record()
    torch.cuda.synchronize(0)
    wall = time.perf_counter() - begin
    cuda = start_event.elapsed_time(end_event) / 1000
    del pooled
    return {"synchronized_wall_seconds": wall, "CUDA_event_seconds": cuda,
            "operation_summary": family.ledger.summary(),
            "CSR_ledger_covers_native_gather_scatter": False,
            "execution_is_native_source": getattr(model, "native_source", False)}


class PropagationTimer:
    """Separate-run aggregation scope; never installed for raw inference.

    Scope begins with normalized states and ends with unparameterized neighbor
    aggregates. Candidate includes mean/centering/combine; native hooks include
    gather and mean aggregation, before SAGE's learned root/neighbor maps.
    CUDA event sums measure GPU scope, with no synchronization inside layers.
    """
    def __init__(self, model, family):
        self.model, self.family = model, family
        self.events, self.pending, self.handles, self.restore = [], [], [], []
        self.kind = None

    def begin(self):
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.pending.append(event)

    def end(self):
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append((self.pending.pop(), event))

    def __enter__(self):
        model, family = self.model, self.family
        if getattr(model, "native_source", False):
            from torch_geometric.nn import MessagePassing
            modules = [m for m in model.source.modules() if isinstance(m, MessagePassing)]
            if len(modules) != 2:
                raise ValueError("Native aggregation hook scope is not the two frozen SAGE layers")
            for module in modules:
                if not hasattr(module, "register_propagate_forward_pre_hook"):
                    raise ValueError("Frozen PyG runtime cannot qualify native propagation timer")
                self.handles.append(module.register_propagate_forward_pre_hook(lambda _m, _i: self.begin()))
                self.handles.append(module.register_propagate_forward_hook(lambda _m, _i, _o: self.end()))
            self.kind = "native_PyG_propagate_gather_mean_before_linear_maps"
        elif model.arm in ARMS[:2] or (model.arm != "single_base" and self.execution == "packed"):
            original = model._propagate
            def timed(*args, **kwargs):
                self.begin()
                result = original(*args, **kwargs)
                self.end()
                return result
            model._propagate = timed
            self.restore.append(lambda: setattr(model, "_propagate", original))
            self.kind = "complete_CSR_preaggregation_mean_center_and_neighbor_aggregate"
        elif model.arm == "naive_shared_sample":
            original = family.naive
            def timed(*args, **kwargs):
                self.begin()
                result = original(*args, **kwargs)
                self.end()
                return result
            family.naive = timed
            self.restore.append(lambda: setattr(family, "naive", original))
            self.kind = "serial_CSR_naive_neighbor_aggregate"
        else:
            original = family.operators["full"]
            timer = self
            class TimedOperator:
                def __getattr__(self, name):
                    return getattr(original, name)
                def __call__(self, features):
                    timer.begin()
                    result = original(features)
                    timer.end()
                    return result
            family.operators["full"] = TimedOperator()
            self.restore.append(lambda: family.operators.__setitem__("full", original))
            self.kind = "serial_CSR_full_neighbor_aggregate"
        return self

    def __exit__(self, _kind, _error, _trace):
        for handle in self.handles:
            handle.remove()
        for undo in self.restore:
            undo()

    def sample(self, model, family, features, execution, budget):
        self.events, self.pending = [], []
        row = raw_sample(model, family, features, execution, budget)
        expected = 2 * (4 if model.arm in ("exact_member", "naive_shared_sample")
                        and (getattr(model, "native_source", False) or execution == "serial") else 1)
        if self.pending or len(self.events) != expected:
            raise ValueError("Propagation scope lacks all declared layer/member aggregates")
        row["propagation_CUDA_event_seconds"] = sum(a.elapsed_time(b) / 1000 for a, b in self.events)
        row["aggregation_calls"] = len(self.events)
        return row


def instantiate(admitted, choice, export, metadata, directory):
    import torch
    cell, variant = choice["cell"], choice["deployment"]
    begin = time.perf_counter()
    raw_edges = export["edge_index"].to("cuda:0") if variant["native_source"] else None
    family = load_execution_family(metadata, directory, cell["arm"], variant)
    model = build_model(cell["task"], cell["arm"], variant, cell["seed"], raw_edges).eval()
    checkpoint = torch.load(checked(choice["selected_checkpoint"]), map_location="cpu", weights_only=True)
    load_canonical(model, checkpoint["canonical_state"], cell["task"], cell["arm"], cell["seed"])
    del checkpoint
    torch.cuda.synchronize(0)
    return model, family, {"operator_model_checkpoint_H2D_setup_seconds": time.perf_counter() - begin,
        "graph_GPU_payload": family.storage_report(),
        "native_raw_edge_GPU_payload_bytes": 0 if raw_edges is None else raw_edges.numel() * raw_edges.element_size()}


def final_test(admitted, binding_ref, lock_path, target_freeze_path, output, context):
    import numpy as np
    import torch
    from propagation_cost_engineering_v1.contract import TASKS as SPECS, check_peak
    from .labels import load_label_bundle
    from .metrics import full_split, gates
    output = within(output, HERE)
    if output != prescribed(admitted, "final_test") or output.exists():
        raise ValueError("Exactly one new complete-family final evaluation is allowed")
    lock = audit_family_lock(lock_path, admitted, binding_ref)
    lock_ref = reference(lock_path)
    target_freeze, target_ref = test_freeze(target_freeze_path, admitted, binding_ref, lock_ref)
    budget = Budget(admitted, prior_seconds(admitted), context)
    configure(admitted)
    output.mkdir(parents=True, exist_ok=False)
    (output / "test_logits").mkdir()
    write(output / "START.json", {"schema": "gnnm-propagation-complete30-test-start-v1",
          "binding": binding_ref, "family_lock": lock_ref, "test_target_freeze": target_ref,
          "live": context, "supervisor_directory": context["supervisor_relative_to_project"],
          "complete_family_test_metric_evaluation_number": 1})
    results, timings, fit_costs, preparations = [], {}, {}, {}
    try:
        for task in TASKS:
            budget.check()
            input_begin = time.perf_counter()
            export, lineage, metadata, directory = load_inputs(admitted, task)
            targets = load_label_bundle(target_freeze["test_exports"][task], "test", task, export,
                {"binding_sha256": binding_ref["sha256"], "gpu_uuid": admitted["gpu_uuid"]})
            torch.cuda.synchronize(0)
            copy_begin = time.perf_counter()
            features = export["x"].to("cuda:0")
            test_ids = export["official_split_node_ids"]["test"].to("cuda:0")
            torch.cuda.synchronize(0)
            H2D = time.perf_counter() - copy_begin
            input_seconds = time.perf_counter() - input_begin
            timings[task], fit_costs[task] = {}, {}
            # Root freezes each arm's attributable once-only preparation. The
            # exact comparator must not absorb candidate color-building costs.
            preparations[task] = admitted["tasks"][task]["preparation_charged_seconds"]
            for seed in range(3):
                choices = [r for r in lock["locked_choices"] if r["cell"]["task"] == task and r["cell"]["seed"] == seed]
                indexed = {r["cell"]["arm"]: r for r in choices}
                timings[task][str(seed)], fit_costs[task][str(seed)] = {}, {}
                # Individual memory/cold/setup measurements and ONCE test metrics.
                for arm in ARMS:
                    choice, variant = indexed[arm], indexed[arm]["deployment"]
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(0)
                    model, family, setup = instantiate(admitted, choice, export, metadata, directory)
                    cold = raw_sample(model, family, features, variant["execution"], budget)
                    begin = time.perf_counter()
                    member, pooled = split_forward(model, family, features, test_ids, variant["execution"])
                    torch.cuda.synchronize(0)
                    forward_copy = time.perf_counter() - begin
                    begin = time.perf_counter()
                    measured = full_split(member, targets, choice["validation_chosen_member"], pooled_logits=pooled)
                    name = ("reddit2" if task == "Reddit2" else "products") + "_seed%d_%s.pt" % (seed, arm)
                    evidence = save_tensor(output / "test_logits" / name,
                        {"schema": "gnnm-propagation-locked-test-member-logits-v1", "cell": choice["cell"],
                         "node_ids": export["official_split_node_ids"]["test"], "member_logits": member,
                         "pooled_logits": pooled, "family_lock": lock_ref,
                         "selected_checkpoint": choice["selected_checkpoint"], "deployment": variant}, budget)
                    metric_save = time.perf_counter() - begin
                    validation = dict(choice["selected_validation"])
                    m = choice["validation_chosen_member"]
                    validation.update(validation_chosen_member=m,
                        chosen_member_CE_minus_pooled_CE=validation["member_CE"][m] - validation["CE"])
                    results.append({**choice["cell"], "validation": validation, "test": measured,
                        "test_logits": evidence, "selected_epoch": choice["selected_epoch"],
                        "checkpoint": choice["selected_checkpoint"], "deployment": variant,
                        "complete_test_forward_slice_copy_seconds": forward_copy,
                        "CPU_metric_and_test_save_seconds": metric_save, "individual_memory": check_peak(),
                        "cold_first_pooled_inference": cold, "setup": setup, "input_H2D_seconds": H2D,
                        "complete_input_replay_and_target_bundle_load_seconds": input_seconds,
                        "complete_data_binding": lineage})
                    fit_receipt = completed(PHASE / choice["fit_supervisor_directory"])
                    fit_costs[task][str(seed)][arm] = fit_receipt["seconds"]
                    del member, pooled, model, family
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(0)
                resident, resident_setup = {}, {}
                for arm in ARMS:
                    model, family, setup = instantiate(admitted, indexed[arm], export, metadata, directory)
                    resident[arm] = (model, family)
                    resident_setup[arm] = setup
                    check_peak()
                # Outcome-independent paired shuffled order across all five arms.
                rng_seed = 20260930 + 50000 + 1000 * SPECS[task]["index"] + seed
                rng = np.random.Generator(np.random.PCG64(rng_seed))
                orders = [[ARMS[i] for i in rng.permutation(5)] for _ in range(60)]
                samples = {arm: [] for arm in ARMS}
                warmups = {arm: [] for arm in ARMS}
                for round_index, order in enumerate(orders):
                    for arm in order:
                        model, family = resident[arm]
                        sample = raw_sample(model, family, features, indexed[arm]["deployment"]["execution"], budget)
                        sample.update(round=round_index, arm_order=order)
                        (warmups if round_index < 10 else samples)[arm].append(sample)
                        check_peak()
                raw_peak = check_peak()
                # Component instrumentation runs after all raw measurements.
                component_seed = 20260930 + 60000 + 1000 * SPECS[task]["index"] + seed
                component_rng = np.random.Generator(np.random.PCG64(component_seed))
                component_orders = [[ARMS[i] for i in component_rng.permutation(5)] for _ in range(60)]
                component_samples = {arm: [] for arm in ARMS}
                component_warmups = {arm: [] for arm in ARMS}
                component_reports, timers = {}, {}
                with ExitStack() as stack:
                    for arm in ARMS:
                        model, family = resident[arm]
                        timer = PropagationTimer(model, family)
                        timer.execution = indexed[arm]["deployment"]["execution"]
                        timers[arm] = stack.enter_context(timer)
                    for round_index, order in enumerate(component_orders):
                        for arm in order:
                            model, family = resident[arm]
                            execution = indexed[arm]["deployment"]["execution"]
                            sample = timers[arm].sample(model, family, features, execution, budget)
                            sample.update(round=round_index, arm_order=order)
                            (component_warmups if round_index < 10 else component_samples)[arm].append(sample)
                            check_peak()
                    for arm in ARMS:
                        component = component_samples[arm]
                        component_reports[arm] = {"scope": timers[arm].kind, "warmup_samples": component_warmups[arm],
                            "raw_samples": component, "median_seconds": statistics.median(
                                r["propagation_CUDA_event_seconds"] for r in component),
                            "instrumented_full_forward_summary": summarize(component),
                            "paired_order_seed": component_seed, "paired_orders": component_orders,
                            "raw_latency_measurements_contaminated": False}
                del timer, timers
                # One separately charged source trace; no target values enter it.
                model, family = resident["candidate"]
                budget.check()
                torch.cuda.synchronize(0)
                begin = time.perf_counter()
                with torch.no_grad():
                    logits, trace = model(family, features, colors=(0, 2), execution="packed", return_trace=True)
                torch.cuda.synchronize(0)
                diagnostic = {"layers": [{k: (float(v.cpu()) if isinstance(v, torch.Tensor) else v)
                                          for k, v in row.items()} for row in trace],
                              "wall_seconds": time.perf_counter() - begin,
                              "scope": "complete node post-LayerNorm states at selected candidate checkpoint",
                              "new_predictive_view_or_target_metric": False}
                del logits, trace
                for arm in ARMS:
                    summary = summarize(samples[arm])
                    summary.update(warmup_samples=warmups[arm], paired_order_seed=rng_seed,
                        paired_orders=orders, deployment=indexed[arm]["deployment"],
                        resident_setup=resident_setup[arm], resident_family_memory=raw_peak,
                        propagation_median_seconds=component_reports[arm]["median_seconds"],
                        propagation=component_reports[arm],
                        candidate_contrast_diagnostic=diagnostic if arm == "candidate" else None)
                    timings[task][str(seed)][arm] = summary
                del model, family, resident
                gc.collect()
                torch.cuda.empty_cache()
            del features, test_ids, targets, export
        budget.check()
        decision = gates(results, timings, fit_costs, preparations)
        write(output / "FINAL_TEST_END.json", {"schema": "gnnm-propagation-complete30-final-evaluation-v1",
            "status": "completed_all_outcomes", "binding": binding_ref, "family_lock": lock_ref,
            "test_target_freeze": target_ref, "fit_count": 30, "results": results,
            "timings": timings, "actual_supervised_fit_costs_seconds": fit_costs,
            "once_per_graph_preparation_seconds": preparations, "gates": decision,
            "failure_or_ambiguity_rejects_promotion": True, "selective_reporting_authorized": False,
            "complete_family_test_metric_evaluation_count": 1, "live": context,
            "own_allocation_wall_seconds_before_final_manifest": time.monotonic() - budget.start,
            "charged_prior_GPU_seconds": budget.prior,
            "reference_hash_cache": hash_cache_report(),
            "qualification": "requires later completed final-evaluator supervisor chain"})
        manifest(output)
    except BaseException as error:
        write(output / "FAILURE.json", {"schema": "gnnm-propagation-complete30-final-failure-v1",
            "type": type(error).__name__, "message": str(error), "partial_outputs_preserved": True,
            "rerun_resume_or_partial_promotion_authorized": False, "live": context})
        raise
