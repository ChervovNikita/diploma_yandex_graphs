"""Two matched four-member complete native TRAIN resource epochs."""
import hashlib
import os
import time
import torch
from guards import require
from gpu_parity import initial_engineering_state, state_digest, train_objective
from runtime import capture_rng, restore_rng, rng_digest, instrument_portable, finite_gradients_and_state, state_bytes, Meter
from train_only_data import epoch_stream


EXPECTED_BATCH_WORK = {
    "encoder_calls": 1, "outer_enumeration_calls": 2, "recursive_enumeration_calls": 16,
    "outer_full_node_xlin_calls": 8, "recursive_full_node_xlin_calls": 16,
    "outer_full_node_xlin_linear_map_calls": 16, "recursive_full_node_xlin_linear_map_calls": 32,
    "recursive_score_calls": 16, "completion_clamp_calls": 16,
    "outer_aggregation_calls": 24, "recursive_aggregation_calls": 16,
    "outer_nonlinear_decode_calls": 8, "recursive_nonlinear_decode_calls": 16}


def run(context, prototype, graph_ops, reference, selected_state, data, device, sampler, setup_meter, progress):
    initial, epoch_rng, initial_sha = initial_engineering_state(prototype, device, setup_meter)
    parity = context["gpu_parity"]
    require(initial_sha == parity["initial_engineering_state_sha256"], "Resource/GPU parity engineering initialization differs")
    require(rng_digest(epoch_rng) == parity["epoch_start_rng_sha256"], "Resource/GPU parity epoch start RNG differs")
    progress["initial_engineering_state_sha256"] = initial_sha
    progress["epoch_start_rng_sha256"] = rng_digest(epoch_rng)
    _, native_utils = reference.native_modules()
    results = progress.setdefault("twins", [])
    stream_reference = None
    progress["setup_accounting_before_twin_peak_resets"] = setup_meter.receipt()
    for mode in ("private", "pooled_after_clamp"):
        progress["current_twin"] = mode
        # Select device before counter reset. Previous twin objects are released.
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        meter = Meter()
        result = {"mode": mode, "status": "IN_PROGRESS", "batches": []}
        results.append(result)
        try:
            result["current_phase"] = "matched_model_Adam_restoration"
            model = meter.call("matched_four_member_model_restoration", lambda: prototype.CompletionTwin(prototype.Recipe()).to(device))
            model.load_state_dict(initial, strict=True)
            initial_bytes = sum(value.numel() * value.element_size() for value in initial.values())
            meter.counts["explicit_host_to_device_tensor_bytes"] += 2 * initial_bytes
            meter.counts["explicit_device_to_host_tensor_bytes"] += initial_bytes
            require(meter.call("restored_initial_state_hash", state_digest, model.state_dict()) == initial_sha, "Twin initial state differs")
            optimizer = prototype.native_optimizer(model)
            result["initial_state_bytes"] = state_bytes(model, optimizer)
            model.train()
            restore_rng(epoch_rng)
            epoch_started = time.perf_counter()
            result["current_phase"] = "native_full_epoch_negative_and_permutation_stream"
            negatives, iterator, stream = epoch_stream(data, native_utils, sampler, meter)
            require(stream == parity["native_stream"], "Resource/GPU parity full epoch negative/permutation identities differ")
            if stream_reference is None:
                stream_reference = stream
            else:
                require(stream == stream_reference, "Twins did not receive matched native negative/permutation streams")
            result["native_stream"] = stream
            with instrument_portable(prototype, model, meter):
                for batch_index, record_ids in enumerate(iterator):
                    result["attempted_batch"] = batch_index
                    require(len(record_ids) == 65536, "Native full minibatch size changed")
                    baseline = meter.counts.copy()
                    enumeration_start = len(meter.enumerations)
                    batch_started = time.perf_counter()
                    meter.call("zero_grad", optimizer.zero_grad, set_to_none=True)
                    result["current_phase"] = "record_mask_full_graph_rebuild"
                    graph = meter.call("full_train_record_mask_graph_rebuild", graph_ops.Graph.mask_train_batch, data["pairs"], record_ids, 235868)
                    meter.counts["source_graph_validation_scalar_synchronizations"] += 2
                    require(graph.nodes == 235868, "Masked graph dropped nodes")
                    # Enumeration instrumentation applies to every native full
                    # positive/negative pass; decoder uses this original graph.
                    meter.counts["encoder_calls"] += 1
                    meter.counts["encoder_full_node_rows"] += len(data["x"])
                    result["current_phase"] = "full_node_encoder"
                    h = meter.call("full_node_encoder", model.encoder, data["x"], graph)
                    meter.finite(h, "full_node_encoder")
                    meter.query_pass = "positive"
                    result["current_phase"] = "complete_positive_decoder"
                    positive = meter.call("complete_positive_decoder", model.decoder, h, graph, data["pairs"][record_ids], mode)
                    meter.query_pass = "negative"
                    result["current_phase"] = "complete_negative_decoder"
                    negative = meter.call("complete_negative_decoder", model.decoder, h, graph, negatives[record_ids], mode)
                    require(tuple(positive.shape) == tuple(negative.shape) == (65536, 4), "Native supervision/member shape changed")
                    loss = meter.call("native_twice_balanced_TRAIN_loss", train_objective, positive, negative)
                    meter.finite(loss, "TRAIN_loss_scalar")
                    # Preserve the serving pool; no rankings or quality metric.
                    served = meter.call("mean_raw_logit_serving_pool", model.serve, positive)
                    require(torch.equal(served, positive.mean(1)), "Mean raw-logit serving pool changed")
                    meter.finite(served, "mean_raw_logit_pool")
                    result["current_phase"] = "backward_and_native_Adam"
                    meter.call("backward", loss.backward)
                    meter.call("gradient_finite_guards", finite_gradients_and_state, model, optimizer, meter)
                    meter.call("native_Adam_step", optimizer.step)
                    meter.call("post_Adam_finite_guards", finite_gradients_and_state, model, optimizer, meter)
                    delta = {name: meter.counts[name] - baseline[name] for name in EXPECTED_BATCH_WORK}
                    require(delta == EXPECTED_BATCH_WORK, "Native complete minibatch execution/work schedule differs")
                    batch_enumerations = meter.enumerations[enumeration_start:]
                    require(len(batch_enumerations) == 18, "Incomplete outer/recursive candidate enumeration accounting")
                    require(sum(row["queries"] for row in batch_enumerations if row["scope"] == "outer") == 131072, "Outer query coverage changed")
                    outer = [row for row in batch_enumerations if row["scope"] == "outer"]
                    require(sum(row["queries"] for row in batch_enumerations if row["scope"] == "recursive") == 4 * sum(row["left"] + row["right"] for row in outer), "Recursive candidate population reduced")
                    result["batches"].append({"batch_index": batch_index, "positive_records": len(record_ids),
                        "negative_records": len(record_ids), "masked_graph_nodes": graph.nodes,
                        "masked_graph_directed_entries": len(graph.row), "work": delta,
                        "full_node_xlin_rows": 24 * graph.nodes, "full_node_xlin_linear_map_rows": 48 * graph.nodes,
                        "encoder_projection_rows": graph.nodes, "full_candidate_enumerations": batch_enumerations,
                        "wall_seconds": time.perf_counter() - batch_started,
                        "cuda_peak_allocated_so_far_bytes": torch.cuda.max_memory_allocated(0),
                        "cuda_peak_reserved_so_far_bytes": torch.cuda.max_memory_reserved(0)})
                    del graph, h, positive, negative, loss, served
                    print("RESOURCE_BATCH_COMPLETE mode=" + mode + " batch=" + str(batch_index + 1) + "/17", flush=True)
            require(len(result["batches"]) == 17, "Incomplete native epoch; no feasibility pass")
            result["complete_native_epoch_seconds_including_sampler_order_guards_and_profiling"] = time.perf_counter() - epoch_started
            result["final_state_bytes"] = state_bytes(model, optimizer)
            result["final_rng_sha256"] = rng_digest(capture_rng())
            if mode == "pooled_after_clamp":
                require(result["final_rng_sha256"] == results[0]["final_rng_sha256"], "Twins consumed different full-epoch RNG/dropout schedules")
            result["current_phase"] = "internal_state_serialization"
            # Internal engineering state, no selection or study donor. All
            # model/Adam/RNG serialization and transfers are charged explicitly.
            blob = meter.call("model_Adam_RNG_state_serialization", selected_state.selected_bytes, model, optimizer, mode,
                {"kind": "complete_native_TRAIN_resource_epoch", "no_predictive_selection": True,
                 "root_admission_sha256": context["admission_sha256"]})
            state_path = context["output"] / ("INTERNAL_RESOURCE_STATE_" + mode + ".pt")
            def write_state():
                descriptor = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(blob)
            meter.call("internal_state_write", write_state)
            result["internal_state"] = {"path": state_path.name, "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(), "scientific_donor_admitted": False}
            del blob
            result["status"] = "COMPLETE_NATIVE_EPOCH_FINITE"
            result["current_phase"] = "complete"
        finally:
            if result["status"] != "COMPLETE_NATIVE_EPOCH_FINITE":
                result["status"] = "FAILED_INCOMPLETE_NATIVE_EPOCH"
            try:
                result["accounting"] = meter.receipt()
            except Exception as accounting_error:
                result["accounting_error_condition"] = type(accounting_error).__name__ + ": " + str(accounting_error)[:1000]
        del model, optimizer, negatives, iterator
    progress["all_required_checks_passed"] = len(results) == 2 and all(result["status"] == "COMPLETE_NATIVE_EPOCH_FINITE" and len(result["batches"]) == 17 for result in results)
    require(progress["all_required_checks_passed"], "Both complete native resource epochs are required")
    progress["outcome_comparisons_computed"] = False
    return "BOTH_COMPLETE_NATIVE_RESOURCE_EPOCHS_FINITE"
