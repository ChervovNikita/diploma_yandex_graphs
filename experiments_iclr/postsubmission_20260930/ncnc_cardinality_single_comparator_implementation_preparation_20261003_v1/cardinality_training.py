"""Future D4-only native17-batch epoch utilities; no executable entry point.

No fit loop or root release is supplied. A future root must bind this packet,
fabricated qualifications, complete-support resource receipts, data/runtime
authority and a fresh disposable/scientific state before calling helpers.
"""
from time import perf_counter
import torch
from torch.nn import functional as F
from cardinality_counter import CompletionKeys
from cardinality_graph import Graph
from cardinality_density import pattern_nll, require
from cardinality_teacher import TrainObservationTeacher, tensor_sha, sealed_support_digest


def finite(model, optimizer, *, gradients, count_head_active=True):
    for name, parameter in model.named_parameters():
        require(bool(torch.isfinite(parameter).all()), "Nonfinite model parameter: " + name)
        if gradients:
            inactive = name.startswith("decoder.ptlin.") or name.startswith("count_head.") and not count_head_active
            require(parameter.grad is not None or inactive, "Missing active gradient: " + name)
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite gradient: " + name)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                require(bool(torch.isfinite(value).all()), "Nonfinite Adam state")


def batch_forward(model, x, train_pairs, negative_pairs, record_ids, teacher, *, epoch, batch):
    require(model.training, "Training dropout required")
    require(type(teacher) is TrainObservationTeacher and teacher.nodes == len(x), "Typed TRAIN teacher required")
    require(teacher.train_records_sha256 == tensor_sha(train_pairs), "Teacher and masked TRAIN snapshots differ")
    require(record_ids.ndim == 1 and record_ids.dtype == torch.long and len(record_ids) > 0, "Native record mask required")
    require(negative_pairs.ndim == 2 and negative_pairs.shape[1] == 2 and len(negative_pairs) >= len(train_pairs), "Complete native epoch negative draw required")
    require(bool(torch.isfinite(x).all()), "Nonfinite node features")
    graph = Graph.mask_train_batch(train_pairs, record_ids, len(x))
    h = model.encode(x, graph)
    queries = (train_pairs[record_ids], negative_pairs[record_ids])
    records = []
    # Combined ordinal is positive then negative; no query-label key exists.
    # Produce BOTH prediction laws/samples/target forwards before any oracle.
    for current, ordinal_start in zip(queries, (0, len(record_ids))):
        keys = CompletionKeys("train", epoch=epoch, batch=batch, ordinal_start=ordinal_start)
        logits, detail, draws = model.query_forward(h, graph, current, keys, route="D4")
        records.append({"query": current, "logits": logits, "detail": detail, "draws": draws,
                        "support_pre_teacher_sha256": sealed_support_digest(current, detail["support"])})
    for record in records:
        # Only fixed counterpart coordinates are exposed to this label oracle.
        labels = teacher.labels(record["detail"]["support"].counterpart_pairs)
        values, observed_count = pattern_nll(record["detail"]["law"], labels)
        record.update(labels=labels, nll_per_query=values, observed_count=observed_count)
    main = -F.logsigmoid(records[0]["logits"]).mean() - F.logsigmoid(-records[1]["logits"]).mean()
    auxiliary = records[0]["nll_per_query"].mean() + records[1]["nll_per_query"].mean()
    total = main + 1. * auxiliary
    require(bool(torch.isfinite(total)), "Nonfinite native four-draw/main/full-pattern objective")
    return main, auxiliary, total, records


def train_batch(model, optimizer, x, train_pairs, negative_pairs, record_ids, teacher, *, epoch, batch, progress=None):
    optimizer.zero_grad(set_to_none=True)
    main, auxiliary, total, records = batch_forward(model, x, train_pairs, negative_pairs, record_ids, teacher, epoch=epoch, batch=batch)
    if progress is not None:
        progress({"phase": "auxiliary_DP_recompute_and_backward_Adam", "epoch": epoch, "batch": batch})
    total.backward()
    head_active = any(len(record["detail"]["law"].t) > 0 for record in records)
    finite(model, optimizer, gradients=True, count_head_active=head_active)
    optimizer.step()
    finite(model, optimizer, gradients=False)
    # Engineering receipts expose coverage/custody/work, never a ranking
    # metric or scientific loss. Dedicated future diagnostics have own scope.
    support_digest, histogram, maximum, cells, slots, classes = [], {}, 0, 0, 0, 0
    for record in records:
        law = record["detail"]["law"]
        sizes = (law.offsets[1:] - law.offsets[:-1]).detach().cpu().tolist()
        for r in sizes:
            histogram[str(r)] = histogram.get(str(r), 0) + 1
            maximum = max(maximum, r)
            cells += (r + 1) * (r + 2) // 2
        slots += len(law.t)
        classes += len(law.log_pi)
        support_digest.append({"before_teacher": record["support_pre_teacher_sha256"],
                               "with_observation_labels": sealed_support_digest(record["query"], record["detail"]["support"], record["labels"])})
    return {"positive_queries": len(record_ids), "negative_queries": len(record_ids), "draws_per_query": 4,
            "encoder_calls": 1, "outer_full_node_xlin_calls": 2, "recursive_full_node_xlin_calls": 4,
            "native_depth_zero_calls": 4, "downstream_decodes": 8, "optimizer_steps": 1,
            "residual_slots": slots, "all_count_classes": classes, "R_histogram": histogram, "R_max": maximum,
            "DP_sampling_forward_queries": 2 * len(record_ids), "DP_auxiliary_forward_queries": 2 * len(record_ids),
            "DP_auxiliary_recomputed_queries": 2 * len(record_ids), "DP_reachable_cells_per_full_pass": cells,
            "mask_record_ids_sha256": tensor_sha(record_ids), "support_sha256": support_digest,
            "main_sampler_gradients": False, "outer_features_live": True, "project_loss_exposed": False}


def train_epoch(model, optimizer, data, native_utils, sampler, teacher, *, epoch, progress=None):
    """Exactly one complete epoch per call; authority/100-state fit is external."""
    require(type(epoch) is int and 1 <= epoch <= 100, "Frozen prospective epoch index required")
    require(type(teacher) is TrainObservationTeacher and teacher.scope == "complete_authenticated_TRAIN", "Project epoch requires authenticated TRAIN teacher")
    require(data["pairs"].shape == (1179052, 2) and data["x"].shape == (235868, 128), "Complete representative dimensions required")
    require(teacher.train_records_sha256 == data["digests"]["train_records"] == tensor_sha(data["pairs"]), "TRAIN snapshot custody differs")
    model.train()
    torch.cuda.synchronize(0)
    started = perf_counter()
    # Pinned native default sampler, same count/law and raw complete graph.
    negatives = sampler(data["raw_edge_index"], len(data["x"]))
    require(negatives.dtype == torch.long and negatives.ndim == 2 and negatives.shape[0] == 2 and negatives.shape[1] >= 1179052,
            "Incomplete native negative draw")
    require(bool((negatives[0] != negatives[1]).all()) and int(negatives.min()) >= 0 and int(negatives.max()) < len(data["x"]), "Invalid native negative endpoints")
    graph_keys = (data["raw_edge_index"][0] * len(data["x"]) + data["raw_edge_index"][1]).sort().values
    negative_keys = negatives[0] * len(data["x"]) + negatives[1]
    from cardinality_graph import membership
    require(not bool(membership(graph_keys, negative_keys).any()), "Native sampler returned forbidden TRAIN edge")
    iterator = native_utils.PermIterator(data["pairs"].device, len(data["pairs"]), 65536)
    require(len(iterator) == 17 and len(iterator.idx) == 1179052, "Native full-batch/drop-last policy differs")
    stream = {"negative_draw_sha256": tensor_sha(negatives), "permutation_sha256": tensor_sha(iterator.idx),
              "dropped_tail_sha256": tensor_sha(iterator.idx[17 * 65536:]), "full_batches": 17,
              "supervised_records": 1114112, "dropped_tail_records": 64940, "negative_rows_drawn": negatives.shape[1]}
    rows = []
    for batch, record_ids in enumerate(iterator, 1):
        require(len(record_ids) == 65536, "Native batch reduced")
        if progress is not None:
            progress({"phase": "record_mask_and_D4_forward", "epoch": epoch, "batch": batch, "completed_batches": batch - 1})
        rows.append(train_batch(model, optimizer, data["x"], data["pairs"], negatives.T, record_ids, teacher,
                                epoch=epoch, batch=batch, progress=progress))
        if progress is not None:
            progress({"phase": "batch_complete", "epoch": epoch, "completed_batches": batch})
    require(len(rows) == 17, "Incomplete TRAIN epoch")
    torch.cuda.synchronize(0)
    return {"epoch": epoch, "full_batches": 17, "positive_queries": 1114112, "negative_queries": 1114112,
            "optimizer_steps": 17, "dropped_positive_tail": 64940, "stream": stream, "batch_work": rows,
            "wall_seconds": perf_counter() - started, "main_route": "C64-D4", "M4_training_updates": 0,
            "byte_identical_JF_native_stream_claimed": False, "all_supports_and_count_classes_complete": True}
