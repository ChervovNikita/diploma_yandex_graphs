"""One native update; fixed target-only, joint, or separated conditional auxiliary."""
from time import perf_counter
import torch
from torch.nn import functional as F
from graph_ops import Graph
from pilot_common import require
from pilot_model import train_flag, named_parameters
from pilot_data import epoch_stream, tensor_sha
from pilot_state import rng_state, rng_digest
from conditional_loss import TrainPatternLabels, training_pattern_losses


def finite(model, optimizer, *, gradients):
    for name, parameter in named_parameters(model):
        require(bool(torch.isfinite(parameter).all()), "Nonfinite parameter " + name)
        if gradients:
            require(parameter.grad is not None or "ptlin." in name, "Missing active gradient " + name)
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite gradient " + name)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                require(bool(torch.isfinite(value).all()), "Nonfinite Adam state")


def train_batch(model, optimizer, data, teacher, negative_pairs, record_ids, arm, progress=None):
    require(arm in ("target_only", "joint", "separate"), "Unknown prospective arm")
    optimizer.zero_grad(set_to_none=True)
    graph = Graph.mask_train_batch(data["pairs"], record_ids, len(data["x"]))
    h = model.encoder(data["x"], graph)
    logits, auxiliary = [], []
    key = {"joint": "J_K", "separate": "J_K_sep"}.get(arm)
    for query in (data["pairs"][record_ids], negative_pairs[record_ids]):
        native, detail = model.decoder.pattern_forward(h, graph, query, auxiliary_grad=key is not None)
        logits.append(native)
        if key is not None:
            rows, bits = teacher.labels(query, detail["neighbors"])
            left_rows, _ = detail["neighbors"].left
            right_rows, _ = detail["neighbors"].right
            require(torch.equal(rows, torch.cat((left_rows, right_rows))), "Teacher/native slot order differs")
            labels = TrainPatternLabels(bits.to(detail["t"].dtype), "complete_TRAIN_observation_membership")
            values = training_pattern_losses(detail["t"][:, :len(left_rows)], detail["t"][:, len(left_rows):],
                                             left_rows, right_rows, labels, len(query))
            auxiliary.append(values[key].mean())
    main = -F.logsigmoid(logits[0]).mean() - F.logsigmoid(-logits[1]).mean()
    aux = sum(auxiliary) if auxiliary else main.new_zeros(())
    total = main + aux  # coefficient1; positive and negative means are summed once.
    require(bool(torch.isfinite(total)), "Nonfinite fixed objective")
    if progress is not None:
        progress({"phase": "native_target_and_selected_auxiliary_backward_Adam"})
    total.backward()
    finite(model, optimizer, gradients=True)
    optimizer.step()
    finite(model, optimizer, gradients=False)
    return {"main_loss": float(main.detach()), "auxiliary_loss": float(aux.detach()),
            "total_loss": float(total.detach()), "record_ids_sha256": tensor_sha(record_ids)}


def train_epoch(model, optimizer, data, mods, sampler, teacher, arm, progress=None):
    train_flag(model, True)
    torch.cuda.synchronize(0)
    started = perf_counter()
    negative_pairs, iterator, stream = epoch_stream(data, mods, sampler)
    rows = []
    for batch, record_ids in enumerate(iterator, start=1):
        before = rng_digest(rng_state())
        if progress is not None:
            progress({"phase": "native_mask_and_forward", "attempted_batch": batch, "completed_batches": batch-1})
        value = train_batch(model, optimizer, data, teacher, negative_pairs, record_ids, arm, progress)
        value.update(batch=batch, start_rng_sha256=before, end_rng_sha256=rng_digest(rng_state()))
        rows.append(value)
        if progress is not None:
            progress({"phase": "batch_complete", "completed_batches": batch})
    require(len(rows) == 17, "Incomplete native full epoch")
    torch.cuda.synchronize(0)
    return {"full_batches": 17, "optimizer_steps": 17, "positive_records": 1114112,
            "negative_records": 1114112, "dropped_positive_tail": 64940, "encoder_calls": 17,
            "decoder_query_passes": 34, "member_count": 4, "stream": stream,
            "target_completion_detached": True, "auxiliary_scorer_gradient": arm != "target_only",
            "batch_receipts": rows, "wall_seconds": perf_counter()-started}
