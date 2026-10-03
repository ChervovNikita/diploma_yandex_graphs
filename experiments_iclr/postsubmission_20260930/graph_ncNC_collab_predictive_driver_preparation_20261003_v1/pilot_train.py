"""Full source-native epoch updates, no quality logging or TEST access."""
from time import perf_counter
from pilot_common import require
from pilot_data import epoch_stream, native_graph
from pilot_model import train_flag, named_parameters


def finite(model, optimizer, *, gradients):
    import torch
    for name, parameter in named_parameters(model):
        require(bool(torch.isfinite(parameter).all()), "Nonfinite model parameter")
        if gradients:
            require(parameter.grad is not None or "ptlin." in name, "Missing active native parameter gradient: " + name)
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite model gradient")
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                require(bool(torch.isfinite(value).all()), "Nonfinite Adam state")


def train_epoch(model, optimizer, data, mods, sampler, *, mode=None, progress=None):
    import torch
    train_flag(model, True)
    torch.cuda.synchronize(0)
    started = perf_counter()
    negatives, iterator, stream = epoch_stream(data, mods, sampler)
    batches = 0
    for record_ids in iterator:
        if progress is not None:
            progress({"phase": "record_mask_and_forward", "attempted_batch": batches + 1, "completed_batches": batches})
        require(len(record_ids) == 65536, "Native full batch changed")
        train_batch(model, optimizer, data, mods, negatives, record_ids, mode=mode,
                    backward_progress=(lambda: progress({"phase": "backward_and_Adam", "attempted_batch": batches + 1, "completed_batches": batches})) if progress is not None else None)
        batches += 1
        if progress is not None:
            progress({"phase": "batch_complete", "attempted_batch": batches, "completed_batches": batches})
    require(batches == 17, "Incomplete native epoch")
    torch.cuda.synchronize(0)
    return {"full_batches": batches, "positive_records": 1114112, "negative_records": 1114112,
            "dropped_positive_tail": 64940, "stream": stream, "wall_seconds": perf_counter() - started,
            "encoder_calls": 17, "decoder_query_passes": 34, "member_count": 4 if mode is not None else 1,
            "optimizer_steps": 17, "project_training_loss_exposed": False}


def train_batch(model, optimizer, data, mods, negatives, record_ids, *, mode=None, backward_progress=None):
    """One native record-masked update; also used for tiny synthetic execution."""
    import torch
    from torch.nn import functional as F
    optimizer.zero_grad(set_to_none=True)
    if isinstance(model, tuple):
        keep = torch.ones(len(data["pairs"]), device=data["pairs"].device, dtype=torch.bool)
        keep[record_ids] = False
        graph = native_graph(data["pairs"][keep], len(data["x"]))
        encoder, decoder = model
        h = encoder(data["x"], graph)
        positive = decoder.multidomainforward(h, graph, data["pairs"][record_ids].T, cndropprobs=[])
        negative = decoder.multidomainforward(h, graph, negatives[record_ids].T, cndropprobs=[])
        members = 1
    else:
        graph = mods["graph_ops"].Graph.mask_train_batch(data["pairs"], record_ids, len(data["x"]))
        h = model.encoder(data["x"], graph)
        positive = model.decoder(h, graph, data["pairs"][record_ids], mode)
        negative = model.decoder(h, graph, negatives[record_ids], mode)
        members = 4
    require(positive.numel() == negative.numel() == len(record_ids) * members, "Native query/member coverage differs")
    loss = -F.logsigmoid(positive).mean() - F.logsigmoid(-negative).mean()
    require(bool(torch.isfinite(loss)), "Nonfinite native loss")
    if backward_progress is not None:
        backward_progress()
    loss.backward()
    finite(model, optimizer, gradients=True)
    optimizer.step()
    finite(model, optimizer, gradients=False)
