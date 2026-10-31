"""Two fixed auxiliary losses on the exact native record-mask stream."""
from time import perf_counter
import torch
from torch.nn import functional as F
from graph_ops import Graph
from pilot_common import require
from pilot_model import train_flag, named_parameters
from pilot_data import epoch_stream
from pattern_objective import losses, diagnostic_sums


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


def batch_forward(model, data, teacher, negative_pairs, record_ids, *, auxiliary_grad=True):
    graph = Graph.mask_train_batch(data["pairs"], record_ids, len(data["x"]))
    h = model.encoder(data["x"], graph)
    records = []
    for query in (data["pairs"][record_ids], negative_pairs[record_ids]):
        logits, detail = model.decoder.pattern_forward(h, graph, query, auxiliary_grad=auxiliary_grad)
        rows, labels = teacher.labels(query, detail["neighbors"])
        values = losses(detail["t"], rows, labels.to(detail["t"].dtype), len(query))
        records.append({"query": query, "logits": logits, "detail": detail, "rows": rows, "labels": labels, "values": values})
    main = -F.logsigmoid(records[0]["logits"]).mean() - F.logsigmoid(-records[1]["logits"]).mean()
    return main, records


def train_batch(model, optimizer, data, teacher, negative_pairs, record_ids, arm, *, progress=None):
    require(arm in ("J", "F"), "Unknown pattern loss")
    optimizer.zero_grad(set_to_none=True)
    main, records = batch_forward(model, data, teacher, negative_pairs, record_ids)
    auxiliary = sum(row["values"][arm].mean() for row in records)
    total = main + auxiliary  # frozen lambda=1; native main gradients unscaled
    require(bool(torch.isfinite(total)), "Nonfinite main/auxiliary objective")
    if progress is not None:
        progress()
    total.backward()
    finite(model, optimizer, gradients=True)
    optimizer.step()
    finite(model, optimizer, gradients=False)
    diagnostics = {name: diagnostic_sums(row["detail"]["t"].detach(), row["detail"]["native_q"], row["rows"], row["labels"], len(record_ids), row["values"], common_rows=row["detail"]["neighbors"].common[0])
                   for name, row in zip(("positive", "negative"), records)}
    return {"main_loss": float(main.detach()), "selected_auxiliary_loss": float(auxiliary.detach()),
            "total_loss": float(total.detach()), "diagnostics": diagnostics,
            "native_member_main_loss": (-F.logsigmoid(records[0]["logits"]).mean(0)-F.logsigmoid(-records[1]["logits"]).mean(0)).detach().cpu().tolist(),
            "support_digest": {name: teacher.support_digest(row["query"], row["detail"]["neighbors"], row["labels"])
                               for name, row in zip(("positive", "negative"), records)}}


def train_epoch(model, optimizer, data, mods, sampler, teacher, arm, *, progress=None):
    train_flag(model, True)
    torch.cuda.synchronize(0)
    started = perf_counter()
    negative_pairs, iterator, stream = epoch_stream(data, mods, sampler)
    private = []
    slots, removed = 0, 0
    for batch, record_ids in enumerate(iterator, start=1):
        if progress is not None:
            progress({"phase": "masked_graph_native_and_pattern_forward", "attempted_batch": batch, "completed_batches": batch - 1})
        row = train_batch(model, optimizer, data, teacher, negative_pairs, record_ids, arm,
                          progress=(lambda: progress({"phase": "main_and_auxiliary_backward_Adam"})) if progress is not None else None)
        private.append(row)
        for d in row["diagnostics"].values():
            slots += d["residual_slots"]; removed += d["synthetic_removed_observed_slots"]
        if progress is not None:
            progress({"phase": "batch_complete", "completed_batches": batch})
    require(len(private) == 17, "Incomplete native epoch")
    torch.cuda.synchronize(0)
    return {"full_batches": 17, "optimizer_steps": 17, "positive_records": 1114112,
            "negative_records": 1114112, "dropped_positive_tail": 64940,
            "encoder_calls": 17, "decoder_query_passes": 34, "member_count": 4,
            "recursive_scorer_gradient_enabled": True, "target_completion_detached": True,
            "residual_slots": slots, "synthetic_removed_observed_slots": removed,
            "support_digests": [row["support_digest"] for row in private],
            "stream": stream, "wall_seconds": perf_counter() - started,
            "private_training_diagnostics": private}
