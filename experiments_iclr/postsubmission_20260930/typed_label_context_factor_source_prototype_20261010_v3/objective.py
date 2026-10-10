"""Native shuffled TRAIN filtering and FP32 paired-context serving; inactive."""
from .caps import CLOSED
from .context import half_inputs
from .runtime_gate import require


def ordered_queries(torch, train_batch, train, paired, caps=CLOSED):
    caps.require("source_bound", "data", "runtime")
    ids = tuple(int(v) for v in train_batch.detach().cpu().tolist())
    require(len(ids) == len(train.ids) and len(set(ids)) == len(ids) and set(ids) == set(train.ids)
            and tuple(train.ids) == paired.plan.train_ids, "Exactly one native complete shuffled TRAIN permutation")
    positions = {node: p for p, node in enumerate(train.ids)}
    selected = []
    for view in paired.views:
        queries = set(view.query_ids)
        ordered = tuple(node for node in ids if node in queries)
        require(set(ordered) == queries and not set(ordered) & set(view.context_ids), "Half query filter preserves native order and excludes its labels")
        selected.append((view, ordered, tuple(positions[node] for node in ordered)))
    require(sum(len(row[1]) for row in selected) == len(ids)
            and not set(selected[0][1]) & set(selected[1][1]), "Every TRAIN target supervised exactly once per member")
    return tuple(selected)


def member_serving(torch, bank, inputs, member, counters, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    require(not bank.training, "Native selected eval and complete nonempty requested cohort")
    device = next(bank.members[member].parameters()).device
    positions = {node: p for p, node in enumerate(inputs.train.ids)}
    train = torch.empty((len(inputs.train.ids), 5), dtype=torch.float32, device="cpu")
    valid_values, context_values, seen = [], [], set()
    for view in inputs.paired.views:
        # TRAIN uses ONLY the context excluding every queried TRAIN label. In
        # particular an all-movie global field must never include that label.
        ids = view.query_ids + inputs.validation.ids
        require(not set(ids) & set(view.context_ids), "TRAIN diagnostics and all VALID labels excluded from their context")
        batch, features, labels, field = half_inputs(torch, view, ids, inputs.feats,
                                                    bank.condition == "shared_global_mul4", caps)
        output = bank.forward_member(member, batch.to(device),
                    {k: v.to(device) for k, v in features.items()},
                    {k: v.to(device) for k, v in labels.items()}, field)
        require(output.shape == (len(ids), 5) and output.dtype == torch.float32
                and torch.isfinite(output).all().item(), "Complete finite native FP32 serving logits")
        output = output.cpu()
        n = len(view.query_ids)
        where = torch.tensor(tuple(positions[node] for node in view.query_ids), dtype=torch.long)
        train[where] = output[:n]
        seen.update(view.query_ids)
        valid_values.append(output[n:])
        context_values.append({"TRAIN_query_ids": list(view.query_ids),
                               "TRAIN_complement_logits": output[:n].clone(),
                               "VALID_logits": output[n:].clone()})
        counters["serving_native_calls"] += 1
        counters["serving_query_rows"] += len(ids)
    require(seen == set(inputs.train.ids), "Complete TRAIN complement diagnostics exactly once per member")
    valid = torch.stack(valid_values).mean(dim=0)  # Two equally weighted frozen contexts.
    return torch.cat((train, valid), dim=0), tuple(context_values)
