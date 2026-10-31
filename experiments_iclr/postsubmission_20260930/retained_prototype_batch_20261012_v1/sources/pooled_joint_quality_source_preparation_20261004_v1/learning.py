"""Prospective literal-replay updates and fixed in-memory fit contract.

No loader, CLI, release or old-state reader. Root supplies authenticated fresh
data/provider/runtime and separately authorizes every engineering/science run.
"""
import torch
from torch.nn import functional as F
from graph_ops import Graph
from pattern_objective import losses
from count_density import require

ARMS = ("P0", "J_P", "F_P", "C_mu")
REPLAY_MASTER_SEED = 2026100401
EPOCHS, BATCHES, TRAIN_BATCH, EVAL_BATCH = 100, 17, 65536, 131072


def batch_forward(model, x, graph, positive, negative, *, auxiliary_grad=True):
    h = model.encode(x, graph)
    output = []
    for query in (positive, negative):
        logits, detail = model.query_forward(h, graph, query, auxiliary_grad=auxiliary_grad)
        output.append({"query": query, "logits": logits, "detail": detail})
    main = -F.logsigmoid(output[0]["logits"]).mean() - F.logsigmoid(-output[1]["logits"]).mean()
    return main, output


def train_replay_batch(model, optimizer, x, replay, batch, teacher, arm):
    require(arm in ARMS, "Unknown prospectively fixed arm")
    require(replay.receipt["master_seed"] == REPLAY_MASTER_SEED, "Different prospective replay choice")
    records, negatives, _ = replay.query_batch(batch)
    pairs, nodes = replay.data["pairs"], replay.data["nodes"]
    require(len(records) == TRAIN_BATCH and len(x) == nodes, "Complete native batch/features required")
    graph = Graph.mask_train_batch(pairs, records, nodes)
    positive, negative = pairs[records], negatives[records]
    optimizer.zero_grad(set_to_none=True)
    main, output = batch_forward(model, x, graph, positive, negative, auxiliary_grad=arm != "P0")
    # Actual full graph, query, common and residual support authenticated before
    # any teacher lookup. Record IDs and receipts never enter predictor inputs.
    replay.assert_actual(batch, {"record_ids": records, "graph": graph,
                         "positive_queries": positive, "negative_queries": negative,
                         "positive_neighbors": output[0]["detail"]["neighbors"],
                         "negative_neighbors": output[1]["detail"]["neighbors"]})
    auxiliary = []
    for row in output:
        rows, labels = teacher.labels(row["query"], row["detail"]["neighbors"])
        if arm == "P0":
            require("t" in row["detail"], "P0 requires the same pooled bank")
            # Teacher remains label-only; there is no source auxiliary branch
            # or scorer gradient at coefficient0. The target/RNG route matches.
            value = main.new_zeros(len(row["query"]))
        elif arm == "C_mu":
            require("eta" in row["detail"], "Count arm/model mismatch")
            value = model.source_nll(row["detail"], labels)
        else:
            require("t" in row["detail"], "Pooled arm/model mismatch")
            value = losses(row["detail"]["t"], rows,
                           labels.to(row["detail"]["t"].dtype), len(row["query"]))[arm[:1]]
        auxiliary.append(value.mean())  # includes every empty-query zero
    aux = sum(auxiliary)
    total = main + aux
    require(bool(torch.isfinite(total)), "Nonfinite main/source objective")
    total.backward()
    gradients = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite gradient " + name)
            gradients.append(name)
    require(gradients, "No learning gradients")
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated state")
    return {"main": float(main.detach()), "auxiliary": float(aux.detach()), "total": float(total.detach()),
            "positive_queries": len(positive), "negative_queries": len(negative),
            "residual_slots": [len(r["detail"]["neighbors"].left[0]) + len(r["detail"]["neighbors"].right[0]) for r in output],
            "literal_replay_actual_context_checked": True}


def train_epoch(model, optimizer, x, replay, teacher, arm):
    require(replay.receipt["full_batches"] == BATCHES
            and replay.receipt["batch_size"] == TRAIN_BATCH, "Changed reduction/tail geometry")
    model.train()
    return [train_replay_batch(model, optimizer, x, replay, i, teacher, arm) for i in range(BATCHES)]


@torch.no_grad()
def score_complete_valid(model, x, train_pairs, positive, negative):
    require(len(positive) == 60084 and len(negative) == 100000, "Complete official VALID pools required")
    model.eval()
    graph = Graph.from_pairs(train_pairs, len(x))
    h = model.encode(x, graph)
    result = []
    for queries in (positive, negative):
        scores = []
        for start in range(0, len(queries), EVAL_BATCH):
            query = queries[start:start + EVAL_BATCH]
            logits, _ = model.query_forward(h, graph, query, auxiliary_grad=False)
            score = model.serve(logits)
            require(score.shape == (len(query),) and score.dtype == torch.float32
                    and bool(torch.isfinite(score).all()), "Incomplete/nonfinite serving output")
            scores.append(score.cpu())
        result.append(torch.cat(scores))
    return result


def fit_fixed(factory, replay_for_epoch, x, train_pairs, teacher, positive, negative,
              metric, state, arm, epoch_commit, selected_roundtrip):
    """Fresh fit only; root-owned receipt/journal policy is a separate gate.

    factory constructs from the fixed seed without a state argument. state is
    the inherited typed snapshot/RNG module. replay_for_epoch authenticates an
    externally bound actual epoch receipt for each of the same 100 epochs.
    epoch_commit must persist every attempted/completed epoch under root's
    released policy; this core itself cannot resume or read a checkpoint.
    selected_roundtrip must write and authenticate this fit's own selected
    snapshot, then return its loaded snapshot; an in-memory copy is inadequate.
    """
    require(arm in ARMS, "Unknown arm")
    model, optimizer = factory()
    initial = state.state_digest(state.snapshot(model, optimizer))
    best, selected, epochs = None, None, []
    for epoch in range(1, EPOCHS + 1):
        replay = replay_for_epoch(epoch)
        require(replay.receipt["epoch"] == epoch, "Wrong prospective epoch")
        work = train_epoch(model, optimizer, x, replay, teacher, arm)
        train_rng = state.rng_state()
        before = state.rng_digest(train_rng)
        served = score_complete_valid(model, x, train_pairs, positive, negative)
        require(state.rng_digest(state.rng_state()) == before, "Selector advanced training RNG")
        quality = float(metric.eval({"y_pred_pos": served[0], "y_pred_neg": served[1]})["hits@50"])
        explicit = float((served[0] > torch.topk(served[1], 50).values[-1]).sum()) / len(served[0])
        require(quality == explicit, "Official shared-negative strict Hits50 differs")
        current = state.snapshot(model, optimizer, rng=train_rng)
        if best is None or quality > best["hits50"]:
            best, selected = {"epoch": epoch, "hits50": quality}, current
        epochs.append({"epoch": epoch, "replay_epoch": replay.receipt["epoch"], "TRAIN": work, "hits50": quality})
        epoch_commit(arm, epoch, current, best, selected, epochs[-1])
    state.restore_snapshot(model, optimizer, selected)
    before = state.rng_digest(state.rng_state())
    reference = score_complete_valid(model, x, train_pairs, positive, negative)
    require(state.rng_digest(state.rng_state()) == before, "Selected replay advanced RNG")
    roundtrip = selected_roundtrip(arm, selected, best)
    require(state.state_digest(roundtrip) == state.state_digest(selected),
            "Own selected serialization changed typed model/Adam/RNG/flags state")
    restored, restored_opt = factory()
    state.restore_snapshot(restored, restored_opt, roundtrip)
    before = state.rng_digest(state.rng_state())
    replayed = score_complete_valid(restored, x, train_pairs, positive, negative)
    require(state.rng_digest(state.rng_state()) == before
            and all(torch.equal(a, b) for a, b in zip(reference, replayed)), "Selected-state exact replay differs")
    require(float(metric.eval({"y_pred_pos": reference[0], "y_pred_neg": reference[1]})["hits@50"]) == best["hits50"],
            "Selected score differs from strict first-best selector")
    return {"arm": arm, "initial_state_sha256": initial, "fresh_initialization": True,
            "selected": best, "selected_snapshot": selected, "epochs": epochs,
            "optimizer_updates": 1700, "selectors": 100, "complete_VALID_evaluations": 102,
            "selected_state_replays": 2, "resource_state_donor": False}
