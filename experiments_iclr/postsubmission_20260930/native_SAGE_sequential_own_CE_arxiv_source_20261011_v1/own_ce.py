"""Own-CE execution hook; caller supplies the admitted native Family and Torch.

No imports of numerical libraries, data reader, optimizer construction or fit
launcher. Simultaneous tapes are a disposable qualification comparator only.
"""
import math


def update(family, model, optimizer, streams, bank, *, mode="sequential",
           qualification=False, capture_gradients=False):
    if mode not in ("sequential", "simultaneous"):
        raise ValueError("Fixed sequential accumulation or qualification comparator only")
    if mode == "simultaneous" and not (qualification and family.training_only):
        raise ValueError("Simultaneous route tapes are qualification-only")
    if capture_gradients and not qualification:
        raise ValueError("Gradient copies belong only to disposable qualification")
    torch, members = family.torch, 4 if bank else 1
    body = model.body if bank else model
    if (bank and (model.kind != "baseline" or model.block is not None
                  or model.members != 4)) or len(streams) != members:
        raise ValueError("Four independent native routes; no cross-route term")
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("Every registered native/builder parameter exactly once")
    if list(body.named_buffers()):
        raise ValueError("This port requires the existing buffer-free LayerNorm owner")
    targets = family.data.y[family.train_ids] if bank else family.data.y[family.data.train_mask]
    model.train()
    optimizer.zero_grad()  # Same default behavior as the completed E owner.
    detached = []
    if mode == "simultaneous" or not bank:
        logits, hidden = family.training_outputs(model, streams, bank)
        if logits.shape != (members, len(targets), 40) or hidden.shape != (members, len(targets), 128):
            raise ValueError("Complete TRAIN/native head dimensions differ")
        loss = family.F.cross_entropy(logits.flatten(0, 1), targets.repeat(members))
        loss.backward()
        own_ce = loss.item()
        detached.append(logits.detach())
        del logits, hidden, loss
        backwards = 1
    else:
        adapter = model.native_adapter  # Captured before the VALUE conv replacement.
        if adapter.body is not body or adapter.width != 128:
            raise ValueError("Reuse the exact native adapter and body")
        batch = dict(family.batch, ids=family.train_ids)
        if any(k in batch for k in ("y", "labels", "train_mask", "valid_mask", "test_mask")):
            raise ValueError("Native route interface must remain label-free")
        adapter.check(batch)
        versions = (batch["x"]._version, batch["edge_index"]._version)
        own_ce = 0.0
        for member in range(4):
            # Identical order/scope to common_routes.SharedRoutes.forward.
            with family.scope(streams, member), family.factors.member_context(body, member):
                full_logits, full_hidden = adapter.native(batch)
            if full_logits.shape != (169343, 40) or full_hidden.shape != (169343, 128):
                raise ValueError("No toy, sampled or induced graph route")
            route_logits = full_logits[family.train_ids]
            route_ce = family.F.cross_entropy(route_logits, targets)
            loss = route_ce / 4
            loss.backward()  # retain_graph=False; no parameter/moment update here.
            own_ce += route_ce.item() / 4
            detached.append(route_logits.detach().unsqueeze(0))
            del loss, route_ce, route_logits, full_logits, full_hidden
            if versions != (batch["x"]._version, batch["edge_index"]._version):
                raise ValueError("A route mutated factual graph/input")
        backwards = 4
    if not math.isfinite(own_ce):
        raise ValueError("Nonfinite full-TRAIN own CE")
    with torch.no_grad():
        train_logits = torch.cat(detached)
        pool_ce = family.probability_pool_ce(train_logits, targets).item() if bank else None
    del train_logits, detached
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in model.named_parameters()} if capture_gradients else None
    optimizer.step()  # Exactly once after all quarter-scaled route backwards.
    return {"own_ce": own_ce, "pool_ce": pool_ce, "backwards": backwards,
            "adam_steps": 1, "native_trajectories": members,
            "gradients": gradients, "mode": mode}
