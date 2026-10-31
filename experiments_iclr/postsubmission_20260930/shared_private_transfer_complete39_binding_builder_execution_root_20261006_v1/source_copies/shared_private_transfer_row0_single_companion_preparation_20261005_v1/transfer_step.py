"""Persistent virtual/shared/recomputed-private learning; no import-time torch."""
from contextlib import contextmanager
from private_adam import step as adam_step, initial_state, detached_state


class DropoutStreams:
    """Only committed inner work advances each private dropout stream once.

    Sampling/geometry own a separate restored native RNG. No global RNG is
    advanced by these functional model calls; outer serving is dropout-off.
    """
    def __init__(self, torch, device, seed, members=4):
        self.torch, self.device = torch, device
        self.devices = [device.index or 0] if device.type == "cuda" else []
        self.rows = []
        for member in range(members):
            with torch.random.fork_rng(devices=self.devices):
                torch.random.default_generator.manual_seed(seed+100000+1009*member)
                if self.devices:
                    with torch.cuda.device(device): torch.cuda.manual_seed(seed+100000+1009*member)
                self.rows.append({"cpu": torch.get_rng_state().clone(),
                                  "cuda": torch.cuda.get_rng_state(device).clone() if self.devices else None,
                                  "committed_calls": 0})

    @contextmanager
    def use(self, member, *, advance=False):
        torch, row = self.torch, self.rows[member]
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(row["cpu"])
            if self.devices: torch.cuda.set_rng_state(row["cuda"], self.device)
            yield
            if advance:
                row["cpu"] = torch.get_rng_state().clone()
                if self.devices: row["cuda"] = torch.cuda.get_rng_state(self.device).clone()
                row["committed_calls"] += 1

    def state_dict(self):
        return {str(i): {"cpu": row["cpu"].clone(),
                         "cuda": None if row["cuda"] is None else row["cuda"].clone(),
                         "committed_calls": row["committed_calls"]}
                for i, row in enumerate(self.rows)}


def own_loss(torch, positive, negative):
    return -torch.nn.functional.logsigmoid(positive).mean()-torch.nn.functional.logsigmoid(-negative).mean()


def outer_loss(torch, positive, negative):
    return .5*own_loss(torch, positive.mean(1), negative.mean(1))+.5*own_loss(torch, positive, negative)


def functional_forward(torch, model, parameters, features, support, queries, *, training, stream=None, route=None, advance=False):
    from torch.func import functional_call
    modes = [(module, module.training) for module in model.modules()]
    model.train(training)
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    try:
        if training:
            if stream is None: raise ValueError("Explicit replayable inner dropout stream required")
            streams, stream_index = stream
            with streams.use(stream_index, advance=advance):
                positive, negative = functional_call(model, (parameters, buffers),
                    (features, support, *queries), {"route": route}, strict=True)
        else:
            positive, negative = functional_call(model, (parameters, buffers),
                (features, support, *queries), {"route": route}, strict=True)
        if positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] != negative.shape[1]:
            raise ValueError("Prediction member geometry differs")
        if not bool(torch.isfinite(positive).all() and torch.isfinite(negative).all()):
            raise FloatingPointError("Nonfinite native logits")
        return positive, negative
    finally:
        for module, mode in modes: module.training = mode


def inner_gradients(torch, model, parameters, private_names, features, support, route_queries,
                    streams, *, create_graph, advance=False):
    """F4 uses four own normalized losses; strong single uses their full union.

    None gradients in an untied route's unrelated parameter block become exact
    zeros. Active private blocks are not averaged across members. Single union
    repeats count exactly as sampled and its one loss is normalized over union.
    Fixed-row0 F1 instead averages four own losses from four replayable streams.
    """
    if len(route_queries) != 4:
        raise ValueError("Four fixed inner streams are required in every matched arm")
    member_count = model.member_count
    four_stream_single = getattr(model, "private_four_streams", False)
    if four_stream_single:
        calls = list(enumerate(route_queries))
    elif member_count == 1:
        calls = [(0, (torch.cat([row[0] for row in route_queries], 1),
                      torch.cat([row[1] for row in route_queries], 1)))]
    else:
        calls = list(enumerate(route_queries))
    gradients = {name: torch.zeros_like(parameters[name]) for name in private_names}
    connected = {name: False for name in private_names}
    values = []
    for member, queries in calls:
        positive, negative = functional_forward(torch, model, parameters, features, support, queries,
            training=True, stream=(streams, member), route=0 if four_stream_single else member, advance=advance)
        loss = own_loss(torch, positive, negative)
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite private inner loss")
        local = torch.autograd.grad(loss, tuple(parameters[name] for name in private_names),
                                    create_graph=create_graph, allow_unused=True)
        for name, gradient in zip(private_names, local):
            if gradient is not None:
                connected[name] = True
                if not bool(torch.isfinite(gradient).all()): raise FloatingPointError("Nonfinite private gradient: "+name)
                gradients[name] = gradients[name]+gradient
        values.append(float(loss.detach()))
    if not all(connected.values()):
        raise ValueError("Declared private blocks disconnected from all own-route inner losses: "+str([name for name in private_names if not connected[name]]))
    if four_stream_single:
        gradients = {name: gradient / 4 for name, gradient in gradients.items()}
    return gradients, values, len(calls)


def virtual_state(torch, model, features, support, route_queries, streams, private_names, previous, *, live):
    parameters = dict(model.named_parameters())
    gradients, values, calls = inner_gradients(torch, model, parameters, private_names,
        features, support, route_queries, streams, create_graph=True, advance=False)
    private = {name: parameters[name] for name in private_names}
    adapted, state = adam_step(torch, private, gradients, previous)
    if not live:
        adapted = {name: value.detach().requires_grad_(True) for name, value in adapted.items()}
    return {**parameters, **adapted}, gradients, state, values, calls


def episode_step(torch, model, shared_names, private_names, shared_optimizer, previous,
                 features, support, inner_queries, outer_queries, streams, *, live, commit="recomputed"):
    """One committed shared Adam step and one recomputed private Adam step.

    Virtual moments are discarded. Parameters/previous moments for the private
    block remain untouched until the shared update has finished. No inference
    adaptation or private gradient from outer labels is committed.
    """
    if commit not in ("recomputed", "stale_virtual"):
        raise ValueError("Only prospectively reviewed recomputed/stale commit rules admitted")
    parameters = dict(model.named_parameters())
    adapted, virtual_gradients, virtual_moments, private_losses, inner_calls = virtual_state(torch, model,
        features, support, inner_queries, streams, private_names, previous, live=live)
    positive, negative = functional_forward(torch, model, adapted, features, support,
                                            outer_queries, training=False)
    objective = outer_loss(torch, positive, negative)
    outer_gradients = torch.autograd.grad(objective, tuple(parameters[name] for name in shared_names))
    if not bool(torch.isfinite(objective)) or not all(bool(torch.isfinite(g).all()) for g in outer_gradients):
        raise FloatingPointError("Nonfinite live/detached shared derivative")
    shared_optimizer.zero_grad(set_to_none=True)
    for name, gradient in zip(shared_names, outer_gradients): parameters[name].grad = gradient.detach()
    shared_optimizer.step()
    objective_value = float(objective.detach())
    stale_private = ({name: adapted[name].detach().clone() for name in private_names}
                     if commit == "stale_virtual" else None)
    stale_moments = detached_state(virtual_moments) if commit == "stale_virtual" else None
    # Drop the entire virtual graph before native inner recomputation. Its
    # parameter/moment update is never committed or used as the next start.
    del adapted, positive, negative, outer_gradients, virtual_moments, virtual_gradients, objective
    recomputed, committed_losses, committed_calls = inner_gradients(torch, model, parameters,
        private_names, features, support, inner_queries, streams, create_graph=False, advance=True)
    new_private, new_moments = adam_step(torch, {name: parameters[name] for name in private_names}, recomputed, previous)
    if commit == "stale_virtual":
        # Recomputed values/moments and full work are paid, then discarded.
        # The stale control differs only in the private serving commitment.
        new_private, new_moments = stale_private, stale_moments
    with torch.no_grad():
        for name, value in new_private.items(): parameters[name].copy_(value)
    if committed_calls != inner_calls: raise ValueError("Virtual/recomputed exposure differs")
    return detached_state(new_moments), {
        "virtual_inner_losses": private_losses, "recomputed_inner_losses": committed_losses,
        "virtual_outer_objective": objective_value,
        "private_optimizer_updates": 1, "shared_optimizer_updates": 1,
        "functional_forward_calls": inner_calls+1+committed_calls,
        "model_encoder_forward_count": inner_calls+committed_calls+(4 if model.member_count == 4 and hasattr(model, "routes") else 1),
        "dropout_off_outer": True, "virtual_moments_discarded": commit == "recomputed",
        "private_outer_label_update": False, "live_adaptation_credit": bool(live),
        "private_commit": commit, "paid_recomputation_discarded": commit == "stale_virtual"}


def ordinary_episode_step(torch, model, optimizer, features, support, inner_queries, outer_queries, streams):
    """Strong paid-exposure ordinary control: inner, outer, repeated inner.

    All parameters receive standard native joint Adam at each pass. Both inner
    passes use the same dropout masks; only the final pass advances streams.
    Native independent4 uses summed own losses so each constituent receives its
    own normalized native gradient; sharedF4/single use their native mean loss.
    This is a declared matched ordinary schedule, not an unchanged donor fit.
    """
    four_stream_single = getattr(model, "private_four_streams", False)
    calls = list(enumerate(inner_queries)) if four_stream_single else [(0, (torch.cat([row[0] for row in inner_queries], 1),
                  torch.cat([row[1] for row in inner_queries], 1)))] if model.member_count == 1 else list(enumerate(inner_queries))
    losses, forward_calls = [], 0
    for pass_index in range(3):
        parameters = dict(model.named_parameters())
        optimizer.zero_grad(set_to_none=True)
        if pass_index == 1:
            positive, negative = functional_forward(torch, model, parameters, features, support, outer_queries, training=False)
            # Match the candidate's aggregate/own outer objective; the
            # separately preserved native references retain their own loss.
            value = outer_loss(torch, positive, negative)
            if hasattr(model, "routes"): value = value*model.member_count
            forward_calls += 1
        else:
            own = []
            for member, queries in calls:
                positive, negative = functional_forward(torch, model, parameters, features, support, queries,
                    training=True, stream=(streams, member), route=0 if four_stream_single else member, advance=pass_index == 2)
                own.append(own_loss(torch, positive, negative)); forward_calls += 1
            value = torch.stack(own).sum() if hasattr(model, "routes") else torch.stack(own).mean()
        if not bool(torch.isfinite(value)): raise FloatingPointError("Nonfinite matched ordinary task loss")
        value.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite matched ordinary gradient")
        disconnected = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        if disconnected: raise ValueError("Ordinary declared parameters disconnected from paid task pass: "+str(disconnected))
        optimizer.step(); losses.append(float(value.detach()))
    return {"task_losses": losses, "native_joint_Adam_updates": 3,
            "functional_forward_calls": forward_calls,
            "model_encoder_forward_count": 2*len(calls)+(4 if hasattr(model, "routes") else 1),
            "same_inner_examples_twice": True, "outer_pass_dropout_off": True,
            "inner_masks_replayed": True, "serving_adaptation": False}
