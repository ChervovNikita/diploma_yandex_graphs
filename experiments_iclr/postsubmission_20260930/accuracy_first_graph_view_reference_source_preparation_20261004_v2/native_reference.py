"""Ordinary native models: no ensemble modulation or neural-operation edits."""
from copy import deepcopy
from admission import reference_device
from contract import fit_spec, pass_plan, stage_at


def optimizer_names(model, optimizer):
    owners = {id(p): name for name, p in model.named_parameters()}
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer.param_groups) != 1 or len(params) != len(owners) or \
            len({id(p) for p in params}) != len(params) or {id(p) for p in params} != set(owners):
        raise ValueError("Every native parameter must be owned once")
    group = optimizer.param_groups[0]
    if (group["lr"], group["weight_decay"], group["betas"], group["eps"], group["amsgrad"]) != \
            (0.001, 0.0, (0.9, 0.999), 1e-8, False) or not all(p.requires_grad for p in params):
        raise ValueError("Native optimizer recipe/trainability differs")
    return [owners[id(p)] for p in params]


def primitives(model):
    return {"local_layers": len(model.local_convs), "global_layers": model.global_attn.num_layers,
            "heads": model.global_attn.heads, "hidden_per_head": model.global_attn.hidden_channels,
            "beta": model.beta, "pre_ln": model.pre_ln, "qk_shared": model.global_attn.qk_shared,
            "in_dropout": model.in_drop, "local_dropout": model.dropout,
            "global_dropout": model.global_attn.dropout}


def build(rt, kind, split, member=None, device="cpu", *, synthetic_recipe=None):
    device = reference_device(rt, device, rt.helpers.resolve_device)
    spec = fit_spec(kind, split, member)
    rt.helpers.seed_all(rt, spec["seed"], device)
    construction = {"selected_device": rt.helpers.device_provenance(device),
                    "after_seed": rt.helpers.cpu_tree(rt, rt.helpers.rng_state(rt, device))}
    recipe = rt.helpers.NATIVE_RECIPE if synthetic_recipe is None else synthetic_recipe
    model = rt.native.Polynormer(**recipe)
    construction["after_constructor_cpu"] = rt.helpers.cpu_tree(rt, rt.helpers.rng_state(rt, device))
    model.to(device)
    model.reset_parameters()
    model._global = False
    construction["after_device_reset"] = rt.helpers.cpu_tree(rt, rt.helpers.rng_state(rt, device))
    optimizer = rt.torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0,
                                    betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    optimizer_names(model, optimizer)
    return model, optimizer, construction


def check_gradients(rt, model):
    dormant = ("pred_local.",) if model._global else ("global_attn.", "ln.", "pred_global.")
    for name, p in model.named_parameters():
        if name.startswith(dormant):
            if p.grad is not None:
                raise ValueError("Inactive native gradient: " + name)
        elif p.grad is None or not rt.torch.isfinite(p.grad).all().item():
            raise ValueError("Disconnected/nonfinite native gradient: " + name)


def train_step(rt, model, optimizer, x, graphs, train_ids, labels, kind, update):
    if type(model._global) is not bool or model._global != stage_at(update):
        raise ValueError("Actual native stage differs")
    torch = rt.torch
    index = torch.tensor(train_ids, dtype=torch.long, device=x.device)
    targets = torch.tensor(labels, dtype=torch.long, device=x.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for graph, coefficient in pass_plan(kind):
        logits = model(x, graphs[graph])
        if logits.dtype != torch.float32 or tuple(logits.shape) != (x.shape[0], 5) or \
                not torch.isfinite(logits).all().item():
            raise ValueError("Complete finite FP32 native logits required")
        logp = torch.nn.functional.log_softmax(logits, dim=1).index_select(0, index)
        loss = torch.nn.functional.nll_loss(logp, targets) * coefficient
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite native CE")
        loss.backward()
        loss_value += float(loss.detach().cpu())
        del logits, logp, loss
    check_gradients(rt, model)
    optimizer.step()
    optimizer_names(model, optimizer)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()) or \
            any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item()
                for state in optimizer.state.values() for v in state.values()):
        raise ValueError("Nonfinite native parameter/Adam state")
    return loss_value


def evaluate(rt, model, x, graph):
    model.eval()
    with rt.torch.no_grad():
        logits = model(x, graph)
    if logits.dtype != rt.torch.float32 or tuple(logits.shape) != (x.shape[0], 5) or \
            not rt.torch.isfinite(logits).all().item():
        raise ValueError("Invalid native evaluation logits")
    return logits


def snapshot(rt, model, optimizer, device, selection, bindings):
    tree = rt.helpers.cpu_tree
    return {"schema": "graph-view-native-reference-state-v1", "bindings": deepcopy(bindings),
            "selection": deepcopy(selection), "model": tree(rt, model.state_dict()),
            "optimizer": tree(rt, optimizer.state_dict()), "optimizer_names": optimizer_names(model, optimizer),
            "primitives": primitives(model), "rng": tree(rt, rt.helpers.rng_state(rt, device)),
            "global_flags": {n: bool(m._global) for n, m in model.named_modules() if hasattr(m, "_global")},
            "modes": {n: m.training for n, m in model.named_modules()},
            "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
            "gradients": {n: tree(rt, p.grad) for n, p in model.named_parameters()}}


def restore_model_adam(rt, model, optimizer, image):
    if optimizer_names(model, optimizer) != image["optimizer_names"] or primitives(model) != image["primitives"]:
        raise ValueError("Native ownership/primitives differ")
    model.load_state_dict(rt.helpers.cpu_tree(rt, image["model"]), strict=True)
    optimizer.load_state_dict(rt.helpers.cpu_tree(rt, image["optimizer"]))
    optimizer_names(model, optimizer)


def restore(rt, model, optimizer, device, image, bindings):
    if image["schema"] != "graph-view-native-reference-state-v1" or image["bindings"] != bindings:
        raise ValueError("Native checkpoint source/context differs")
    modules, params = dict(model.named_modules()), dict(model.named_parameters())
    if set(modules) != set(image["modes"]) or set(params) != set(image["requires_grad"]) or \
            set(params) != set(image["gradients"]) or \
            {n for n, m in modules.items() if hasattr(m, "_global")} != set(image["global_flags"]):
        raise ValueError("Native state inventory differs")
    restore_model_adam(rt, model, optimizer, image)
    for n, value in image["global_flags"].items():
        modules[n]._global = value
    if type(model._global) is not bool or model._global != image["selection"]["global"]:
        raise ValueError("Selected native stage differs")
    for n, value in image["modes"].items():
        if type(value) is not bool:
            raise ValueError("Boolean native mode required")
        modules[n].training = value
    for n, p in params.items():
        p.requires_grad_(image["requires_grad"][n])
        gradient = image["gradients"][n]
        p.grad = None if gradient is None else gradient.to(p.device).clone()
    rt.helpers.restore_rng(rt, device, image["rng"])


def transition(rt, model, optimizer, device, selected_local, actual_update):
    if actual_update != 200 or model._global or selected_local["selection"]["global"]:
        raise ValueError("Actual200 selected-local transition required")
    live = rt.helpers.cpu_tree(rt, rt.helpers.rng_state(rt, device))
    restore_model_adam(rt, model, optimizer, selected_local)
    model._global = True
    rt.helpers.restore_rng(rt, device, live)
    return {"actual_update": 200, "restored_from": selected_local["selection"]["actual_update"],
            "live_RNG_preserved": True}
