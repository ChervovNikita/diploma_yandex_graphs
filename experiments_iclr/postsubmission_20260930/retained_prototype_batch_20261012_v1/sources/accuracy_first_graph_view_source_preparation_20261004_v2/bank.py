"""Untied ownership only: every neural forward comes from byte-bound v6."""
from copy import deepcopy
import random

NATIVE_RECIPE = dict(in_channels=300, hidden_channels=256, out_channels=5,
                     local_layers=10, global_layers=1, in_dropout=0.2,
                     dropout=0.3, global_dropout=0.3, heads=2, beta=-1, pre_ln=False)


def cpu_tree(rt, value):
    if isinstance(value, rt.torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(rt, v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(rt, v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(rt, v) for v in value)
    return deepcopy(value)


def resolve_device(rt, device):
    device = rt.torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = rt.torch.device("cuda", rt.torch.cuda.current_device())
    return device


def device_provenance(device):
    return {"type": device.type, "index": device.index, "canonical": str(device)}


def seed_all(rt, seed, device):
    device = resolve_device(rt, device)
    random.seed(seed)
    rt.numpy.random.seed(seed)
    rt.torch.manual_seed(seed)
    if device.type == "cuda":
        rt.torch.cuda.manual_seed_all(seed)


def rng_state(rt, device):
    device = resolve_device(rt, device)
    state = rt.numpy.random.get_state()
    return {"selected_device": device_provenance(device), "python": random.getstate(),
            "numpy": {"name": state[0], "keys": state[1].tolist(), "position": int(state[2]),
                      "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu": rt.torch.get_rng_state().clone(),
            "cuda": rt.torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None}


def restore_rng(rt, device, state):
    device = resolve_device(rt, device)
    if state.get("selected_device") != device_provenance(device):
        raise ValueError("Selected-device RNG provenance differs")
    if device.type == "cuda" and state["cuda"] is None:
        raise ValueError("Missing selected-device CUDA stream")
    if device.type != "cuda" and state["cuda"] is not None:
        raise ValueError("Unexpected CUDA stream for non-CUDA device")
    random.setstate(state["python"])
    n = state["numpy"]
    rt.numpy.random.set_state((n["name"], rt.numpy.asarray(n["keys"], dtype=rt.numpy.uint32),
                               n["position"], n["has_gauss"], n["cached_gaussian"]))
    rt.torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        rt.torch.cuda.set_rng_state(state["cuda"], device)


def untie(rt, tied):
    """Deepcopy each interior/W and retain exactly the corresponding factor row.

    No constructor, reset or random draw follows the initialized tied image.
    The one-member clones still call the immutable adapter's forward_member.
    """
    torch = rt.torch

    class UntiedBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.members = 4
            copies = []
            for member in range(4):
                family = deepcopy(tied)
                family.members = 1
                for site in (family.stem, family.local_head, family.global_head):
                    site.members = 1
                    for name in ("R", "S", "B"):
                        setattr(site, name, torch.nn.Parameter(getattr(site, name)[member:member + 1].detach().clone()))
                copies.append(family)
            self.copies = torch.nn.ModuleList(copies)

        def set_global_stage(self, enabled):
            for family in self.copies:
                family.set_global_stage(enabled)

        def forward_member(self, features, edge_index, member):
            if member not in range(4):
                raise IndexError("Member outside bank")
            return self.copies[member].forward_member(features, edge_index, 0)

        def forward(self, features, edge_index):
            return torch.stack([self.forward_member(features, edge_index, m) for m in range(4)])

    return UntiedBank()


def build_bank(rt, condition, seed, device="cpu", *, synthetic_recipe=None):
    from schedule import CONDITIONS
    if condition not in CONDITIONS:
        raise ValueError("Unknown bank condition")
    device = resolve_device(rt, device)
    seed_all(rt, seed, device)
    construction = {"selected_device": device_provenance(device),
                    "after_seed": cpu_tree(rt, rng_state(rt, device))}
    recipe = NATIVE_RECIPE if synthetic_recipe is None else synthetic_recipe
    native = rt.native.Polynormer(**recipe)
    construction["after_constructor_cpu"] = cpu_tree(rt, rng_state(rt, device))
    native.to(device)
    native.reset_parameters()
    native._global = False
    construction["after_device_reset"] = cpu_tree(rt, rng_state(rt, device))
    tied = rt.boundary.PolynormerBoundaryFamily(native, members=4)
    construction["after_factor_wrap"] = cpu_tree(rt, rng_state(rt, device))
    model = untie(rt, tied) if condition == "untied_persistent" else tied
    construction["after_ownership_clone"] = cpu_tree(rt, rng_state(rt, device))
    optimizer = rt.torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0,
                                    betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    optimizer_names(model, optimizer)
    return model, optimizer, construction


def families(model):
    return tuple(model.copies) if hasattr(model, "copies") else (model,)


def stage(model):
    values = [family.core._global for family in families(model)]
    if any(type(v) is not bool for v in values) or len(set(values)) != 1:
        raise ValueError("All bank core stage flags must agree")
    return values[0]


def set_stage(model, enabled):
    model.set_global_stage(bool(enabled))
    if stage(model) is not bool(enabled):
        raise ValueError("Stage assignment failed")


def optimizer_names(model, optimizer):
    owners = {id(p): name for name, p in model.named_parameters()}
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer.param_groups) != 1 or len(parameters) != len(owners) or \
            {id(p) for p in parameters} != set(owners) or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Adam must own every bank parameter exactly once in one group")
    group = optimizer.param_groups[0]
    if (group["lr"], group["weight_decay"], group["betas"], group["eps"], group["amsgrad"]) != \
            (0.001, 0.0, (0.9, 0.999), 1e-8, False):
        raise ValueError("Adam recipe changed")
    if not all(p.requires_grad for p in parameters):
        raise ValueError("Common and private parameters must remain trainable")
    return [owners[id(p)] for p in parameters]


def check_gradients(rt, model):
    dormant = set()
    for family in families(model):
        sites = (family.local_head,) if stage(model) else \
                (family.global_head, family.core.global_attn, family.core.ln)
        dormant.update(id(p) for site in sites for p in site.parameters())
    for name, p in model.named_parameters():
        if id(p) in dormant:
            if p.grad is not None:
                raise ValueError("Inactive-stage gradient: " + name)
        elif p.grad is None or not rt.torch.isfinite(p.grad).all().item():
            raise ValueError("Disconnected/nonfinite active gradient: " + name)


def primitives(model):
    records = []
    for family in families(model):
        core = family.core
        records.append({"members": family.members, "local_layers": len(core.local_convs),
                        "global_layers": core.global_attn.num_layers,
                        "hidden_per_head": core.global_attn.hidden_channels, "heads": core.global_attn.heads,
                        "beta": core.beta, "pre_ln": core.pre_ln,
                        "input_dropout": core.in_drop, "local_dropout": core.dropout,
                        "global_dropout": core.global_attn.dropout, "qk_shared": core.global_attn.qk_shared})
    return {"bank_members": model.members, "untied": hasattr(model, "copies"), "families": records}


def snapshot(rt, model, optimizer, device, selection, bindings):
    return {"schema": "accuracy-first-bank-state-v2", "bindings": deepcopy(bindings),
            "selection": deepcopy(selection), "model": cpu_tree(rt, model.state_dict()),
            "optimizer": cpu_tree(rt, optimizer.state_dict()),
            "optimizer_parameter_names": optimizer_names(model, optimizer),
            "primitives": primitives(model),
            "rng": cpu_tree(rt, rng_state(rt, device)),
            "global_flags": {n: bool(m._global) for n, m in model.named_modules() if hasattr(m, "_global")},
            "modes": {n: m.training for n, m in model.named_modules()},
            "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
            "gradients": {n: cpu_tree(rt, p.grad) for n, p in model.named_parameters()}}


def restore_model_adam(rt, model, optimizer, image):
    if optimizer_names(model, optimizer) != image["optimizer_parameter_names"] or \
            primitives(model) != image["primitives"]:
        raise ValueError("Optimizer ownership or native primitive configuration differs")
    model.load_state_dict(cpu_tree(rt, image["model"]), strict=True)
    optimizer.load_state_dict(cpu_tree(rt, image["optimizer"]))
    optimizer_names(model, optimizer)


def restore(rt, model, optimizer, device, image, bindings):
    if image["schema"] != "accuracy-first-bank-state-v2" or image["bindings"] != bindings:
        raise ValueError("Source/protocol/views/schedule/checkpoint binding differs")
    modules, parameters = dict(model.named_modules()), dict(model.named_parameters())
    flag_names = {n for n, m in modules.items() if hasattr(m, "_global")}
    if set(modules) != set(image["modes"]) or set(parameters) != set(image["requires_grad"]) or \
            set(parameters) != set(image["gradients"]) or flag_names != set(image["global_flags"]):
        raise ValueError("Exact module/parameter/state inventory required")
    restore_model_adam(rt, model, optimizer, image)
    for name, value in image["global_flags"].items():
        if type(value) is not bool:
            raise ValueError("Boolean stage flag required")
        modules[name]._global = value
    if stage(model) != image["selection"]["global"]:
        raise ValueError("Selected stage differs")
    for name, value in image["modes"].items():
        if type(value) is not bool:
            raise ValueError("Boolean module mode required")
        modules[name].training = value
    for name, p in parameters.items():
        p.requires_grad_(image["requires_grad"][name])
        grad = image["gradients"][name]
        p.grad = None if grad is None else grad.to(p.device).clone()
    restore_rng(rt, device, image["rng"])


def transition(rt, model, optimizer, device, selected_local, clock):
    clock.require_transition()
    if stage(model) or selected_local["selection"]["global"]:
        raise ValueError("Selected-local transition required")
    live_rng = cpu_tree(rt, rng_state(rt, device))
    restore_model_adam(rt, model, optimizer, selected_local)
    set_stage(model, True)
    # Restore explicitly as an invariant, even though load_state_dict consumes no RNG.
    restore_rng(rt, device, live_rng)
    return {"restored_from_actual_update": selected_local["selection"]["actual_update"],
            "live_actual_clock": clock.record(), "live_RNG_preserved": True, "global": True}
