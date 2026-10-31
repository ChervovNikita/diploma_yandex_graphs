"""Unexecuted factories over pinned Model/AR APIs; backend is injected.

No module-scope backend import, model construction, RNG draw or training.
Initialization choices are supplied only by the future root scientific freeze.
"""
import hashlib
import math
from functools import wraps


FACTOR_NAMES = ("r_x", "s_x", "r_h", "s_h", "r_o", "s_o")
RECURRENT_FACTOR_NAMES = ("r_x", "s_x", "r_h", "s_h")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def full_validation(function):
    @wraps(function)
    def measured(*args, **kwargs):
        with globals()["__ar_diagnostic_meter__"].block("explicit/" + function.__name__):
            return function(*args, **kwargs)
    return measured


def role_seed(model_seed, role):
    text = "ar_fitted_screen_v1|" + str(model_seed) + "|" + role
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def generator(torch, model_seed, role):
    return torch.Generator(device="cpu").manual_seed(role_seed(model_seed, role))


def initialize(torch, tensor, function, stream):
    # Pinned Torch 2.1 compatibility: restore caller RNG and advance only stream.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(stream.get_state())
        function(tensor)
        stream.set_state(torch.get_rng_state())


def matrices(torch, model_seed, hidden, role):
    wx = torch.empty(3 * hidden, 18)
    wh = torch.empty(3 * hidden, hidden)
    wo = torch.empty(2, hidden)
    gx = generator(torch, model_seed, role + "/input")
    gh = generator(torch, model_seed, role + "/recurrent")
    go = generator(torch, model_seed, role + "/output")
    for gate in range(3):
        initialize(torch, wx[gate * hidden:(gate + 1) * hidden], torch.nn.init.xavier_uniform_, gx)
        initialize(torch, wh[gate * hidden:(gate + 1) * hidden], torch.nn.init.orthogonal_, gh)
    initialize(torch, wo, torch.nn.init.xavier_uniform_, go)
    return wx, wh, wo


def private_tables(torch, model_seed, members, hidden, initialization):
    require(initialization["kind"] == "independent_gaussian", "Root-frozen private initialization required")
    sigma = initialization["standard_deviation"]
    require(isinstance(sigma, (float, int)) and math.isfinite(sigma) and sigma > 0,
            "A positive finite scale must be frozen before any draw")
    require(members > 1, "Private pairing tables are for multiple members")
    tables = {}
    for name, size in (("b_x", 3 * hidden), ("b_h", 3 * hidden), ("b_o", 2), ("h0", hidden)):
        stream = generator(torch, model_seed, "paired_private/h" + str(hidden) + "/M" + str(members) + "/" + name)
        value = torch.randn((members, size), generator=stream) * sigma
        require(any(not torch.equal(value[0], value[m]) for m in range(1, members)),
                "Private member initialization unexpectedly collapsed")
        tables[name] = value
    return tables


def decoder_template(torch, arm, model_seed, initialization):
    M, h = arm["members"], arm["hidden"]
    role = "decoder/h" + str(h)
    wx, wh, wo = matrices(torch, model_seed, h, role)
    ro = torch.empty(M, h).bernoulli_(0.5, generator=generator(torch, model_seed, role + "/output_r/M" + str(M))).mul_(2).sub_(1)
    if arm["family"] == "shared":
        tensors = {"W_x": wx, "W_h": wh, "b_x": torch.zeros(3 * h),
                   "b_h": torch.zeros(3 * h), "h0": torch.zeros(h), "b_o": torch.zeros(M, 2)}
        if arm["mode"] == "ordinary_heads":
            tensors["W_o"] = torch.stack([wo * ro[m][None, :] for m in range(M)])
        else:
            require(arm["mode"] == "factored_heads", "Pinned shared-head mode required")
            tensors.update(W_o=wo, r_o=ro, s_o=torch.ones(M, 2))
        return tensors
    tensors = {"W_x": wx, "W_h": wh, "W_o": wo, "b_x": torch.zeros(M, 3 * h),
               "b_h": torch.zeros(M, 3 * h), "b_o": torch.zeros(M, 2), "h0": torch.zeros(M, h)}
    if arm["mode"] == "factored":
        tensors.update(private_tables(torch, model_seed, M, h, initialization))
        rx = torch.empty(M, 18).bernoulli_(0.5, generator=generator(torch, model_seed, role + "/input_r/M" + str(M))).mul_(2).sub_(1)
        tensors.update(r_x=rx, s_x=torch.ones(M, 3 * h), r_h=torch.ones(M, h),
                       s_h=torch.ones(M, 3 * h), r_o=ro, s_o=torch.ones(M, 2))
        if arm["fixed_factor_names"]:
            for name in arm["fixed_factor_names"]:
                tensors[name] = torch.ones_like(tensors[name])
    elif arm["mode"] == "untied":
        private = [matrices(torch, model_seed, h, role + "/untied/" + str(m)) for m in range(M)]
        tensors.update(W_x=torch.stack([row[0] for row in private]),
                       W_h=torch.stack([row[1] for row in private]), W_o=torch.stack([row[2] for row in private]))
    else:
        require(arm["mode"] == "single" and M == 1, "Pinned private mode required")
    return tensors


def initial_encoder_state(torch, Model, model_seed, recipe):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(role_seed(model_seed, "source_constructor"))
        core = Model(**recipe).to(device="cpu", dtype=torch.float64)
    stream = generator(torch, model_seed, "source_encoder_xavier")
    with torch.no_grad():
        for name, parameter in core.named_parameters():
            if parameter.ndim >= 2:
                initialize(torch, parameter, torch.nn.init.xavier_uniform_, stream)
            else:
                require(name.endswith("bias"), "Pinned Identity-normalized encoder changed")
                parameter.zero_()
    require(sum(p.numel() for p in core.parameters()) == 2992, "Pinned encoder count changed")
    return {name: value.detach().clone() for name, value in core.state_dict().items()}


@full_validation
def parameter_inventory(torch, encoder, decoder):
    parameters = tuple(("encoder/" + name, p) for name, p in encoder.named_parameters()) + tuple(("decoder/" + name, p) for name, p in decoder.named_parameters())
    addresses = [p.untyped_storage().data_ptr() for _, p in parameters]
    require(len(addresses) == len(set(addresses)), "Parameter tensors alias storage")
    require(all(p.dtype == torch.float64 and p.device.type == "cpu" and p.is_contiguous() for _, p in parameters),
            "Pinned CPU float64 contiguous storage required")
    records = [{"name": name, "shape": list(p.shape), "stored_elements": p.numel(),
                "trainable": p.requires_grad} for name, p in parameters]
    return parameters, {"tensors": records, "stored_elements": sum(p.numel() for _, p in parameters),
                        "trainable_elements": sum(p.numel() for _, p in parameters if p.requires_grad),
                        "fixed_elements": sum(p.numel() for _, p in parameters if not p.requires_grad)}


def construct_decoder(torch, modules, arm, template, audit):
    tensors = {name: value.detach().clone() for name, value in template.items()}
    if arm["family"] == "shared":
        ledger = modules["shared_work"].SharedTransitionLedger(audit)
        decoder = modules["shared"].make_shared_transition_decoder(torch_api=torch, head_mode=arm["mode"],
            n_branches=arm["members"], embedding_dimension=16, hidden_dimension=arm["hidden"], parameter_tensors=tensors, ledger=ledger)
    else:
        ledger = modules["work"].WorkLedger(audit)
        decoder = modules["private"].make_recurrent_decoder(torch_api=torch, mode=arm["mode"],
            n_branches=arm["members"], embedding_dimension=16, hidden_dimension=arm["hidden"], parameter_tensors=tensors, ledger=ledger)
    if arm["fixed_factor_names"]:
        require(arm["family"] == "private" and arm["mode"] == "factored", "Identity must keep private recurrence/state")
        require(tuple(arm["fixed_factor_names"]) in (FACTOR_NAMES, RECURRENT_FACTOR_NAMES), "Root-frozen factor ablation required")
        for name in arm["fixed_factor_names"]:
            parameter = getattr(decoder, name)
            require(torch.equal(parameter, torch.ones_like(parameter)), "Identity factor was not initialized to one")
            parameter.requires_grad_(False)
        require(all(getattr(decoder, name).requires_grad for name in ("b_x", "b_h", "b_o", "h0")),
                "Identity private biases and initial states must remain trainable")
    return decoder, ledger


def construct_system(torch, modules, Model, arm, model_seed, encoder_state, template, audit, recipe):
    decoder, ledger = construct_decoder(torch, modules, arm, template, audit)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(role_seed(model_seed, "source_constructor"))
        core = Model(**recipe).to(device="cpu", dtype=torch.float64)
    core.load_state_dict(encoder_state)
    encoder = modules["encoder"].make_static_source_encoder(torch_api=torch, source_core=core, audit=audit, ledger=ledger)
    parameters, inventory = parameter_inventory(torch, encoder, decoder)
    require(inventory["stored_elements"] == 2992 + arm["expected_decoder_stored_elements"], "Stored arm count changed")
    require(inventory["trainable_elements"] == 2992 + arm["expected_decoder_trainable_elements"], "Trainable arm count changed")
    return encoder, decoder, ledger, parameters, inventory


@full_validation
def assert_fixed_identity(torch, decoder, arm):
    if arm["fixed_factor_names"]:
        for name in arm["fixed_factor_names"]:
            parameter = getattr(decoder, name)
            require(not parameter.requires_grad and parameter.grad is None and torch.equal(parameter, torch.ones_like(parameter)),
                    "Identity factor changed or entered the gradient/optimizer path")
