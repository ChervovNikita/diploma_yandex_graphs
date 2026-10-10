"""Exact-order native prefix/exchange/tail forward; no runner or numerical import.

The existing full-member replay is not used. All four prefixes remain live
through the exchange; there is no detach, frozen teacher or cached prediction.
Native source files and native module forwards are never modified.
"""
from contextlib import ExitStack

MODEL_CONFIG = {
    "beta": -1, "dropout": .5, "global_dropout": .5, "global_layers": 2,
    "heads": 1, "hidden_channels": 512, "in_channels": 300,
    "in_dropout": .5, "local_layers": 7, "out_channels": 10, "pre_ln": False,
}
NATIVE_SHA = "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"
CORE_SHA = {
    "factors.py": "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3",
    "models.py": "2be6d872e962d9fdd341883b0d1060170fe0a176f7a8d6829d988fc6f8e7f9c3",
    "objectives.py": "9208e677ab5cfb286862216991453f8592a1d373120019a6e2ff14c38fad5bcf",
    "selection.py": "c48c08adfea5d08b945a38e3edf55522d6dbfa44418bf902fb46700a8cc11d20",
}
BLOCK_NAME = "live_route_block"


def require_interface(session):
    if session.task != "wikics" or session.config["model"] != MODEL_CONFIG:
        raise ValueError("Exact native width512 WikiCS interface required")
    if session.native_provenance != {"polynormer_model_sha256": NATIVE_SHA}:
        raise ValueError("Exact pinned native Polynormer source required")
    if session.core_provenance != CORE_SHA:
        raise ValueError("Exact original public core required")
    model = session.model
    if model.independent or model.members != 4 or len(model.models) != 1:
        raise ValueError("One shared body with four private-factor routes")
    if model.contrastive:
        raise ValueError("Plain own F only; no auxiliary or relation-J policy")
    wrapper = model.models[0]
    body = wrapper.body
    if body.pre_ln or len(body.local_convs) != 7 or body.global_attn.num_layers != 2:
        raise ValueError("Exact native local/global path required")
    if not hasattr(wrapper, BLOCK_NAME) or len(session.streams) != 4:
        raise ValueError("Fresh registered block and original four RNG streams")
    return wrapper, body, getattr(wrapper, BLOCK_NAME)


def _run_route(session, member, operation):
    """Sequentially select existing factors/scorers and resume this RNG stream.

    Prefix and tail use the same persistent per-member stream. Mutable bank
    selectors are never used concurrently, and context closure restores them.
    """
    torch = session.torch
    wrapper = session.model.models[0]
    stream = session.streams[member]
    old_member = getattr(wrapper, "member", 0)
    if hasattr(wrapper, "member"):
        wrapper.member = member
    try:
        with torch.random.fork_rng(devices=session.cuda_devices):
            torch.set_rng_state(stream["cpu"])
            if session.cuda_index is not None:
                torch.cuda.set_rng_state(stream["cuda"], session.cuda_index)
            with ExitStack() as stack:
                stack.enter_context(session.core["models"].member_context(wrapper, member))
                controller = getattr(session, "private_local_attention", None)
                if controller is not None:
                    if controller.body is not wrapper.body or controller.members != 4:
                        raise ValueError("Existing matching private-scorer controller required")
                    stack.enter_context(controller.member_context(member))
                result = operation()
            stream["cpu"] = torch.get_rng_state()
            if session.cuda_index is not None:
                stream["cuda"] = torch.cuda.get_rng_state(session.cuda_index)
            return result
    finally:
        if hasattr(wrapper, "member"):
            wrapper.member = old_member


def _local_block(torch, body, x, edge_index, i):
    """Pinned native lines156-167, in their original operation order."""
    F = torch.nn.functional
    if body.pre_ln:
        x = body.pre_lns[i](x)
    h = body.h_lins[i](x)
    h = F.relu(h)
    x = body.local_convs[i](x, edge_index) + body.lins[i](x)
    x = F.relu(x)
    x = F.dropout(x, p=body.dropout, training=body.training)
    if body.beta < 0:
        beta = F.sigmoid(body.betas[i]).unsqueeze(0)
    else:
        beta = body.betas[i].unsqueeze(0)
    return (1-beta)*body.lns[i](h*x) + beta*x


def forward(session, batch):
    """Return original [member, selected object, class/representation] axes.

    One new site is between local block0 and block1. The original first term
    in x_local is retained; all six remaining local blocks and native optional
    two-block global path are evaluated for every route, with its own head.
    """
    wrapper, body, block = require_interface(session)
    torch = session.torch
    if tuple(batch["x"].shape) != (11701, 300):
        raise ValueError("The complete11701-node feature matrix is required")
    if tuple(batch["edge_index"].shape) != (2, 442907):
        raise ValueError("The unchanged complete442907-edge support is required")
    F = torch.nn.functional
    prefixes = []
    for member in range(4):
        def prefix():
            x = F.dropout(batch["x"], p=body.in_drop, training=body.training)
            x = body.lin_in(x)
            x = F.dropout(x, p=body.dropout, training=body.training)
            x_local = 0
            x = _local_block(torch, body, x, batch["edge_index"], 0)
            x_local = x_local + x
            return x, x_local
        prefixes.append(_run_route(session, member, prefix))
    raw = torch.stack([row[0] for row in prefixes], dim=1)
    exchanged = block(raw)
    rows = []
    for member in range(4):
        def tail():
            x = exchanged[:, member, :]
            x_local = prefixes[member][1]
            for i in range(1, 7):
                x = _local_block(torch, body, x, batch["edge_index"], i)
                x_local = x_local + x
            if body._global:
                representation = body.global_attn(body.ln(x_local))
                logits = body.pred_global(representation)
            else:
                representation = x_local
                logits = body.pred_local(representation)
            ids = batch["ids"]
            return logits[ids], representation[ids]
        rows.append(_run_route(session, member, tail))
    return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])
