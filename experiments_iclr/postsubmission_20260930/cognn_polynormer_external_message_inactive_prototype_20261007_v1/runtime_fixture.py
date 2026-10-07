"""Bounded CPU qualification SOURCE; disabled before any Torch/model import.

No dataset, training runner, server, CLI enable switch, or scientific grid.
Future invocation: run(pinned_native_path, factor_source_path), after an explicit
review of runtime activation. Current task executes only static_checks.py.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import importlib.util
import json

from prototype import ActionNoise, build_runtime

NATIVE_SHA256 = "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"


def run(pinned_native_path, factor_source_path):
    api = build_runtime()  # Inactive guard raises BEFORE numerical imports.
    import torch
    from torch_geometric.nn import GATConv

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    native_path = Path(pinned_native_path)
    if sha256(native_path.read_bytes()).hexdigest() != NATIVE_SHA256:
        raise RuntimeError("Native Polynormer source pin mismatch")
    bindings = json.loads((Path(__file__).parent / "SOURCE_BINDINGS.json").read_text())
    factor_pin = next(r for r in bindings["files"] if r["role"] == "fixture_factor_source")
    if sha256(Path(factor_source_path).read_bytes()).hexdigest() != factor_pin["sha256"]:
        raise RuntimeError("Existing factor fixture source pin mismatch")
    native_source = load(native_path, "fixture_pinned_polynormer")
    factors = load(factor_source_path, "fixture_existing_factors")
    dtype = torch.float64
    close = lambda a, b: torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)

    # Directed edges, one empty row, and a duplicate self edge; support is kept.
    edge = torch.tensor([[0, 1, 2, 0, 0, 1, 2, 0],
                         [1, 2, 0, 2, 0, 1, 2, 0]], dtype=torch.long)
    x = torch.tensor([[1., 2.], [2., 1.], [3., 2.], [4., 1.]], dtype=dtype)
    conv = GATConv(2, 1, heads=2, add_self_loops=False, bias=False).to(dtype)
    with torch.no_grad():
        conv.lin.weight.fill_(0.5)
        conv.att_src.zero_()
        conv.att_dst.zero_()
    conv.eval()
    identities = tuple(id(p) for p in conv.parameters())
    ones = x.new_ones(x.shape[0])
    all_keep, alpha, gate = api.gat(conv, x, edge, ones, ones)
    close(all_keep, conv(x, edge))
    assert identities == tuple(id(p) for p in conv.parameters())

    def manual(alpha, gate):
        projected = conv.lin(x).view(x.shape[0], conv.heads, conv.out_channels)
        messages = alpha.unsqueeze(-1) * projected.index_select(0, edge[0])
        weighted = messages * gate[:, None, None]
        out = torch.index_add(x.new_zeros((x.shape[0], conv.heads, conv.out_channels)),
                              0, edge[1], weighted)
        return out.reshape(x.shape[0], -1)

    patterns = [
        (ones, ones),
        (torch.tensor([0., 1., 1., 0.], dtype=dtype), ones),  # receive only
        (ones, torch.tensor([1., 0., 1., 0.], dtype=dtype)),  # send only
        (torch.zeros_like(ones), torch.zeros_like(ones)),  # external isolate
        (torch.tensor([0., 1., 1., 0.], dtype=dtype),
         torch.tensor([1., 0., 1., 0.], dtype=dtype)),  # broadcast/listen/standard/isolate
    ]
    for receive, send in patterns:
        out, original_alpha, actual_gate = api.gat(conv, x, edge, receive, send)
        expected = torch.where(edge[0] == edge[1], ones[edge[0]],
                               receive[edge[1]] * send[edge[0]])
        close(actual_gate, expected)
        close(original_alpha, alpha)
        close(out, manual(alpha, expected))
        assert bool(torch.isfinite(out).all())
        assert out[3].abs().sum().item() == 0  # Empty GAT row, before own/root map.

    # ST hard-on/off cases. Compare the PRODUCT's declared first-order surrogate:
    # derivative of each soft gate times the OTHER gate's HARD forward value.
    # Comparing to a product of two wholly soft gates would be a different rule.
    for receive_on, send_on in ((True, True), (False, True), (True, False), (False, False)):
        lr = torch.zeros((4, 2), dtype=dtype, requires_grad=True)
        ls = torch.zeros((4, 2), dtype=dtype, requires_grad=True)
        nr = torch.tensor([[2., -2.] if receive_on else [-2., 2.]], dtype=dtype).expand(4, 2)
        ns = torch.tensor([[2., -2.] if send_on else [-2., 2.]], dtype=dtype).expand(4, 2)
        tau = x.new_full((4, 1), 0.75)
        r, sr = api.st_keep(lr, tau, nr)
        s, ss = api.st_keep(ls, tau, ns)
        out, native_alpha, actual_gate = api.gat(conv, x, edge, r, s)
        actual_grads = torch.autograd.grad(out.sum(), (lr, ls), retain_graph=True)
        u, v = edge
        hr, hs = r.detach(), s.detach()
        surrogate = (hr[v] * hs[u] + (sr[v] - sr[v].detach()) * hs[u]
                     + hr[v] * (ss[u] - ss[u].detach()))
        surrogate = torch.where(u == v, torch.ones_like(surrogate), surrogate)
        expected_grads = torch.autograd.grad(manual(native_alpha.detach(), surrogate).sum(), (lr, ls))
        for actual, expected in zip(actual_grads, expected_grads):
            close(actual, expected)
            assert bool(torch.isfinite(actual).all())
        assert (actual_grads[0].abs().sum().item() > 0) == send_on
        assert (actual_grads[1].abs().sum().item() > 0) == receive_on

    # Tiny complete native model, existing factors and all four member contexts.
    model = native_source.Polynormer(2, 2, 3, local_layers=2, global_layers=1,
                                    heads=2, in_dropout=0.2, dropout=0.15,
                                    global_dropout=0.1).to(dtype)
    factors.install_factors(model, 4)  # Native only; policies constructed AFTER this.
    native_ids = tuple(id(p) for p in model.parameters())
    before_rng = torch.get_rng_state().clone()
    policies = api.make_policies(4, 4, 0.5, 17, dtype=dtype, device="cpu")
    assert torch.equal(before_rng, torch.get_rng_state())
    assert set(native_ids).isdisjoint(id(p) for p in policies.parameters())
    prepared = torch.tensor([[0, 1, 2, 0, 0, 1, 2, 3],
                             [1, 2, 0, 2, 0, 1, 2, 3]], dtype=torch.long)

    def complete(use_graft):
        outputs = []
        for member in range(4):
            with factors.member_context(model, member):
                outputs.append(api.forward(model, policies, x, prepared, member=member,
                                           view=0, force_all_keep=True).logits
                               if use_graft else model(x, prepared))
        return torch.stack(outputs)

    def compare_nested(a, b):
        if isinstance(a, torch.Tensor):
            close(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare_nested(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b):
                compare_nested(left, right)
        else:
            assert a == b

    for global_mode in (False, True):
        model._global = global_mode
        model.train()
        initial = deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        snapshots = []
        for use_graft in (False, True):
            model.load_state_dict(initial)
            model.zero_grad(set_to_none=True)
            torch.set_rng_state(rng)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            logits = complete(use_graft)
            logits.square().mean().backward()
            gradients = {n: None if p.grad is None else p.grad.detach().clone()
                         for n, p in model.named_parameters()}
            optimizer.step()  # Qualification fixture only; never run in this packet.
            snapshots.append((logits.detach().clone(), gradients,
                              deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())))
        compare_nested(snapshots[0], snapshots[1])
        assert native_ids == tuple(id(p) for p in model.parameters())

    # Distinct live tensors across four members/two views/two local layers.
    model.eval()
    noise = ActionNoise(x.new_tensor([[2., -2.]]).expand(4, 2),
                        x.new_tensor([[2., -2.]]).expand(4, 2))
    panel = {(m, v, layer): noise for m in range(4) for v in range(2) for layer in range(2)}
    records = []
    for member in range(4):
        for view in range(2):
            with factors.member_context(model, member):
                records.append(api.forward(model, policies, x, prepared, member=member,
                                           view=view, noise_panel=panel))
    gates = [g for record in records for g in record.gates]
    for name in ("receive", "send", "edge_gate", "attention"):
        assert len({id(getattr(g, name)) for g in gates}) == len(gates)
    private = tuple(policies.parameters())
    losses = [record.logits.square().mean()
              + (i + 1) * sum(g.receive.sum() + g.send.sum() for g in record.gates)
              for i, record in enumerate(records)]
    # Direct gate terms here test graph retention, not a scientific objective.
    separate = [torch.autograd.grad(loss, private, retain_graph=True, allow_unused=True)
                for loss in losses]
    joint = torch.autograd.grad(sum(losses), private, allow_unused=True)
    for index, actual in enumerate(joint):
        terms = [row[index] for row in separate if row[index] is not None]
        if not terms:
            assert actual is None
        else:
            close(actual, sum(terms))
    assert native_ids == tuple(id(p) for p in model.parameters())
    return {"all_keep_outputs_gradients_adam_local_global": "checked",
            "actions_support_empty_rows": "checked", "ST_product_surrogate": "checked",
            "four_members_two_views_tensor_lifetime": "checked"}


if __name__ == "__main__":
    raise SystemExit("Inactive fixture source; no runtime or model import performed")
