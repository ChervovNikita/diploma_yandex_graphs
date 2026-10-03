"""Prospective leading-axis/transpose/all-private higher-order native adjoints."""
from dataclasses import replace
from native_source.adapter import build_all_layer_polyformer
from native_source.tokens import build_uncached_mono_tokens, pullback_mono_gradients
from prepared_native_reference_checks import unequal_factors
from prepared_native_view_checks import fixture_views


def native_adjoint_axes_and_higher_order(spec):
    import torch
    # Nonsymmetric fixed operator exercises P^T. This is a helper fixture only;
    # it does not admit directed native graphs or a different scientific mask.
    dense = torch.tensor([[.2, .3, 0, 0, 0], [0, .1, .7, 0, 0], [0, 0, .4, .2, 0],
                          [.5, 0, 0, .2, 0], [0, .1, 0, 0, .6]], dtype=torch.float32, device="cpu")
    operator = dense.to_sparse()
    gradients = tuple(torch.sin(torch.arange(4 * 2 * 5 * 3, dtype=torch.float32, device="cpu") / (7 + k))
                      .reshape(4, 2, 5, 3) for k in range(3))
    actual = pullback_mono_gradients(gradients, operator)
    reference = torch.zeros_like(actual)
    power = torch.eye(5, dtype=torch.float32, device="cpu")
    for gradient in gradients:
        for member in range(4):
            for target in range(2):
                reference[member, target] += power.T @ gradient[member, target]
        power = power @ dense
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
    assert torch.equal(pullback_mono_gradients((gradients[0],), operator), gradients[0])
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long, device="cpu")
    assert torch.equal(pullback_mono_gradients(tuple(g.index_select(0, permutation) for g in gradients), operator),
                       actual.index_select(0, permutation))
    wrong_transpose = torch.zeros_like(actual)
    power = torch.eye(5, dtype=torch.float32, device="cpu")
    for gradient in gradients:
        for member in range(4):
            for target in range(2):
                wrong_transpose[member, target] += power @ gradient[member, target]
        power = power @ dense
    assert not torch.allclose(actual, wrong_transpose, atol=1e-6, rtol=1e-5)

    # Native K=0 and the admitted engineering representative K retain complete
    # original-node/feature axes; five selected score targets differ from M=4.
    for order in (0, spec.K):
        x0, _, edge_views, complete, _, _, _ = fixture_views(order)
        x = x0.detach().clone().requires_grad_()
        identity = dict(complete.banks)["native"].identity
        bank = build_uncached_mono_tokens(x, edge_views["native"], None, identity)
        local_spec = replace(spec, K=order)
        model = build_all_layer_polyformer(local_spec, seed=17,
                                          composition_receipt="synthetic_native_adjoint_only").eval()
        unequal_factors(model)
        model.begin_private_continuation(warm_bank_receipt="synthetic_adjoint_not_fitted_warm")
        direct_scores = model(torch.stack(bank.tokens, dim=1))
        cached = tuple(t.detach().clone().requires_grad_() for t in bank.tokens)
        cached_scores = model(torch.stack(cached, dim=1))
        assert torch.equal(direct_scores, cached_scores)
        targets = (0, 1, 6, 7, 19)
        direct, token_rows = [], [[] for _ in range(order + 1)]
        for member in range(4):
            for target in targets:
                score_class = target % 2  # synthetic raw-logit score, not a FoRDE choice
                direct.append(torch.autograd.grad(direct_scores[member, target, score_class], x,
                                                  create_graph=True, retain_graph=True)[0])
                token_gradients = torch.autograd.grad(cached_scores[member, target, score_class], cached,
                                                       create_graph=True, retain_graph=True)
                for rows, gradient in zip(token_rows, token_gradients):
                    rows.append(gradient)
        direct = torch.stack(direct).reshape(4, len(targets), 96, 3)
        leading = tuple(torch.stack(rows).reshape(4, len(targets), 96, 3) for rows in token_rows)
        pulled = pullback_mono_gradients(leading, bank.operator)
        looped = torch.stack([pullback_mono_gradients(tuple(g[m, t] for g in leading), bank.operator)
                              for m in range(4) for t in range(len(targets))]).reshape_as(pulled)
        assert torch.equal(pulled, looped)
        assert torch.allclose(direct, pulled, atol=1e-5, rtol=1e-4)
        assert torch.equal(pullback_mono_gradients(tuple(g.index_select(0, permutation) for g in leading), bank.operator),
                           pulled.index_select(0, permutation))
        names_and_private = tuple((n, p) for n, p in model.named_parameters() if p.requires_grad)
        parameters = tuple(p for _, p in names_and_private)
        direct_second = torch.autograd.grad(direct.square().sum(), parameters, retain_graph=True)
        pulled_second = torch.autograd.grad(pulled.square().sum(), parameters)
        assert len(parameters) == 2 * len(model.intermediate_sites)
        for (name, _), a, b in zip(names_and_private, direct_second, pulled_second):
            assert name.endswith((".R", ".S")) and torch.isfinite(a).all() and torch.isfinite(b).all()
            assert torch.allclose(a, b, atol=2e-5, rtol=2e-4)
        assert sum(float(g.abs().sum()) for g in pulled_second) > 0
        assert all(p.grad is None for p in model.parameters())
        try:
            build_uncached_mono_tokens(x.double(), edge_views["native"], None, identity)
        except ValueError:
            pass
        else:
            raise AssertionError("Float64 native token input was accepted")
        weights = torch.ones(edge_views["native"].shape[1], dtype=torch.float32, device="cpu", requires_grad=True)
        try:
            build_uncached_mono_tokens(x, edge_views["native"], weights, identity)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported edge-weight derivative was accepted")
