"""CPU engineering checks only; no scientific data or predictive claim."""
import json
import time
import torch


def run(module):
    started = time.monotonic()
    torch.set_num_threads(1)
    dtype = torch.float64
    generator = torch.Generator().manual_seed(106005)
    cfg = module.Config()
    assignment_checks = []
    for count, members in [(1, 4), (6, 4), (9, 4), (6, 1)]:
        raw = torch.randn(count, members, generator=generator, dtype=dtype)
        cost, _, _ = module._normalized_cost(raw, cfg.response_epsilon)
        affinity = torch.zeros(count, count, dtype=dtype)
        for i in range(count - 1):
            affinity[i, i + 1] = affinity[i + 1, i] = 1
        affinity /= 1 + affinity.sum(1).max()
        laplacian = torch.diag(affinity.sum(1)) - affinity
        q = module._assignment_map(cost, laplacian, cfg)
        uniform = torch.full_like(q, 1 / members)

        def objective(value):
            return ((cost * value).sum()
                    + cfg.entropy * (value * (members * value).log()).sum()
                    + cfg.graph / 2 * (value * (laplacian @ value)).sum())

        row_error = (q.sum(1) - 1).abs().max().item()
        column_error = (q.sum(0) - count / members).abs().max().item()
        descent = (objective(q) - objective(uniform)).item()
        assert q.min() > 0 and row_error < 2e-13 and column_error < 2e-13
        assert descent <= 2e-13
        assignment_checks.append(dict(count=count, members=members,
                                      row_error=row_error, column_error=column_error,
                                      minimum=q.min().item(), objective_change=descent))

    # A deterministic nonlinear tensor fixture checks the mixed derivative only.
    # It is not a model comparison or a representative dataset experiment.
    x = torch.randn(12, 4, generator=generator, dtype=dtype)
    theta = {'weight': torch.randn(4, 5, generator=generator, dtype=dtype) / 2,
             'bias': torch.randn(5, generator=generator, dtype=dtype) / 5}
    phis = tuple({'head': torch.randn(5, 3, generator=generator, dtype=dtype) / 2,
                  'bias': torch.randn(3, generator=generator, dtype=dtype) / 5}
                 for _ in range(4))

    def forward(core, private):
        hidden = torch.tanh(x @ core['weight'] + core['bias'])
        return hidden @ private['head'] + private['bias']

    support = torch.arange(6)
    query = torch.arange(6, 12)
    labels_s = torch.tensor([0, 1, 2, 0, 1, 2])
    labels_r = torch.tensor([2, 0, 1, 1, 2, 0])
    affinity = torch.zeros(6, 6, dtype=dtype)
    for i in range(5):
        affinity[i, i + 1] = affinity[i + 1, i] = 1
    module._validate(support, labels_s, query, labels_r, affinity, 3, phis, cfg, 'live')
    pairs = module._pair_blocks(labels_s, affinity, 3)

    def response(core, control='live', diagnostics=False):
        return module._private_response(core, phis, forward, support, labels_s,
                                        pairs, 3, cfg, control, diagnostics)

    def outer(core, control='live'):
        adapted, _ = response(core, control)
        return module._query_objective(core, adapted, forward, query, labels_r, cfg)

    actual, info = response(theta, diagnostics=True)
    stopped, _ = response(theta, 'stop_q')
    assert all(torch.equal(actual[m][name], stopped[m][name])
               for m in range(4) for name in phis[m])
    anchors = info['assignments']

    # Independent plain-autograd partial with Q held fixed checks ownership.
    manual = []
    for member, original in enumerate(phis):
        private = {name: value.clone().requires_grad_() for name, value in original.items()}
        logits = forward(theta, private)[support]
        loss = torch.nn.functional.cross_entropy(logits, labels_s)
        allocated = logits.new_zeros(())
        for q, pair in zip(anchors, pairs):
            gaps = logits[pair.nodes, pair.competitors] - logits[pair.nodes, pair.targets]
            allocated = allocated + (q[:, member] * torch.nn.functional.softplus(gaps)).sum()
        loss = loss + cfg.extra_margin * 4 / (6 * 2) * allocated
        gradients = torch.autograd.grad(loss, tuple(private.values()))
        manual.append({name: value.detach() - cfg.eta_private * gradient.detach()
                       for (name, value), gradient in zip(private.items(), gradients)})
    partial_error = max((actual[m][name] - manual[m][name]).abs().max().item()
                        for m in range(4) for name in phis[m])
    assert partial_error < 2e-13

    live_gradient = torch.func.grad(outer)(theta)
    stopped_gradient = torch.func.grad(lambda core: outer(core, 'stop_q'))(theta)
    direction = {name: torch.randn(value.shape, generator=generator, dtype=dtype)
                 for name, value in theta.items()}
    norm = sum(value.square().sum() for value in direction.values()).sqrt()
    direction = {name: value / norm for name, value in direction.items()}
    analytic = sum((live_gradient[name] * direction[name]).sum() for name in theta).item()
    finite_differences = []
    for epsilon in [1e-3, 3e-4, 1e-4]:
        plus = {name: value + epsilon * direction[name] for name, value in theta.items()}
        minus = {name: value - epsilon * direction[name] for name, value in theta.items()}
        numerical = ((outer(plus) - outer(minus)) / (2 * epsilon)).item()
        error = abs(numerical - analytic)
        assert error < 5e-8
        finite_differences.append(dict(epsilon=epsilon, numerical=numerical,
                                       analytic=analytic, absolute_error=error))

    # stop_q must equal differentiating a genuinely fixed-Q function.
    def fixed_q_outer(core):
        partial = torch.func.grad(module._main_loss, argnums=1)
        adapted = tuple(module._sgd(phi, partial(core, phi, anchors, member,
                                               forward, support, labels_s, pairs,
                                               3, cfg), cfg.eta_private)
                        for member, phi in enumerate(phis))
        return module._query_objective(core, adapted, forward, query, labels_r, cfg)

    anchored_gradient = torch.func.grad(fixed_q_outer)(theta)
    stopped_error = max((stopped_gradient[name] - anchored_gradient[name]).abs().max().item()
                        for name in theta)
    assert stopped_error < 2e-13
    q_chain_norm = sum((live_gradient[name] - stopped_gradient[name]).square().sum()
                       for name in theta).sqrt().item()
    return dict(status='PASS_CPU_MATH_FIXTURE_ONLY', torch_version=torch.__version__,
                dtype='float64', source_released=module.SOURCE_RELEASED,
                assignment_checks=assignment_checks, independent_private_partial_error=partial_error,
                live_core_directional_finite_differences=finite_differences,
                stopped_Q_vs_fixed_Q_gradient_error=stopped_error,
                observed_live_Q_chain_gradient_norm=q_chain_norm,
                native_model_qualified=False, scientific_data_access=False,
                model_fits=0, TEST_access=False, predictive_evidence=False,
                elapsed_seconds=time.monotonic() - started)
