def qualify_gradient_interface(logits_fn, theta0, train_rows, train_labels, seed):
    """Prospective precision amendment. Train labels and source inputs only.

    FP32 per-example CE is averaged in FP64 for finite-difference measurement.
    Original FP32 reduced-loss diagnostics remain reported beside it. Model,
    training gradients, initializer and all frozen directions/steps/tolerances
    remain unchanged. This version was designed after the disclosed Photo17
    failure and exact-state arithmetic diagnostic; it is not confirmatory.

    Three deterministic directions check JVP/VJP duality, centered-logit central
    differences and CE directional derivatives at two fixed epsilon values.
    Every diagnostic is retained. Unsupported AD is an explicit failure, never
    silently replaced by coordinate repulsion. Caller charges all passes/time.
    """
    import torch
    import torch.nn.functional as F
    generator = torch.Generator(device='cpu').manual_seed(seed)
    z, pullback = torch.func.vjp(logits_fn, theta0)
    residual = torch.zeros_like(z)
    residual[train_rows] = (z[train_rows].softmax(-1).detach()
                           - F.one_hot(train_labels, z.shape[1]).to(z.dtype))/train_rows.numel()
    g = pullback(residual)[0]
    records = []
    for index in range(3):
        d = torch.randn(theta0.shape, generator=generator, dtype=theta0.dtype).to(theta0.device)
        d = d/d.norm().clamp_min(1e-20)*(theta0.numel()**0.5)
        cotangent = torch.randn(z.shape, generator=generator, dtype=z.dtype).to(z.device)
        _, jvp = torch.func.jvp(logits_fn, (theta0,), (d,))
        vjp = pullback(cotangent)[0]
        lhs = float((jvp.double()*cotangent.double()).sum())
        rhs = float((vjp.double()*d.double()).sum())
        dual_error = abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-8)
        derivative = float((g.double()*d.double()).sum())
        checks = []
        for epsilon in (1e-3, 3e-4):
            with torch.no_grad():
                plus, minus = logits_fn(theta0+epsilon*d), logits_fn(theta0-epsilon*d)
                finite_difference = (plus-minus)/(2*epsilon)
                centered_fd = finite_difference-finite_difference.mean(-1, keepdim=True)
                centered_jvp = jvp-jvp.mean(-1, keepdim=True)
                logits_error = float((centered_fd-centered_jvp).double().norm()
                                     /centered_jvp.double().norm().clamp_min(1e-8))
                original_ce_fd = float((F.cross_entropy(plus[train_rows], train_labels)
                              - F.cross_entropy(minus[train_rows], train_labels))/(2*epsilon))
                plus_loss = F.cross_entropy(plus[train_rows], train_labels, reduction='none').double().mean()
                minus_loss = F.cross_entropy(minus[train_rows], train_labels, reduction='none').double().mean()
                ce_fd = float((plus_loss-minus_loss)/(2*epsilon))
                original_ce_error = abs(original_ce_fd-derivative)/max(abs(original_ce_fd), abs(derivative), 1e-3)
                ce_error = abs(ce_fd-derivative)/max(abs(ce_fd), abs(derivative), 1e-3)
            checks.append(dict(epsilon=epsilon, logits_relative_error=logits_error,
                               ce_directional_error=ce_error, analytic_ce_derivative=derivative,
                               finite_ce_derivative=ce_fd,
                               original_FP32_finite_ce_derivative=original_ce_fd,
                               original_FP32_ce_directional_error=original_ce_error,
                               FP64_reduced_plus_CE=float(plus_loss), FP64_reduced_minus_CE=float(minus_loss)))
        records.append(dict(direction=index, dual_relative_error=dual_error,
                            finite_differences=checks))
    # Finite differences can be noisy near ReLU kinks. One of TWO PREDECLARED
    # steps must agree for every direction; no adaptive epsilon/seed search.
    passed = all(row['dual_relative_error'] <= 2e-4 and
                 any(check['logits_relative_error'] <= 0.05 and
                     check['ce_directional_error'] <= 0.05 for check in row['finite_differences'])
                 for row in records)
    original_passed = all(row['dual_relative_error'] <= 2e-4 and
                 any(check['logits_relative_error'] <= 0.05 and
                     check['original_FP32_ce_directional_error'] <= 0.05 for check in row['finite_differences'])
                 for row in records)
    return dict(passed=passed, qualification_measurement='FP32_per_example_CE_FP64_mean_v1',
                original_FP32_measurement_would_pass=original_passed,
                original_training_gradient_unchanged=True,
                thresholds=dict(dual_rtol=2e-4, logits_FD_rtol=.05, CE_FD_rtol=.05, CE_denominator_floor=1e-3),
                records=records, vjp_forwards=1, vjp_calls=4,
                jvp_calls=3, finite_difference_forward_calls=12,
                source_labels='train_only', epsilons=[1e-3, 3e-4], random_directions=3)
