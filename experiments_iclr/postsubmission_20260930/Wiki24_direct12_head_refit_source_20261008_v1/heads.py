"""One FP64 affine kernel and one fixed effective-head objective."""
from common import require


def train_rms(np, H):
    require(H.ndim == 3 and np.isfinite(H).all(), 'Finite bank TRAIN features')
    rms = np.sqrt(np.mean(H * H, axis=(0, 1), dtype=np.float64))
    zero = rms == 0
    rms[zero] = 1
    return rms, np.flatnonzero(zero)


def prepare(np, a):
    H = a['H_train'].astype(np.float64)
    D = a['H_dev'].astype(np.float64)
    require(H.shape[1:] == (580, 512) and D.shape == (H.shape[0], 5274, 512), 'Complete frozen features')
    require(np.isfinite(H).all() and np.isfinite(D).all(), 'Finite features')
    rms, zero = train_rms(np, H)
    result = dict(H_train=H / rms, H_dev=D / rms, scale=rms, zero_RMS_indices=zero,
                  y_train=a['y_train'].copy(), y_dev=a['y_dev'].copy(),
                  train_ids=a['train_ids'].copy(), dev_ids=a['dev_ids'].copy())
    if 'W' in a:
        # Freeze multiplication order: scale shared W first, then r, then s.
        W = a['W'].astype(np.float64) * rms
        r, s, b = (a[key].astype(np.float64) for key in ('r', 's', 'bias'))
        A = W[None] * r[:, None, :] * s[:, :, None]
        result.update(W=W, r=r, s=s, A0=A.copy(), bias=b)
    else:
        result.update(A0=a['A'].astype(np.float64) * rms, bias=a['bias'].astype(np.float64))
    require(all(np.isfinite(v).all() for v in result.values()), 'Finite RMS transform and start')
    return result


def parameters(torch, a, condition):
    names = ('W', 'r', 's', 'bias') if condition == 'BE_factor_refit' else ('A0', 'bias')
    return {key: torch.tensor(a[key], dtype=torch.float64, device='cpu', requires_grad=True) for key in names}


def effective(p):
    return p['W'][None] * p['r'][:, None, :] * p['s'][:, :, None] if 'W' in p else p['A0']


def logits(torch, H, A, bias):
    require(H.device.type == A.device.type == bias.device.type == 'cpu', 'CPU-only affine kernel')
    return torch.matmul(H, A.transpose(-1, -2)) + (bias[None, None] if bias.ndim == 1 else bias[:, None])


def objective(torch, H, y, A, bias):
    scores = logits(torch, H, A, bias)
    require(bool(torch.isfinite(scores).all()), 'Nonfinite affine logits')
    logp = scores.log_softmax(-1)
    ce = -logp.gather(-1, y[None, :, None].expand(A.shape[0], -1, 1)).mean()
    centered = A - A.mean(dim=1, keepdim=True)
    penalty = .001 / 2 * centered.square().sum(dim=(1, 2)).mean()
    return ce + penalty, ce, penalty


def paired_start(torch, prepared):
    factor = parameters(torch, prepared, 'BE_factor_refit')
    full = parameters(torch, prepared, 'BE_full_refit')
    require(torch.allclose(effective(factor), effective(full), rtol=1e-12, atol=1e-12)
            and torch.equal(factor['bias'], full['bias']), 'Same paired effective start/bias')
    for role in ('H_train', 'H_dev'):
        H = torch.from_numpy(prepared[role])
        first = logits(torch, H, effective(factor), factor['bias']).softmax(-1)
        second = logits(torch, H, effective(full), full['bias']).softmax(-1)
        require(torch.allclose(first, second, rtol=1e-12, atol=1e-12), 'Actual paired initial probabilities differ')
    return dict(passed=True, effective_weights_equal=True, common_bias_equal=True,
                TRAIN_and_development_initial_probabilities_equal=True,
                probability_rtol=1e-12, probability_atol=1e-12,
                original_native_FP32_parity_claimed=False)


def fit(torch, np, a, condition, deadline):
    import time
    p = parameters(torch, a, condition)
    H, y = torch.from_numpy(a['H_train']), torch.from_numpy(a['y_train'])
    calls = 0
    optimizer = torch.optim.LBFGS(list(p.values()), lr=1., max_iter=1000, max_eval=50000,
        tolerance_grad=1e-6, tolerance_change=0., history_size=10, line_search_fn='strong_wolfe')

    def closure():
        nonlocal calls
        require(time.monotonic() < deadline, 'Finite active deadline expired')
        calls += 1
        require(calls <= 50000, 'Fixed closure evaluation bound expired')
        optimizer.zero_grad(set_to_none=True)
        require(all(bool(torch.isfinite(v).all()) for v in p.values()), 'Nonfinite solver parameters')
        total, _, _ = objective(torch, H, y, effective(p), p['bias'])
        require(bool(torch.isfinite(total)), 'Nonfinite TRAIN objective')
        total.backward()
        require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in p.values()), 'Missing/nonfinite raw gradient')
        return total

    optimizer.step(closure)  # One optimizer trajectory; no external restart or selector.
    final = closure()
    grad_inf = max(float(v.grad.abs().max()) for v in p.values())
    A = effective(p).detach()
    _, ce, penalty = objective(torch, H, y, A, p['bias'].detach())
    free = A.clone().requires_grad_(True)
    free_total, _, _ = objective(torch, H, y, free, p['bias'].detach())
    effective_grad = torch.autograd.grad(free_total, free)[0]
    state = optimizer.state[next(iter(p.values()))]
    status = 'converged' if grad_inf <= 1e-6 else 'finite_nonconverged'
    endpoint = {key: value.detach().numpy().copy() for key, value in p.items()}
    endpoint['A'] = A.numpy().copy()
    receipt = dict(status=status, diagnostic_inconclusive_for_factor_capacity=status != 'converged' or condition == 'BE_factor_refit',
        TRAIN_objective=float(final), TRAIN_CE=float(ce), effective_L2_penalty=float(penalty),
        raw_parameter_gradient_infinity=grad_inf, effective_A_gradient_infinity=float(effective_grad.abs().max()),
        LBFGS_iterations=int(state.get('n_iter', 0)), optimizer_function_evaluations=int(state.get('func_evals', 0)),
        actual_closure_calls_including_final_check=calls, restarts=0, development_selector=False,
        optimizer='torch.optim.LBFGS', line_search='strong_wolfe', max_iter=1000, max_eval=50000,
        tolerance_grad=1e-6, tolerance_change=0., history_size=10, gradient_check='raw final infinity norm')
    return endpoint, receipt
